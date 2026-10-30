import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from .conftest import strip_net  # noqa: E402
from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import stage  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402

# trash_panda.offload.stage is the function, so the module has to come from importlib
stage_module = importlib.import_module("trash_panda.offload.stage")

HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
PLAN = dict(compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0)


def _cuda(available):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available), nn=torch.nn)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Run on the CPU whatever the machine has, with no CUDA seen by stage(), and nothing remembered at the start."""
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", _cuda(False))
    monkeypatch.setattr(stage_module, "_activations", {})


@pytest.fixture
def measured(monkeypatch):
    """Pretend there is a GPU and that every block peaked at 123 bytes. Returns the calls made to _peak_since."""
    calls = []

    def peak_since(before, plan=None, device=None):
        calls.append((before, plan))
        return 123

    monkeypatch.setattr(stage_module, "torch", _cuda(True))
    monkeypatch.setattr(stage_module, "_start_peak", lambda device=None: 7)
    monkeypatch.setattr(stage_module, "_peak_since", peak_since)
    return calls


@pytest.fixture
def planned(monkeypatch):
    """The activation_vram that each plan() call received, in order."""
    seen = []
    real = stage_module.plan

    def spy(model, path, **kw):
        seen.append(kw.get("activation_vram"))
        return real(model, path, **kw)

    monkeypatch.setattr(stage_module, "plan", spy)
    return seen


def test_the_model_is_loaded_inside_the_block_and_released_after(tiny_checkpoint):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (model, p):
        assert not any(t.is_meta for t in model.parameters())
        assert model(torch.zeros(1, 16)).shape == (1, 4)
    assert all(t.is_meta for t in model.parameters())


def test_the_model_is_released_when_the_block_raises(tiny_checkpoint):
    _, path = tiny_checkpoint
    held = []
    with pytest.raises(RuntimeError, match="boom"):
        with stage(meta_model, path, **PLAN) as (model, _):
            held.append(model)
            raise RuntimeError("boom")
    assert all(t.is_meta for t in held[0].parameters())


def test_the_peak_is_remembered_and_used_by_the_next_plan(tiny_checkpoint, measured, planned):
    _, path = tiny_checkpoint
    with stage(meta_model, path, key="a", **PLAN):
        pass
    with stage(meta_model, path, key="a", **PLAN):
        pass
    assert planned == [None, 123]


def test_the_peak_is_measured_from_the_start_of_the_block_with_the_plan(tiny_checkpoint, measured):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (_, p):
        pass
    assert measured == [(7, p)]


def test_a_value_passed_to_stage_wins_over_the_remembered_one(tiny_checkpoint, measured, planned):
    _, path = tiny_checkpoint
    with stage(meta_model, path, key="a", **PLAN):
        pass
    with stage(meta_model, path, key="a", activation_vram="2GB", **PLAN):
        pass
    assert planned == [None, "2GB"]


def test_each_key_is_remembered_on_its_own(tiny_checkpoint, measured, planned):
    _, path = tiny_checkpoint
    with stage(meta_model, path, key="a", **PLAN):
        pass
    with stage(meta_model, path, key="b", **PLAN):
        pass
    assert planned == [None, None]


def test_the_checkpoint_path_is_the_default_key(tiny_checkpoint, measured, planned):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN):
        pass
    with stage(meta_model, path, **PLAN):
        pass
    assert planned == [None, 123]
    assert list(stage_module._activations) == [str(path)]


def test_nothing_is_remembered_when_the_block_raises(tiny_checkpoint, measured):
    _, path = tiny_checkpoint
    with pytest.raises(RuntimeError):
        with stage(meta_model, path, **PLAN):
            raise RuntimeError("boom")
    assert stage_module._activations == {}
    assert measured == []


def test_key_map_reaches_both_the_planner_and_the_loader(renamed_checkpoint):
    ref, path = renamed_checkpoint
    x = torch.randn(1, 16)
    with stage(meta_model, path, key_map=strip_net, **PLAN) as (model, p):
        assert all(b.stored_bytes > 0 for b in p.blocks)
        assert torch.allclose(model(x), ref(x))


def test_skip_prefixes_reach_both_the_planner_and_the_loader(tiny_checkpoint):
    _, path = tiny_checkpoint
    with stage(meta_model, path, skip_prefixes=("blocks.3.", "head."), **PLAN) as (model, p):
        assert len(p.blocks) == 3
        assert model.head.weight.is_meta
        assert not model.blocks[0].lin1.weight.is_meta


def test_nothing_is_measured_without_cuda(tiny_checkpoint, planned):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN):
        pass
    assert stage_module._activations == {}
