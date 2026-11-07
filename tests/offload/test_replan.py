"""preload(): the memory a model can count on is predicted from the plan of the running stage, and the plan is fitted to the
VRAM that is really free when the model is entered."""
import importlib

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, parse_bytes, preload  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import BlockPlan, ram_overhead, refit_gpu_blocks, vram_kept_back  # noqa: E402

stage_module = importlib.import_module("trash_panda.offload.stage")

BLOCK = 4224                                         # one block of the tiny model, in float32
HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
PLAN = dict(compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0)
PRE = dict(PLAN, max_vram="10GB", max_ram="1GB", resident_blocks=2)        # blocks 0 and 1 on the GPU, 2 and 3 in RAM


def _cuda(available):
    from types import SimpleNamespace
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available), nn=torch.nn)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Run on the CPU whatever the machine has, with nothing remembered at the start."""
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", _cuda(False))
    monkeypatch.setattr(stage_module, "_activations", {})


# -- the memory after the release is predicted from the plan of the running stage

def _running_plan(gpu=5000, ram=7000, other=300):
    running = Plan.manual(compute_dtype="fp32", resident=["a."], streamed=["b."])
    running.blocks[0].stored_bytes, running.blocks[1].stored_bytes, running.other_bytes = gpu, ram, other
    return running


def _machine(vram_free, ram_available):
    return Hardware(vram_total=10 ** 10, vram_free=vram_free, ram_total=16 * 10 ** 9, ram_available=ram_available)


def test_the_memory_after_the_release_is_what_is_free_plus_what_the_running_stage_holds():
    vram, ram = stage_module._expected_after(_running_plan(), _machine(10_000, 20_000), None, None)
    assert vram == 10_000 + 300 + 5000                              # its weights outside the blocks and its GPU blocks
    assert ram == 20_000 + 7000                                     # its RAM tier


def test_a_figure_that_is_given_wins_over_the_prediction():
    vram, ram = stage_module._expected_after(_running_plan(), _machine(10_000, 20_000), "1KB", "2KB")
    assert (vram, ram) == (1000, 2000)
    assert stage_module._expected_after(None, _machine(10_000, 20_000), "1KB", None) == (1000, None)


def test_preload_plans_for_the_predicted_memory(tiny_checkpoint):
    _, path = tiny_checkpoint
    hw = _machine(10 ** 9, 10 ** 9)
    pending = preload(meta_model, path, running=_running_plan(), **dict(PLAN, hardware=hw), resident_blocks=2)
    assert pending._vram_assumed == 10 ** 9 + 300 + 5000
    assert pending.plan.ram_budget + ram_overhead(pending.plan, hw, 0) == 10 ** 9 + 7000     # less the pinned buffers
    pending.cancel()


# -- the plan is fitted to the VRAM that is really free when the model is entered

def _entered_with(monkeypatch, tiny_checkpoint, available, **kw):
    ref, path = tiny_checkpoint
    monkeypatch.setattr(stage_module, "_available_now", lambda device: available[0] if isinstance(available, list) else available)
    pending = preload(meta_model, path, **dict(PRE, **kw))
    assert pending._done.wait(10)
    return ref, pending


def _room_for(pending, blocks):
    """The free VRAM at which exactly `blocks` of the blocks that are not on the GPU yet fit."""
    p = pending.plan
    return p.headroom_bytes + p.stream_buffer_bytes + p.lookahead * BLOCK + blocks * BLOCK + 10


def test_blocks_that_do_not_fit_go_to_the_ram_tier(tiny_checkpoint, monkeypatch):
    x = torch.randn(3, 16)
    ref, pending = _entered_with(monkeypatch, tiny_checkpoint, 0)
    with pytest.warns(RuntimeWarning, match="2 blocks planned for the GPU were put in the RAM tier"):
        with pending as (model, p, _handle):
            assert p.prefixes("gpu") == ()
            assert p.prefixes("ram") == ("blocks.0.", "blocks.1.", "blocks.2.", "blocks.3.")
            assert model._trash_panda_state.deferred is None
            assert any("RAM tier" in w for w in p.warnings)
            assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_the_blocks_that_fit_stay_on_the_gpu_and_the_rest_go_to_ram(tiny_checkpoint, monkeypatch):
    x = torch.randn(3, 16)
    free = [0]
    ref, pending = _entered_with(monkeypatch, tiny_checkpoint, free)
    free[0] = _room_for(pending, 1)
    with pytest.warns(RuntimeWarning, match="1 blocks"):
        with pending as (model, p, _handle):
            assert p.prefixes("gpu") == ("blocks.0.",)
            assert "blocks.1." in p.prefixes("ram")
            assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_extra_room_takes_ram_blocks_that_are_in_memory(tiny_checkpoint, monkeypatch):
    x = torch.randn(3, 16)
    ref, pending = _entered_with(monkeypatch, tiny_checkpoint, 10 ** 12)
    with pending as (model, p, _handle):
        assert p.prefixes("gpu") == ("blocks.0.", "blocks.1.", "blocks.2.", "blocks.3.")
        assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_blocks_that_stay_mapped_are_not_moved_to_the_gpu(tiny_checkpoint, monkeypatch):
    hw = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=BLOCK + 100)
    _, pending = _entered_with(monkeypatch, tiny_checkpoint, 10 ** 12, hardware=hw)
    assert pending._memory == ("blocks.2.",)
    with pending as (_model, p, _handle):
        assert p.prefixes("gpu") == ("blocks.0.", "blocks.1.", "blocks.2.")
        assert p.prefixes("ram") == ("blocks.3.",)


def test_the_headroom_is_the_one_the_plan_kept_even_when_the_vram_budget_was_clamped_to_zero(tiny_checkpoint, monkeypatch):
    # the plan counted on 100 KB, less than its default headroom of 1.5 GB, so its VRAM budget is 0
    _, pending = _entered_with(monkeypatch, tiny_checkpoint, 10 ** 9, max_vram="100KB")
    assert pending.plan.vram_budget == 0 and pending.plan.headroom_bytes == parse_bytes("1.5GB")
    with pytest.warns(RuntimeWarning, match="2 blocks"):          # 1 GB free is less than the 1.5 GB kept back
        with pending as (_model, p, _handle):
            assert p.prefixes("gpu") == ()


def test_the_plan_stands_when_the_free_vram_is_not_known(tiny_checkpoint, monkeypatch):
    _, pending = _entered_with(monkeypatch, tiny_checkpoint, None)
    with pending as (_model, p, _handle):
        assert p.prefixes("gpu") == ("blocks.0.", "blocks.1.")


def test_blocks_loaded_early_are_not_demoted(tiny_checkpoint, monkeypatch):
    x = torch.randn(3, 16)
    ref, pending = _entered_with(monkeypatch, tiny_checkpoint, 0)
    state = pending.model._trash_panda_state
    with state.checkpoint:
        state.deferred.load("blocks.0.")
    with pytest.warns(RuntimeWarning, match="1 blocks"):
        with pending as (model, p, _handle):
            assert "blocks.0." in p.prefixes("gpu")
            assert "blocks.1." in p.prefixes("ram")
            assert torch.allclose(model(x), ref(x), atol=1e-6)


# -- the decision itself, with the sizes written out

def _blocks(*tiers):
    plan = Plan.manual(compute_dtype="fp32")
    plan.blocks = [BlockPlan(f"b{i}.", t, 100, 100) for i, t in enumerate(tiers)]
    return plan


def test_refit_demotes_what_does_not_fit_and_keeps_a_later_block_that_does():
    p = _blocks("gpu", "gpu", "ram")
    p.blocks[1].stored_bytes = 50                                   # the second block is smaller than the first
    p.stream_buffer_bytes, p.lookahead = 0, 0
    assert refit_gpu_blocks(p, p.blocks, 60, 0, {"b0.", "b1."}, set()) == ([], ["b0."])


def test_refit_gives_the_left_over_room_to_the_ram_blocks_in_run_order_up_to_the_first_miss():
    p = _blocks("gpu", "ram", "ram", "ram")
    p.stream_buffer_bytes, p.lookahead = 0, 0
    p.blocks[2].stored_bytes = 500
    assert refit_gpu_blocks(p, p.blocks, 260, 0, {"b0."}, {"b1.", "b2.", "b3."}) == (["b1."], [])    # b3 is not tried


def test_refit_keeps_room_for_the_stream_buffer_and_the_staged_blocks():
    p = _blocks("gpu", "ram")
    p.stream_buffer_bytes, p.lookahead = 40, 2                      # 40 + 2 * 100 = 240 kept back
    assert refit_gpu_blocks(p, p.blocks, 339, 0, {"b0."}, set()) == ([], ["b0."])
    assert refit_gpu_blocks(p, p.blocks, 340, 0, {"b0."}, set()) == ([], [])


def test_refit_keeps_the_headroom():
    p = _blocks("gpu")
    p.stream_buffer_bytes, p.lookahead = 0, 0
    assert refit_gpu_blocks(p, p.blocks, 150, 60, {"b0."}, set()) == ([], ["b0."])
    assert refit_gpu_blocks(p, p.blocks, 160, 60, {"b0."}, set()) == ([], [])


def test_refit_keeps_nothing_back_for_streaming_when_everything_ends_on_the_gpu():
    p = _blocks("gpu", "gpu")
    p.stream_buffer_bytes, p.lookahead = 40, 2
    assert refit_gpu_blocks(p, p.blocks, 200, 0, {"b0.", "b1."}, set()) == ([], [])             # no expansion: no buffer
    # one block would be left out, so the buffer and the staged blocks (240) are needed, and then none fits
    assert refit_gpu_blocks(p, p.blocks, 199, 0, {"b0.", "b1."}, set()) == ([], ["b0.", "b1."])
    assert refit_gpu_blocks(p, p.blocks, 339, 0, {"b0.", "b1."}, set()) == ([], [])


# -- what the plan keeps free is also what the early loader keeps free

def test_the_vram_kept_back_is_the_headroom_and_the_room_for_streaming():
    p = _blocks("gpu", "ram")
    p.stream_buffer_bytes, p.lookahead, p.headroom_bytes = 40, 2, 1000
    assert vram_kept_back(p) == 1000 + 40 + 2 * 100                 # blocks are outside the GPU: buffer and staged blocks
    q = _blocks("gpu", "gpu")
    q.stream_buffer_bytes, q.lookahead, q.headroom_bytes = 40, 2, 1000
    assert vram_kept_back(q) == 1000                                # all on the GPU, no expansion: nothing for streaming
    q.blocks[0].expanded_bytes = 200
    assert vram_kept_back(q) == 1000 + 40


def test_the_early_loader_leaves_the_kept_back_vram_free(monkeypatch):
    from trash_panda.offload import early
    total = 10 ** 10
    monkeypatch.setattr(early, "_vram_available", lambda device: (5 * 10 ** 9, total))
    loader = early._EarlyLoader(None, None, [], {}, None, "cuda")
    assert loader._room(4 * 10 ** 9)                                # 1 GB left is more than the 5% margin
    loader = early._EarlyLoader(None, None, [], {}, None, "cuda", keep=2 * 10 ** 9)
    assert not loader._room(4 * 10 ** 9)                            # but less than what the plan keeps back
    assert loader._room(3 * 10 ** 9)
