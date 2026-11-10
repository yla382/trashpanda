import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Coordinator, PendingStage, Step, preload, stage  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.hooks import _PinnedPool  # noqa: E402

stage_module = importlib.import_module("trash_panda.offload.stage")

HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 ** 10)
PLAN = dict(compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0)
OPTIONS = dict(PLAN, max_vram="1GB", max_ram="1GB", resident_blocks=2)      # blocks 0 and 1 on the GPU, 2 and 3 in RAM


def _cuda(available):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available), nn=torch.nn)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Run on the CPU whatever the machine has, with no CUDA seen by stage(), and nothing remembered at the start."""
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", _cuda(False))
    monkeypatch.setattr(stage_module, "_activations", {})


def _coordinator(path, *names, **options):
    return Coordinator(*(Step(n, meta_model, path, **dict(OPTIONS, **options)) for n in names))


# -- running the steps ------------------------------------------------------------------------------------------------

def test_every_step_gives_the_same_output_as_the_reference(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    co = _coordinator(path, "a", "b", "c")
    with co:
        for name in ("a", "b", "c"):
            with co.stage(name) as (model, p, handle):
                assert torch.allclose(model(x), ref(x), atol=1e-6)
            assert all(t.is_meta for t in model.parameters())          # released when its block ended


def test_a_step_can_run_again(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    co = _coordinator(path, "a", "b")
    with co:
        for name in ("a", "b", "a", "b"):
            with co.stage(name) as (model, _p, _handle):
                assert torch.allclose(model(x), ref(x), atol=1e-6)


# -- preloading the next step ---------------------------------------------------------------------------------------

def test_a_stage_starts_loading_the_next_step_and_the_next_stage_takes_it(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a"):
            name, pending = co._pending
            assert name == "b" and isinstance(pending, PendingStage)
        with co.stage("b") as (model, p, handle):
            assert p is pending.plan and handle is pending.handle and model is pending.model
            assert torch.allclose(model(x), ref(x), atol=1e-6)
            assert co._pending is None                                 # nothing follows the last step


def test_the_next_step_is_preloaded_with_the_plan_of_the_running_one(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    given = {}
    real = stage_module.preload

    def spy(*args, **kw):
        given.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr("trash_panda.offload.coordinator.preload", spy)
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a") as (_model, p, handle):
            assert given["running"] is p and given["pool"] is co.pool
            assert given["after"] is handle._prefetcher.releases        # the queue the final pass announces on


@pytest.fixture
def preload_calls(monkeypatch):
    """The keyword arguments of every preload() call the Coordinator makes."""
    calls = []
    real = stage_module.preload

    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr("trash_panda.offload.coordinator.preload", spy)
    return calls


@pytest.mark.parametrize("flags, ram, queue_given", [
    ({}, True, True),                                               # both on by default
    ({"ram_preload": False}, False, True),                          # RAM tier stays mapped, GPU blocks still load early
    ({"vram_preload": False}, True, False),                         # RAM tier read, GPU blocks load at entry
])
def test_the_flags_say_what_is_prepared_for_the_next_step(tiny_checkpoint, preload_calls, flags, ram, queue_given):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a", **flags) as (_model, _p, handle):
            assert co._pending[0] == "b"
            assert len(preload_calls) == 1 and preload_calls[0]["ram_preload"] is ram
            after = preload_calls[0]["after"]
            if queue_given:
                assert after is not None and after is handle._prefetcher.releases    # the queue of the final pass
            else:
                assert after is None and handle._prefetcher.releases is None         # nothing even subscribed


def test_with_both_flags_off_nothing_is_prepared_and_the_next_step_loads_normally(tiny_checkpoint, preload_calls):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a", ram_preload=False, vram_preload=False):
            assert co._pending is None
        with co.stage("b") as (model, _p, _handle):
            assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert preload_calls == []                                      # "b" was the last step, so none for it either


def test_then_none_prepares_nothing_whatever_the_flags_say(tiny_checkpoint, preload_calls):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a", then=None, ram_preload=True, vram_preload=True):
            assert co._pending is None
    assert preload_calls == []


def test_the_flags_apply_to_the_step_that_then_names(tiny_checkpoint, preload_calls):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b", "c")
    with co:
        with co.stage("a", then="c", ram_preload=False):
            assert co._pending[0] == "c" and preload_calls[0]["ram_preload"] is False


def test_then_changes_which_step_is_preloaded(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b", "c")
    with co:
        with co.stage("a", then="c"):
            assert co._pending[0] == "c"
        with co.stage("c", then=None):
            assert co._pending is None
        with co.stage("a", then=None):
            assert co._pending is None


def test_asking_for_another_step_than_the_preloaded_one_gives_that_one_up(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a"):
            _, first = co._pending                                      # "b" is loading
        with co.stage("a") as (model, _p, _handle):                     # but "a" is asked for again
            assert first._used and all(t.is_meta for t in first.model.parameters())
            assert co._pending[1] is not first                          # a new "b" is loading now
            assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_leaving_the_coordinator_gives_up_a_model_nobody_asked_for(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a"):
            _, pending = co._pending
        assert co.pool is not None
    assert pending._used and all(t.is_meta for t in pending.model.parameters())
    assert co._pending is None and co.pool is None


def test_leaving_the_coordinator_after_an_error_still_gives_everything_up(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with pytest.raises(ZeroDivisionError):
        with co:
            with co.stage("a"):
                _, pending = co._pending
                1 / 0
    assert pending._used and co._pending is None and co.pool is None


# -- the pool ---------------------------------------------------------------------------------------------------------

def test_every_stage_gets_the_pool_of_the_coordinator(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    pools = []
    real = stage_module.install_offload

    def spy(model, *args, **kw):
        pools.append(kw.get("pool"))
        return real(model, *args, **kw)

    monkeypatch.setattr(stage_module, "install_offload", spy)
    co = _coordinator(path, "a", "b")
    with co:
        pool = co.pool
        assert isinstance(pool, _PinnedPool)
        with co.stage("a"):                                             # loaded by stage()
            pass
        with co.stage("b"):                                             # loaded by preload()
            pass
    assert pools == [pool, pool]


def test_stage_and_preload_tell_the_planner_what_the_pool_holds(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    seen = []
    real = stage_module.plan
    monkeypatch.setattr(stage_module, "plan", lambda *a, **kw: seen.append(kw.get("pinned_held")) or real(*a, **kw))
    pool = _PinnedPool(2)
    pool.buffers[0] = torch.empty(1000, dtype=torch.uint8)
    with stage(meta_model, path, pool=pool, **PLAN):
        pass
    preload(meta_model, path, pool=pool, **OPTIONS).cancel()
    with stage(meta_model, path, **PLAN):
        pass
    assert seen == [[1024, 0], [1024, 0], None]                         # the size of each buffer; none without a pool


def test_a_pinned_held_that_the_caller_gives_wins(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    seen = []
    real = stage_module.plan
    monkeypatch.setattr(stage_module, "plan", lambda *a, **kw: seen.append(kw.get("pinned_held")) or real(*a, **kw))
    pool = _PinnedPool(2)
    pool.buffers[0] = torch.empty(1000, dtype=torch.uint8)
    with stage(meta_model, path, pool=pool, pinned_held=[7], **PLAN):
        pass
    assert seen == [[7]]


def test_the_pinned_memory_is_given_back_when_the_coordinator_ends(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    calls = []
    monkeypatch.setattr("trash_panda.offload.coordinator._empty_host_cache", lambda: calls.append(1))
    co = _coordinator(path, "a")
    with co:
        with co.stage("a"):
            pass
        assert calls == []                                              # kept between the stages
    assert calls == [1]


def test_a_pool_that_a_stage_enlarged_goes_back_to_the_configured_size_when_the_stage_ends(tiny_checkpoint, monkeypatch):
    from trash_panda.offload import config
    _, path = tiny_checkpoint
    calls = []
    monkeypatch.setattr("trash_panda.offload.coordinator._empty_host_cache", lambda: calls.append(1))
    co = _coordinator(path, "a", "b")
    wanted = config.current().pinned_buffers
    with co:
        with co.stage("a"):
            co.pool.grow(wanted + 2)                                    # what the look-ahead tuner does
            for i in range(wanted + 2):
                co.pool.buffers[i] = torch.empty(100, dtype=torch.uint8)
            assert len(co.pool.buffers) == wanted + 2                   # kept while the stage runs
        assert len(co.pool.buffers) == len(co.pool.events) == wanted
        assert calls == [1]                                             # and the memory is given back at once
        with co.stage("b"):
            pass
        assert calls == [1]                                             # nothing to give back the second time


# -- misuse -----------------------------------------------------------------------------------------------------------

def test_a_stage_needs_the_with_block(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a")
    with pytest.raises(RuntimeError, match="with"):
        with co.stage("a"):
            pytest.fail("the block must not run")


def test_a_stage_cannot_start_inside_another(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with co:
        with co.stage("a"):
            with pytest.raises(RuntimeError, match="running"):
                with co.stage("b"):
                    pytest.fail("the block must not run")
        with co.stage("b"):                                             # and the coordinator is usable afterwards
            pass


def test_a_failed_stage_leaves_the_coordinator_usable(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a", "b")
    with co:
        with pytest.raises(ZeroDivisionError):
            with co.stage("a"):
                1 / 0
        with co.stage("b") as (model, _p, _handle):                     # the "b" that "a" had started loading
            model(torch.randn(1, 16))


def test_an_unknown_step_is_named_in_the_error(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a")
    with co:
        with pytest.raises(KeyError, match="'x'"):
            with co.stage("x"):
                pytest.fail("the block must not run")
        with pytest.raises(KeyError, match="'x'"):
            with co.stage("a", then="x"):
                pytest.fail("the block must not run")


def test_the_steps_have_to_be_given_and_to_have_different_names(tiny_checkpoint):
    _, path = tiny_checkpoint
    with pytest.raises(ValueError, match="at least one"):
        Coordinator()
    with pytest.raises(ValueError, match="different"):
        _coordinator(path, "a", "a")


@pytest.mark.parametrize("name", ["pool", "running", "after"])
def test_a_step_cannot_set_what_the_coordinator_sets(tiny_checkpoint, name):
    _, path = tiny_checkpoint
    with pytest.raises(TypeError, match=name):
        Step("a", meta_model, path, **{name: None})


def test_the_coordinator_cannot_be_entered_twice(tiny_checkpoint):
    _, path = tiny_checkpoint
    co = _coordinator(path, "a")
    with co:
        with pytest.raises(RuntimeError, match="already"):
            co.__enter__()
