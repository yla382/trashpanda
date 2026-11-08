"""The look-ahead that follows the passes (adaptive_lookahead, off by default)."""
import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, hooks, install_offload, stream_load  # noqa: E402
from trash_panda.offload.depth import DepthTuner  # noqa: E402


# -- the decision, with the pass figures written out

def feed(tuner, passes):
    """The depth after each (took, waited) pass."""
    return [tuner.update(took, waited) for took, waited in passes]


def tuner(**kw):
    return DepthTuner(2, 4, warm=0, window=1, **kw)


def test_it_stops_at_once_when_the_main_thread_hardly_waits_and_there_is_nothing_to_give_back():
    t = DepthTuner(1, 4, warm=0, window=1)
    assert feed(t, [(1.0, 0.05), (1.0, 0.05)]) == [1, 1]            # 5% of the pass is below the 10% that is worth a try
    assert t.done


def test_without_waiting_the_shallowest_look_ahead_is_tried_and_kept_if_it_is_not_slower():
    t = tuner()
    # depth 2 with a 1.00 s pass and no waiting -> try 1 (one pass is left out) -> 1.01 s is within the 2% -> kept
    assert feed(t, [(1.00, 0.01), (9, 9), (1.01, 0.0), (1.0, 0.0)]) == [1, 1, 1, 1]
    assert t.done


def test_the_shallowest_look_ahead_is_given_up_if_it_is_slower():
    t = tuner()
    assert feed(t, [(1.00, 0.01), (9, 9), (1.05, 0.0), (1.0, 0.0)]) == [1, 1, 2, 2]     # 5% slower is more than the 2%
    assert t.done


def test_the_limit_of_the_slowdown_that_is_accepted_is_two_percent():
    assert feed(tuner(), [(1.00, 0.0), (9, 9), (1.019, 0.0)])[-1] == 1
    assert feed(tuner(), [(1.00, 0.0), (9, 9), (1.021, 0.0)])[-1] == 2


def test_a_deeper_look_ahead_that_pays_off_is_kept_and_the_climb_goes_on_up_to_the_maximum():
    t = tuner()
    # depth 2: 1.00 s -> try 3 (one pass is left out) 0.90 s, 10% faster, kept -> try 4 (one left out) 0.80 s -> done
    assert feed(t, [(1.00, 0.5), (9, 9), (0.90, 0.4), (9, 9), (0.80, 0.3), (0.80, 0.3)]) == [3, 3, 4, 4, 4, 4]
    assert t.done and t.base == pytest.approx(0.80)


def test_a_deeper_look_ahead_that_does_not_pay_off_is_given_up():
    t = tuner()
    assert feed(t, [(1.00, 0.5), (9, 9), (0.99, 0.5), (0.5, 0.0)]) == [3, 3, 2, 2]     # 1% is below the 3% gain asked for
    assert t.done


def test_it_stops_climbing_when_the_waiting_is_gone_even_if_the_pass_got_faster():
    t = tuner()
    assert feed(t, [(1.0, 0.5), (9, 9), (0.8, 0.05), (0.7, 0.5)]) == [3, 3, 3, 3]
    assert t.done


def test_it_never_changes_the_depth_again_once_it_has_stopped():
    t = DepthTuner(1, 4, warm=0, window=1)
    feed(t, [(1.0, 0.05)])
    assert feed(t, [(1.0, 0.9), (0.1, 0.9)]) == [1, 1]


def test_the_cold_passes_and_the_pass_after_a_change_are_left_out():
    t = DepthTuner(2, 4, warm=2, window=1)
    # two cold passes, then the baseline (waiting), then one pass that is left out after the change, then the trial,
    # which is faster, so it is kept and the next depth is tried
    assert feed(t, [(9.0, 9.0), (9.0, 9.0), (1.0, 0.5), (5.0, 0.0), (0.8, 0.4)]) == [2, 2, 3, 3, 4]


def test_the_median_of_a_window_decides_not_one_odd_pass():
    t = DepthTuner(2, 4, warm=0, window=3)
    assert feed(t, [(1.0, 0.5)] * 3)[-1] == 3                      # baseline of three passes, then the depth goes up
    out = feed(t, [(9.0, 0.0), (0.8, 0.4), (0.8, 0.4), (0.8, 0.4)])
    assert out[-1] == 4                                             # the skipped pass was the odd one; the median is 0.8


def test_the_start_depth_above_the_maximum_is_the_maximum():
    t = DepthTuner(6, 4, warm=0, window=1)
    assert feed(t, [(1.0, 0.5)]) == [6] and t.done


# -- the pinned pool

def test_the_pinned_pool_grows_and_does_not_shrink():
    pool = hooks._PinnedPool(2)
    pool.grow(4)
    assert len(pool.buffers) == len(pool.events) == 4
    pool.grow(3)
    assert len(pool.buffers) == 4


# -- the prefetcher applies the answer

def _prefetcher(monkeypatch, free, depth=2, largest=100 << 20, total=12 << 30):
    pre = object.__new__(hooks._Prefetcher)
    pre.device = torch.device("cuda")
    pre.lookahead = depth
    pre.largest = largest
    pre.pool = hooks._PinnedPool(2)
    pre.tuner = DepthTuner(depth, 4, warm=0, window=1)
    monkeypatch.setattr(hooks.torch.cuda, "mem_get_info", lambda device=None: (free, total))
    return pre


def test_a_deeper_look_ahead_is_taken_when_there_is_free_vram_for_it(monkeypatch):
    pre = _prefetcher(monkeypatch, free=4 << 30)
    pre._retune(1.0, 0.5)
    assert pre.lookahead == 3 and len(pre.pool.buffers) >= 3


def test_it_is_not_taken_when_the_free_vram_would_fall_below_the_margin(monkeypatch):
    pre = _prefetcher(monkeypatch, free=(600 << 20) + (50 << 20))      # 5% of 12 GB is 614 MB: 650 MB - 100 MB is not enough
    pre._retune(1.0, 0.5)
    assert pre.lookahead == 2 and len(pre.pool.buffers) == 2
    assert pre.tuner.done and pre.tuner.depth == 2                  # and it does not ask again


def test_a_shallower_look_ahead_needs_no_check_and_keeps_the_buffers(monkeypatch):
    pre = _prefetcher(monkeypatch, free=4 << 30)
    pre._retune(1.0, 0.5)                                           # 3
    pre._retune(1.0, 0.5)                                           # the pass after the change is left out
    pre._retune(1.0, 0.5)                                           # not faster: back to 2
    assert pre.lookahead == 2 and len(pre.pool.buffers) >= 3


# -- the setting

def test_it_is_off_by_default(tiny_checkpoint):
    _, path = tiny_checkpoint
    pre = _blocks_prefetcher(path)
    assert pre.tuner is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_with_the_setting_on_the_look_ahead_stays_within_its_limits_and_the_output_is_right(tiny_checkpoint, monkeypatch):
    ref, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_ADAPTIVE_LOOKAHEAD", "1")
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD_MAX", "3")
    model = meta_model()
    stream_load(model, path, plan=_plan(), device="cuda")
    handle = install_offload(model, order=[f"blocks.{i}" for i in range(4)], profile=False)
    pre = handle._prefetcher
    assert pre.tuner is not None
    x = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = ref.to("cuda").to(torch.bfloat16)(x)
        for _ in range(25):
            out = model(x)
            assert 1 <= pre.lookahead <= 3 and len(pre.pool.buffers) >= pre.lookahead
    assert torch.allclose(out.float(), expected.float(), rtol=5e-2, atol=5e-2)
    assert pre.tuner.done                                           # a handful of passes is enough for it to settle
    handle.remove()


def _plan():
    """bf16 compute from an fp32 file: the blocks are expanded when they run, so they are wrapped."""
    return Plan.manual(compute_dtype="bf16", resident=["blocks.0."], streamed=["blocks.1.", "blocks.2.", "blocks.3."])


def _blocks_prefetcher(path):
    model = meta_model()
    stream_load(model, path, plan=_plan(), device="cpu")
    handle = install_offload(model, order=[f"blocks.{i}" for i in range(4)], profile=False)
    pre = handle._prefetcher
    handle.remove()
    return pre


# -- the staging that is not used is given to the next model

import queue  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from trash_panda.offload.early import _EarlyLoader  # noqa: E402


def test_lowering_the_look_ahead_tells_the_plan_how_much_staging_is_not_used(monkeypatch):
    pre = _prefetcher(monkeypatch, free=4 << 30, largest=100)
    pre.plan, pre.planned = Plan.manual(compute_dtype="bf16"), 2
    pre._retune(1.0, 0.0)                                           # no waiting: the shallowest look-ahead is tried
    assert pre.lookahead == 1 and pre.plan.spare_staging_bytes == 100       # one block of the two that were planned
    pre._retune(9, 9)                                               # the pass after the change is left out
    pre._retune(1.01, 0.0)                                          # not slower: kept
    assert pre.lookahead == 1 and pre.plan.spare_staging_bytes == 100


def test_the_staging_is_taken_back_when_the_shallow_look_ahead_is_slower(monkeypatch):
    pre = _prefetcher(monkeypatch, free=4 << 30, largest=100)
    pre.plan, pre.planned = Plan.manual(compute_dtype="bf16"), 2
    pre._retune(1.0, 0.0)
    pre._retune(9, 9)
    pre._retune(1.5, 0.0)
    assert pre.lookahead == 2 and pre.plan.spare_staging_bytes == 0


def test_a_deeper_look_ahead_than_planned_leaves_no_spare(monkeypatch):
    pre = _prefetcher(monkeypatch, free=4 << 30, largest=100)
    pre.plan, pre.planned = Plan.manual(compute_dtype="bf16"), 2
    pre._retune(1.0, 0.5)                                           # waiting: deeper
    assert pre.lookahead == 3 and pre.plan.spare_staging_bytes == 0


class _Deferred:
    def __init__(self):
        self.loaded = []

    def load(self, prefix, copy=None):
        self.loaded.append(prefix)


def _loader(credit, sizes):
    deferred = _Deferred()
    loader = _EarlyLoader(deferred, nullcontext(), list(sizes), sizes, queue.SimpleQueue(), "cpu", credit=credit)
    return loader, deferred


def test_the_spare_staging_lets_the_next_model_load_before_anything_is_freed():
    import threading
    loader, deferred = _loader(lambda: 250, {"a.": 100, "b.": 100, "c.": 100})
    thread = threading.Thread(target=loader.run)
    thread.start()
    thread.join(0.5)
    assert deferred.loaded == ["a.", "b."]                          # 250 covers two blocks; the third waits for a release
    assert thread.is_alive()
    loader.stop()
    thread.join(2)
    assert not thread.is_alive()


def test_without_credit_nothing_loads_until_memory_is_freed():
    import threading
    loader, deferred = _loader(None, {"a.": 100})
    thread = threading.Thread(target=loader.run)
    thread.start()
    thread.join(0.3)
    assert thread.is_alive() and deferred.loaded == []              # it waits for the queue
    loader.stop()
    thread.join(2)
    assert not thread.is_alive()


def test_the_credit_is_read_each_time_so_a_later_tuner_decision_counts():
    spare = [0]
    loader, deferred = _loader(lambda: spare[0], {"a.": 100})
    import threading
    thread = threading.Thread(target=loader.run)
    thread.start()
    thread.join(0.2)
    assert deferred.loaded == []                                    # no credit yet
    spare[0] = 100
    thread.join(2)
    assert deferred.loaded == ["a."] and not thread.is_alive()


def test_preload_gives_the_early_loader_the_spare_staging_of_the_running_plan(tiny_checkpoint, monkeypatch):
    import importlib
    from trash_panda.offload import preload
    from trash_panda.offload.hardware import Hardware
    stage_module = importlib.import_module("trash_panda.offload.stage")
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    _, path = tiny_checkpoint
    running = Plan.manual(compute_dtype="fp32")
    hw = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
    pending = preload(meta_model, path, running=running, max_vram="1GB", max_ram="1GB", resident_blocks=2,
                      compute_dtype="fp32", hardware=hw, min_free_ram=0, after=queue.SimpleQueue())
    assert pending._done.wait(10)
    running.spare_staging_bytes = 123
    assert pending._make_early().credit() == 123
    pending.cancel()
