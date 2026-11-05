import queue
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


def _loaded(path, dtype="fp32", **tiers):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype=dtype, **tiers), device="cpu")
    return model, install_offload(model)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class _Slow(list):
    """Stands in for an entry's stored tensors: dropping it takes a while, and it notes where and when it happened."""

    def __init__(self, log, seconds=0.0):
        super().__init__()
        self.log, self.seconds = log, seconds

    def clear(self):
        time.sleep(self.seconds)
        super().clear()
        self.log.append((threading.current_thread().name, time.perf_counter()))


def test_there_is_no_drop_thread_without_a_compute_stream(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, streamed=ALL, dtype="bf16")
    assert handle._prefetcher.background_drop is False


def test_a_ram_block_is_dropped_on_its_own_thread_and_announced_afterwards(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, streamed=ALL, dtype="bf16")
    pre = handle._prefetcher
    pre.background_drop = True
    q = handle.subscribe()
    log = []
    started = time.perf_counter()
    pre.drop([_Slow(log, 0.3), {}], "ram", ("blocks.0", "ram", 123))
    assert time.perf_counter() - started < 0.2         # the caller did not wait for the drop
    assert q.empty()                                    # and the block is not announced before it is gone
    pre.settle()
    (thread, done), = log
    assert thread.startswith("trash-panda-drop")
    (release,) = _drain(q)
    assert (release.name, release.tier, release.nbytes) == ("blocks.0", "ram", 123)
    assert release.queued >= done
    handle.remove()


def test_other_tiers_are_dropped_on_the_calling_thread_even_with_the_drop_thread_on(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, streamed=ALL, dtype="bf16")
    pre = handle._prefetcher
    pre.background_drop = True
    q = handle.subscribe()
    log = []
    for tier in ("gpu", "disk"):
        pre.drop([_Slow(log), {}], tier, (f"b.{tier}", tier, 0))
    assert [name for name, _ in log] == [threading.current_thread().name] * 2
    assert [r.tier for r in _drain(q)] == ["gpu", "disk"]      # announced at once
    assert pre.dropper is None
    handle.remove()


def test_blocks_are_announced_in_the_order_they_were_handed_over(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, streamed=ALL, dtype="bf16")
    pre = handle._prefetcher
    pre.background_drop = True
    q = handle.subscribe()
    log = []
    for i, seconds in enumerate((0.05, 0.0, 0.1, 0.0)):
        pre.drop([_Slow(log, seconds), {}], "ram", (f"blocks.{i}", "ram", i))
    pre.settle()
    assert [r.name for r in _drain(q)] == [f"blocks.{i}" for i in range(4)]
    handle.remove()


def test_removing_the_handle_waits_for_a_drop_that_is_still_running(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, streamed=ALL, dtype="bf16")
    pre = handle._prefetcher
    pre.background_drop = True
    q = handle.subscribe()
    log = []
    pre.drop([_Slow(log, 0.3), {}], "ram", ("blocks.0", "ram", 1))
    assert not log
    handle.remove()
    assert len(log) == 1                                # the drop had finished when remove() returned
    assert [r.name for r in _drain(q)] == ["blocks.0"]
    assert pre.dropper is None


def test_the_profile_counts_blocks_dropped_on_the_thread(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    handle._prefetcher.background_drop = True
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    text = handle._prefetcher.report()                  # waits for the thread first
    assert "final pass freed 4 blocks" in text
    assert "ram 4 blocks" in text


def test_a_whole_final_pass_with_the_drop_thread_frees_everything_in_order(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    handle._prefetcher.background_drop = True
    q = handle.subscribe()
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    handle._prefetcher.settle()
    got = _drain(q)
    assert [r.name for r in got] == [f"blocks.{i}" for i in range(4)]
    assert [r.tier for r in got] == ["ram"] * 4
    assert all(p.is_meta for p in model.blocks.parameters())
    handle.remove()
