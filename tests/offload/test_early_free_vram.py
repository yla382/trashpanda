"""The early loader fills VRAM that is free already, as far as the running model keeps its own room, and still uses the
memory the running model's final pass frees."""
import queue
import threading
import time
from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.early import _EarlyLoader  # noqa: E402
from trash_panda.offload.hooks import Release  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MB = 1 << 20


class _Deferred:
    """Stands in for the DeferredLoad of a pending model: it only records which blocks were loaded."""

    def __init__(self, on_load=None):
        self.calls = []
        self.on_load = on_load

    def load(self, prefix, copy=None):
        self.calls.append(prefix)
        if self.on_load is not None:
            self.on_load(prefix)


def _loader(deferred, names, releases=None, hold=0):
    return _EarlyLoader(deferred, nullcontext(), names, {n: MB for n in names}, releases or queue.SimpleQueue(),
                        "cuda", hold=hold)


def _run(loader):
    thread = threading.Thread(target=loader.run, daemon=True)
    thread.start()
    return thread


def _until(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


@needs_gpu
def test_blocks_are_loaded_into_free_vram_without_any_release():
    deferred = _Deferred()
    loader = _loader(deferred, ["a", "b", "c"])                   # nothing is ever put on the queue
    thread = _run(loader)
    assert _until(lambda: loader.loaded == 3)
    thread.join(5)
    assert deferred.calls == ["a", "b", "c"] and not thread.is_alive()


@needs_gpu
def test_the_vram_the_running_model_keeps_for_itself_is_left_alone():
    free, total = torch.cuda.mem_get_info()
    deferred = _Deferred()
    loader = _loader(deferred, ["a", "b"], hold=total)            # more than any GPU has free
    thread = _run(loader)
    time.sleep(0.4)
    assert deferred.calls == [] and loader.loaded == 0            # it waits for the running model, and loads nothing
    loader.stop()
    thread.join(5)
    assert not thread.is_alive()


@needs_gpu
def test_released_memory_is_still_used_when_the_free_vram_is_held_back():
    _, total = torch.cuda.mem_get_info()
    releases = queue.SimpleQueue()
    deferred = _Deferred()
    loader = _loader(deferred, ["a", "b"], releases, hold=total)
    thread = _run(loader)
    time.sleep(0.2)
    assert deferred.calls == []
    releases.put(Release("x", "gpu", 2 * MB, None, time.perf_counter()))      # the running model freed two blocks' worth
    assert _until(lambda: loader.loaded == 2)
    thread.join(5)
    assert deferred.calls == ["a", "b"]


@needs_gpu
def test_a_block_loaded_into_free_vram_does_not_use_up_released_bytes():
    _, total = torch.cuda.mem_get_info()
    releases = queue.SimpleQueue()
    releases.put(Release("x", "gpu", 2 * MB, None, time.perf_counter()))      # exactly what b and c need
    loader = _loader(None, ["a", "b", "c"], releases)

    def after_a(prefix):                                          # from the second block on, no spot is free any more
        if prefix == "a":
            loader.hold = total

    loader.deferred = _Deferred(after_a)
    thread = _run(loader)
    assert _until(lambda: loader.loaded == 3)                     # a took no credit, so b and c are covered
    thread.join(5)
    assert loader.deferred.calls == ["a", "b", "c"]


def test_without_a_gpu_the_loader_still_waits_for_the_releases():
    deferred = _Deferred()
    releases = queue.SimpleQueue()
    loader = _EarlyLoader(deferred, nullcontext(), ["a"], {"a": MB}, releases, "cpu")
    thread = _run(loader)
    time.sleep(0.2)
    assert deferred.calls == []                                   # no free VRAM to fill without CUDA
    releases.put(Release("x", "gpu", MB, None, time.perf_counter()))
    assert _until(lambda: loader.loaded == 1)
    thread.join(5)
