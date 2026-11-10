import importlib
import queue
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Tier, preload  # noqa: E402
from trash_panda.offload.checkpoint import Checkpoint  # noqa: E402
from trash_panda.offload.early import _StagedCopy  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.hooks import Release  # noqa: E402
from trash_panda.offload.order_file import order_path, write_order  # noqa: E402

stage_module = importlib.import_module("trash_panda.offload.stage")

HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
PLAN = dict(compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0)
PRE = dict(PLAN, max_vram="1GB", max_ram="1GB", resident_blocks=2)        # blocks 0 and 1 on the GPU, 2 and 3 in RAM


def _cuda(available):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available), nn=torch.nn)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", _cuda(False))
    monkeypatch.setattr(stage_module, "_activations", {})


@pytest.fixture
def reads(monkeypatch):
    """How many times each checkpoint key was read with Checkpoint.get_tensor()."""
    seen = Counter()
    real = Checkpoint.get_tensor

    def get_tensor(self, key):
        seen[key] += 1
        return real(self, key)

    monkeypatch.setattr(Checkpoint, "get_tensor", get_tensor)
    return seen


def _release(nbytes, tier="gpu", name="x"):
    return Release(name, tier, nbytes, None, time.perf_counter())


def _until(condition, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


def _loaded(model, i):
    return not model.blocks[i].lin1.weight.is_meta


def _stored(pending, i):
    return pending.plan.blocks[i].stored_bytes


def test_nothing_is_loaded_until_the_running_model_has_freed_memory(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    assert pending._done.wait(10)
    time.sleep(0.2)
    assert not _loaded(pending.model, 0) and not _loaded(pending.model, 1)
    pending.cancel()


def test_a_block_is_loaded_when_enough_gpu_memory_has_been_freed(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    releases.put(_release(_stored(pending, 0)))
    assert _until(lambda: _loaded(pending.model, 0))
    time.sleep(0.2)
    assert not _loaded(pending.model, 1)                          # only one block's worth was freed
    releases.put(_release(_stored(pending, 1)))
    assert _until(lambda: _loaded(pending.model, 1))
    pending.cancel()


def test_a_block_waits_until_the_freed_memory_adds_up(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    half = _stored(pending, 0) // 2
    releases.put(_release(half))
    time.sleep(0.2)
    assert not _loaded(pending.model, 0)
    releases.put(_release(_stored(pending, 0) - half))
    assert _until(lambda: _loaded(pending.model, 0))
    pending.cancel()


def test_memory_freed_in_the_other_tiers_does_not_count(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    releases.put(_release(10 ** 9, tier="ram"))
    releases.put(_release(10 ** 9, tier="disk"))
    time.sleep(0.2)
    assert not _loaded(pending.model, 0)
    pending.cancel()


def test_the_blocks_are_loaded_in_the_order_they_will_run(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(order_path(path), ["blocks.1", "blocks.0"])        # the order a real run left
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    releases.put(_release(_stored(pending, 1)))
    assert _until(lambda: _loaded(pending.model, 1))
    time.sleep(0.2)
    assert not _loaded(pending.model, 0)
    pending.cancel()


def test_an_order_given_wins_over_the_file(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(order_path(path), ["blocks.0", "blocks.1"])
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, order=["blocks.1"], **PRE)
    releases.put(_release(_stored(pending, 1)))
    assert _until(lambda: _loaded(pending.model, 1))
    assert not _loaded(pending.model, 0)
    pending.cancel()


def test_the_model_gives_the_same_output_and_every_tensor_is_read_once(tiny_checkpoint, reads):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    releases.put(_release(_stored(pending, 0)))                    # block 0 early, block 1 when it is asked for
    assert _until(lambda: _loaded(pending.model, 0))
    with pending as (model, _p, _handle):
        assert not any(t.is_meta for t in model.parameters())
        assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert max(reads.values()) == 1


def test_asking_for_the_model_does_not_wait_for_the_running_one(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()                                 # nothing will ever be put on it
    pending = preload(meta_model, path, after=releases, **PRE)
    result = []
    thread = threading.Thread(target=lambda: result.append(pending.__enter__()))
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and result
    pending.__exit__(None, None, None)


def test_cancel_returns_while_the_loader_is_waiting(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    thread = threading.Thread(target=pending.cancel)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert all(t.is_meta for t in pending.model.parameters())


def test_a_failure_in_the_early_loading_only_warns_and_the_rest_loads_later(tiny_checkpoint, monkeypatch):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)

    def broken(self, src, dest):
        raise OSError("the copy failed")

    monkeypatch.setattr(_StagedCopy, "__call__", broken)
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **PRE)
    assert pending._done.wait(10)
    with pytest.warns(RuntimeWarning, match="early stopped"):
        releases.put(_release(_stored(pending, 0)))
        pending._thread.join(5)                                    # the loader returns after its error
    assert isinstance(pending._early.error, OSError)
    with pending as (model, _p, _handle):
        assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_a_model_without_gpu_blocks_has_no_early_loader(tiny_checkpoint):
    _, path = tiny_checkpoint
    releases = queue.SimpleQueue()
    pending = preload(meta_model, path, after=releases, **{**PRE, "resident_blocks": 0})
    assert pending._done.wait(10)
    assert pending._early is None
    pending.cancel()


def test_without_the_queue_nothing_is_loaded_early(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    assert pending._done.wait(10)
    assert pending._early is None
    assert not _loaded(pending.model, 0)
    pending.cancel()


def test_the_gpu_blocks_are_the_ones_of_the_plan(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, after=queue.SimpleQueue(), **PRE)
    assert [b.prefix for b in pending.plan.blocks if b.tier == Tier.GPU] == ["blocks.0.", "blocks.1."]
    pending.cancel()


def test_stream_load_defers_blocks_that_a_second_thread_can_load_once(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    assert pending._done.wait(10)
    state = pending.model._trash_panda_state
    with state.checkpoint:
        state.deferred.load("blocks.0.")
        state.deferred.load("blocks.0.")                           # a second call does nothing
    assert all(n == 1 for k, n in reads.items() if k.startswith("blocks.0."))
    assert "blocks.0." in state.deferred.loaded
    pending.cancel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_staged_copy_gives_the_same_tensor_on_the_gpu():
    copier = _StagedCopy(chunk=1 << 20)
    cases = ((torch.float32, (700, 513)), (torch.uint8, (3_000_001,)), (torch.float32, ()),
             (torch.bfloat16, (333, 1001)), (torch.float8_e4m3fn, (700, 513)), (torch.float8_e4m3fn, (3_000_001,)))
    for dtype, shape in cases:
        src = (torch.randn(shape) * 100).to(dtype)
        out = copier(src, "cuda")
        assert out.device.type == "cuda" and out.dtype == dtype and out.shape == src.shape
        assert torch.equal(out.cpu().reshape(-1).view(torch.uint8), src.reshape(-1).view(torch.uint8))     # the same bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_staged_copy_still_works_when_pinning_fails(monkeypatch):
    from trash_panda.offload.hooks import _PinnedPool
    monkeypatch.setattr(_PinnedPool, "acquire", lambda self, nbytes: None)
    src = torch.randn(1000, 1000)
    assert torch.equal(_StagedCopy(chunk=1 << 20)(src, "cuda").cpu(), src)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_gpu_must_have_room_for_the_block_and_a_margin():
    from trash_panda.offload.early import _EarlyLoader
    loader = _EarlyLoader(None, None, [], {}, queue.SimpleQueue(), "cuda")
    assert loader._room(1)
    assert not loader._room(10 ** 15)                              # more than the GPU has
    free, total = torch.cuda.mem_get_info()
    assert not loader._room(free)                                  # it would leave nothing for the running model


def test_without_cuda_there_is_always_room(tiny_checkpoint):
    from trash_panda.offload.early import _EarlyLoader
    assert _EarlyLoader(None, None, [], {}, queue.SimpleQueue(), "cpu")._room(10 ** 15)
