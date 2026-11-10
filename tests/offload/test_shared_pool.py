"""A pinned pool that outlives the models that use it."""
import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402
from trash_panda.offload.hooks import _PinnedPool, _Prefetcher  # noqa: E402
from trash_panda.offload.planner import _pinned_size  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
BLOCKS = [f"blocks.{i}." for i in range(4)]


def test_sizes_gives_each_buffer_at_the_size_pytorch_gives_it():
    pool = _PinnedPool(3)
    assert pool.sizes() == [0, 0, 0]
    pool.buffers[0] = torch.empty(1000, dtype=torch.uint8)
    pool.buffers[2] = torch.empty(5000, dtype=torch.uint8)
    assert pool.sizes() == [_pinned_size(1000), 0, _pinned_size(5000)]


class _Event:
    def __init__(self):
        self.waited = False

    def synchronize(self):
        self.waited = True


def test_trim_keeps_the_first_buffers_and_waits_for_the_copies_that_read_the_others():
    pool = _PinnedPool(4)
    for i in range(4):
        pool.buffers[i] = torch.empty(100, dtype=torch.uint8)
    kept_events = [_Event(), _Event()]
    last = _Event()
    pool.events[:] = [*kept_events, None, last]                            # the third buffer has no copy in flight
    pool.cursor = 3
    kept = pool.buffers[:2]
    assert pool.trim(2) is True
    assert pool.buffers == kept and len(pool.events) == 2
    assert pool.events == kept_events and not any(e.waited for e in kept_events)
    assert last.waited                                                     # a copy that reads from a dropped buffer
    assert pool.cursor == 1                                                # still a valid position
    assert pool.trim(2) is False and pool.trim(5) is False                 # nothing more to drop


def test_trim_always_keeps_one_buffer():
    pool = _PinnedPool(3)
    assert pool.trim(0) is True and len(pool.buffers) == 1 and pool.cursor == 0


def test_a_prefetcher_without_a_stream_has_no_pool_even_when_given_one():
    pool = _PinnedPool(2)
    pre = _Prefetcher(torch.device("cpu"), pool=pool)
    assert pre.pool is None and pre.shared
    assert not _Prefetcher(torch.device("cpu")).shared


@needs_gpu
def test_a_prefetcher_uses_the_pool_it_is_given_and_leaves_it_alone_when_closed():
    pool = _PinnedPool(2)
    pool.wait_s = 3.0                                    # left over from an earlier model
    pre = _Prefetcher(torch.device("cuda"), pool=pool)
    assert pre.pool is pool and pre.shared and pool.wait_s == 0.0
    pre.close()
    assert pre.pool is None                              # it let go of its reference ...
    assert len(pool.buffers) == 2                        # ... and the pool is as it was


@needs_gpu
def test_two_models_in_a_row_stage_through_the_same_buffers(tiny_checkpoint):
    _, path = tiny_checkpoint
    pool = _PinnedPool(2)
    x = torch.randn(3, 16, device="cuda")
    seen = []
    for _ in range(2):
        model = meta_model()
        plan = Plan.manual(compute_dtype="fp32", resident=BLOCKS[:1], streamed=BLOCKS[1:])
        stream_load(model, path, plan=plan, device="cuda")
        handle = install_offload(model, order=[b.rstrip(".") for b in BLOCKS], profile=False, lookahead=2, pool=pool)
        with torch.no_grad():
            model(x)
        handle.remove()
        seen.append([b.data_ptr() if b is not None else None for b in pool.buffers])
    assert any(ptr is not None for ptr in seen[0])       # the first model pinned some buffers ...
    assert seen[0] == seen[1]                            # ... and the second one did not pin them again
    assert sum(pool.sizes()) > 0
