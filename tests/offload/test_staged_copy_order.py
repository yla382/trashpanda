"""The early loader copies into GPU memory that the running model freed a moment ago, while the CPU is ahead of the GPU.

The allocator hands that memory out at once, but kernels of the running model that read or write it may still be queued.
So a copy must not run before everything queued on the compute stream at the moment it allocates, however old the event
of the last released block is. These tests keep the GPU busy with a spinning kernel to make the order visible."""
import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.early import _StagedCopy  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
SIZE = 32 << 20                     # bytes: one chunk, so one copy
BUSY = 10 ** 9                      # cycles the GPU spins: about half a second, long enough for the copy to finish first


def _warmed_up_copier():
    """A copier whose pinned buffer exists, and every kernel the tests launch loaded already: the first launch of a
    kernel (and pinning) can wait for the GPU, which would hide the order."""
    copier = _StagedCopy()
    copier(torch.zeros(SIZE, dtype=torch.uint8), "cuda")
    torch.cuda._sleep(1)
    small = torch.ones(8, dtype=torch.uint8, device="cuda")
    small.mul_(0)
    small.sum(dtype=torch.int64)
    torch.cuda.synchronize()
    return copier


def _still_queued():
    """Skip the test unless the GPU has not caught up with the CPU, as in a forward pass: otherwise there is no window."""
    marker = torch.cuda.Event()
    marker.record()
    if marker.query():
        pytest.skip("the GPU ran the queued work before the copy started; the order cannot be tested here")


@needs_gpu
def test_a_copy_does_not_land_before_queued_work_that_writes_the_same_memory():
    copier = _warmed_up_copier()
    src = torch.randint(1, 255, (SIZE,), dtype=torch.uint8)
    victim = torch.empty(SIZE, dtype=torch.uint8, device="cuda")
    torch.cuda._sleep(BUSY)                            # the GPU is busy for a while ...
    victim.mul_(0)                                     # ... and then writes here
    del victim                                         # freed on the CPU side: the allocator can hand it out now
    _still_queued()
    out = copier(src, "cuda")                          # same size, so the same memory
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), src)


@needs_gpu
def test_a_copy_does_not_overwrite_memory_that_queued_work_has_still_to_read():
    copier = _warmed_up_copier()
    victim = torch.ones(SIZE, dtype=torch.uint8, device="cuda")
    torch.cuda._sleep(BUSY)
    total = victim.sum(dtype=torch.int64)              # queued: reads the weights of the model that is running
    del victim
    _still_queued()
    copier(torch.zeros(SIZE, dtype=torch.uint8), "cuda")
    torch.cuda.synchronize()
    assert total.item() == SIZE                        # it read its own weights, not the new ones
