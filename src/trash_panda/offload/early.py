"""Loading the GPU blocks of the next model while the running model frees the VRAM they need."""

from __future__ import annotations

import queue
import threading
import time
import warnings
from contextlib import ExitStack, nullcontext
from typing import Dict, Iterable, Optional

import torch

from .hooks import _PinnedPool

_CHUNK = 64 << 20          # bytes copied through a pinned buffer at a time
_POLL  = 0.05              # seconds between looks at the stop flag while the queue is empty
_MARGIN_FRACTION = 0.05    # GPU memory kept free for the running model, as a share of the GPU's total ...
_MARGIN_MIN      = 256 << 20       # ... and at least this many bytes


class _StagedCopy:
    """Moves a tensor to the GPU through two small pinned buffers, on a copy stream of its own.

    The destination is allocated on the stream that is current on the calling thread (the compute stream, normally), so
    it can reuse the GPU memory that a running model freed on that stream. The copy stream first waits for `after`, the
    event of the last block the running model freed, which marks the end of that model's work on the memory it gave up
    (an event recorded now would also wait for everything the running model has queued so far, which in a test was most
    of its forward call). The call returns when the copy is done. A tensor read from a memory-mapped file is copied in
    pieces from pinned memory because copying it straight from the mapped pages held up the compute in a probe, even on
    a separate stream. Without CUDA (or when pinning fails) it falls back to a plain `.to()`."""

    def __init__(self, chunk: int = _CHUNK) -> None:
        self.chunk  = chunk
        self.pool   = None
        self.stream = None
        self.after  = None                                       # CUDA event of the last block that was freed

    def __call__(self, src: torch.Tensor, dest) -> torch.Tensor:
        dest = torch.device(dest)
        if dest.type != "cuda" or src.numel() == 0:
            return src.to(dest)
        if self.stream is None:
            self.stream = torch.cuda.Stream(dest)
            self.pool   = _PinnedPool(2)
        out      = torch.empty(src.shape, dtype=src.dtype, device=dest)
        flat_in  = src.reshape(-1).view(torch.uint8)
        flat_out = out.reshape(-1).view(torch.uint8)
        ready = self.after
        if ready is None:                                        # no event from the running model: wait for its queue
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(dest))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            for a in range(0, flat_in.numel(), self.chunk):
                b   = min(a + self.chunk, flat_in.numel())
                got = self.pool.acquire(b - a)
                if got is None:                                 # pinning failed: the rest goes through ordinary memory
                    flat_out[a:].copy_(flat_in[a:])
                    break
                i, buf = got
                buf[: b - a].copy_(flat_in[a:b])
                flat_out[a:b].copy_(buf[: b - a], non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.stream)
                self.pool.release(i, done)
        self.stream.synchronize()
        return out


def _vram_available(device):
    """(VRAM that a new allocation can use, total VRAM) of a CUDA device: what the driver says is free plus what this
    process holds in its allocator without using it."""
    free, total = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free + cached, total


class _EarlyLoader:
    """Loads a pending model's GPU blocks, one at a time, as the running model frees the same number of bytes.

    `releases` is the queue of the running model's final pass (`handle.subscribe()`). A block is loaded when the GPU
    blocks it has announced add up to more than what was loaded here so far plus the block's size, and when the GPU has
    room for it: the memory the driver reports free plus what PyTorch holds cached and unused must cover the block and
    leave a margin (at least `keep`, the headroom and streaming room that the plan counted on). Without that check, a GPU filled to the brim sent PyTorch into its out-of-memory recovery (freeing
    and allocating cached memory again) and held the running model up for seconds in a test. `run()` returns when
    every block is loaded, when `stop()` is called, or after an error (which is kept in `error`; the blocks that are
    not loaded are loaded later, when the model is asked for)."""

    def __init__(self, deferred, checkpoint, order: Iterable[str], sizes: Dict[str, int], releases, device,
                 keep: int = 0) -> None:
        self.deferred   = deferred
        self.checkpoint = checkpoint
        self.order      = list(order)
        self.sizes      = sizes
        self.releases   = releases
        self.device     = torch.device(device)
        self.keep       = keep                                   # bytes of VRAM that stay free for the model's own use
        self.copier     = _StagedCopy()
        self.loaded     = 0                                      # blocks this loader put on the device
        self.error: Optional[BaseException] = None
        self._stop      = threading.Event()

    def stop(self) -> None:
        """Ask for `run()` to return after the block it is loading."""
        self._stop.set()

    def _room(self, need: int) -> bool:
        """True if `need` bytes can be allocated on the GPU and the margin stays free. Always true without CUDA."""
        if self.device.type != "cuda":
            return True
        available, total = _vram_available(self.device)
        return available - need >= max(int(_MARGIN_FRACTION * total), _MARGIN_MIN, self.keep)

    def run(self) -> None:
        freed = used = 0
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext())
                stack.enter_context(self.checkpoint)             # memory-mapped reads, as in stream_load()
                for prefix in self.order:
                    need = self.sizes.get(prefix, 0)
                    while freed - used < need:
                        if self._stop.is_set():
                            return
                        try:
                            item = self.releases.get(timeout=_POLL)
                        except queue.Empty:
                            continue
                        if item.tier == "gpu":
                            freed += item.nbytes
                            if item.event is not None:
                                self.copier.after = item.event
                    while not self._room(need):                          # the memory is announced, but is it free?
                        if self._stop.is_set():
                            return
                        time.sleep(_POLL)
                    if self._stop.is_set():
                        return
                    self.deferred.load(prefix, copy=self.copier)
                    used += need
                    self.loaded += 1
        except Exception as err:                                 # the rest is loaded when the model is asked for
            self.error = err
            warnings.warn(f"trash-panda: loading the next model early stopped ({err!r}); the rest of it loads when it "
                          "is used.", RuntimeWarning)
