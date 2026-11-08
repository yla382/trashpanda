"""Streaming blocks through the compute device during the forward pass.

For every wrapped block, forward() first expands the block's compact weights on the compute device
(reading them from the checkpoint file first, for disk blocks), runs, and then ALWAYS puts the compact
originals back (try/finally), so a failed forward pass cannot leave a block expanded.
"""

from __future__ import annotations

import atexit
import gc
import queue
import threading
import time
import types
import warnings
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import config
from .blocks import module_nbytes
from .checkpoint import Checkpoint
from .depth import DepthTuner
from .formats import SCALE_SUFFIX, expand
from .hardware import _process_memory, _read_ram
from .loader import OffloadState, _navigate
from .order_file import order_path, read_order, write_order
from .planner import Plan, Tier


class Release(NamedTuple):
    """One block that a final pass has freed, as it is put on the queue of `OffloadHandle.subscribe()`.

    `nbytes` is what the block held in its tier: its weights on the GPU for "gpu" (a block that stays on the GPU), its
    stored weights in CPU memory for "ram" (with the memory-mapped RAM tier the pages stay in the file cache until the
    system needs them), and 0 for "disk", which holds nothing in memory between uses. The block is already freed when
    this is queued. `event` marks the end of the block's work on the compute stream: GPU memory it held may be reused
    for work on another stream once the event has completed. It is None without CUDA."""

    name: Optional[str]
    tier: str
    nbytes: int
    event: Optional[torch.cuda.Event]
    queued: float                       # time.perf_counter() when it was put on the queue


class OffloadHandle:
    """Returned by the install functions. remove() puts everything back the way it was."""

    def __init__(self) -> None:
        self._undo: List[Callable[[], None]] = []
        self._prefetcher = None

    def _add(self, fn: Callable[[], None]) -> None:
        self._undo.append(fn)

    def extend(self, other: "OffloadHandle") -> None:
        self._undo.extend(other._undo)

    def remove(self) -> None:
        while self._undo:
            self._undo.pop()()

    def final_pass(self) -> None:
        """From now on each block is freed right after it runs. The model cannot run again."""
        self._prefetcher.final = True

    def subscribe(self) -> "queue.SimpleQueue[Release]":
        """A queue that gets a `Release` for every block the final pass frees, in the order they are freed. Nothing is
        queued until this is called, and nothing is queued outside a final pass. The queue is not bounded: read it."""
        return self._prefetcher.subscribe()


class DiskReader:
    """Reads one block's tensors from the checkpoint file."""

    def __init__(self, checkpoint: Checkpoint, keys: Dict[str, Tuple[str, Optional[str]]], prefix: str):
        self.checkpoint = checkpoint
        self.entries = {name[len(prefix):]: v for name, v in keys.items() if name.startswith(prefix)}

    def read(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        wanted: List[str] = []
        for key, skey in self.entries.values():
            wanted.append(key)
            if skey:
                wanted.append(skey)
        tensors = self.checkpoint.read(wanted)
        return {
            name: (tensors[key], tensors[skey] if skey else None)
            for name, (key, skey) in self.entries.items()
        }

    def specs(self):
        """(name, kind, TensorInfo) for every tensor of the block, read from the checkpoint's header only. kind 0 is the weight, kind 1 is its scale."""
        out = []
        for name, (key, skey) in self.entries.items():
            out.append((name, 0, self.checkpoint.info(key)))
            if skey:
                out.append((name, 1, self.checkpoint.info(skey)))
        return out


class _Entry:
    """One wrapped block: what it holds, which block ran after it, and its prefetched tensors."""

    def __init__(self, slots, scales, reader):
        self.slots  = slots
        self.scales = scales 
        self.reader = reader
        self.next   = None
        self.staged = None
        self.name   = None

    def needs_copy(self) -> bool:
        """True if the block's weights are on disk or in CPU memory, so they must be copied to the GPU."""
        return self.reader is not None or any(slot[4].device.type == "cpu" for slot in self.slots)

    def tier(self) -> str:
        """Where the block's weights are kept between uses: "disk", "ram" or "gpu"."""
        if self.reader is not None:
            return "disk"
        return "ram" if any(slot[4].device.type == "cpu" for slot in self.slots) else "gpu"

    def stored_nbytes(self) -> int:
        """Bytes of the block's stored weights and scales (none for a disk block, which keeps nothing in memory)."""
        if self.reader is not None:
            return 0
        return sum(t.numel() * t.element_size() for t in [slot[4] for slot in self.slots] + list(self.scales.values()))


_ALIGN = 256

def _aligned(n: int) -> int:
    """Round a byte count up so every tensor in a pinned buffer starts at a properly asigned offset."""
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class _PinnedPool:
    """A few page-locked (pinned) host buffers that blocks are copied through on their way to the GPU.

    Only these buffers are pinned, not the whole RAM tier. A buffer is reused only after the GPU copy that
    read from it has finished."""

    def __init__(self, count: Optional[int] = None) -> None:
        count = config.current().pinned_buffers if count is None else count
        self.buffers = [None] * count #uint8 tensors, grown when a bigger block shows up
        self.events  = [None] * count  #the GPU copy that last read from each buffer
        self.cursor  = 0
        self.wait_s  = 0.0             #seconds the worker spent waiting for a buffer's last GPU copy to finish
        self._warned = False

    def acquire(self, nbytes: int):
        """(index, buffer) with room for nbytes, or None if pinned memory could not be allocated."""
        i = self.cursor
        self.cursor = (i + 1) % len(self.buffers)
        if self.events[i] is not None:
            started = time.perf_counter()
            self.events[i].synchronize()
            self.wait_s += time.perf_counter() - started
            self.events[i] = None
        buf = self.buffers[i]
        if buf is None or buf.numel() < nbytes:
            try:
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as err: # pinning failed (not enough memory, or no CUDA)
                if not self._warned:
                    self._warned = True
                    warnings.warn(
                        f"trash-panda: could not allocate {nbytes / 1e9:.2f} GB of pinned (page-locked) memory ({err}). "
                        "Blocks will be copied through ordinary memory, which is slower and does not overlap with compute.",
                        RuntimeWarning,
                    )
                return None
            self.buffers[i] = buf
        return i, buf


    def release(self, i: int, event) -> None:
        """Remember which GPU copy still reads from buffer i"""
        self.events[i] = event

    def grow(self, count: int) -> None:
        """Make room for at least `count` buffers (they are pinned when first used). Never makes the pool smaller."""
        while len(self.buffers) < count:
            self.buffers.append(None)
            self.events.append(None)


_PASS_KEYS = ("ram_copy_s", "ram_copy_gb", "disk_read_s", "disk_read_gb")      # the worker's figures that are also kept per pass


class _Prefetcher:
    """Copies the next block's compact tensors to the device on a second CUDA stream"""

    profile    = False
    order_path = None
    dropper    = None
    background_drop = False
    plan       = None           # the Plan the prefetcher serves, if known: it is told how much staging is not used
    planned    = 0              # the look-ahead the plan counted on
    largest    = 0              # bytes of the biggest block staged so far
    tuner      = None           # DepthTuner, when the look-ahead follows the passes (adaptive_lookahead)

    def __init__(self, device: torch.device, lookahead: Optional[int] = None, profile: Optional[bool] = None):
        cfg = config.current()
        self.device      = device
        self.lookahead   = cfg.lookahead if lookahead is None else lookahead
        self.index      = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None
        self.stream      = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.last        = None
        self.head        = None                      # the first block in the order the blocks were given
        self.hinted      = None                      # the last block handed over, in the order the blocks were given
        self.warned      = False
        self.seen        = {}                        # block names in the order they first ran
        self.known       = {}                        # profile: name -> bytes of every named block that was handed over
        self.order_path  = None                      # where to save that order when the prefetcher closes
        self.order_read  = False                     # True if the order in use came from that file
        self.pending     = {}                        # block name -> Event set when its weights are loaded (deferred blocks)
        self.ready       = set()                     # names of the deferred blocks that are loaded
        self.failure     = None                      # why the loader stopped, if it did
        self.window      = []
        self.in_flight   = deque()
        self.pool        = _PinnedPool(cfg.pinned_buffers) if self.stream is not None else None
        self.worker      = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trash-panda") if self.stream is not None else None
        self.final       = False
        self.dropper     = None                      # thread that lets go of RAM-tier tensors in a final pass, made when needed
        self.background_drop = self.stream is not None   # only where there is a compute stream to keep going
        self.releases    = None                      # queue.SimpleQueue of Release, once somebody has subscribed
        self.release_log = []                        # profile: (time, tier, bytes) of every block the final pass freed
        self.release_peak = 0                        # profile: the most Releases waiting on the queue at once
        # profiling: where does the time go? Off by default; on with profile=True or TRASH_PANDA_PROFILE=1
        self.profile     = cfg.profile if profile is None else bool(profile)
        self.stats       = defaultdict(float)
        self.first       = None
        self.where       = {}                       # id(entry) -> its position in the pass, in the order first seen
        self.block_wait  = defaultdict(float)       # position -> seconds the main thread waited for that block
        self.block_late  = defaultdict(int)         # position -> times it was not ready when asked for
        self.pass_log    = []                       # per finished pass: (seconds it took, seconds spent waiting for staged blocks)
        self.pass_copy   = []                       # per finished pass: the worker's _PASS_KEYS figures, as they grew during it
        self.first_pass  = []                       # per block of the first pass: (position, name, seconds on the main thread, of which waiting for the staged copy, of which waiting for the GPU)
        self._pass_mark  = {}
        self._block_mark = None
        self._pass_start = None
        self._pass_end   = None
        self._pass_wait0 = 0.0
        self.waited      = 0.0                      # seconds the main thread has waited for staged blocks; always measured
        self.pass_waits  = deque(maxlen=64)         # per finished pass: (seconds it took, seconds spend waiting for staged blocks)
        self.began       = None                     # when the pass in progress started
        self.waited0     = 0.0                      # self.waited at that moment
        if cfg.adaptive_lookahead and self.stream is not None and self.lookahead > 0:
            self.tuner = DepthTuner(self.lookahead, cfg.lookahead_max)
        self._reported   = False
        if self.profile:
            atexit.register(self.report)

    def subscribe(self) -> "queue.SimpleQueue[Release]":
        if self.releases is None:
            self.releases = queue.SimpleQueue()
        return self.releases

    @property
    def watching(self) -> bool:
        """True if a freed block is reported: somebody subscribed, or the profile is on."""
        return self.releases is not None or self.profile

    def _compute_event(self):
        """A CUDA event recorded now on the current stream of the calling thread (the compute stream), or None without CUDA."""
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def announce(self, name: Optional[str], tier: str, nbytes: int, event=None) -> None:
        """A block of the final pass has been freed: queue it for the subscriber and note it for the profile. Called
        after the block ran and was freed, on the compute thread or (for RAM-tier blocks, see drop()) on the drop
        thread, so it never waits for whoever reads the queue. `event` is a CUDA event recorded on the compute stream
        after the block was freed; without one, a new event is recorded on the current stream."""
        now = time.perf_counter()
        if self.releases is not None:
            if event is None:
                event = self._compute_event()
            self.releases.put(Release(name, tier, nbytes, event, now))
        if self.profile:
            self.release_log.append((now, tier, nbytes))
            if self.releases is not None:
                self.release_peak = max(self.release_peak, self.releases.qsize())

    def drop(self, held, tier: str, freed) -> None:
        """Final pass: let go of the tensors a block stored (`held`: the entry's slots and scales, already taken out of
        the entry) and announce the block (`freed`: its name, tier and bytes, or None if nobody is watching).

        The tensors of a RAM-tier block may be memory-mapped from the checkpoint, and the last of them to go takes the
        mapping with it: 1.1 s for 6.95 GB of touched pages, and Python's lock is not held meanwhile. So where there
        is a compute stream, RAM-tier tensors are dropped one block at a time on a thread of their own, in the order
        the blocks ran, and the block is announced when its memory is really gone. The compute thread goes on at once.
        close() and report() wait for the thread."""
        if tier == "ram" and self.background_drop:
            event = self._compute_event() if freed is not None and self.releases is not None else None
            if self.dropper is None:
                self.dropper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trash-panda-drop")
            self.dropper.submit(self._drop_job, held, freed, event)
            return
        self._drop_job(held, freed, None)

    def _drop_job(self, held, freed, event) -> None:
        for part in held:
            part.clear()
        if freed is not None:
            if event is None:
                self.announce(*freed)
            else:
                self.announce(*freed, event=event)

    def settle(self) -> None:
        """Wait until every drop handed to the drop thread has finished."""
        if self.dropper is not None:
            try:
                self.dropper.submit(lambda: None).result()
            except RuntimeError:                         # the interpreter is shutting down
                pass

    def _retune(self, took: float, waited: float) -> None:
        """Show the pass that just finished to the DepthTuner and use the look-ahead it answers with from the next pass on.
        A deeper look-ahead stages more blocks on the GPU than the plan counted on, so it is only taken when the driver
        reports free VRAM (outside PyTorch's cache, which the activations may need) for the extra blocks and a margin of 5%
        of the GPU (at least 256 MB); otherwise the depth stays and the tuner stops. The pinned buffers grow to match."""
        depth = self.tuner.update(took, waited)
        if depth == self.lookahead:
            return
        if depth > self.lookahead:
            free, total = torch.cuda.mem_get_info(self.device)
            if free - (depth - self.lookahead) * self.largest < max(int(0.05 * total), 256 << 20):
                self.tuner.depth = self.lookahead
                self.tuner.stop()
                return
            self.pool.grow(depth)
        self.lookahead = depth
        if self.plan is not None:                     # the staging the plan reserved and the look-ahead no longer uses
            self.plan.spare_staging_bytes = max(self.planned - self.lookahead, 0) * self.largest

    def learn(self, entry) -> None:
        if entry.name is not None:
            self.note(entry.name)
        if self.first is None:
            self.first = entry
        if entry is self.first:
            now = time.perf_counter()
            if self.began is not None:
                self.pass_waits.append((now - self.began, self.waited - self.waited0))
                if self.tuner is not None:
                    self._retune(*self.pass_waits[-1])
            self.began, self.waited0 = now, self.waited
        if self.profile:
            if entry is self.first:
                self._end_pass()
                self.stats["passes"] += 1               # the first block starts a new pass through the model
                self._pass_start = time.perf_counter()
                self._pass_wait0 = self.stats["wait_staged_s"]
                self._pass_mark  = {k: self.stats[k] for k in _PASS_KEYS}
            self.where.setdefault(id(entry), len(self.where))
            self._block_mark = (time.perf_counter(), self.stats["wait_staged_s"], self.stats["wait_gpu_s"])
        late = entry.name in self.pending                # a block whose weights were loaded while the model ran has no place in the order given
        if self.last is None:
            if self.head is not None and entry is not self.head and not late and not self.warned:
                self._warn_order(f"{entry.name or 'a block'} ran first, but {self.head.name or 'another block'} was expected first.")
        else:
            expected = self.last.next
            if expected is not None and expected is not entry and not late and not self.warned:
                self._warn_order(
                    f"{entry.name or 'a block'} ran after {self.last.name or 'a block'}, "
                    f"but {expected.name or 'another block'} was expected."
                )
            self.last.next = entry
        self.last = entry

    def _warn_order(self, what: str) -> None:
        self.warned = True
        warnings.warn(
            f"block order differs from the order given: {what} Pass order=[...] to stage() or install_offload(), "
            "or list the blocks in the order they run, so the first pass is prefetched correctly."
        )

    def note(self, name: str) -> None:
        """Remember that the block called `name` has started, the first time only: this is the order it first ran in."""
        self.seen.setdefault(name, None)

    def hold(self, names) -> None:
        """Blocks whose weights are not loaded yet: a block that starts before it is marked ready waits for it."""
        for name in names:
            self.pending.setdefault(name, threading.Event())

    def mark_ready(self, name: str) -> None:
        self.ready.add(name)
        self.pending[name].set()

    def fail(self, error: BaseException) -> None:
        """The loader stopped with `error`: nobody waits any longer, and a block that was not loaded raises it."""
        self.failure = error
        for event in self.pending.values():
            event.set()

    def wait_ready(self, name) -> None:
        event = self.pending.get(name)
        if event is None:
            return
        started = time.perf_counter() if self.profile else 0.0
        event.wait()
        if self.profile:
            self.stats["wait_loader_s"] += time.perf_counter() - started
        if name not in self.ready:
            raise RuntimeError(f"the weights of {name} were not loaded: {self.failure!r}") from self.failure

    def expect(self, entry) -> None:
        """Link `entry` after the previous one in the order the blocks were given, before any pass has run."""
        if self.hinted is None:
            self.head = entry
        else:
            self.hinted.next = entry
        self.hinted = entry

    def _end_pass(self) -> None:
        """Close the pass in progress: how long it took (until its last block finished) and how long the main thread waited for staged blocks during it."""
        if self._pass_start is None or self._pass_end is None:
            return
        self.pass_log.append((self._pass_end - self._pass_start, self.stats["wait_staged_s"] - self._pass_wait0))
        self.pass_copy.append(tuple(self.stats[k] - self._pass_mark.get(k, 0.0) for k in _PASS_KEYS))
        self._pass_start = None

    def _log_block(self) -> None:
        """In the first pass, keep how the main thread spent the time of the block that just ran."""
        if self.stats["passes"] != 1 or self._block_mark is None or self.last is None:
            return
        started, staged, gpu = self._block_mark
        self.first_pass.append((
            self.where.get(id(self.last)), self.last.name, self._pass_end - started,
            self.stats["wait_staged_s"] - staged, self.stats["wait_gpu_s"] - gpu,
        ))
        self._block_mark = None

    def stage(self, entry) -> None:
        if self.stream is None or entry is  None or entry.staged is not None:
            return
        entry.staged = self.worker.submit(self._stage_job, entry)

    def stage_ahead(self, entry) -> None:
        """Stage the next `lookahead` blocks that need a copy, following the order learned so far."""
        if self.stream is None or self.lookahead <= 0:
            return
        wanted , seen, nxt = [], {id(entry)}, entry.next
        while nxt is not None and id(nxt) not in seen and len(wanted) < self.lookahead:
            seen.add(id(nxt))
            if nxt.needs_copy():
                wanted.append(nxt)
            nxt = nxt.next
        for old in self.window:
            if old not in wanted:
                self._drop(old)
        for item in wanted:
            self.stage(item)
        self.window = wanted

    def _drop(self, entry) -> None:
        future, entry.staged = entry.staged, None
        if future is not None:
            future.cancel()

    def wait_for_room(self, limit: Optional[int] = None) -> None:
        """Let the GPU catch up, so that at most `limit - 1` earlier blocks are still running (default from config)."""
        if self.stream is None or self.lookahead <= 0:
            return
        limit = config.current().blocks_in_flight if limit is None else limit
        started = time.perf_counter() if self.profile else 0.0
        while len(self.in_flight) >= limit:
            self.in_flight.popleft().synchronize()
        if self.profile:
            self.stats["wait_gpu_s"] += time.perf_counter() - started

    def block_done(self) -> None:
        """Mark the end of a block's work on the ocmpute stream."""
        if self.profile:
            self._pass_end = time.perf_counter()
            self._log_block()
        if self.stream is None or self.lookahead <= 0:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.in_flight.append(event)

    def _stage_job(self, entry):
        """Runs on the worker thread: copy the block through a pinned buffer and start the GPU copies."""
        torch.cuda.set_device(self.index)
        if entry.reader is not None:
            return self._stage_disk_job(entry)
        sources = []

        for name, _owner, _leaf, _is_param, orig in entry.slots:
            sources.append((name, 0, orig))
            scale = entry.scales.get(name)
            if scale is not None:
                sources.append((name, 1, scale))

        total = sum(_aligned(t.numel() * t.element_size()) for _n, _k, t in sources if t.device.type == "cpu")
        self.largest = max(self.largest, total)
        got = self.pool.acquire(total) if total else None

        staged = {}
        with torch.cuda.stream(self.stream):
            offset = 0
            for name, kind, t  in sources:
                if t.device.type != "cpu" or got is None:
                    staged[(name, kind)] = t.to(self.device, non_blocking=True)
                else:
                    n = t.numel() * t.element_size()
                    view = got[1][offset:offset + n].view(t.dtype).reshape(t.shape)
                    started = time.perf_counter() if self.profile else 0.0
                    view.copy_(t)
                    if self.profile:
                        self.stats["ram_copy_s"] += time.perf_counter() - started
                        self.stats["ram_copy_gb"] += n / 1e9
                    offset += _aligned(n)
                    staged[(name, kind)] = view.to(self.device, non_blocking=True)
            
            event = torch.cuda.Event()
            event.record(self.stream)
        if got is not None:
            self.pool.release(got[0], event)

        tensors = {name: (staged[(name, 0)], staged.get((name, 1))) for name, *_ in entry.slots}
        return tensors, event

    def _stage_disk_job(self, entry):
        """Runs on teh worker thread: read a disk block straight into a pinned buffer and start the GPU copies."""
        specs = entry.reader.specs()
        if self.profile:
            self.stats["disk_blocks"] += 1
        total = sum(_aligned(info.nbytes) for _name, _kind, info in specs)
        self.largest = max(self.largest, total)
        got   = self.pool.acquire(total)

        staged = {}
        with torch.cuda.stream(self.stream):
            offset = 0
            for name, kind, info in specs:
                if got is None:
                    part = torch.empty(info.nbytes, dtype=torch.uint8)
                else:
                    part = got[1][offset:offset + info.nbytes]
                    offset += _aligned(info.nbytes)
                started = time.perf_counter() if self.profile else 0.0
                entry.reader.checkpoint.read_into(info.key, part)
                if self.profile:
                    self.stats["disk_read_s"] += time.perf_counter() - started
                    self.stats["disk_read_gb"] += info.nbytes / 1e9
                tensor = part.view(info.torch_dtype).reshape(info.shape)
                staged[(name, kind)] = tensor.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        if got is not None:
            self.pool.release(got[0], event)

        tensors = {name: (staged[(name, 0)], staged.get((name, 1))) for name in entry.reader.entries}
        return tensors, event

    def take(self, entry):
        """The staged tensors, once their copy has finished, or None if nothing was staged."""
        staged, entry.staged = entry.staged, None
        if entry in self.window:
            self.window.remove(entry)
        if staged is None:
            return None
        ready = staged.done()
        started = time.perf_counter()
        try:
            tensors, event = staged.result()  # wait until the worker has issued the copies
        except Exception:
            import traceback
            traceback.print_exc()             # never hide a failure in the worker
            return None                       # fall back to the plain copy in bring()
        waited = time.perf_counter() - started
        self.waited += waited
        if self.profile:
            self.stats["wait_staged_s"] += waited
            position = self.where.get(id(entry))
            if position is not None:
                self.block_wait[position] += waited
                self.block_late[position] += 0 if ready else 1
            self.stats["takes"] += 1
            self.stats["takes_ready"] += 1 if ready else 0

        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(event)

        for raw, scale in tensors.values():
            raw.record_stream(compute)
            if scale is not None:
                scale.record_stream(compute)
        return tensors

    def report(self) -> str:
        """A short summary of where the prefetching spent its time. Printed at exit when profiling is on."""
        self.settle()
        if not self.profile or self._reported or not (self.stats or self.release_log or self._unused()):
            return ""
        s = self.stats
        self._end_pass()
        self._reported = True
        passes = max(int(s["passes"]), 1)
        lines = [f"trash-panda prefetch profile ({passes} passes through the blocks, figures are per pass)"]
        if s["disk_read_s"]:
            lines.append(
                f"  worker, disk reads: {s['disk_read_gb'] / passes:.2f} GB in {s['disk_read_s'] / passes:.2f} s "
                f"({s['disk_read_gb'] * 1000 / s['disk_read_s']:.0f} MB/s), {int(s['disk_blocks'] / passes)} blocks"
            )
        if s["ram_copy_s"]:
            lines.append(
                f"  worker, RAM copies into the pinned buffers: {s['ram_copy_gb'] / passes:.2f} GB in "
                f"{s['ram_copy_s'] / passes:.2f} s ({s['ram_copy_gb'] / s['ram_copy_s']:.1f} GB/s)"
            )
        if s["takes"]:
            lines.append(
                f"  main thread waited for staged blocks: {s['wait_staged_s'] / passes:.2f} s "
                f"({int(s['takes_ready'])} of {int(s['takes'])} were ready at once)"
            )
        if self.pass_log:
            lines.append("  per pass (time, of which the main thread waited for staged blocks):")
            shown = list(enumerate(self.pass_log, 1))
            if len(shown) > 10:
                shown = shown[:5] + [None] + shown[-2:]
            for item in shown:
                if item is None:
                    lines.append("    ...")
                else:
                    n, (took, waited) = item
                    lines.append(f"    pass {n}: {took:.2f} s, waited {waited:.2f} s{self._copied(n - 1)}")
        if self.first_pass:
            total = sum(b[2] for b in self.first_pass)
            lines.append(
                f"  first pass, main thread time by block ({len(self.first_pass)} blocks, {total:.2f} s; "
                "staged = waiting for the copy, gpu = waiting for the GPU, the rest is launching work):"
            )
            for position, name, ran, staged, gpu in sorted(self.first_pass, key=lambda b: b[2], reverse=True)[:6]:
                lines.append(f"    {name or f'block {position}'}: {ran:.2f} s (staged {staged:.2f}, gpu {gpu:.2f})")
        if s["load_s"]:
            lines.append(
                f"  loader thread: {int(s['load_blocks'])} blocks in {s['load_s']:.2f} s, "
                f"main thread waited for it {s['wait_loader_s']:.2f} s (once, not per pass)"
            )
        unused = self._unused()
        if unused:
            names = ", ".join(name for name, _ in unused[:6]) + (f" and {len(unused) - 6} more" if len(unused) > 6 else "")
            lines.append(
                f"  blocks that never ran: {names} ({sum(n for _, n in unused) / 1e9:.2f} GB). "
                "If the model never uses them, consider leaving them out with skip_prefixes"
            )
        if self.release_log:
            held = {tier: [0, 0] for tier in ("gpu", "ram", "disk")}
            for _at, tier, nbytes in self.release_log:
                held[tier][0] += 1
                held[tier][1] += nbytes
            parts = [f"{tier} {n} blocks {nbytes / 1e9:.2f} GB" for tier, (n, nbytes) in held.items() if n]
            span = self.release_log[-1][0] - self.release_log[0][0]
            line = f"  final pass freed {len(self.release_log)} blocks over {span:.2f} s ({', '.join(parts)})"
            if self.releases is not None:
                line += f"; the queue held at most {self.release_peak}, {self.releases.qsize()} unread now"
            lines.append(line)

        slowest = sorted(self.block_wait, key=self.block_wait.get, reverse=True)[:5]
        if slowest:
            lines.append("  slowest waits (block position in the pass, seconds per pass, passes it was late):")
            for position in slowest:
                lines.append(f"    block {position}: {self.block_wait[position] / passes:.2f} s, late in {self.block_late[position]} of {passes}")
        if self.pool is not None:
            lines.append(f"  worker waited for a free pinned buffer: {self.pool.wait_s / passes:.2f} s")
        lines.append(f"  main thread waited for the GPU to catch up: {s['wait_gpu_s'] / passes:.2f} s")
        text = "\n".join(lines)
        print(text)
        return text

    def _unused(self) -> List[Tuple[str, int]]:
        """(name, bytes) of the named blocks that never started, once some block has. Only known with the profile on."""
        if not self.seen:
            return []
        return [(name, nbytes) for name, nbytes in self.known.items() if name not in self.seen]

    def _copied(self, index: int) -> str:
        """What the worker read and copied while pass `index` ran, for its line in the report."""
        if index >= len(self.pass_copy):
            return ""
        ram_s, ram_gb, disk_s, disk_gb = self.pass_copy[index]
        parts = []
        if ram_s:
            parts.append(f"RAM copies {ram_gb:.2f} GB in {ram_s:.2f} s ({ram_gb / ram_s:.2f} GB/s)")
        if disk_s:
            parts.append(f"disk reads {disk_gb:.2f} GB in {disk_s:.2f} s ({disk_gb * 1000 / disk_s:.0f} MB/s)")
        return f", {', '.join(parts)}" if parts else ""

    def _save_order(self) -> None:
        """Save the order that was seen, unless there is nothing to save or the file in use was right."""
        if self.order_path is None or not self.seen or (self.order_read and not self.warned):
            return
        if write_order(self.order_path, self.seen):
            print(f"trash-panda: saved the order of {len(self.seen)} blocks to {self.order_path}")

    def close(self) -> None:
        """Stop the worker thread and let go of the pinned buffers and anything still staged."""
        self._save_order()
        self.report()
        if self.dropper is not None:
            self.dropper.shutdown(wait=True)
            self.dropper = None
        if self.worker is not None:
            self.worker.shutdown(wait=True)
            self.worker = None
        if self.stream is not None:
            self.stream.synchronize()

        self.pool   = None
        self.stream = None
        self.last   = None
        self.first  = None
        self.window = []
        self.in_flight.clear()
        


def _needs_wrapping(slots, scales, device: torch.device, dtype: torch.dtype, disk) -> bool:
    if disk is not None:
        return True
    for name, _owner, _leaf, _is_param, orig in slots:
        if name in scales:
            return True
        if orig.device.type != device.type:
            return True
        if orig.is_floating_point() and orig.dtype != dtype:
            return True
    return False


def _loaded_nbytes(module: nn.Module) -> int:
    """Bytes of the module's parameters and buffers that are not on the meta device, so 0 for a block already freed."""
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()] if not t.is_meta)


def _to_meta(module: nn.Module) -> None:
    for m in module.modules():
        for name in list(m._parameters):
            p = m._parameters[name]
            if p is not None and not p.is_meta:
                m._parameters[name] = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)
        for name in list(m._buffers):
            b = m._buffers[name]
            if b is not None and not b.is_meta:
                m._buffers[name] = torch.empty_like(b, device="meta")


def offload_blocks(
    blocks: Iterable[nn.Module],
    *,
    device: Union[str, torch.device, None] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    disk: Optional[Callable[[nn.Module], Optional[DiskReader]]] = None,
    prefetcher: Optional["_Prefetcher"] = None,
    lookahead: Optional[int] = None,
    profile: Optional[bool] = None,
    names: Optional[Iterable[str]] = None,
    hint: bool = True,
) -> OffloadHandle:
    """Low-level: wrap each block so its weights are expanded on `device` only while it runs.

    Blocks whose weights are already on the device in the compute dtype are left untouched.
    `disk` (advanced) maps a block to a DiskReader when its weights live in the checkpoint file.
    `lookahead` is how many blocks ahead the next weights are copied (0 turns prefetching off; the default comes
    from config).
    `names` gives each block a name, for the order file and for messages. `hint` is False for a block that is
    installed while the model runs, which has no place in the order given."""
    names  = list(names) if names is not None else None
    dev    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    handle = OffloadHandle()
    if prefetcher is None:
        prefetcher = _Prefetcher(dev, lookahead, profile)
        handle._add(prefetcher.close)
    handle._prefetcher = prefetcher

    for i, block in enumerate(blocks):
        label  = names[i] if names is not None else None
        if label is not None and prefetcher.profile:
            prefetcher.known[label] = module_nbytes(block)
        scales = {
            n[: -len(SCALE_SUFFIX)]: b for n, b in block.named_buffers() if n.endswith(SCALE_SUFFIX)
        }
        slots = []        # (name, owner, leaf, is_param, original object)
        for name, p in block.named_parameters():
            owner, leaf = _navigate(block, name)
            slots.append((name, owner, leaf, True, p))
        for name, b in block.named_buffers():
            if name.endswith(SCALE_SUFFIX):
                continue
            owner, leaf = _navigate(block, name)
            slots.append((name, owner, leaf, False, b))

        reader = disk(block) if disk is not None else None
        if not _needs_wrapping(slots, scales, dev, compute_dtype, reader):
            def free_after_run(module, _inputs, _output, _label=label):
                if prefetcher.final:
                    held = _loaded_nbytes(module) if prefetcher.watching else 0
                    _to_meta(module)
                    if held:
                        prefetcher.announce(_label, "gpu", held)

            hook = block.register_forward_hook(free_after_run)
            handle._add(hook.remove)
            if label is not None:                        # not wrapped, so only note when it starts
                note = block.register_forward_pre_hook(lambda _module, _args, n=label: prefetcher.note(n))
                handle._add(note.remove)
            continue

        entry  = _Entry(slots, scales, reader)
        entry.name = label
        if hint:
            prefetcher.expect(entry)

        def bring(entry=entry):
            staged = prefetcher.take(entry)
            if staged is None and prefetcher.lookahead > 0 and entry.needs_copy():
                prefetcher.stage(entry)
                staged = prefetcher.take(entry)
            data = None
            if staged is None and entry.reader is not None:
                data = entry.reader.read()
            for name, owner, leaf, is_param, orig in entry.slots:
                if data is not None:
                    raw, scale = data[name]
                elif staged is not None:
                    raw, scale = staged[name]
                else:
                    raw, scale = orig, entry.scales.get(name)
                t = expand(raw, scale, compute_dtype, dev) if raw.is_floating_point() else  raw.to(dev)

                if is_param:
                    owner._parameters[leaf]  = nn.Parameter(t, requires_grad=False)
                else:
                    owner._buffers[leaf] = t

        def restore(slots=slots):
            for _name, owner, leaf, is_param, orig in slots:
                if is_param:
                    owner._parameters[leaf] = orig
                else:
                    owner._buffers[leaf] = orig

        original_forward = block.forward

        def wrapped(*args, _orig=original_forward, _bring=bring, _restore=restore, _entry=entry, _block=block, **kwargs):
            prefetcher.learn(_entry)
            prefetcher.wait_for_room()
            _bring()
            prefetcher.stage_ahead(_entry)

            try:
                return _orig(*args, **kwargs)
            finally:
                if prefetcher.final:
                    tier = _entry.tier()
                    freed = (_entry.name, tier, _entry.stored_nbytes()) if prefetcher.watching else None
                    held = [list(_entry.slots), dict(_entry.scales)]   # the entry holds the original tensors: these keep the memory alive
                    _entry.slots.clear()
                    _entry.scales.clear()
                    _to_meta(_block)
                    prefetcher.drop(held, tier, freed)
                else:
                    _restore()
                prefetcher.block_done()

        def undo(block=block, restore=restore, entry=entry):
            restore()
            block.__dict__.pop("forward", None)
            entry.staged = None
            entry.slots.clear()                  # the prefetcher keeps the entries for as long as the handle lives, and these
            entry.scales.clear()                 # hold the original tensors, which release() is meant to give back

        block.forward = wrapped
        handle._add(undo)

    return handle


def _defer_block(block: nn.Module, name: str, pre: "_Prefetcher", handle: OffloadHandle, install: Callable) -> None:
    """Until the block's weights are loaded, calling the block waits for them; then the real hooks go in.
    `install()` puts them in and returns their handle."""
    state = {"inner": None}

    def gate(*args, **kwargs):
        pre.wait_ready(name)
        block.__dict__.pop("forward", None)               # so that install() sees the block's own forward
        state["inner"] = install()
        pre.note(name)
        if "forward" in block.__dict__:                   # wrapped: its own forward does the rest
            return block.forward(*args, **kwargs)
        out = block.forward(*args, **kwargs)              # not wrapped: the hook that frees it starts with the next call
        if pre.final:
            held = _loaded_nbytes(block) if pre.watching else 0
            _to_meta(block)
            if held:
                pre.announce(name, "gpu", held)
        return out

    def undo():
        if state["inner"] is not None:
            state["inner"].remove()
        elif block.__dict__.get("forward") is gate:
            block.__dict__.pop("forward", None)

    block.forward = gate
    handle._add(undo)


class _BlockLoader:
    """Loads the deferred resident blocks one at a time, in the order given, on a worker thread."""

    def __init__(self, deferred, order: Iterable[str], prefetcher: "_Prefetcher", device: torch.device) -> None:
        self.deferred   = deferred
        self.order      = list(order)
        self.prefetcher = prefetcher
        self.device     = device
        self._stop      = threading.Event()
        self.thread     = threading.Thread(target=self._run, name="trash-panda-loader", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """Ask the thread to stop after the block it is loading, and wait for it."""
        self._stop.set()
        self.thread.join()

    def _run(self) -> None:
        try:
            with torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext():
                for prefix in self.order:
                    if self._stop.is_set():
                        break
                    started = time.perf_counter()
                    self.deferred.load(prefix)
                    if self.prefetcher.profile:
                        self.prefetcher.stats["load_s"] += time.perf_counter() - started
                        self.prefetcher.stats["load_blocks"] += 1
                    self.prefetcher.mark_ready(prefix.rstrip("."))
            if self._stop.is_set():
                self.prefetcher.fail(RuntimeError("the loader was stopped"))
        except BaseException as e:
            self.prefetcher.fail(e)
            warnings.warn(f"loading the weights failed: {e!r}")


def _in_order(blocks, order):
    """The plan's blocks with the ones named in `order` first, in that order; the rest follow in plan order."""
    names   = [p.rstrip(".") for p in order]
    by_name = {b.prefix.rstrip("."): b for b in blocks}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"order names blocks that are not in the plan: {unknown}")
    first = [by_name[n] for n in dict.fromkeys(names)]
    named = set(names)
    return first + [b for b in blocks if b.prefix.rstrip(".") not in named]


def install_offload(
    model: nn.Module,
    plan: Optional[Plan] = None,
    *,
    device: Union[str, torch.device, None] = None,
    compute_dtype: Optional[torch.dtype] = None,
    lookahead: Optional[int] = None,
    profile: Optional[bool] = None,
    order: Optional[Iterable[str]] = None,
    order_file_destination: Union[str, Path, None] = None,
) -> OffloadHandle:
    """Install the offload hooks for every block in the plan (defaults to the plan used by stream_load).
    `order` names blocks (by prefix) that run first, in that order; the others follow in the plan's order. It tells the
    prefetcher what to stage before the first pass has shown the real order.
    When `order` is None, the order the blocks ran in is saved when the handle is removed, in
    `<checkpoint name>_order.json`, and read the next time. `order_file_destination` is the folder of that file (by
    default the checkpoint's own folder); it is only used when `order` is None."""
    state: Optional[OffloadState] = getattr(model, "_trash_panda_state", None)
    plan = plan or (state.plan if state is not None else None)
    if plan is None:
        raise ValueError(
            "install_offload() needs a Plan (pass plan=..., or load with stream_load(model, path, plan=...)). "
            "For manual control without a plan, use offload_blocks(blocks, ...)."
        )
    dev = device or (state.device if state is not None else None)
    cdt = compute_dtype or (state.compute_dtype if state is not None else plan.compute_dtype)
    lookahead = plan.lookahead if lookahead is None else lookahead

    handle = OffloadHandle()
    pre    = _Prefetcher(torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu")), lookahead, profile)
    handle._add(pre.close) #registered first, so it runs last on remove()
    handle._prefetcher = pre
    pre.plan, pre.planned = plan, pre.lookahead
    if order is None and state is not None:
        pre.order_path = order_path(state.checkpoint.path, order_file_destination)
        saved = read_order(pre.order_path, {b.prefix.rstrip(".") for b in plan.blocks})
        if saved is not None:
            order, pre.order_read = saved, True
            print(f"trash-panda: using the block order saved in {pre.order_path}")
    blocks = plan.blocks if order is None else _in_order(plan.blocks, order)
    deferred = state.deferred if state is not None else None
    for b in blocks:
        block = model.get_submodule(b.prefix.rstrip("."))
        name  = b.prefix.rstrip(".")
        if deferred is not None and b.prefix in deferred.items:
            if pre.profile:
                pre.known[name] = module_nbytes(block)
            pre.hold([name])                             # the weights are still being loaded; the hooks go in when they are
            _defer_block(
                block, name, pre, handle,
                lambda block=block, name=name: offload_blocks(
                    [block], device=dev, compute_dtype=cdt, prefetcher=pre, names=[name], hint=False
                ),
            )
            continue
        reader_factory = None
        if b.tier == Tier.DISK:
            if state is None:
                raise ValueError("Disk blocks need a model loaded with stream_load().")
            def reader_factory(_blk, p=b.prefix):
                return DiskReader(state.checkpoint, state.disk_keys, p)
        handle.extend(offload_blocks(
            [block], device=dev, compute_dtype=cdt, disk=reader_factory, prefetcher=pre, names=[name]
        ))
    if deferred is not None:
        loader = _BlockLoader(
            deferred, [b.prefix for b in blocks if b.prefix in deferred.items], pre, torch.device(pre.device)
        )
        handle._add(loader.stop)                         # added last, so that remove() stops the thread first
        loader.start()
    return handle


def _cpu_tensors(smallest: int = 1 << 20) -> list:
    """The CPU tensors that are alive right now and whose storage holds at least `smallest` bytes, one per storage, as
    (bytes, weak reference)."""
    found, seen = [], set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # looking at every object trips deprecation warnings in some modules
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                    continue
                storage = obj.untyped_storage()
                key, size = storage.data_ptr(), storage.nbytes()
                if size >= smallest and key not in seen:
                    seen.add(key)
                    found.append((size, weakref.ref(obj)))
            except Exception:
                continue
    return found


def _who(obj) -> str:
    """A few words that say what an object is."""
    if isinstance(obj, dict):
        return "dict with keys " + ", ".join(repr(k) for k in list(obj)[:4])
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} of {len(obj)}"
    name = getattr(obj, "__qualname__", None)
    return f"{type(obj).__name__} {name}" if name else type(obj).__name__


def _holders(obj, limit: int = 6) -> list:
    """What refers to `obj`, and what refers to each of those, one line each."""
    lines, refs = [], gc.get_referrers(obj)
    for ref in refs:
        if isinstance(ref, types.FrameType):
            continue
        owners = [_who(o) for o in gc.get_referrers(ref) if o is not refs and not isinstance(o, types.FrameType)][:3]
        lines.append(f"{_who(ref)}  <-  {'; '.join(owners) or 'nothing'}")
    return lines[:limit]


class _RamTrace:
    """With profiling on, remember at a few points of release() how much RAM was free, how much this process holds and
    which big CPU tensors are still alive, to see which step gives memory back and what keeps it when none does."""

    def __init__(self) -> None:
        self.on      = config.current().profile
        self.points  = []                           # (what, RAM available, (working set, private) or None, tensors alive, their bytes)
        self.commits = []                           # bytes the system would still let processes commit
        self.biggest = None                         # weak reference to the biggest CPU tensor alive at the last mark

    def mark(self, what: str) -> None:
        if not self.on:
            return
        _total, available, commit = _read_ram()
        found = _cpu_tensors()
        self.points.append((what, available, _process_memory(), len(found), sum(size for size, _ in found)))
        self.biggest = max(found, key=lambda item: item[0])[1] if found else None
        if commit is not None:
            self.commits.append(commit)

    def report(self) -> None:
        if not self.on or not self.points:
            return
        lines = ["trash-panda: memory while releasing (free RAM; this process: working set / private; CPU tensors of 1 MB or more alive):"]
        for what, free, process, count, held in self.points:
            used = f", process {process[0] / 1e9:.2f} / {process[1] / 1e9:.2f} GB" if process else ""
            lines.append(f"  {what:15} {free / 1e9:6.2f} GB free{used}, {count} tensors of {held / 1e9:.2f} GB")
        if len(self.commits) >= 2:
            lines.append(f"  commit headroom {self.commits[0] / 1e9:.2f} -> {self.commits[-1] / 1e9:.2f} GB")
        big = self.biggest() if self.biggest is not None else None
        if big is not None and self.points[-1][4] > 10 ** 8:
            lines.append("  still alive at the end, the biggest tensor is held by:")
            lines.extend(f"    {line}" for line in _holders(big))
        print("\n".join(lines))


def release(model: nn.Module, handle: Optional[OffloadHandle] = None) -> None:
    """Give back the memory a model holds: remove the offload hooks, drop every weight, free the GPU cache and
    PyTorch's cache of pinned memory.

    Pass the handle that install_offload() / offload_blocks() returned; without it the hooks keep the weights alive.
    The model ends up on the meta device and cannot run again. Safe to call twice."""
    trace = _RamTrace()
    trace.mark("start")
    if handle is not None:
        handle.remove()
    trace.mark("hooks removed")
    model.__dict__.pop("_trash_panda_state", None)
    _to_meta(model)
    trace.mark("weights dropped")

    gc.collect()
    trace.mark("gc")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # PyTorch keeps freed pinned (page-locked) memory for reuse, and empty_cache() does not touch it. Measured
        # on the test machine, the pinned buffers stayed in RAM until this call. It is a private function of
        # PyTorch, so it is only called when it exists.
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is not None:
            empty_host_cache()
    trace.mark("caches emptied")
    trace.report()
