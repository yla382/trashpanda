"""Streaming blocks through the compute device during the forward pass.

For every wrapped block, forward() first expands the block's compact weights on the compute device
(reading them from the checkpoint file first, for disk blocks), runs, and then ALWAYS puts the compact
originals back (try/finally), so a failed forward pass cannot leave a block expanded.
"""

from __future__ import annotations

import atexit
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import config
from .checkpoint import Checkpoint
from .formats import SCALE_SUFFIX, expand
from .loader import OffloadState, _navigate
from .planner import Plan, Tier


class OffloadHandle:
    """Returned by the install functions. remove() puts everything back the way it was."""

    def __init__(self) -> None:
        self._undo: List[Callable[[], None]] = []

    def _add(self, fn: Callable[[], None]) -> None:
        self._undo.append(fn)

    def extend(self, other: "OffloadHandle") -> None:
        self._undo.extend(other._undo)

    def remove(self) -> None:
        while self._undo:
            self._undo.pop()()


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

    def needs_copy(self) -> bool:
        """True if the block's weights are on disk or in CPU memory, so they must be copied to the GPU."""
        return self.reader is not None or any(slot[4].device.type == "cpu" for slot in self.slots)


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

    def acquire(self, nbytes: int):
        """(index, buffer) with room for nbytes, or None if pinned memory could not be allocated."""
        i = self.cursor
        self.cursor = (i + 1) % len(self.buffers)
        if self.events[i] is not None:
            self.events[i].synchronize()
            self.events[i] = None
        buf = self.buffers[i]
        if buf is None or buf.numel() < nbytes:
            try:
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except RuntimeError: # pinning failed (not enough memory, or no CUDA)
                return None
            self.buffers[i] = buf
        return i, buf


    def release(self, i: int, event) -> None:
        """Remember which GPU copy still reads from buffer i"""
        self.events[i] = event


class _Prefetcher:
    """Copies the next block's compact tensors to the device on a second CUDA stream"""

    profile = False

    def __init__(self, device: torch.device, lookahead: Optional[int] = None, profile: Optional[bool] = None):
        cfg = config.current()
        self.device    = device
        self.lookahead = cfg.lookahead if lookahead is None else lookahead
        self.index     = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None
        self.stream    = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.last      = None
        self.window    = []
        self.in_flight = deque()
        self.pool      = _PinnedPool(cfg.pinned_buffers) if self.stream is not None else None
        self.worker    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trash-panda") if self.stream is not None else None
        # profiling: where does the time go? Off by default; on with profile=True or TRASH_PANDA_PROFILE=1
        self.profile   = cfg.profile if profile is None else bool(profile)
        self.stats     = defaultdict(float)
        self.first     = None
        self.where     = {}                        # id(entry) -> its position in the pass, in the order first seen
        self.block_wait = defaultdict(float)       # position -> seconds the main thread waited for that block
        self.block_late = defaultdict(int)         # position -> times it was not ready when asked for
        self._reported = False
        if self.profile:
            atexit.register(self.report)

    def learn(self, entry) -> None:
        if self.profile:
            if self.first is None:
                self.first = entry
            if entry is self.first:
                self.stats["passes"] += 1               # the first block starts a new pass through the model
            self.where.setdefault(id(entry), len(self.where))
        if self.last is not None:
            self.last.next = entry
        self.last = entry

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
        started = time.perf_counter() if self.profile else 0.0
        try:
            tensors, event = staged.result()  # wait until the worker has issued the copies
        except Exception:
            import traceback
            traceback.print_exc()             # never hide a failure in the worker
            return None                       # fall back to the plain copy in bring()
        if self.profile:
            waited = time.perf_counter() - started
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
        if not self.profile or self._reported or not self.stats:
            return ""
        s = self.stats
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
        slowest = sorted(self.block_wait, key=self.block_wait.get, reverse=True)[:5]
        if slowest:
            lines.append("  slowest waits (block position in the pass, seconds per pass, passes it was late):")
            for position in slowest:
                lines.append(f"    block {position}: {self.block_wait[position] / passes:.2f} s, late in {self.block_late[position]} of {passes}")
        lines.append(f"  main thread waited for the GPU to catch up: {s['wait_gpu_s'] / passes:.2f} s")
        text = "\n".join(lines)
        print(text)
        return text

    def close(self) -> None:
        """Stop the worker thread and let go of the pinned buffers and anything still staged."""
        self.report()
        if self.worker is not None:
            self.worker.shutdown(wait=True)
            self.worker = None
        if self.stream is not None:
            self.stream.synchronize()

        self.pool   = None
        self.stream = None
        self.last   = None
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


def offload_blocks(
    blocks: Iterable[nn.Module],
    *,
    device: Union[str, torch.device, None] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    disk: Optional[Callable[[nn.Module], Optional[DiskReader]]] = None,
    prefetcher: Optional["_Prefetcher"] = None,
    lookahead: Optional[int] = None,
    profile: Optional[bool] = None,
) -> OffloadHandle:
    """Low-level: wrap each block so its weights are expanded on `device` only while it runs.

    Blocks whose weights are already on the device in the compute dtype are left untouched.
    `disk` (advanced) maps a block to a DiskReader when its weights live in the checkpoint file.
    `lookahead` is how many blocks ahead the next weights are copied (0 turns prefetching off; the default comes
    from config)."""
    dev    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    handle = OffloadHandle()
    if prefetcher is None:
        prefetcher = _Prefetcher(dev, lookahead, profile)
        handle._add(prefetcher.close)

    for block in blocks:
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
            continue

        entry  = _Entry(slots, scales, reader)

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

        def wrapped(*args, _orig=original_forward, _bring=bring, _restore=restore, _entry=entry, **kwargs):
            prefetcher.learn(_entry)
            prefetcher.wait_for_room()
            _bring()
            prefetcher.stage_ahead(_entry)

            try:
                return _orig(*args, **kwargs)
            finally:
                _restore()
                prefetcher.block_done()

        def undo(block=block, restore=restore, entry=entry):
            restore()
            block.__dict__.pop("forward", None)
            entry.staged = None

        block.forward = wrapped
        handle._add(undo)

    return handle


def install_offload(
    model: nn.Module,
    plan: Optional[Plan] = None,
    *,
    device: Union[str, torch.device, None] = None,
    compute_dtype: Optional[torch.dtype] = None,
    lookahead: Optional[int] = None,
    profile: Optional[bool] = None,
) -> OffloadHandle:
    """Install the offload hooks for every block in the plan (defaults to the plan used by stream_load)."""
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
    for b in plan.blocks:
        block = model.get_submodule(b.prefix.rstrip("."))
        reader_factory = None
        if b.tier == Tier.DISK:
            if state is None:
                raise ValueError("Disk blocks need a model loaded with stream_load().")
            reader_factory = lambda _blk, p=b.prefix: DiskReader(state.checkpoint, state.disk_keys, p)
        handle.extend(offload_blocks([block], device=dev, compute_dtype=cdt, disk=reader_factory, prefetcher=pre))
    return handle
