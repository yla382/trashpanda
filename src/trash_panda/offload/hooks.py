"""Streaming blocks through the compute device during the forward pass.

For every wrapped block, forward() first expands the block's compact weights on the compute device
(reading them from the checkpoint file first, for disk blocks), runs, and then ALWAYS puts the compact
originals back (try/finally), so a failed forward pass cannot leave a block expanded.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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
) -> OffloadHandle:
    """Low-level: wrap each block so its weights are expanded on `device` only while it runs.

    Blocks whose weights are already on the device in the compute dtype are left untouched.
    `disk` (advanced) maps a block to a DiskReader when its weights live in the checkpoint file."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    handle = OffloadHandle()

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

        def bring(slots=slots, scales=scales, reader=reader):
            data = reader.read() if reader is not None else None
            for name, owner, leaf, is_param, orig in slots:
                if data is not None:
                    raw, scale = data[name]
                else:
                    raw, scale = orig, scales.get(name)
                t = expand(raw, scale, compute_dtype, dev) if raw.is_floating_point() else raw.to(dev)
                if is_param:
                    owner._parameters[leaf] = nn.Parameter(t, requires_grad=False)
                else:
                    owner._buffers[leaf] = t

        def restore(slots=slots):
            for _name, owner, leaf, is_param, orig in slots:
                if is_param:
                    owner._parameters[leaf] = orig
                else:
                    owner._buffers[leaf] = orig

        original_forward = block.forward

        def wrapped(*args, _orig=original_forward, _bring=bring, _restore=restore, **kwargs):
            _bring()
            try:
                return _orig(*args, **kwargs)
            finally:
                _restore()

        def undo(block=block, restore=restore):
            restore()
            block.__dict__.pop("forward", None)

        block.forward = wrapped
        handle._add(undo)

    return handle


def install_offload(
    model: nn.Module,
    plan: Optional[Plan] = None,
    *,
    device: Union[str, torch.device, None] = None,
    compute_dtype: Optional[torch.dtype] = None,
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

    handle = OffloadHandle()
    for b in plan.blocks:
        block = model.get_submodule(b.prefix.rstrip("."))
        reader_factory = None
        if b.tier == Tier.DISK:
            if state is None:
                raise ValueError("Disk blocks need a model loaded with stream_load().")
            reader_factory = lambda _blk, p=b.prefix: DiskReader(state.checkpoint, state.disk_keys, p)
        handle.extend(offload_blocks([block], device=dev, compute_dtype=cdt, disk=reader_factory))
    return handle
