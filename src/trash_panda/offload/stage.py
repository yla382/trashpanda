"""Running one model of a pipeline: build, plan, load, run, release."""

from __future__ import annotations
from contextlib import contextmanager
from typing import Dict
import torch
from .hooks import install_offload, release
from .loader import stream_load
from .measure import _peak_since, _start_peak
from .planner import plan

_activations: Dict[str, int] = {}  # key -> measured activation bytes, kept for the life of the process

@contextmanager
def stage(build, path, *, key=None, key_map=None, skip_prefixes=(), order=None, order_file_destination=None, **plan_kwargs):
    """Load one model for the `with` block and give its memory back afterwards.
    `build` returns the model on the meta device. `path` is its checkpoint. `plan_kwargs` go to plan().
    The peak GPU memory of the block is remembered under `key` (default: the checkpoint path) and used as 
    `activation_vram` the next time the same key is planned, unless plan_kwargs set it. Nothing is remembered when the block raises.
    `order` names the blocks that run first (see install_offload()); the rest follow in the plan's order. When it is
    None, the order the blocks ran in is saved to `<checkpoint name>_order.json` (in `order_file_destination`, by
    default next to the checkpoint) and used the next time.
    The block gets `(model, plan, handle)`; `handle.final_pass()` frees each block as it runs during the last call.
    The model cannot be used after the block."""
    k = key or str(path)
    model = build()
    plan_kwargs.setdefault("activation_vram", _activations.get(k))
    p = plan(model, path, key_map=key_map, skip_prefixes=skip_prefixes, **plan_kwargs)
    stream_load(model, path, plan=p, key_map=key_map, skip_prefixes=skip_prefixes)
    handle = install_offload(model, order=order, order_file_destination=order_file_destination)
    before = _start_peak() if torch.cuda.is_available() else None
    try:
        yield model, p, handle
        if before is not None:
            _activations[k] = _peak_since(before, p)
    finally:
        release(model, handle)
