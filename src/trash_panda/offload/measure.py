"""Measuring how much GPU memory the activation of a real run need."""

from __future__ import annotations
from typing import Callable, Optional
import torch
from .planner import Plan

def measure_activation_vram(fn: Callable[[], object], plan: Optional[Plan] = None, device=None) -> int:
    """Run fn() once and return the peak GPU memory it needed beyond what was already allocated, minus the streaming buffer the plan already accounts for (int bytes)."""
    dev = torch.device(device or "cuda")
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev)
    fn()
    torch.cuda.synchronize(dev)
    extra  = torch.cuda.max_memory_allocated(dev) - before
    buffer = plan.stream_buffer_bytes if plan is not None else 0
    return max(0, extra - buffer)