"""Measuring how much GPU memory the activation of a real run need."""

from __future__ import annotations
from typing import Callable, Optional
import torch
from .planner import Plan

def _start_peak(device=None) -> int:
    """Wait for the GPU, reset the peak counter and return the memory allocated now (int bytes)"""
    dev = torch.device(device or "cuda")
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    return torch.cuda.memory_allocated(dev)


def _peak_since(before: int, plan: Optional[Plan] = None, device=None) -> int:
    """The peak GPU memory since _start_peak() returned `before`, minus the streaming buffer and the staged-ahead blocks the plan already accounts for (int bytes)."""
    dev = torch.device(device or "cuda")
    torch.cuda.synchronize(dev)
    extra  = torch.cuda.max_memory_allocated(dev) - before
    reserved = (plan.stream_buffer_bytes + plan.staging_bytes) if plan is not None else 0
    return max(0, extra - reserved)


def measure_activation_vram(fn: Callable[[], object], plan: Optional[Plan] = None, device=None) -> int:
    """Run fn() once and return the peak GPU memory it needed beyond what was already allocated, minus the streaming buffer the plan already accounts for (int bytes)."""
    before = _start_peak(device)
    fn()
    return _peak_since(before, plan, device)