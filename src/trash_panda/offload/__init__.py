"""Memory-tiered model loading and block offloading (Phase 1 of trash-panda)."""

from .blocks import BlockGroup, find_blocks, module_nbytes
from .checkpoint import Checkpoint, TensorInfo
from .formats import Kind, expand, parse_dtype, quantize_fp8
from .hardware import Hardware, detect_hardware
from .hooks import DiskReader, OffloadHandle, install_offload, offload_blocks
from .loader import stream_load
from .planner import BlockPlan, Plan, Tier, parse_bytes, plan
from .report import LoadReport

__all__ = [
    "BlockGroup", "BlockPlan", "Checkpoint", "DiskReader", "Hardware", "Kind", "LoadReport",
    "OffloadHandle", "Plan", "TensorInfo", "Tier",
    "detect_hardware", "expand", "find_blocks", "install_offload", "module_nbytes",
    "offload_blocks", "parse_bytes", "parse_dtype", "plan", "quantize_fp8", "stream_load",
]
