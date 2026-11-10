"""Memory-tiered model loading and block offloading (Phase 1 of trash-panda)."""

from .blocks import BlockGroup, find_blocks, module_nbytes
from .checkpoint import Checkpoint, TensorInfo
from .formats import Kind, expand, parse_dtype, quantize_fp8
from .hardware import Hardware, detect_hardware
from .hooks import DiskReader, OffloadHandle, install_offload, offload_blocks, release
from .loader import stream_load
from .planner import BlockPlan, Plan, Tier, parse_bytes, plan
from .report import LoadReport
from .measure import measure_activation_vram
from .stage import PendingStage, preload, record_order, stage
from .coordinator import Coordinator, Step

__all__ = [
    "BlockGroup", "BlockPlan", "Checkpoint", "Coordinator", "DiskReader", "Hardware", "Kind", "LoadReport",
    "OffloadHandle", "PendingStage", "Plan", "Step", "TensorInfo", "Tier",
    "detect_hardware", "expand", "find_blocks", "install_offload", "measure_activation_vram", "module_nbytes",
    "offload_blocks", "release", "parse_bytes", "parse_dtype", "plan", "preload", "quantize_fp8", "record_order", "stream_load", "stage"
]
