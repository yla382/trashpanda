"""trash-panda: run models bigger than your GPU by streaming weights across GPU, RAM and disk."""

__version__ = "0.0.16"

# The offload API is imported lazily so that `import trash_panda` works even where torch is not installed yet.
_OFFLOAD_NAMES = (
    "BlockGroup", "BlockPlan", "Checkpoint", "DiskReader", "Hardware", "Kind", "LoadReport",
    "OffloadHandle", "Plan", "TensorInfo", "Tier",
    "detect_hardware", "expand", "find_blocks", "install_offload", "measure_activation_vram", "module_nbytes",
    "offload_blocks", "release", "parse_bytes", "parse_dtype", "plan", "quantize_fp8", "stream_load", "stage"
)

__all__ = ["__version__", *_OFFLOAD_NAMES]


def __getattr__(name):
    if name in _OFFLOAD_NAMES:
        from . import offload

        return getattr(offload, name)
    raise AttributeError(f"module 'trash_panda' has no attribute {name!r}")
