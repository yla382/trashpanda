"""`trash-panda hardware`: what this machine has, for a bug report."""

import platform

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def _gib(n) -> str:
    """Bytes in GiB (1024 ** 3), the unit that memory is sold and shown in, unlike the decimal GB of the rest of the library."""
    return f"{n / 1024 ** 3:.2f} GiB"

# What the rows that need a word of explanation mean. A row that is not here has an empty note.
NOTES = {
    "Device": "Where a model runs by default.",
    "Unified memory": "The GPU and the CPU share one memory pool, so splitting a model between them frees nothing.",
    "Commit headroom": "Windows: how much more memory the system lets programs reserve (RAM plus page file).",
    "CUDA": "The CUDA version this PyTorch was built for.",
    "Compute capability": "The GPU's architecture version, which decides what it supports.",
    "bf16": "The GPU supports bfloat16, a 16-bit float format used as a compute dtype.",
    "Native fp8": "The GPU computes in fp8 itself (compute capability 8.9 or higher).",
}


def rows() -> list:
    """(item, value) for each fact about this machine, in the order they are printed."""
    import torch

    from .. import __version__
    from ..offload.hardware import detect_hardware

    hw = detect_hardware()
    out = [
        ("trash-panda", f"v{__version__}"),
        ("Python", f"v{platform.python_version()}"),
        ("PyTorch", f"v{torch.__version__}"),
        ("OS", f"{platform.system()} {platform.release()} {platform.version()} ({platform.machine()})"),
        ("Device", hw.device),
        ("Unified memory", "yes" if hw.unified_memory else "no"),
        ("RAM", f"{_gib(hw.ram_available)} available of {_gib(hw.ram_total)}"),
    ]
    if hw.commit_available is not None:
        out.append(("Commit headroom", _gib(hw.commit_available)))
    if hw.device == "cuda":
        out += [
            ("GPU", hw.gpu_name),
            ("VRAM", f"{_gib(hw.vram_free)} free of {_gib(hw.vram_total)}"),
            ("CUDA", f"v{torch.version.cuda}" if torch.version.cuda else "none"),
            ("Compute capability", ".".join(str(n) for n in hw.compute_capability)),
            ("bf16", "yes" if hw.bf16_supported else "no"),
            ("Native fp8", "yes" if hw.native_fp8 else "no"),
        ]
    elif hw.unified_memory:
        out.append(("GPU memory", "shared with RAM"))
    else:
        out.append(("GPU", "none detected"))
    return out


def hardware_command() -> None:
    """Print the device, memory and versions of this machine. Paste it into a bug report."""
    table = Table()
    table.add_column("Item", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_column("Note", overflow="fold")
    for item, value in rows():
        table.add_row(item, Text(str(value)), Text(NOTES.get(item, ""), style="dim"))
    console.print(table)