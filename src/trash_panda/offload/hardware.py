"""Reading what the machine has: VRAM, RAM and (on Windows) the commit limit."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .report import fmt_bytes


@dataclass
class Hardware:
    vram_total: int = 0
    vram_free: int = 0
    ram_total: int = 0
    ram_available: int = 0
    commit_available: Optional[int] = None      # Windows: how much more the system will let processes commit
    gpu_name: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None
    bf16_supported: bool = False
    native_fp8: bool = False

    def summary(self) -> str:
        lines = []
        if self.gpu_name:
            lines.append(f"GPU: {self.gpu_name}, VRAM {fmt_bytes(self.vram_free)} free of {fmt_bytes(self.vram_total)}")
        else:
            lines.append("GPU: none detected")
        lines.append(f"RAM: {fmt_bytes(self.ram_available)} available of {fmt_bytes(self.ram_total)}")
        if self.commit_available is not None:
            lines.append(f"Commit limit headroom (RAM + page file): {fmt_bytes(self.commit_available)}")
        return "\n".join(lines)


def _windows_memory() -> Optional[Tuple[int, int, int]]:
    """(total physical, available physical, available commit) in bytes."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys), int(status.ullAvailPageFile)


def _read_ram() -> Tuple[int, int, Optional[int]]:
    """(total, available, commit_available or None)"""
    total = available = 0
    commit: Optional[int] = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        total, available = int(vm.total), int(vm.available)
    except ImportError:
        pass

    if sys.platform == "win32":
        win = _windows_memory()
        if win is not None:
            w_total, w_avail, commit = win
            total = total or w_total
            available = available or w_avail
    elif not total and os.path.exists("/proc/meminfo"):
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                name, _, rest = line.partition(":")
                info[name] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", total // 2)
    elif not total:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            available = total // 2                       # unknown: assume half
        except (ValueError, OSError, AttributeError):
            pass
    return total, available, commit


def detect_hardware(device_index: int = 0) -> Hardware:
    hw = Hardware()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        free, total = torch.cuda.mem_get_info(device_index)
        hw.vram_total, hw.vram_free = int(total), int(free)
        hw.gpu_name = props.name
        hw.compute_capability = (props.major, props.minor)
        hw.bf16_supported = bool(torch.cuda.is_bf16_supported())
        hw.native_fp8 = hw.compute_capability >= (8, 9)
    hw.ram_total, hw.ram_available, hw.commit_available = _read_ram()
    return hw
