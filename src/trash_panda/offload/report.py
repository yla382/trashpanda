"""Plain-language summaries of what the loader did."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def fmt_bytes(n: float) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{int(n)} B"


_TIER_LABELS = {
    "normal": "loaded to the compute device (expanded)",
    "gpu_raw": "kept in compact form on the GPU",
    "ram": "kept in compact form in RAM",
    "disk": "left on disk (read on demand)",
}


@dataclass
class LoadReport:
    counts: Dict[str, int] = field(default_factory=dict)
    nbytes: Dict[str, int] = field(default_factory=dict)
    kinds: Dict[str, int] = field(default_factory=dict)
    unexpected: List[str] = field(default_factory=list)
    skipped: int = 0
    aux_keys: int = 0
    warnings: List[str] = field(default_factory=list)

    def add(self, tier: str, nbytes: int, kind: Optional[str] = None) -> None:
        self.counts[tier] = self.counts.get(tier, 0) + 1
        self.nbytes[tier] = self.nbytes.get(tier, 0) + int(nbytes)
        if kind is not None:
            self.kinds[kind] = self.kinds.get(kind, 0) + 1

    def __str__(self) -> str:
        lines = ["Load report:"]
        for tier in ("normal", "gpu_raw", "ram", "disk"):
            if tier in self.counts:
                lines.append(
                    f"  {_TIER_LABELS[tier]}: {self.counts[tier]} tensors, {fmt_bytes(self.nbytes[tier])}"
                )
        if self.kinds:
            lines.append("  stored as: " + ", ".join(f"{k} x{v}" for k, v in sorted(self.kinds.items())))
        if self.skipped:
            lines.append(f"  skipped (skip_prefixes): {self.skipped}")
        if self.aux_keys:
            lines.append(f"  auxiliary keys ignored (.comfy_quant): {self.aux_keys}")
        if self.unexpected:
            lines.append(
                f"  WARNING: {len(self.unexpected)} checkpoint keys are not in the model "
                f"(ignored), e.g. {self.unexpected[:3]}"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)
