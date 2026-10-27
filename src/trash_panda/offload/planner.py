"""Deciding where each block lives: GPU (resident), RAM (streamed) or disk (streamed).

A `Plan` is a plain, editable object. `plan()` builds one automatically from the checkpoint's sizes and
the machine's free memory; `Plan.manual()` builds one by hand. Everything downstream (loader, hooks) just
executes whatever plan it is given, so automatic and manual plans behave identically afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from . import config
from .blocks import BlockGroup, find_blocks
from .checkpoint import Checkpoint
from .formats import expanded_nbytes, parse_dtype, stored_nbytes
from .hardware import Hardware, detect_hardware
from .report import fmt_bytes

_GB = 1024 ** 3
_PINNED_BUFFERS = config.DEFAULTS.pinned_buffers   # kept for older imports; plan() reads config.current()

def _auto_ram_reserve(hw: Hardware, cfg: Optional[config.Defaults] = None) -> int:
    """RAM kept free for the OS and other programs: a share of the machine's RAM, within a lower and an upper limit
    (8%, between 1 and 4 GB by default, see config)"""
    cfg = cfg or config.current()
    if not hw.ram_total:
        return int(cfg.ram_reserve_unknown_gb * _GB)
    return int(min(max(cfg.ram_reserve_fraction * hw.ram_total, cfg.ram_reserve_min_gb * _GB), cfg.ram_reserve_max_gb * _GB))


class Tier(str, Enum):
    GPU = "gpu"      # resident: kept on the compute device
    RAM = "ram"      # waits in system RAM, streamed to the device when its block runs
    DISK = "disk"    # stays in the checkpoint file, read when its block runs


@dataclass
class BlockPlan:
    prefix: str                     # e.g. "blocks.3."
    tier: Tier = Tier.RAM
    stored_bytes: int = 0           # size while waiting (compact form)
    expanded_bytes: int = 0         # size once expanded to the compute dtype


@dataclass
class Plan:
    blocks: List[BlockPlan]
    compute_dtype: torch.dtype = torch.bfloat16
    storage_dtype: Optional[torch.dtype] = None
    other_bytes: int = 0                 # weights outside any block (kept expanded on the device)
    stream_buffer_bytes: int = 0         # room for one expanded streamed block (+ temporaries)
    staging_bytes: int = 0               # room for the blocks staged on the device ahead of time
    lookahead: int = field(default_factory=lambda: config.current().lookahead)   # how many blocks ahead they are staged
    vram_budget: int = 0
    ram_budget: int = 0
    warnings: List[str] = field(default_factory=list)
    source: str = "auto"                 # "auto" or "manual"

    # -- reading the plan
    def prefixes(self, tier: Tier) -> Tuple[str, ...]:
        return tuple(b.prefix for b in self.blocks if b.tier == tier)

    @property
    def resident(self) -> Tuple[str, ...]:
        return self.prefixes(Tier.GPU)

    @property
    def streamed(self) -> Tuple[str, ...]:
        return self.prefixes(Tier.RAM)

    @property
    def disk(self) -> Tuple[str, ...]:
        return self.prefixes(Tier.DISK)

    # -- editing the plan
    def assign(
        self,
        tier: Tier,
        *,
        indices: Optional[Iterable[int]] = None,
        prefixes: Optional[Iterable[str]] = None,
    ) -> "Plan":
        """Move blocks to `tier`, selected by position (`indices`) and/or `prefixes`. Returns self."""
        wanted = {_norm_prefix(p) for p in (prefixes or [])}
        idx = set(indices or [])
        for i, b in enumerate(self.blocks):
            if i in idx or b.prefix in wanted:
                b.tier = Tier(tier)
        return self

    @classmethod
    def manual(
        cls,
        *,
        resident: Sequence[str] = (),
        streamed: Sequence[str] = (),
        disk: Sequence[str] = (),
        compute_dtype: Union[str, torch.dtype] = "bf16",
        storage_dtype: Union[None, str, torch.dtype] = "auto",
    ) -> "Plan":
        """Build a plan by hand from lists of block prefixes ("blocks.0." ...)."""
        blocks = (
            [BlockPlan(_norm_prefix(p), Tier.GPU) for p in resident]
            + [BlockPlan(_norm_prefix(p), Tier.RAM) for p in streamed]
            + [BlockPlan(_norm_prefix(p), Tier.DISK) for p in disk]
        )
        return cls(
            blocks=blocks,
            compute_dtype=parse_dtype(compute_dtype) or torch.bfloat16,
            storage_dtype=parse_dtype(storage_dtype),
            source="manual",
        )

    def summary(self) -> str:
        dtype_name = str(self.compute_dtype).replace("torch.", "")
        lines = [f"Plan ({self.source}): {len(self.blocks)} blocks, compute dtype {dtype_name}"]
        for tier, label in (
            (Tier.GPU, "resident on the GPU"),
            (Tier.RAM, "streamed from RAM"),
            (Tier.DISK, "streamed from disk"),
        ):
            chosen = [b for b in self.blocks if b.tier == tier]
            if chosen:
                size = fmt_bytes(sum(b.stored_bytes for b in chosen)) if any(b.stored_bytes for b in chosen) else "size unknown"
                lines.append(f"  {len(chosen):>3} blocks {label}: {size}")
        if self.other_bytes:
            lines.append(f"  weights outside blocks (on the device): {fmt_bytes(self.other_bytes)}")
        if self.stream_buffer_bytes:
            lines.append(f"  room reserved for one streamed block: {fmt_bytes(self.stream_buffer_bytes)}")
        if self.staging_bytes:
            lines.append(f"  room reserved for blocks staged ahead (look-ahead {self.lookahead}): {fmt_bytes(self.staging_bytes)}")
        if self.vram_budget:
            lines.append(f"  VRAM budget: {fmt_bytes(self.vram_budget)}   RAM budget: {fmt_bytes(self.ram_budget)}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    __str__ = summary


def _norm_prefix(p: str) -> str:
    return p if p.endswith(".") else p + "."


_UNITS = {
    "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
    "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4,
}


def parse_bytes(value: Union[None, int, float, str]) -> Optional[int]:
    """'9GB' / '8GiB' / 1500000000 -> bytes. None -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace(" ", "")
    for unit in sorted(_UNITS, key=len, reverse=True):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * _UNITS[unit])
    return int(float(s))


def _is_scale_key(key: str, ck: Checkpoint) -> bool:
    return key.endswith("_scale") and key[: -len("_scale")] in ck

def _spread_disk_blocks(rest: List[BlockPlan], ram_budget: int) -> set:
    """Choose which of the blocks outside the GPU go to disk: as few as it takes for the others to fit in `ram_budget`, 
    spared evenly so that each disk read has other blocks' compute next to it to hide behind."""
    total = sum(b.stored_bytes for b in rest)
    m     = len(rest)
    for k in range(m + 1):
        chosen = {int((i + 0.5) * m / k) for i in range(k)} if k else set()
        if total - sum(rest[i].stored_bytes for i in chosen) <= ram_budget:
            return chosen
    return set(range(m))


def plan(
    model: nn.Module,
    checkpoint: Union[str, Checkpoint],
    *,
    compute_dtype: Union[str, torch.dtype] = "bf16",
    storage_dtype: Union[None, str, torch.dtype] = "auto",
    resident_blocks: Optional[int] = None,
    max_vram: Union[None, int, str] = None,
    headroom_vram: Union[None, int, str] = None,
    activation_vram: Union[None, int, str] = None,
    max_ram: Union[None, int, str] = None,
    min_free_ram: Union[None, int, str] = None,
    allow_disk: bool = True,
    lookahead: Optional[int] = None,
    spread_disk: bool = True,
    hardware: Optional[Hardware] = None,
    groups: Optional[List[BlockGroup]] = None,
) -> Plan:
    """Decide where each block lives. Every argument that is set explicitly overrides the automatic choice.
    Arguments left as None take their default from config (see config.py, which also lists the TRASH_PANDA_<NAME>
    environment overrides).

    resident_blocks  force this many blocks (from the start) to be GPU-resident instead of "as many as fit"
    max_vram         VRAM to use (default: what is free right now)
    headroom_vram    VRAM kept free for activations, the CUDA context and other programs (default "1.5GB")
    activation_vram  measured GPU memory the activations need (see measure_activation_vram); replaces headroom_cram
    max_ram          RAM to use for streamed blocks (default: what is available right now)
    min_free_ram     RAM kept free for the operating system and other programs ("auto": 8% of the machine's RAM, 1 to 4GB)
    allow_disk       blocks that do not fit in RAM stay in the checkpoint file (slow) instead of RAM
    lookahead        how many blocks ahead the next weights are staged on the GPU(0 turns prefetching off); the plan leaves room for them
    spread_disk      put the disk blocks evenly among the others instead of at the end, so reads can overlap with compute
    """
    cfg = config.current()
    headroom_vram = cfg.headroom_vram if headroom_vram is None else headroom_vram
    min_free_ram  = cfg.min_free_ram if min_free_ram is None else min_free_ram
    lookahead     = cfg.lookahead if lookahead is None else lookahead
    cdt = parse_dtype(compute_dtype) or torch.bfloat16
    sdt = parse_dtype(storage_dtype)
    ck = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
    hw = hardware if hardware is not None else detect_hardware()
    found = list(groups) if groups is not None else find_blocks(model)    # its thresholds also come from config
    warnings: List[str] = []
    if not found:
        warnings.append("No repeating blocks found in the model; nothing can be offloaded.")

    # -- sizes from the checkpoint headers
    blocks: List[BlockPlan] = []
    in_block = set()
    for g in found:
        for prefix in g.prefixes:
            keys = ck.keys_with_prefix(prefix)
            if not keys:
                warnings.append(f"No checkpoint tensors found for block '{prefix}'.")
            stored = expanded = 0
            for k in keys:
                info = ck.info(k)
                in_block.add(k)
                stored += stored_nbytes(k, info.shape, info.torch_dtype, info.nbytes, sdt)
                if not _is_scale_key(k, ck) and not k.endswith(".comfy_quant"):
                    expanded += expanded_nbytes(info.shape, info.torch_dtype, cdt)
            blocks.append(BlockPlan(prefix, Tier.RAM, stored, expanded))

    other = 0
    for k in ck.keys():
        if k in in_block or k.endswith(".comfy_quant") or _is_scale_key(k, ck):
            continue
        info = ck.info(k)
        other += expanded_nbytes(info.shape, info.torch_dtype, cdt)

    # -- budgets
    activation = parse_bytes(activation_vram)
    if activation is not None:
        margin   = max(int(cfg.activation_margin_fraction * hw.vram_total), int(cfg.activation_margin_min_gb * _GB)) # allocator slack and fragmentation
        headroom = activation + margin
    else:
        headroom = parse_bytes(headroom_vram) or 0
    
    vram_cap = parse_bytes(max_vram)
    vram_budget = (vram_cap if vram_cap is not None else hw.vram_free) - headroom
    vram_budget = max(vram_budget, 0)
    if hw.vram_total == 0 and vram_cap is None:
        warnings.append("No CUDA GPU detected and no max_vram given: nothing can be placed on the GPU.")

    buffer  = int(max((b.expanded_bytes for b in blocks), default=0) * cfg.stream_buffer_factor)   # expanded copy + temporaries
    staging = max(int(lookahead), 0) * max((b.stored_bytes for b in blocks), default=0) # blocks waiting on the gPU
    needs_expand = any(b.stored_bytes != b.expanded_bytes for b in blocks)

    # -- GPU-resident blocks
    n = len(blocks)
    if resident_blocks is not None:
        n_res = max(0, min(int(resident_blocks), n))
    else:
        everything = other + sum(b.stored_bytes for b in blocks) + (buffer if needs_expand else 0)
        if everything <= vram_budget:
            n_res = n
        else:
            room = vram_budget - other - buffer - staging
            n_res = 0
            for b in blocks:
                if b.stored_bytes <= room:
                    room -= b.stored_bytes
                    n_res += 1
                else:
                    break
    for i, b in enumerate(blocks):
        b.tier = Tier.GPU if i < n_res else Tier.RAM

    resident_bytes = sum(b.stored_bytes for b in blocks[:n_res])
    reserved = (buffer + staging) if n_res < n else (buffer if needs_expand else 0)
    if other + resident_bytes + reserved > vram_budget:
        warnings.append(
            f"The GPU part of the plan needs {fmt_bytes(other + resident_bytes + reserved)} but the VRAM budget is "
            f"{fmt_bytes(vram_budget)}; expect out-of-memory errors or slow spilling into shared memory."
        )

    # -- RAM vs disk for the rest
    ram_cap = parse_bytes(max_ram)
    if isinstance(min_free_ram, str) and min_free_ram.strip().lower() == "auto":
        ram_reserve  = _auto_ram_reserve(hw, cfg)
    else:
        ram_reserve = parse_bytes(min_free_ram) or 0
   
    # the prefetcher's pinned staging buffers: page-locked RAM that cannot be paged out, only when there is a GPU
    pinned     = cfg.pinned_buffers * max((b.stored_bytes for b in blocks[n_res:]), default=0) if hw.vram_total else 0
    ram_budget = (ram_cap if ram_cap is not None else hw.ram_available) - ram_reserve - pinned
    ram_budget = max(ram_budget, 0)
    
    rest = blocks[n_res:]
    if spread_disk and allow_disk:
        on_disk = _spread_disk_blocks(rest, ram_budget)
        for i, b in enumerate(rest):
            b.tier = Tier.DISK if i in on_disk else Tier.RAM
    else:
        ram_left = ram_budget
        for b in rest:
            if b.stored_bytes <= ram_left:
                b.tier = Tier.RAM
                ram_left -= b.stored_bytes
            elif allow_disk:
                b.tier = Tier.DISK
            else:
                b.tier = Tier.RAM
                warnings.append(f"Block '{b.prefix}' does not fit in the RAM budget and disk is not allowed.")

    n_disk = sum(1 for b in blocks if b.tier == Tier.DISK)
    if n_disk:
        warnings.append(
            f"{n_disk} blocks will be read from disk on every pass; this is slow unless the drive is fast (NVMe)."
        )
        if sdt is not None:
            warnings.append("storage_dtype is not applied to disk blocks: they are read as stored in the file.")
    ram_used = sum(b.stored_bytes for b in blocks if b.tier == Tier.RAM)
    if hw.commit_available is not None and ram_used > cfg.commit_warn_fraction * hw.commit_available:
        warnings.append(
            f"Streamed blocks need {fmt_bytes(ram_used)} but only {fmt_bytes(hw.commit_available)} can still be "
            "committed (RAM + page file); loading may fail with 'paging file too small'."
        )

    return Plan(
        blocks=blocks,
        compute_dtype=cdt,
        storage_dtype=sdt,
        other_bytes=other,
        stream_buffer_bytes=buffer,
        staging_bytes=staging if n_res < n else 0,
        lookahead=lookahead,
        vram_budget=vram_budget,
        ram_budget=ram_budget,
        warnings=warnings,
        source="auto",
    )
