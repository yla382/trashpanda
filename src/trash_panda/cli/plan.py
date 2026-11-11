"""`trash-panda plan`: where each block of a model would live on this machine."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .names import group_pattern, index_range

console = Console()
_COLORS = {"gpu": "green", "ram": "cyan", "disk": "yellow"}


def _blocks(n):
    return f"{n} block" if n == 1 else f"{n} blocks"


def _hardware_lines(hw):
    from ..offload.report import fmt_bytes

    if hw.gpu_name:
        lines = [f"GPU: {hw.gpu_name}, VRAM {fmt_bytes(hw.vram_free)} free of {fmt_bytes(hw.vram_total)}"]
    else:
        lines = ["GPU: none detected"]
    lines.append(f"RAM: {fmt_bytes(hw.ram_available)} available of {fmt_bytes(hw.ram_total)}")
    if hw.commit_available is not None:          # only measured on some systems
        lines.append(f"Memory the system will still let programs reserve (RAM plus page file or swap): "
                     f"{fmt_bytes(hw.commit_available)}")
    return lines


def _groups_table(p, groups):
    """What the model is made of: one row per group of repeated blocks, and how many of them wait in each tier."""
    from ..offload.planner import Tier
    from ..offload.report import fmt_bytes

    name_of = {prefix: g.name for g in groups for prefix in g.prefixes}
    by_group = {}
    for b in p.blocks:
        by_group.setdefault(name_of.get(b.prefix, b.prefix), []).append(b)

    def cell(blocks):
        return f"{len(blocks)} ({fmt_bytes(sum(b.stored_bytes for b in blocks))})" if blocks else "-"

    def row(label, numbers, blocks):
        return [label, numbers, str(len(blocks))] + [cell([b for b in blocks if b.tier == t]) for t in (Tier.GPU, Tier.RAM, Tier.DISK)]

    def numbers(name, blocks):
        """The block numbers of a group: its prefixes are `name.<number>.`"""
        return index_range([int(b.prefix[len(name) + 1:-1]) for b in blocks])

    table = Table()
    table.add_column("Group", overflow="fold")      # a name is wrapped, never cut
    table.add_column("N", justify="right")
    table.add_column("Blocks", justify="right")
    for label, tier in (("GPU", "gpu"), ("RAM", "ram"), ("Disk", "disk")):
        table.add_column(label, header_style=_COLORS[tier], justify="right")
    for name, blocks in by_group.items():
        table.add_row(*row(escape(group_pattern(name)), numbers(name, blocks), blocks))
    if len(by_group) > 1:
        table.add_section()
        table.add_row(*row("Total", "", p.blocks))
    return table


def _blocks_table(p):
    """Every block with its tier (the `--blocks` option)."""
    from ..offload.report import fmt_bytes

    table = Table()
    table.add_column("Block", overflow="fold")
    table.add_column("Tier")
    table.add_column("Size", justify="right")
    for b in p.blocks:
        table.add_row(escape(b.prefix), f"[{_COLORS[b.tier.value]}]{b.tier.value}[/]", fmt_bytes(b.stored_bytes))
    return table


def _memory_table(p, hw, max_vram, max_ram, min_free_ram):
    """The GPU and RAM budgets and what fills them, so the numbers of the plan can be followed. Returns the table and
    the lines that say where each budget comes from."""
    from ..offload.planner import Tier, parse_bytes, ram_overhead, vram_kept_back
    from ..offload.report import fmt_bytes

    table = Table()
    table.add_column("")
    table.add_column("Budget")
    table.add_column("What it holds")
    table.add_column("Size", justify="right")

    def tier_rows(label, color, budget, items, budget_bytes=None):
        items = [(text, size) for text, size in items if size]
        if budget_bytes is not None:
            unused = budget_bytes - sum(size for _, size in items)
            if unused > 0:
                items.append(("Unused", unused))
            elif unused < 0:
                items.append(("[red]Over the budget[/]", -unused))
        if not items:
            items = [("Nothing", 0)]
        for i, (text, size) in enumerate(items):
            table.add_row(f"[{color}]{label}[/]" if i == 0 else "", budget if i == 0 else "", text,
                          fmt_bytes(size) if size else "-")

    by_tier = {t: [b for b in p.blocks if b.tier == t] for t in (Tier.GPU, Tier.RAM, Tier.DISK)}
    stored = lambda blocks: sum(b.stored_bytes for b in blocks)    # noqa: E731

    workspace = vram_kept_back(p) - p.headroom_bytes
    tier_rows("GPU", _COLORS["gpu"], fmt_bytes(p.vram_budget), [
        (f"{_blocks(len(by_tier[Tier.GPU]))} kept on the GPU" if by_tier[Tier.GPU] else "", stored(by_tier[Tier.GPU])),
        ("Input and output layers", p.other_bytes),
        ("Workspace for streaming blocks in", max(workspace, 0)),
    ], p.vram_budget)
    table.add_section()
    tier_rows("RAM", _COLORS["ram"], fmt_bytes(p.ram_budget), [
        (_blocks(len(by_tier[Tier.RAM])) if by_tier[Tier.RAM] else "", stored(by_tier[Tier.RAM])),
    ], p.ram_budget)
    if by_tier[Tier.DISK]:
        table.add_section()
        tier_rows("Disk", _COLORS["disk"], "none", [
            (f"{_blocks(len(by_tier[Tier.DISK]))} that did not fit in RAM", stored(by_tier[Tier.DISK])),
        ])

    vram_base = parse_bytes(max_vram) if max_vram else hw.vram_free
    ram_base = parse_bytes(max_ram) if max_ram else hw.ram_available
    notes = [
        f"GPU budget: {fmt_bytes(vram_base)} {'allowed by --max-vram' if max_vram else 'free now'}, "
        f"less {fmt_bytes(p.headroom_bytes)} kept for activations",
        f"RAM budget: {fmt_bytes(ram_base)} {'allowed by --max-ram' if max_ram else 'available now'}, "
        f"less {fmt_bytes(ram_overhead(p, hw, min_free_ram))} kept for the system and copy buffers",
    ]
    return table, notes


def plan_command(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True, help="A .safetensors file, or a sharded checkpoint."),
    max_vram: Optional[str] = typer.Option(None, help='VRAM to plan for, e.g. "9GB" (default: what is free now).'),
    max_ram: Optional[str] = typer.Option(None, help='RAM to plan for, e.g. "10GB" (default: what is available now).'),
    min_free_ram: Optional[str] = typer.Option(None, help='RAM kept free for the system: "auto" or a size such as "2GB".'),
    resident_blocks: Optional[int] = typer.Option(None, help="Force this many blocks onto the GPU."),
    headroom_vram: Optional[str] = typer.Option(None, help="VRAM kept free for activations."),
    compute_dtype: str = typer.Option("bf16", help="Dtype a block is expanded to when it runs."),
    storage_dtype: str = typer.Option("auto", help='Dtype blocks wait in: "auto", "fp8", ...'),
    no_disk: bool = typer.Option(False, "--no-disk", help="Keep blocks in RAM instead of reading them from disk."),
    skip: List[str] = typer.Option([], "--skip", help="Leave out checkpoint entries with this prefix (repeatable)."),
    blocks: bool = typer.Option(False, "--blocks", help="Also list every block with its tier."),
) -> None:
    """Print where each block of a model would live on this machine, without loading it.

    This is an estimate: the blocks are found from the names and sizes in the checkpoint, not from the model's code.
    stage() plans from the real model and can differ, for example when a block has extra tensors or the checkpoint
    needs a key_map.
    """
    from ..offload.blocks import CheckpointNames, find_blocks_in_checkpoint     # torch is imported here, not at startup
    from ..offload.checkpoint import Checkpoint
    from ..offload.hardware import detect_hardware
    from ..offload.planner import plan
    from ..offload.report import fmt_bytes

    ck = Checkpoint(checkpoint)
    hw = detect_hardware()
    groups = find_blocks_in_checkpoint(ck, skip_prefixes=skip)
    p = plan(CheckpointNames(ck.keys()), ck, groups=groups, hardware=hw, skip_prefixes=skip,
             max_vram=max_vram, max_ram=max_ram, min_free_ram=min_free_ram, resident_blocks=resident_blocks,
             headroom_vram=headroom_vram, compute_dtype=compute_dtype, storage_dtype=storage_dtype,
             allow_disk=not no_disk)

    for line in _hardware_lines(hw):
        console.print(line, markup=False)
    console.print(f"{checkpoint.name}: {len(p.blocks)} blocks, {fmt_bytes(ck.total_bytes())}", markup=False)
    console.print(_groups_table(p, groups))
    if blocks:
        console.print(_blocks_table(p))
    memory, notes = _memory_table(p, hw, max_vram, max_ram, min_free_ram)
    console.print(memory)
    for note in notes:
        console.print(note, markup=False)
    if p.warnings:
        console.print()
    for w in p.warnings:
        console.print(f"WARNING: {w}", style="yellow", markup=False)
