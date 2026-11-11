"""`trash-panda info`: what is inside a checkpoint."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .names import group_pattern, index_range

console = Console()
_COLLAPSE_AT = 4        # this many tensors under one name are shown as one row ("name.*"), fewer are listed one by one


def _dtype_name(info):
    dtype = info.torch_dtype
    return str(dtype).replace("torch.", "") if dtype is not None else info.dtype


def _shape(info):
    return " x ".join(str(d) for d in info.shape) or "scalar"


def _share(part, total):
    if not total or not part:
        return "0%"
    pct = 100 * part / total
    return "<1%" if pct < 1 else f"{pct:.0f}%"


def _split_blocks(ck, groups):
    """(per group: {block index: [bytes, tensors]}, the keys that are in no block)."""
    sizes = {g.name: {} for g in groups}
    outside = []
    for key in ck.keys():
        for g in groups:
            head = g.name + "."
            index = key[len(head):].split(".", 1)[0] if key.startswith(head) else ""
            if index.isdigit():
                entry = sizes[g.name].setdefault(int(index), [0, 0])
                entry[0] += ck.info(key).nbytes
                entry[1] += 1
                break
        else:
            outside.append(key)
    return sizes, outside


def _dtypes_table(ck):
    from ..offload.report import fmt_bytes

    totals = {}
    for key in ck.keys():
        info = ck.info(key)
        entry = totals.setdefault(_dtype_name(info), [0, 0])
        entry[0] += 1
        entry[1] += info.nbytes
    table = Table(title="Data types", title_justify="left")
    table.add_column("Dtype")
    table.add_column("Tensors", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Share", justify="right")
    total = ck.total_bytes()
    for name, (count, nbytes) in sorted(totals.items(), key=lambda item: -item[1][1]):
        table.add_row(escape(name), f"{count:,}", fmt_bytes(nbytes), _share(nbytes, total))
    return table


def _groups_table(groups, sizes):
    from ..offload.report import fmt_bytes

    table = Table(title="Block groups", title_justify="left")
    table.add_column("Group", overflow="fold")      # a name is wrapped, never cut
    table.add_column("N", justify="right")
    table.add_column("Blocks", justify="right")
    table.add_column("Per block", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Tensors per block", justify="right")
    for g in groups:
        blocks = sizes[g.name]
        each = [blocks[i][0] for i in sorted(blocks)]
        per_block = fmt_bytes(each[0]) if min(each) == max(each) else f"{fmt_bytes(min(each))} to {fmt_bytes(max(each))}"
        table.add_row(escape(group_pattern(g.name)), index_range(blocks), str(len(each)), per_block,
                      fmt_bytes(sum(each)), str(blocks[min(blocks)][1]))
    return table


def _outside_table(ck, outside):
    """The tensors that are in no block. Many under one name become one row; a few are listed by their full names."""
    from ..offload.report import fmt_bytes

    by_name = {}
    for key in outside:
        by_name.setdefault(key.split(".")[0], []).append(key)
    rows = []                                                       # (name, tensors, bytes)
    for name, keys in by_name.items():
        if len(keys) >= _COLLAPSE_AT:
            rows.append((f"{name}.*", len(keys), sum(ck.info(k).nbytes for k in keys)))
        else:
            rows.extend((k, 1, ck.info(k).nbytes) for k in keys)
    table = Table(title=f"Outside the blocks: {fmt_bytes(sum(ck.info(k).nbytes for k in outside))}", title_justify="left")
    table.add_column("Name", overflow="fold")
    table.add_column("Tensors", justify="right")
    table.add_column("Size", justify="right")
    for name, count, nbytes in sorted(rows, key=lambda row: -row[2]):
        table.add_row(escape(name), f"{count:,}", fmt_bytes(nbytes))
    return table


def _tensors_table(ck, keys, title, strip=""):
    from ..offload.report import fmt_bytes

    table = Table(title=title, title_justify="left")
    table.add_column("Tensor", overflow="fold")
    table.add_column("Dtype")
    table.add_column("Shape")
    table.add_column("Size", justify="right")
    for key in keys:
        info = ck.info(key)
        table.add_row(escape(key[len(strip):]), escape(_dtype_name(info)), _shape(info), fmt_bytes(info.nbytes))
    return table


def info_command(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True, help="A .safetensors file, or a sharded checkpoint."),
    keys: bool = typer.Option(False, "--keys", help="Also list every tensor."),
) -> None:
    """Show what is inside a checkpoint: its size, data types, groups of repeated blocks and the rest.

    The blocks are found from the names and sizes in the checkpoint, the same way `plan` does. Sizes are as stored in
    the file, so the part outside the blocks can differ from what `plan` counts, which is its size expanded to the
    compute dtype.
    """
    from ..offload.blocks import find_blocks_in_checkpoint     # torch is imported here, not at startup
    from ..offload.checkpoint import Checkpoint
    from ..offload.report import fmt_bytes

    ck = Checkpoint(checkpoint)
    groups = find_blocks_in_checkpoint(ck)
    sizes, outside = _split_blocks(ck, groups)

    files = len(ck.files)
    console.print(f"{checkpoint.name}   {fmt_bytes(ck.total_bytes())}   {len(ck):,} tensors   "
                  f"{files} file{'s' if files != 1 else ''}", markup=False)
    console.print(_dtypes_table(ck))
    if groups:
        console.print(_groups_table(groups, sizes))
    else:
        console.print("No repeating blocks found.", markup=False)
    if outside:
        console.print(_outside_table(ck, outside))
    if groups:
        first = groups[0].prefixes[0]
        inside = sorted((k for k in ck.keys() if k.startswith(first)), key=lambda k: -ck.info(k).nbytes)     # biggest first
        console.print(_tensors_table(ck, inside, f"Inside one block ({first})", first))
    if keys:
        console.print(_tensors_table(ck, ck.keys(), "Every tensor"))
