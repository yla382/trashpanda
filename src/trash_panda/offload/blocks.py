"""Finding the repeating blocks (transformer layers) of a model and measuring them."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn

from . import config
from .checkpoint import Checkpoint


@dataclass
class BlockGroup:
    name: str                 # module path of the container, e.g. "blocks"
    prefixes: List[str]       # one parameter-name prefix per block, e.g. ["blocks.0.", "blocks.1.", ...]
    nbytes: int               # size of the whole group in the model's own dtype


def module_nbytes(module: nn.Module) -> int:
    """Parameter + buffer bytes of a module (works for meta-device modules too)."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


def find_blocks(model: nn.Module, *, min_fraction: Optional[float] = None, min_blocks: Optional[int] = None) -> List[BlockGroup]:
    """Find groups of repeated blocks: ModuleList / Sequential containers whose children all have the
    same class and that hold at least `min_fraction` of the model's bytes (default 0.01) in at least `min_blocks`
    blocks (default 2). The defaults come from config.

    Outermost groups win (a list nested inside a chosen block is not reported separately)."""
    cfg = config.current()
    min_fraction = cfg.find_blocks_min_fraction if min_fraction is None else min_fraction
    min_blocks   = cfg.find_blocks_min_blocks if min_blocks is None else min_blocks
    total = module_nbytes(model)
    groups: List[BlockGroup] = []
    for name, mod in model.named_modules():
        if not name:
            continue
        if any(name.startswith(g.name + ".") for g in groups):
            continue
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(mod.named_children())
        if len(children) < min_blocks:
            continue
        if len({type(c) for _, c in children}) != 1:
            continue
        nbytes = module_nbytes(mod)
        if total and nbytes / total >= min_fraction:
            groups.append(BlockGroup(name, [f"{name}.{cname}." for cname, _ in children], nbytes))
    return groups


class CheckpointNames:
    """Stands in for a model when only the checkpoint is known: plan() asks it for the parameter names."""

    def __init__(self, names):
        self._names = list(names)

    def named_parameters(self):
        return ((n, None) for n in self._names)

    def named_buffers(self):
        return iter(())


def find_blocks_in_checkpoint(ck: Checkpoint, skip_prefixes=(), min_fraction=None, min_blocks=None) -> List[BlockGroup]:
    """Like find_blocks(), but from the key names: a container whose children are numbered 0, 1, 2 ... and
    all hold the same tensors. Outermost groups win."""
    cfg = config.current()
    min_fraction = cfg.find_blocks_min_fraction if min_fraction is None else min_fraction
    min_blocks = cfg.find_blocks_min_blocks if min_blocks is None else min_blocks
    total = ck.total_bytes()
    found = defaultdict(lambda: defaultdict(dict))       # container -> index -> {rest of the key: bytes}
    for key in ck.keys():
        parts = key.split(".")
        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                found[".".join(parts[:i])][int(part)][".".join(parts[i + 1:])] = ck.info(key).nbytes
    groups: List[BlockGroup] = []
    for name in sorted(found, key=lambda n: (n.count("."), n)):          # outermost first
        if not name or (name + ".").startswith(tuple(skip_prefixes)) or any(name.startswith(g.name + ".") for g in groups):
            continue
        blocks = found[name]
        if sorted(blocks) != list(range(len(blocks))) or len(blocks) < min_blocks:
            continue
        if len({frozenset(b) for b in blocks.values()}) != 1:             # every block holds the same tensors
            continue
        nbytes = sum(sum(b.values()) for b in blocks.values())
        if total and nbytes / total >= min_fraction:
            groups.append(BlockGroup(name, [f"{name}.{i}." for i in range(len(blocks))], nbytes))
    return groups
