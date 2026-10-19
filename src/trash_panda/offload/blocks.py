"""Finding the repeating blocks (transformer layers) of a model and measuring them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch.nn as nn


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


def find_blocks(model: nn.Module, *, min_fraction: float = 0.01, min_blocks: int = 2) -> List[BlockGroup]:
    """Find groups of repeated blocks: ModuleList / Sequential containers whose children all have the
    same class and that hold at least `min_fraction` of the model's bytes.

    Outermost groups win (a list nested inside a chosen block is not reported separately)."""
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
