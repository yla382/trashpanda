"""A tiny transformer-like model used by the tests (runs on CPU in milliseconds)."""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, d * 2, bias=False)
        self.lin2 = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return x + self.lin2(torch.relu(self.lin1(self.norm(x))))


class Tiny(nn.Module):
    def __init__(self, d: int = 16, n: int = 4):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n))
        self.head = nn.Linear(d, 4)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def meta_model(d: int = 16, n: int = 4) -> Tiny:
    """The model structure without any weights in memory, as stream_load() expects."""
    with torch.device("meta"):
        return Tiny(d, n).eval()
