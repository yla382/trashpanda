import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from .tiny_model import Tiny, meta_model  # noqa: E402
from trash_panda.offload.blocks import find_blocks, module_nbytes  # noqa: E402


def test_finds_the_repeated_blocks():
    groups = find_blocks(meta_model())
    assert [g.name for g in groups] == ["blocks"]
    assert groups[0].prefixes == [f"blocks.{i}." for i in range(4)]


def test_module_nbytes_counts_parameters():
    layer = nn.Linear(10, 10, bias=False)          # 100 float32 values
    assert module_nbytes(layer) == 400


def test_works_on_meta_models():
    model = meta_model()
    assert module_nbytes(model) == module_nbytes(Tiny())


def test_ignores_lists_of_different_modules():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    assert find_blocks(model) == []


def test_ignores_tiny_lists():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.big = nn.Linear(1000, 1000)
            self.small = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    assert find_blocks(M()) == []
