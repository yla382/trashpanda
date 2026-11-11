import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload.blocks import CheckpointNames, find_blocks, find_blocks_in_checkpoint  # noqa: E402
from trash_panda.offload.checkpoint import Checkpoint  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.offload.planner import plan  # noqa: E402

_GB = 1024 ** 3


def checkpoint_of(tmp_path, names, size=8):
    """A checkpoint with one float32 tensor of `size` values per name."""
    path = tmp_path / "m.safetensors"
    save_file({n: torch.zeros(size) for n in names}, str(path))
    return Checkpoint(path)


def block_names(container, count, parts=("attn.weight", "mlp.weight")):
    return [f"{container}.{i}.{p}" for i in range(count) for p in parts]


def test_finds_the_same_blocks_as_the_model(tiny_checkpoint):
    _, path = tiny_checkpoint
    from_names = find_blocks_in_checkpoint(Checkpoint(path))
    from_model = find_blocks(meta_model())
    assert [g.name for g in from_names] == [g.name for g in from_model]
    assert from_names[0].prefixes == from_model[0].prefixes


def test_the_plan_matches_the_plan_made_from_the_model(tiny_checkpoint):
    _, path = tiny_checkpoint
    ck = Checkpoint(path)
    hw = Hardware(vram_total=_GB, vram_free=_GB, ram_total=8 * _GB, ram_available=8 * _GB)
    real = plan(meta_model(), path, hardware=hw, resident_blocks=1)
    estimate = plan(CheckpointNames(ck.keys()), ck, groups=find_blocks_in_checkpoint(ck), hardware=hw, resident_blocks=1)

    def rows(p):
        return [(b.prefix, b.tier, b.stored_bytes, b.expanded_bytes) for b in p.blocks]

    assert rows(estimate) == rows(real)
    assert estimate.other_bytes == real.other_bytes


def test_checkpoint_names_lists_the_keys_as_parameters(tmp_path):
    names = CheckpointNames(["a.weight", "b.bias"])
    assert [n for n, _ in names.named_parameters()] == ["a.weight", "b.bias"]
    assert list(names.named_buffers()) == []


def test_a_numbered_list_inside_a_block_is_not_reported_separately(tmp_path):
    names = block_names("blocks", 4) + [f"blocks.{i}.attn.layers.{j}.w" for i in range(4) for j in range(3)]
    groups = find_blocks_in_checkpoint(checkpoint_of(tmp_path, names))
    assert [g.name for g in groups] == ["blocks"]


def test_a_gap_in_the_numbers_is_not_a_group(tmp_path):
    names = [f"blocks.{i}.w" for i in (0, 1, 3)]
    assert find_blocks_in_checkpoint(checkpoint_of(tmp_path, names)) == []


def test_blocks_that_hold_different_tensors_are_not_a_group(tmp_path):
    names = block_names("blocks", 4) + ["blocks.4.attn.weight"]         # the last block lacks "mlp.weight"
    names += ["blocks.4.extra.weight"]
    assert find_blocks_in_checkpoint(checkpoint_of(tmp_path, names)) == []


def test_fewer_blocks_than_the_minimum_are_not_a_group(tmp_path):
    assert find_blocks_in_checkpoint(checkpoint_of(tmp_path, block_names("blocks", 1))) == []
    assert len(find_blocks_in_checkpoint(checkpoint_of(tmp_path, block_names("blocks", 2)))) == 1


def test_a_group_that_holds_too_little_of_the_model_is_left_out(tmp_path):
    ck = checkpoint_of(tmp_path, block_names("blocks", 3))
    assert len(find_blocks_in_checkpoint(ck)) == 1
    assert find_blocks_in_checkpoint(ck, min_fraction=1.5) == []


def test_skip_prefixes_leave_a_group_out(tmp_path):
    names = block_names("encoder.layers", 3) + block_names("decoder.layers", 3)
    groups = find_blocks_in_checkpoint(checkpoint_of(tmp_path, names), skip_prefixes=("encoder.",))
    assert [g.name for g in groups] == ["decoder.layers"]


def test_a_skip_prefix_with_a_trailing_dot_leaves_out_a_group_of_that_name(tmp_path):
    ck = checkpoint_of(tmp_path, block_names("blocks", 3))
    assert find_blocks_in_checkpoint(ck, skip_prefixes=("blocks.",)) == []
    assert find_blocks_in_checkpoint(ck, skip_prefixes=("other.",)) != []


def test_two_groups_are_both_found(tmp_path):
    names = block_names("encoder.layers", 3) + block_names("decoder.layers", 4)
    groups = find_blocks_in_checkpoint(checkpoint_of(tmp_path, names))
    assert sorted(g.name for g in groups) == ["decoder.layers", "encoder.layers"]


def test_a_checkpoint_without_numbered_blocks_has_no_groups(tmp_path):
    assert find_blocks_in_checkpoint(checkpoint_of(tmp_path, ["embed.weight", "head.weight"])) == []
