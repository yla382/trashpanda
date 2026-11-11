import pytest

pytest.importorskip("torch")
pytest.importorskip("typer")
pytest.importorskip("rich")
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from .helpers import cells, run  # noqa: E402


def checkpoint_of(tmp_path, tensors):
    path = tmp_path / "m.safetensors"
    save_file(tensors, str(path))
    return path


def test_the_header_line_has_the_name_size_and_counts(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("info", path)
    assert result.exit_code == 0
    assert "tiny.safetensors" in text and "1 file" in text
    assert "tensors" in text


def test_data_types_are_listed_with_their_share(tmp_path):
    path = checkpoint_of(tmp_path, {
        "a.weight": torch.zeros(100, dtype=torch.bfloat16),            # 200 bytes
        "b.weight": torch.zeros(150, dtype=torch.float32),             # 600 bytes
    })
    result, _ = run("info", path)
    assert cells(result, "float32")[1:] == ["1", "600 B", "75%"]
    assert cells(result, "bfloat16")[1:] == ["1", "200 B", "25%"]


def test_a_group_row_says_how_many_blocks_and_what_each_holds(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("info", path)
    group = cells(result, "blocks.<N>.")
    assert group[1] == "0 to 3"                                        # the blocks are blocks.0. to blocks.3.
    assert group[2] == "4"                                             # four blocks
    assert group[5] == "4"                                             # norm weight and bias, and two linear weights


def test_blocks_of_different_sizes_show_a_range(tmp_path):
    tensors = {f"blocks.{i}.w": torch.zeros(10 * (i + 1)) for i in range(3)}
    result, _ = run("info", checkpoint_of(tmp_path, tensors))
    assert cells(result, "blocks.<N>.")[3] == "40 B to 120 B"


def test_each_group_gets_a_row(tmp_path):
    tensors = {f"blocks.{i}.w": torch.zeros(8) for i in range(3)}
    tensors.update({f"fusion.layers.{i}.w": torch.zeros(8) for i in range(2)})
    result, _ = run("info", checkpoint_of(tmp_path, tensors))
    assert cells(result, "blocks.<N>.")[1:3] == ["0 to 2", "3"]
    assert cells(result, "fusion.layers.<N>.")[1:3] == ["0 to 1", "2"]


def test_a_few_tensors_outside_the_blocks_are_listed_by_their_full_names(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("info", path)
    assert "Outside the blocks" in text
    for name in ("embed.weight", "embed.bias", "head.weight", "head.bias"):
        assert cells(result, name)[1] == "1"


def test_many_tensors_under_one_name_become_one_row(tmp_path):
    tensors = {f"blocks.{i}.w": torch.zeros(8) for i in range(3)}
    tensors.update({f"proj.layer{i}.weight": torch.zeros(4) for i in range(5)})
    tensors["final.weight"] = torch.zeros(4)
    result, _ = run("info", checkpoint_of(tmp_path, tensors))
    assert cells(result, "proj.*")[1] == "5"
    assert cells(result, "final.weight")[1] == "1"
    with pytest.raises(AssertionError):
        cells(result, "proj.layer0.weight")


def test_the_outside_total_is_in_the_title(tmp_path):
    tensors = {f"blocks.{i}.w": torch.zeros(8) for i in range(3)}
    tensors["head.weight"] = torch.zeros(250)                          # 1000 bytes
    _, text = run("info", checkpoint_of(tmp_path, tensors))
    assert "Outside the blocks: 1.0 KB" in text or "Outside the blocks: 1000 B" in text


def test_one_block_is_shown_with_its_tensors(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("info", path)
    assert "Inside one block (blocks.0.)" in text
    row = cells(result, "lin1.weight")
    assert row[1] == "float32" and row[2] == "32 x 16"
    assert cells(result, "norm.bias")[2] == "16"


def test_a_checkpoint_without_blocks_says_so(tmp_path):
    result, text = run("info", checkpoint_of(tmp_path, {"a.weight": torch.zeros(4), "b.weight": torch.zeros(4)}))
    assert result.exit_code == 0
    assert "No repeating blocks found" in text
    assert "Inside one block" not in text


def test_keys_lists_every_tensor_only_when_asked(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, plain = run("info", path)
    result, listed = run("info", path, "--keys")
    assert "blocks.3.lin2.weight" not in plain
    assert cells(result, "blocks.3.lin2.weight")[1] == "float32"
    assert "Every tensor" in listed


def test_a_missing_file_is_an_error(tmp_path):
    result, text = run("info", tmp_path / "missing.safetensors")
    assert result.exit_code == 2
    assert "does not exist" in text


def test_the_root_help_lists_info():
    _, text = run("--help")
    assert "info" in text and "plan" in text
