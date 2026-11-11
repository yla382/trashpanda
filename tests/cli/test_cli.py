import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("typer")
pytest.importorskip("rich")
from safetensors.torch import save_file  # noqa: E402

import torch  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402

from .helpers import cells, run  # noqa: E402

_GB = 1024 ** 3
_SRC = Path(__file__).resolve().parents[2] / "src"


def test_plan_puts_everything_on_the_gpu_when_it_fits(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("plan", path)
    assert result.exit_code == 0
    assert "4 blocks" in text
    group = cells(result, "blocks.<N>.")
    assert group[1] == "0 to 3" and group[2] == "4"
    assert group[3].startswith("4 (") and group[4] == "-" and group[5] == "-"


def test_resident_blocks_splits_the_blocks_between_gpu_and_ram(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("plan", path, "--resident-blocks", 1)
    assert result.exit_code == 0
    group = cells(result, "blocks.<N>.")
    assert group[3].startswith("1 (") and group[4].startswith("3 (") and group[5] == "-"


def test_no_vram_sends_the_blocks_to_ram(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("plan", path, "--resident-blocks", 0)
    assert result.exit_code == 0
    group = cells(result, "blocks.<N>.")
    assert group[3] == "-" and group[4].startswith("4 (")


def test_a_tight_ram_budget_sends_blocks_to_disk_unless_it_is_switched_off(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, with_disk = run("plan", path, "--resident-blocks", 0, "--max-ram", "1B", "--min-free-ram", "0")
    _, without = run("plan", path, "--resident-blocks", 0, "--max-ram", "1B", "--min-free-ram", "0", "--no-disk")
    assert "read from disk on every pass" in with_disk
    assert "read from disk on every pass" not in without
    assert "disk is not allowed" in without


def test_skip_leaves_the_blocks_out(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("plan", path, "--skip", "blocks.")
    assert result.exit_code == 0
    assert "No repeating blocks found" in text


def test_a_checkpoint_without_blocks_prints_a_warning(tmp_path):
    path = tmp_path / "flat.safetensors"
    save_file({"a.weight": torch.zeros(4, 4)}, str(path))
    result, text = run("plan", path)
    assert result.exit_code == 0
    assert "0 blocks" in text and "No repeating blocks found" in text


def test_the_hardware_is_printed(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, text = run("plan", path)
    assert "GPU: none detected" not in text
    assert "VRAM" in text and "RAM:" in text


def test_a_missing_file_is_an_error(tmp_path):
    result, text = run("plan", tmp_path / "missing.safetensors")
    assert result.exit_code == 2
    assert "does not exist" in text


def test_a_bad_size_is_an_error(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("plan", path, "--max-vram", "lots")
    assert result.exit_code != 0


def test_help_says_the_plan_is_an_estimate():
    _, text = run("plan", "--help")
    assert "estimate" in text
    assert "--min-free-ram" in text


def test_the_root_help_lists_the_plan_command():
    _, text = run("--help")
    assert "plan" in text


def _python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(_SRC)})


def test_importing_the_cli_does_not_import_torch():
    done = _python("import sys, trash_panda.cli; print('torch' in sys.modules)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_without_typer_the_cli_says_which_extra_to_install():
    done = _python("import sys; sys.modules['typer'] = None; import trash_panda.cli")
    assert done.returncode != 0
    assert "dumpsterpanda[cli]" in done.stderr


def test_the_budgets_are_broken_down(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, text = run("plan", path, "--resident-blocks", 2)
    assert result.exit_code == 0
    gpu, ram = cells(result, "GPU"), cells(result, "RAM")
    assert gpu[2] == "2 blocks kept on the GPU"
    assert ram[2] == "2 blocks"
    assert "Input and output layers" in text and "Unused" in text
    assert "kept for activations" in text and "kept for the system and copy buffers" in text


def test_a_ram_budget_that_is_used_up_says_where_the_blocks_stay(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("plan", path, "--resident-blocks", 0, "--max-ram", "1B", "--min-free-ram", "0")
    assert cells(result, "RAM")[1] == "0 B"
    assert cells(result, "Disk")[2] == "4 blocks that did not fit in RAM"


def test_blocks_that_do_not_fit_when_disk_is_off_show_as_over_the_budget(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, text = run("plan", path, "--resident-blocks", 0, "--max-ram", "1B", "--min-free-ram", "0", "--no-disk")
    assert "Over the budget" in text


def two_group_checkpoint(tmp_path):
    path = tmp_path / "two.safetensors"
    tensors = {"head.weight": torch.zeros(8, 8)}
    for i in range(4):
        tensors[f"blocks.{i}.w"] = torch.zeros(64, 64)
    for i in range(2):
        tensors[f"fusion.layers.{i}.w"] = torch.zeros(64, 64)
    save_file(tensors, str(path))
    return path


def test_each_group_gets_a_row_and_there_is_a_total(tmp_path):
    result, _ = run("plan", two_group_checkpoint(tmp_path))
    assert result.exit_code == 0
    assert cells(result, "blocks.<N>.")[1:3] == ["0 to 3", "4"]
    assert cells(result, "fusion.layers.<N>.")[1:3] == ["0 to 1", "2"]
    assert cells(result, "Total")[2] == "6"


def test_a_single_group_has_no_total_row(tiny_checkpoint):
    _, path = tiny_checkpoint
    result, _ = run("plan", path)
    with pytest.raises(AssertionError):
        cells(result, "Total")


def test_blocks_lists_every_block_only_when_asked(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, plain = run("plan", path)
    result, listed = run("plan", path, "--blocks", "--resident-blocks", 1)
    assert "blocks.0." not in plain
    assert [cells(result, f"blocks.{i}.")[1] for i in range(4)] == ["gpu", "ram", "ram", "ram"]


def test_the_limits_come_from_the_arguments_when_they_are_given(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, given = run("plan", path, "--max-vram", "1GB", "--max-ram", "1GB")
    _, free = run("plan", path)
    assert "allowed by --max-vram" in given and "allowed by --max-ram" in given
    assert "free now" in free and "available now" in free


def test_the_reservable_memory_line_appears_only_when_it_is_known(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    _, without = run("plan", path)
    assert "still let programs reserve" not in without
    known = Hardware(vram_total=16 * _GB, vram_free=16 * _GB, ram_total=64 * _GB, ram_available=64 * _GB,
                     commit_available=40 * _GB, gpu_name="Test GPU")
    monkeypatch.setattr("trash_panda.offload.hardware.detect_hardware", lambda *a, **k: known)
    _, with_it = run("plan", path)
    assert "still let programs reserve (RAM plus page file or swap)" in with_it
    assert "Windows" not in with_it and "Commit limit" not in with_it
