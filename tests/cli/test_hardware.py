import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

from trash_panda import __version__  # noqa: E402
from trash_panda.cli import hardware as hardware_cli  # noqa: E402
from trash_panda.offload.hardware import Hardware, detect_hardware  # noqa: E402

from .helpers import cells, run  # noqa: E402

_GIB = 1024 ** 3        # the command shows GiB, as memory is sold


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    monkeypatch.setattr(hardware_cli.console, "width", 200)


def _machine(monkeypatch, **fields):
    """Makes the command see this machine instead of the one running the tests."""
    hw = Hardware(**fields)
    monkeypatch.setattr("trash_panda.offload.hardware.detect_hardware", lambda *a, **k: hw)


def test_a_cuda_machine_shows_the_gpu_and_the_vram(monkeypatch):
    _machine(monkeypatch, device="cuda", gpu_name="Test GPU", vram_total=16 * _GIB, vram_free=12 * _GIB,
             ram_total=64 * _GIB, ram_available=32 * _GIB, compute_capability=(8, 6), bf16_supported=True)
    result, _ = run("hardware")
    assert result.exit_code == 0
    assert cells(result, "Device")[1] == "cuda"
    assert cells(result, "GPU")[1] == "Test GPU"
    assert cells(result, "VRAM")[1] == "12.00 GiB free of 16.00 GiB"
    assert cells(result, "Compute capability")[1] == "8.6"
    assert cells(result, "bf16")[1] == "yes"
    assert cells(result, "Native fp8")[1] == "no"
    assert cells(result, "Unified memory")[1] == "no"
    assert cells(result, "RAM")[1] == "32.00 GiB available of 64.00 GiB"


def test_the_versions_are_shown(monkeypatch):
    _machine(monkeypatch)
    result, _ = run("hardware")
    assert cells(result, "trash-panda")[1] == f"v{__version__}"
    for item in ("Python", "PyTorch"):
        assert cells(result, item)[1].startswith("v")
    assert cells(result, "OS")[1]


def test_a_cuda_machine_with_shared_memory_shows_both(monkeypatch):
    _machine(monkeypatch, device="cuda", unified_memory=True, gpu_name="Test GPU", vram_total=16 * _GIB, vram_free=12 * _GIB,
             compute_capability=(12, 1))
    result, _ = run("hardware")
    assert cells(result, "Unified memory")[1] == "yes"
    assert cells(result, "GPU")[1] == "Test GPU"
    assert cells(result, "VRAM")[1] == "12.00 GiB free of 16.00 GiB"


def test_a_mac_says_its_gpu_memory_is_shared(monkeypatch):
    _machine(monkeypatch, device="mps", unified_memory=True, ram_total=32 * _GIB, ram_available=16 * _GIB)
    result, _ = run("hardware")
    assert cells(result, "Device")[1] == "mps"
    assert cells(result, "Unified memory")[1] == "yes"
    assert cells(result, "GPU memory")[1] == "shared with RAM"


def test_a_machine_without_a_gpu_says_so(monkeypatch):
    _machine(monkeypatch, ram_total=8 * _GIB, ram_available=4 * _GIB)
    result, _ = run("hardware")
    assert cells(result, "Device")[1] == "cpu"
    assert cells(result, "GPU")[1] == "none detected"


def test_the_commit_headroom_is_shown_only_when_there_is_one(monkeypatch):
    _machine(monkeypatch, commit_available=5 * _GIB)
    result, _ = run("hardware")
    assert cells(result, "Commit headroom")[1] == "5.00 GiB"
    _machine(monkeypatch)
    result, text = run("hardware")
    assert "Commit headroom" not in text


def test_detect_hardware_finds_the_apple_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    hw = detect_hardware()
    assert (hw.device, hw.unified_memory, hw.vram_total) == ("mps", True, 0)


def test_detect_hardware_without_any_gpu_falls_back_to_the_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    hw = detect_hardware()
    assert (hw.device, hw.unified_memory, hw.gpu_name) == ("cpu", False, None)


def test_the_root_help_lists_the_hardware_command():
    _, text = run("--help")
    assert "hardware" in text
