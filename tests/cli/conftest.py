import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from trash_panda.offload.hardware import Hardware  # noqa: E402

from ..offload.tiny_model import Tiny  # noqa: E402

_GB = 1024 ** 3


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """(reference model with real weights, path of its safetensors checkpoint) in float32."""
    torch.manual_seed(0)
    ref = Tiny().eval()
    path = tmp_path / "tiny.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    return ref, path


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Tables are drawn 200 columns wide, so a name is never wrapped over two lines and the tests can read whole rows."""
    from trash_panda.cli import info, plan
    monkeypatch.setattr(plan.console, "width", 200)
    monkeypatch.setattr(info.console, "width", 200)


@pytest.fixture(autouse=True)
def fixed_hardware(monkeypatch):
    """A machine with plenty of memory, whatever runs the tests."""
    hw = Hardware(vram_total=16 * _GB, vram_free=16 * _GB, ram_total=64 * _GB, ram_available=64 * _GB, gpu_name="Test GPU")
    monkeypatch.setattr("trash_panda.offload.hardware.detect_hardware", lambda *a, **k: hw)
