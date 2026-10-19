import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from .tiny_model import Tiny  # noqa: E402


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """(reference model with real weights, path of its safetensors checkpoint) in float32."""
    torch.manual_seed(0)
    ref = Tiny().eval()
    path = tmp_path / "tiny.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    return ref, path
