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


def strip_net(key):
    return key[len("net."):]


@pytest.fixture
def renamed_checkpoint(tmp_path):
    """Like tiny_checkpoint, but every key in the file starts with "net.". Load it with key_map=strip_net."""
    torch.manual_seed(0)
    ref = Tiny().eval()
    path = tmp_path / "renamed.safetensors"
    save_file({"net." + k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    return ref, path


class WrittenTraces:
    """The trace files that recordings leave in the trace folder of a test."""

    def __init__(self, directory):
        self.directory = directory

    def files(self):
        return sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []

    def exists(self):
        return bool(self.files())

    def read_text(self, encoding="utf-8"):
        return self.files()[-1].read_text(encoding=encoding)          # the newest: the names are the times


@pytest.fixture
def trace_file(tmp_path, monkeypatch):
    """Switches recording on, into a folder of the test. Returns what has been written there."""
    directory = tmp_path / "traces"
    monkeypatch.setenv("TRASH_PANDA_TRACE", "1")
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(directory))
    return WrittenTraces(directory)
