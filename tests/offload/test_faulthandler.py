import faulthandler
import io

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


@pytest.fixture
def watched(monkeypatch):
    """faulthandler is off and every call to enable() is recorded; pytest has its own faulthandler on, so the real one is replaced."""
    calls = []
    monkeypatch.delenv("TRASH_PANDA_FAULTHANDLER", raising=False)
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)
    monkeypatch.setattr(faulthandler, "enable", lambda **kw: calls.append(kw))
    return calls


def _load(path):
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")


def test_loading_a_model_turns_faulthandler_on_for_every_thread(tiny_checkpoint, watched):
    _, path = tiny_checkpoint
    _load(path)
    assert watched == [{"all_threads": True}]


def test_it_stays_as_it_is_when_faulthandler_is_on_already(tiny_checkpoint, watched, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: True)
    _load(path)
    assert watched == []


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_the_setting_turns_it_off(tiny_checkpoint, watched, monkeypatch, value):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_FAULTHANDLER", value)
    _load(path)
    assert watched == []


def test_a_stderr_without_a_file_descriptor_is_not_an_error(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_FAULTHANDLER", raising=False)
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)

    def no_descriptor(**kw):
        raise io.UnsupportedOperation("fileno")

    monkeypatch.setattr(faulthandler, "enable", no_descriptor)
    _load(path)


def test_nothing_is_printed(tiny_checkpoint, watched, capsys):
    _, path = tiny_checkpoint
    _load(path)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
