import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

from .helpers import run  # noqa: E402


@pytest.fixture
def served(monkeypatch):
    """Replaces the server with a recorder, so the command returns at once."""
    calls = []

    def fake_serve(path, port=0, open_browser=True, echo=print):
        calls.append({"path": path, "port": port, "open_browser": open_browser})

    monkeypatch.setattr("trash_panda.ui.server.serve", fake_serve)
    return calls


def test_without_a_file_the_demo_is_shown_and_said_to_be_made_up(served):
    result, text = run("trace")
    assert result.exit_code == 0
    assert "demo" in text and "not a real run" in text
    assert served[0]["path"].name == "dummy_trace.json"


def test_a_given_file_is_served(served, tmp_path):
    from trash_panda.ui.load import DEMO

    path = tmp_path / "mine.json"
    path.write_text(DEMO.read_text(encoding="utf-8"), encoding="utf-8")
    result, text = run("trace", path)
    assert result.exit_code == 0
    assert served[0]["path"] == path
    assert "demo" not in text


def test_the_port_and_no_open_are_passed_on(served):
    run("trace", "--port", "8123", "--no-open")
    assert served[0]["port"] == 8123 and served[0]["open_browser"] is False


def test_the_browser_is_opened_by_default(served):
    run("trace")
    assert served[0]["open_browser"] is True


def test_a_missing_file_is_an_error(served, tmp_path):
    result, text = run("trace", tmp_path / "missing.json")
    assert result.exit_code == 2
    assert "does not exist" in text and not served


def test_a_file_that_is_not_a_trace_is_an_error_and_nothing_is_served(served, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    result, text = run("trace", path)
    assert result.exit_code == 1
    assert "bad.json" in text and "traceEvents" in text
    assert not served


def test_the_root_help_lists_trace():
    _, text = run("--help")
    assert "trace" in text


def test_trace_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, trash_panda.cli.trace, trash_panda.ui.load, trash_panda.ui.server; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
