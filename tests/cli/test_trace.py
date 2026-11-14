import os

import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

from typer.testing import CliRunner  # noqa: E402

from trash_panda.cli.app import app  # noqa: E402

from ..ui.helpers import SIMPLE  # noqa: E402
from .helpers import run  # noqa: E402


@pytest.fixture
def served(monkeypatch):
    """Replaces the server with a recorder, so the command returns at once."""
    calls = []

    def fake_serve(path, port=0, open_browser=True, echo=print, directory=None):
        calls.append({"path": path, "port": port, "open_browser": open_browser, "directory": directory})

    monkeypatch.setattr("trash_panda.ui.server.serve", fake_serve)
    return calls


def test_a_file_is_served(served):
    result, text = run("trace", SIMPLE)
    assert result.exit_code == 0
    assert served[0]["path"] == SIMPLE


def test_the_trace_folder_is_given_to_the_server_for_its_sidebar(served, tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path))
    run("trace", SIMPLE)
    assert served[0]["directory"] == tmp_path


def test_without_a_file_the_newest_of_the_trace_folder_is_served(served, tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path))
    for name in ("20261112-100000-000.json", "20261114-093012-482.json", "20261113-080000-100.json"):
        (tmp_path / name).write_text(SIMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    result, _ = run("trace")
    assert result.exit_code == 0
    assert served[0]["path"] == tmp_path / "20261114-093012-482.json"


def test_without_a_file_and_without_recordings_it_says_so(served, tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path / "none"))
    result, text = run("trace")
    assert result.exit_code == 1 and "No trace files" in text and "TRASH_PANDA_TRACE=1" in text and not served


def test_the_port_and_no_open_are_passed_on(served):
    run("trace", SIMPLE, "--port", "8123", "--no-open")
    assert served[0]["port"] == 8123 and served[0]["open_browser"] is False


def test_the_browser_is_opened_by_default(served):
    run("trace", SIMPLE)
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


# -- --list and --clean

def folder(tmp_path, monkeypatch, names=("20261112-100000-000.json", "20261113-080000-100.json", "20261114-093012-482.json")):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path))
    for i, name in enumerate(names):
        path = tmp_path / name
        path.write_text(SIMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        os.utime(path, (1_700_000_000 + i * 86400 * 10, 1_700_000_000 + i * 86400 * 10))
    return [tmp_path / n for n in names]


def test_list_shows_the_files_newest_first_with_their_models(served, tmp_path, monkeypatch):
    folder(tmp_path, monkeypatch)
    result, text = run("trace", "--list")
    assert result.exit_code == 0 and not served
    assert text.index("20261114-093012-482.json") < text.index("20261113-080000-100.json") < text.index("20261112-100000-000.json")
    assert "Models" in text and "Duration" in text


def test_list_marks_a_file_that_is_not_a_trace_as_invalid(served, tmp_path, monkeypatch):
    folder(tmp_path, monkeypatch, names=("20261114-093012-482.json",))
    (tmp_path / "20261115-000000-000.json").write_text("{}", encoding="utf-8")
    _, text = run("trace", "--list")
    assert "invalid" in text


def test_list_of_an_empty_folder_says_so(served, tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path))
    result, text = run("trace", "--list")
    assert result.exit_code == 0 and "No trace files" in text


def test_list_and_clean_take_no_file(served):
    for flag in ("--list", "--clean"):
        result, text = run("trace", SIMPLE, flag)
        assert result.exit_code == 2 and "give no file" in text


def test_clean_keeps_the_newest_and_asks_first(served, tmp_path, monkeypatch):
    files = folder(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["trace", "--clean", "--keep", "1"], input="n" + chr(10))
    assert result.exit_code == 1 and all(f.exists() for f in files)               # declined: nothing is removed
    result = CliRunner().invoke(app, ["trace", "--clean", "--keep", "1"], input="y" + chr(10))
    assert result.exit_code == 0
    assert [f.exists() for f in files] == [False, False, True]


def test_clean_with_yes_does_not_ask(served, tmp_path, monkeypatch):
    files = folder(tmp_path, monkeypatch)
    result, text = run("trace", "--clean", "--keep", "2", "--yes")
    assert result.exit_code == 0 and "Removed 1 file" in text
    assert [f.exists() for f in files] == [False, True, True]


def test_clean_older_than_removes_only_the_old_files(served, tmp_path, monkeypatch):
    files = folder(tmp_path, monkeypatch)
    os.utime(files[2], None)                                                      # the newest was just written
    run("trace", "--clean", "--older-than", "30d", "--yes")
    assert [f.exists() for f in files] == [False, False, True]


def test_clean_with_both_limits_removes_only_what_breaks_both(served, tmp_path, monkeypatch):
    files = folder(tmp_path, monkeypatch)
    run("trace", "--clean", "--keep", "2", "--older-than", "30d", "--yes")
    assert [f.exists() for f in files] == [False, True, True]                     # all are old; only one is beyond the newest two


def test_clean_without_limits_removes_every_file_after_asking(served, tmp_path, monkeypatch):
    files = folder(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["trace", "--clean"], input="y" + chr(10))
    assert result.exit_code == 0 and not any(f.exists() for f in files)


def test_clean_with_nothing_to_remove_says_so(served, tmp_path, monkeypatch):
    folder(tmp_path, monkeypatch)
    result, text = run("trace", "--clean", "--keep", "5", "--yes")
    assert result.exit_code == 0 and "Nothing to remove" in text


def test_clean_rejects_a_bad_age_and_stray_options(served, tmp_path, monkeypatch):
    folder(tmp_path, monkeypatch)
    result, text = run("trace", "--clean", "--older-than", "soon", "--yes")
    assert result.exit_code == 2 and "not an age" in text
    result, text = run("trace", "--keep", "1")
    assert result.exit_code == 2 and "belong to --clean" in text
    result, text = run("trace", "--list", "--clean")
    assert result.exit_code == 2 and "not both" in text
