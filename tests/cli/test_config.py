import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

from trash_panda.cli import config as config_cli  # noqa: E402
from trash_panda.offload import config  # noqa: E402

from .helpers import cells, run  # noqa: E402


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No TRASH_PANDA_ variable from the machine that runs the tests."""
    import os

    for name in list(os.environ):
        if name.upper().startswith("TRASH_PANDA_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config_cli.console, "width", 200)


def test_every_setting_and_both_folders_get_a_row():
    result, _ = run("config")
    assert result.exit_code == 0
    names = [var for var, _, _ in config_cli.rows()]
    assert names[:2] == ["TRASH_PANDA_HOME", "TRASH_PANDA_TRACE_DIR"]
    assert {config.ENV_PREFIX + f.name.upper() for f in config.fields(config.DEFAULTS)} <= set(names)
    for name in names:
        cells(result, name)


def test_a_value_nobody_set_is_marked_default():
    result, _ = run("config")
    assert cells(result, "TRASH_PANDA_LOOKAHEAD")[1] == f"{config.DEFAULTS.lookahead} (default)"
    assert cells(result, "TRASH_PANDA_TRACE")[1] == "false (default)"


def test_a_value_from_the_environment_is_shown_without_default(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "0")
    monkeypatch.setenv("TRASH_PANDA_TRACE", "yes")
    result, _ = run("config")
    assert cells(result, "TRASH_PANDA_LOOKAHEAD")[1] == "0"
    assert cells(result, "TRASH_PANDA_TRACE")[1] == "true"


def test_the_folders_show_where_files_go(monkeypatch, tmp_path):
    result, _ = run("config")
    assert cells(result, "TRASH_PANDA_HOME")[1].endswith("(default)")
    monkeypatch.setenv("TRASH_PANDA_HOME", str(tmp_path))
    result, _ = run("config")
    assert "(default)" not in cells(result, "TRASH_PANDA_HOME")[1]
    assert "(default)" in cells(result, "TRASH_PANDA_TRACE_DIR")[1]      # still <cache folder>/traces
    assert str(tmp_path) in cells(result, "TRASH_PANDA_TRACE_DIR")[1]


def test_an_empty_variable_counts_as_not_set(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "")
    result, _ = run("config")
    assert cells(result, "TRASH_PANDA_LOOKAHEAD")[1].endswith("(default)")


def test_a_value_that_cannot_be_read_is_marked_and_the_exit_code_is_1(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHEAD", "many")
    result, text = run("config")
    assert result.exit_code == 1
    assert cells(result, "TRASH_PANDA_LOOKAHEAD")[1] == "many (invalid)"
    assert "cannot be read" in text


def test_an_unknown_variable_is_listed_as_ignored(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_LOOKAHED", "1")
    result, text = run("config")
    assert result.exit_code == 0
    assert "TRASH_PANDA_LOOKAHED" in text and "ignored" in text


def test_the_root_help_lists_the_config_command():
    _, text = run("--help")
    assert "config" in text
