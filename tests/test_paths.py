"""Where trash-panda keeps its files: the cache folder, the trace folder and the names of trace files."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from trash_panda import paths


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("TRASH_PANDA_HOME", raising=False)
    monkeypatch.delenv("TRASH_PANDA_TRACE_DIR", raising=False)


def test_the_cache_folder_is_in_the_home_folder_by_default():
    assert paths.cache_dir() == Path.home() / ".cache" / "trash_panda"
    assert paths.trace_dir() == Path.home() / ".cache" / "trash_panda" / "traces"


def test_trash_panda_home_moves_the_cache_folder_and_with_it_the_trace_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_HOME", str(tmp_path / "home"))
    assert paths.cache_dir() == tmp_path / "home"
    assert paths.trace_dir() == tmp_path / "home" / "traces"


def test_the_trace_dir_variable_wins_over_the_cache_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path / "mine"))
    assert paths.trace_dir() == tmp_path / "mine"


def test_a_blank_variable_is_the_same_as_none(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", "  ")
    monkeypatch.setenv("TRASH_PANDA_HOME", "")
    assert paths.trace_dir() == Path.home() / ".cache" / "trash_panda" / "traces"


def test_a_new_trace_path_is_named_by_the_time_and_the_folder_is_made(tmp_path):
    path = paths.new_trace_path(tmp_path / "a" / "b", datetime(2026, 11, 14, 9, 30, 12, 482_999))
    assert path == tmp_path / "a" / "b" / "20261114-093012-482.json" and path.parent.is_dir()


def test_a_name_that_is_taken_gets_a_number(tmp_path):
    when = datetime(2026, 11, 14, 9, 30, 12, 482_000)
    first = paths.new_trace_path(tmp_path, when)
    first.write_text("{}", encoding="utf-8")
    second = paths.new_trace_path(tmp_path, when)
    second.write_text("{}", encoding="utf-8")
    assert [first.name, second.name, paths.new_trace_path(tmp_path, when).name] == [
        "20261114-093012-482.json", "20261114-093012-482-2.json", "20261114-093012-482-3.json"]


def test_the_files_are_listed_newest_first_and_only_json_files(tmp_path):
    for name in ("20261112-100000-000.json", "20261114-093012-482.json", "20261113-080000-100.json", "notes.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "sub.json").mkdir()
    assert [p.name for p in paths.trace_files(tmp_path)] == ["20261114-093012-482.json","20261113-080000-100.json", "20261112-100000-000.json"]
    assert paths.latest_trace(tmp_path).name == "20261114-093012-482.json"


def test_a_missing_or_empty_folder_has_no_files(tmp_path):
    assert paths.trace_files(tmp_path / "none") == [] and paths.latest_trace(tmp_path / "none") is None
    assert paths.latest_trace(tmp_path) is None


@pytest.mark.parametrize("name", ["", "../x.json", "..\\x.json", "a\\b.json", "a/b.json", "a.txt", "/abs.json", "x.json.exe", "missing.json"])
def test_safe_trace_accepts_only_a_plain_name_of_a_file_that_is_there(tmp_path, name):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert paths.safe_trace(tmp_path, name) is None


def test_safe_trace_returns_the_file(tmp_path):
    (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
    assert paths.safe_trace(tmp_path, "ok.json") == tmp_path / "ok.json"


@pytest.mark.parametrize("text, seconds", [("90m", 5400), ("12h", 43200), ("30d", 30 * 86400), ("2w", 14 * 86400), (" 1.5d ", 129600)])
def test_ages(text, seconds):
    assert paths.parse_age(text) == seconds


@pytest.mark.parametrize("text", ["", "30", "d", "soon", "-1d", "1y"])
def test_a_bad_age_is_an_error(text):
    with pytest.raises(ValueError, match="not an age"):
        paths.parse_age(text)


def make_files(tmp_path, ages_days, now):
    files = []
    for i, age in enumerate(ages_days):                       # newest first
        path = tmp_path / f"{i}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (now - age * 86400, now - age * 86400))
        files.append(path)
    return files


def test_clean_beyond_the_newest(tmp_path):
    files = make_files(tmp_path, [1, 5, 40, 90], 10 ** 9)
    assert paths.to_clean(files, keep=2, now=10 ** 9) == files[2:]
    assert paths.to_clean(files, keep=0, now=10 ** 9) == files
    assert paths.to_clean(files, keep=9, now=10 ** 9) == []


def test_clean_older_than(tmp_path):
    files = make_files(tmp_path, [1, 5, 40, 90], 10 ** 9)
    assert paths.to_clean(files, older_than=30 * 86400, now=10 ** 9) == files[2:]


def test_clean_with_both_limits_takes_only_what_is_beyond_the_newest_and_old(tmp_path):
    files = make_files(tmp_path, [1, 5, 40, 90], 10 ** 9)
    assert paths.to_clean(files, keep=3, older_than=30 * 86400, now=10 ** 9) == files[3:]
    assert paths.to_clean(files, keep=1, older_than=30 * 86400, now=10 ** 9) == files[2:]


def test_clean_with_no_limit_takes_every_file(tmp_path):
    files = make_files(tmp_path, [1, 5], 10 ** 9)
    assert paths.to_clean(files, now=10 ** 9) == files


def test_sizes():
    assert [paths.fmt_size(n) for n in (0, 999, 2048, 5 * 1024 ** 2, 3 * 1024 ** 3)] == ["0 B", "999 B", "2.0 KB", "5.0 MB", "3.0 GB"]


def test_the_module_does_not_import_torch():
    out = subprocess.run([sys.executable, "-c", "import sys, trash_panda.paths; print('torch' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
