"""Where trash-panda keeps its files, and the trace files in that place. It uses only the standard library, so the command line can
use it without importing torch.

    TRASH_PANDA_HOME        the cache folder; default: ~/.cache/trash_panda
    TRASH_PANDA_TRACE_DIR   where trace files are written; default: <cache folder>/traces
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path

SUFFIX = ".json"
_UNIT = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def cache_dir() -> Path:
    home = os.environ.get("TRASH_PANDA_HOME", "").strip()
    return Path(home).expanduser() if home else Path.home() / ".cache" / "trash_panda"


def trace_dir() -> Path:
    chosen = os.environ.get("TRASH_PANDA_TRACE_DIR", "").strip()
    return Path(chosen).expanduser() if chosen else cache_dir() / "traces"


def new_trace_path(directory: Path, when: datetime) -> Path:
    """`<directory>/<date>-<time>-<ms>.json`; if that name is taken, with -2, -3, ... after it. The folder is made if it is missing."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = when.strftime("%Y%m%d-%H%M%S-") + f"{when.microsecond // 1000:03d}"
    path, n = directory / (stem + SUFFIX), 1
    while path.exists():
        n += 1
        path = directory / f"{stem}-{n}{SUFFIX}"
    return path


def trace_files(directory: Path = None) -> list:
    """The trace files of the folder, newest first (the names start with the time, so the order of the names is the order of time)."""
    directory = trace_dir() if directory is None else Path(directory)
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix == SUFFIX), key=lambda p: p.name, reverse=True)


def latest_trace(directory: Path = None):
    files = trace_files(directory)
    return files[0] if files else None


def safe_trace(directory: Path, name: str):
    """The file `name` of the folder, or None unless it is a plain file name of a trace file that is there."""
    if not name or name != Path(name).name or "/" in name or "\\" in name or not name.endswith(SUFFIX):
        return None
    path = Path(directory) / name
    return path if path.is_file() else None


def parse_age(text: str) -> float:
    """Seconds in `30d`, `12h`, `2w` or `90m`."""
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([mhdw])\s*", text or "")
    if not found:
        raise ValueError(f"{text!r} is not an age: use a number and m, h, d or w, for example 30d")
    return float(found.group(1)) * _UNIT[found.group(2)]


def to_clean(files, keep=None, older_than=None, now=None) -> list:
    """The files (newest first) that a cleanup removes: those beyond the newest `keep`, and those older than `older_than` seconds.
    With both limits, only a file that is beyond the newest ones and old; with none, every file."""
    now = time.time() if now is None else now
    chosen = []
    for i, path in enumerate(files):
        if keep is not None and i < keep:
            continue
        if older_than is not None and now - path.stat().st_mtime < older_than:
            continue
        chosen.append(path)
    return chosen


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
