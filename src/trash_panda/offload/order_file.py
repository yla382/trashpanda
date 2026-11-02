"""Keeping the order in which the blocks of a model run, so the next run can prefetch from its first pass."""

import json
import warnings
from pathlib import Path


def order_path(checkpoint, folder=None) -> Path:
    """Where the order of `checkpoint` is kept: `<name>_order.json` in `folder`, or next to the checkpoint."""
    p     = Path(checkpoint)
    name  = p.name if p.is_dir() else p.stem
    where = Path(folder) if folder is not None else (p if p.is_dir() else p.parent)
    return where / f"{name}_order.json"


def read_order(path, known):
    """The saved block names, or None if there is no file, it cannot be read, or it names a block the model lacks."""
    try:
        names = json.loads(Path(path).read_text(encoding="utf-8"))["order"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(names, list) or any(n not in known for n in names):
        return None
    return names


def write_order(path, names) -> bool:
    """Save `names` through a temporary file, so a half-written file is never read. False if it cannot be written."""
    path = Path(path)
    tmp  = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"order": list(names)}, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        warnings.warn(f"could not save the block order to {path}: {e}")
        return False
    return True
