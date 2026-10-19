"""Reading safetensors checkpoints lazily: one file, a folder of shards, or an index json."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from safetensors import safe_open

_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
_TORCH_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_DTYPES["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _TORCH_DTYPES["F8_E5M2"] = torch.float8_e5m2


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str                     # safetensors dtype string, e.g. "BF16", "F8_E4M3"
    shape: Tuple[int, ...]
    file: Path

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPE_BYTES.get(self.dtype, 4)

    @property
    def torch_dtype(self) -> Optional[torch.dtype]:
        return _TORCH_DTYPES.get(self.dtype)


def _resolve_files(path: Union[str, Path]) -> List[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".json":                     # model.safetensors.index.json
            index = json.loads(p.read_text(encoding="utf-8"))
            names = sorted(set(index["weight_map"].values()))
            return [p.parent / n for n in names]
        return [p]
    if p.is_dir():
        index = p / "model.safetensors.index.json"
        if index.exists():
            return _resolve_files(index)
        files = sorted(p.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {p}")
        return files
    raise FileNotFoundError(f"Checkpoint not found: {p}")


class Checkpoint:
    """Header-level view of a (possibly sharded) safetensors checkpoint.

    Creating it only reads the headers. Tensors are read on demand. Use it as a context manager to keep
    the files open while reading many tensors; otherwise each read opens the file briefly."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.files = _resolve_files(path)
        self._index: Dict[str, TensorInfo] = {}
        for f in self.files:
            with safe_open(str(f), framework="pt") as fh:
                for key in fh.keys():
                    sl = fh.get_slice(key)
                    self._index[key] = TensorInfo(key, sl.get_dtype(), tuple(sl.get_shape()), f)
        self._stack: Optional[ExitStack] = None
        self._handles: Dict[Path, object] = {}

    # -- header queries
    def keys(self) -> List[str]:
        return list(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def info(self, key: str) -> TensorInfo:
        return self._index[key]

    def keys_with_prefix(self, prefix: str) -> List[str]:
        return [k for k in self._index if k.startswith(prefix)]

    def total_bytes(self) -> int:
        return sum(i.nbytes for i in self._index.values())

    # -- tensor reads
    def __enter__(self) -> "Checkpoint":
        self._stack = ExitStack()
        self._handles = {f: self._stack.enter_context(safe_open(str(f), framework="pt")) for f in self.files}
        return self

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack, self._handles = None, {}

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self._index[key]
        handle = self._handles.get(info.file)
        if handle is not None:
            return handle.get_tensor(key)
        with safe_open(str(info.file), framework="pt") as fh:
            return fh.get_tensor(key)

    def read(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        """Read several tensors, opening each file once."""
        by_file: Dict[Path, List[str]] = {}
        for k in keys:
            by_file.setdefault(self._index[k].file, []).append(k)
        out: Dict[str, torch.Tensor] = {}
        for f, ks in by_file.items():
            handle = self._handles.get(f)
            if handle is not None:
                for k in ks:
                    out[k] = handle.get_tensor(k)
            else:
                with safe_open(str(f), framework="pt") as fh:
                    for k in ks:
                        out[k] = fh.get_tensor(k)
        return out
