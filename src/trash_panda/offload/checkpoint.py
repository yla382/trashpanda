"""Reading safetensors checkpoints lazily: one file, a folder of shards, or an index json."""

from __future__ import annotations

import json
import struct
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
    start: int = 0                 # absolute byte offsets of the data inside `file`
    end: int = 0

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


def _read_header(f: Path) -> Tuple[Dict[str, dict], int]:
    """The json header of a safetensors file and the offset where the tensor data begins."""
    with open(f, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class Checkpoint:
    """Header-level view of a (possibly sharded) safetensors checkpoint.

    Creating it only reads the headers. Tensors are read on demand. Use it as a context manager to keep
    the files open while reading many tensors; otherwise each read opens the file briefly."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.files = _resolve_files(path)
        self._index: Dict[str, TensorInfo] = {}
        for f in self.files:
            header, base = _read_header(f)
            for key, meta in header.items():
                a, b = meta["data_offsets"]
                self._index[key] = TensorInfo(key, meta["dtype"], tuple(meta["shape"]), f, base + a, base + b)
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
    # Two modes:
    #  * inside `with checkpoint:` (loading) each file is opened once with safetensors' memory map, as a normal
    #    safe_open load does. Tensors that wait in RAM then stay file-backed, which Windows can drop and re-read
    #    instead of writing them to the pagefile when memory runs short.
    #  * outside it (reading disk-tier blocks while the model runs) plain seek-and-read of just the bytes needed.
    #    Mapping the whole file for every read charges the commit limit on Windows and fails with
    #    "paging file too small" (error 1455) on big checkpoints.
    def __enter__(self) -> "Checkpoint":
        self._stack = ExitStack()
        self._handles = {f: self._stack.enter_context(safe_open(str(f), framework="pt")) for f in self.files}
        return self

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack, self._handles = None, {}

    @staticmethod
    def _read_one(fh, info: TensorInfo) -> torch.Tensor:
        fh.seek(info.start)
        buf = bytearray(info.end - info.start)
        fh.readinto(buf)
        dtype = info.torch_dtype
        if dtype is None:
            raise ValueError(f"Unsupported dtype {info.dtype} for {info.key!r}")
        if not buf:
            return torch.empty(info.shape, dtype=dtype)
        return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(info.shape)

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self._index[key]
        handle = self._handles.get(info.file)
        if handle is not None:
            return handle.get_tensor(key)
        with open(info.file, "rb") as fh:
            return self._read_one(fh, info)

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
                with open(f, "rb") as fh:
                    for k in sorted(ks, key=lambda k: self._index[k].start):
                        out[k] = self._read_one(fh, self._index[k])
        return out
