"""Storage formats: how a tensor is kept while it waits, and how it is expanded for compute.

Three kinds of stored tensor are understood:

* CAST        an ordinary floating tensor (fp32 / fp16 / bf16 ...). Expanding is a dtype cast.
* FP8_SCALED  float8 data plus a sibling "<key>_scale" tensor (ComfyUI "fp8 scaled" layout):
              value = fp8 * scale.
* FP8_PLAIN   float8 data without a scale (treated as scale 1.0).

Keeping the compact form while a block waits (fp8 is 1 byte, bf16 is 2) is what lets large
models fit; expansion to the compute dtype happens on the compute device, one block at a time.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, Union

import torch

SCALE_SUFFIX = "__scale"          # name suffix of the buffer that holds a weight's scale

FP8_DTYPES = tuple(
    getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
)
_FP8_MAX = {"float8_e4m3fn": 448.0, "float8_e5m2": 57344.0}

_ALIASES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
}
if hasattr(torch, "float8_e4m3fn"):
    for _name in ("fp8", "fp8_e4m3fn", "float8_e4m3fn"):
        _ALIASES[_name] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    for _name in ("fp8_e5m2", "float8_e5m2"):
        _ALIASES[_name] = torch.float8_e5m2


def parse_dtype(value: Union[None, str, torch.dtype]) -> Optional[torch.dtype]:
    """'bf16' / 'fp16' / 'fp32' / 'fp8' or a torch dtype -> torch dtype. None / 'auto' -> None (keep as stored)."""
    if value is None or isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    if key in ("", "auto", "none", "keep"):
        return None
    if key not in _ALIASES:
        raise ValueError(f"Unknown dtype {value!r}; use one of {sorted(_ALIASES)} or 'auto'")
    return _ALIASES[key]


def is_fp8(dtype: Optional[torch.dtype]) -> bool:
    return dtype is not None and dtype in FP8_DTYPES


def itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


class Kind(str, Enum):
    CAST = "cast"
    FP8_SCALED = "fp8_scaled"
    FP8_PLAIN = "fp8_plain"


def classify(dtype: torch.dtype, has_scale: bool) -> Kind:
    if is_fp8(dtype):
        return Kind.FP8_SCALED if has_scale else Kind.FP8_PLAIN
    return Kind.CAST


def expand(
    raw: torch.Tensor,
    scale: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Expand a stored tensor to `compute_dtype` on `device`.

    The compact tensor is moved first and converted there, so fewer bytes cross the PCIe bus
    and the conversion runs on the (fast) compute device."""
    raw = raw.to(device, non_blocking=True)
    if not raw.is_floating_point():
        return raw
    if scale is not None:
        return (raw.to(torch.float32) * scale.to(device).to(torch.float32)).to(compute_dtype)
    return raw.to(compute_dtype)


def quantize_fp8(w: torch.Tensor, dtype: Optional[torch.dtype] = None):
    """Per-tensor fp8 quantization. Returns (q, scale) with value ~= q * scale. Lossy."""
    dtype = dtype or torch.float8_e4m3fn
    fmax = _FP8_MAX[str(dtype).replace("torch.", "")]
    w32 = w.to(torch.float32)
    scale = (w32.abs().max() / fmax).clamp(min=1e-12)
    q = (w32 / scale).clamp(-fmax, fmax).to(dtype)
    return q, scale


def should_quantize(key: str, shape: Sequence[int], min_numel: int = 4096) -> bool:
    """Only large 2-D weights are quantized; norms, biases and small tensors stay as they are."""
    return key.endswith(".weight") and len(shape) == 2 and _numel(shape) >= min_numel


def stored_nbytes(
    key: str,
    shape: Sequence[int],
    dtype: Optional[torch.dtype],
    raw_nbytes: int,
    storage_dtype: Optional[torch.dtype],
) -> int:
    """Bytes a tensor will occupy while waiting, given the requested storage dtype (None = as stored)."""
    if dtype is None or not dtype.is_floating_point or is_fp8(dtype) or storage_dtype is None:
        return raw_nbytes
    n = _numel(shape)
    if is_fp8(storage_dtype):
        return n + 4 if should_quantize(key, shape) else raw_nbytes
    return n * itemsize(storage_dtype)


def expanded_nbytes(shape: Sequence[int], dtype: Optional[torch.dtype], compute_dtype: torch.dtype) -> int:
    """Bytes a tensor occupies once expanded to the compute dtype."""
    n = _numel(shape)
    if dtype is None:
        return n * 4
    if dtype.is_floating_point:
        return n * itemsize(compute_dtype)
    return n * itemsize(dtype)
