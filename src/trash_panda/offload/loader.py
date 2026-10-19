"""Streaming checkpoint loading: build the model on the meta device, then fill it tensor by tensor.

Each tensor is read lazily from the checkpoint and placed according to its tier:

  normal    (not in any block)       expanded to the compute dtype on the compute device
  gpu_raw   (resident block)         kept in its compact form on the compute device
  ram       (streamed block)         kept in its compact form in system RAM
  disk      (disk block)             not read at all; the hooks read it from the file when needed

"Compact form" means the file's own dtype (for example fp8 with its scale), or the `storage_dtype` you ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .checkpoint import Checkpoint
from .formats import (
    SCALE_SUFFIX,
    Kind,
    classify,
    expand,
    is_fp8,
    parse_dtype,
    quantize_fp8,
    should_quantize,
)
from .planner import Plan, Tier
from .report import LoadReport

KeyMap = Union[None, Dict[str, str], Callable[[str], str]]


@dataclass
class OffloadState:
    """Remembered on the model so install_offload() knows what the loader did."""
    plan: Optional[Plan]
    device: str
    compute_dtype: torch.dtype
    checkpoint: Checkpoint
    disk_keys: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)   # param name -> (key, scale key)


def _navigate(module: nn.Module, name: str) -> Tuple[nn.Module, str]:
    *path, leaf = name.split(".")
    for part in path:
        module = getattr(module, part)
    return module, leaf


def _assign(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    owner, leaf = _navigate(model, name)
    if leaf in owner._parameters:
        owner._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in owner._buffers:
        owner._buffers[leaf] = tensor
    else:
        raise KeyError(name)


def _set_scale(model: nn.Module, name: str, scale: torch.Tensor) -> None:
    owner, leaf = _navigate(model, name)
    buf = leaf + SCALE_SUFFIX
    if buf in owner._buffers:
        owner._buffers[buf] = scale
    else:
        owner.register_buffer(buf, scale, persistent=False)


def _to_meta(model: nn.Module, name: str) -> None:
    """Free a real parameter's memory when its data stays on disk."""
    owner, leaf = _navigate(model, name)
    if leaf in owner._parameters and owner._parameters[leaf] is not None:
        p = owner._parameters[leaf]
        if not p.is_meta:
            owner._parameters[leaf] = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)


def stream_load(
    model: nn.Module,
    path,
    plan: Optional[Plan] = None,
    *,
    offload_prefixes: Sequence[str] = (),
    resident_prefixes: Sequence[str] = (),
    disk_prefixes: Sequence[str] = (),
    skip_prefixes: Sequence[str] = (),
    compute_dtype: Union[str, torch.dtype] = "bf16",
    storage_dtype: Union[None, str, torch.dtype] = "auto",
    device: Optional[str] = None,
    key_map: KeyMap = None,
) -> LoadReport:
    """Fill a meta-device `model` from the checkpoint at `path` (file, shard folder or index json).

    Either pass a `plan` (from plan() or Plan.manual()) or give the prefixes directly:
      offload_prefixes   tensors whose names start with these wait in RAM in compact form
      resident_prefixes  ... wait on the compute device in compact form (subset of the above, wins over it)
      disk_prefixes      ... are not loaded; they are read from the file when needed (wins over both)
      skip_prefixes      model entries with these prefixes are neither loaded nor required
    Anything else is expanded to `compute_dtype` on `device` right away.

    key_map  optional dict or function mapping checkpoint key -> model name.
    Returns a LoadReport (print it) and remembers the settings on the model for install_offload().
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if plan is not None:
        cdt, sdt = plan.compute_dtype, plan.storage_dtype
        offload_prefixes = tuple(offload_prefixes) + plan.prefixes(Tier.RAM)
        resident_prefixes = tuple(resident_prefixes) + plan.prefixes(Tier.GPU)
        disk_prefixes = tuple(disk_prefixes) + plan.prefixes(Tier.DISK)
    else:
        cdt, sdt = parse_dtype(compute_dtype) or torch.bfloat16, parse_dtype(storage_dtype)
    offload_prefixes, resident_prefixes = tuple(offload_prefixes), tuple(resident_prefixes)
    disk_prefixes, skip_prefixes = tuple(disk_prefixes), tuple(skip_prefixes)

    expected = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    skipped = {n for n in expected if n.startswith(skip_prefixes)} if skip_prefixes else set()
    quant_device = "cuda" if torch.cuda.is_available() else "cpu"

    def resolve(key: str) -> Optional[str]:
        k = key_map(key) if callable(key_map) else (key_map or {}).get(key, key)
        if k in expected:
            return k
        if k.startswith("model."):                       # newer transformers nest the LM under language_model
            alt = "model.language_model." + k[len("model."):]
            if alt in expected:
                return alt
        return None

    def tier_of(name: str) -> str:
        if disk_prefixes and name.startswith(disk_prefixes):
            return "disk"
        if resident_prefixes and name.startswith(resident_prefixes):
            return "gpu_raw"
        if offload_prefixes and name.startswith(offload_prefixes):
            return "ram"
        return "normal"

    ck = path if isinstance(path, Checkpoint) else Checkpoint(path)
    report = LoadReport()
    loaded = set()
    disk_keys: Dict[str, Tuple[str, Optional[str]]] = {}

    with ck:
        for key in ck.keys():
            if key.endswith(".comfy_quant"):
                report.aux_keys += 1
                continue
            if key.endswith("_scale") and key[: -len("_scale")] in ck:
                continue                                    # a sibling scale, read together with its weight
            target = resolve(key)
            if target is None:
                report.unexpected.append(key)
                continue
            if target in skipped:
                report.skipped += 1
                continue

            info = ck.info(key)
            tdtype = info.torch_dtype
            if (tdtype is None or not tdtype.is_floating_point) and key.endswith(".weight") \
                    and key[: -len(".weight")] + ".comfy_quant" in ck:
                raise NotImplementedError(
                    f"'{key}' uses a quantization format this loader does not support (integer data with "
                    "ComfyUI 'comfy_quant' metadata, e.g. int8/nvfp4). Only float and fp8 checkpoints are supported."
                )

            scale_key = key + "_scale"
            has_scale = scale_key in ck
            tier = tier_of(target)

            if tier == "disk":
                disk_keys[target] = (key, scale_key if has_scale else None)
                _to_meta(model, target)
                report.add("disk", info.nbytes)
                loaded.add(target)
                continue

            raw = ck.get_tensor(key)
            scale = ck.get_tensor(scale_key) if has_scale else None
            kind = classify(raw.dtype, has_scale) if raw.is_floating_point() else None

            if tier == "normal":
                tensor = expand(raw, scale, cdt, device) if raw.is_floating_point() else raw.to(device)
                _assign(model, target, tensor)
                report.add("normal", tensor.numel() * tensor.element_size())
            else:
                if kind == Kind.CAST and sdt is not None:
                    if is_fp8(sdt) and should_quantize(key, tuple(raw.shape)):
                        raw, scale = quantize_fp8(raw.to(quant_device), sdt)
                        kind = Kind.FP8_SCALED
                    elif not is_fp8(sdt):
                        raw = raw.to(sdt)
                dest = device if tier == "gpu_raw" else "cpu"
                raw = raw.to(dest)
                _assign(model, target, raw)
                nbytes = raw.numel() * raw.element_size()
                if scale is not None:
                    scale = scale.to(dest)
                    _set_scale(model, target, scale)
                    nbytes += scale.numel() * scale.element_size()
                report.add(tier, nbytes, kind.value if kind is not None else None)
            loaded.add(target)

    missing = expected - loaded - skipped
    if missing:
        names = sorted(missing)
        raise RuntimeError(
            f"{len(names)} model tensors were not found in the checkpoint, e.g. {names[:5]}. "
            "Use skip_prefixes for parts you do not need, or key_map to translate names."
        )

    n_plain = report.kinds.get(Kind.FP8_PLAIN.value, 0)
    if n_plain:
        report.warnings.append(
            f"{n_plain} fp8 tensors have no scale tensor; they are used as stored (scale 1.0). "
            "A scaled fp8 checkpoint is usually more accurate."
        )

    model._trash_panda_state = OffloadState(
        plan=plan, device=device, compute_dtype=cdt, checkpoint=ck, disk_keys=disk_keys
    )
    return report
