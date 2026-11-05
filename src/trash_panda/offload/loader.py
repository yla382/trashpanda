"""Streaming checkpoint loading: build the model on the meta device, then fill it tensor by tensor.

Each tensor is read lazily from the checkpoint and placed according to its tier:

  normal    (not in any block)       expanded to the compute dtype on the compute device
  gpu_raw   (resident block)         kept in its compact form on the compute device
  ram       (streamed block)         kept in its compact form in system RAM
  disk      (disk block)             not read at all; the hooks read it from the file when needed

"Compact form" means the file's own dtype (for example fp8 with its scale), or the `storage_dtype` you ask for.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from . import config
from .checkpoint import Checkpoint, name_resolver
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


def _read_pair(ck: Checkpoint, key: str, scale_key: Optional[str]):
    """A tensor and its scale, each read with one big read into memory of its own. Runs on a reader thread."""
    return ck.read_tensor(key), (ck.read_tensor(scale_key) if scale_key is not None else None)


class DeferredLoad:
    """The tensors of resident blocks that stream_load() left on the meta device, to be loaded one block at a time."""

    def __init__(self, items: Dict[str, list], store: Callable) -> None:
        self.items  = items           # block prefix -> [(key, target, has_scale, scale_key), ...]
        self._store = store

    def prefixes(self) -> list:
        return list(self.items)

    def load(self, prefix: str) -> None:
        """Read the block's tensors from the checkpoint and put them on the compute device."""
        for key, target, has_scale, scale_key in self.items.get(prefix, ()):
            self._store(key, target, "gpu_raw", has_scale, scale_key)


@dataclass
class OffloadState:
    """Remembered on the model so install_offload() knows what the loader did."""
    plan: Optional[Plan]
    device: str
    compute_dtype: torch.dtype
    checkpoint: Checkpoint
    disk_keys: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)   # param name -> (key, scale key)
    deferred: Optional[DeferredLoad] = None


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
    defer_prefixes: Sequence[str] = (),
    ram_in_memory: Optional[bool] = None,
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
      defer_prefixes     resident blocks (a subset of resident_prefixes) whose tensors are left on the meta device;
                         `model._trash_panda_state.deferred.load(prefix)` loads one of them later (install_offload()
                         does that on a worker thread, and the block waits for it)
      ram_in_memory      read the RAM tier into memory of its own, not mapped from the file (default: the
                         TRASH_PANDA_RAM_IN_MEMORY setting). A mapped RAM tier is not read at all here, only on first use
    Anything else is expanded to `compute_dtype` on `device` right away.

    key_map  optional dict or function mapping checkpoint key -> model name.
    Returns a LoadReport (print it) and remembers the settings on the model for install_offload().
    """
    began_all = time.perf_counter()
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
    defer_prefixes = tuple(defer_prefixes)

    expected = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    skipped = {n for n in expected if n.startswith(skip_prefixes)} if skip_prefixes else set()
    quant_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolve = name_resolver(expected, key_map)

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
    deferred_items: Dict[str, list] = {}
    cfg = config.current()
    in_memory = cfg.ram_in_memory if ram_in_memory is None else bool(ram_in_memory)
    spent = defaultdict(float)                               # seconds per tier, for the profile
    ram_jobs: list = []                                      # streamed tensors to read into memory of their own, after the rest
    ram_read = {"seconds": 0.0, "bytes": 0}                  # what that took, for the profile

    def store(key: str, target: str, tier: str, has_scale: bool, scale_key: str, preloaded=None) -> None:
        """Put one tensor where its tier says. It is read here, unless `preloaded` already holds it and its scale."""
        if preloaded is None:
            raw = ck.get_tensor(key)
            scale = ck.get_tensor(scale_key) if has_scale else None
        else:
            raw, scale = preloaded
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

            if tier == "gpu_raw" and defer_prefixes and target.startswith(defer_prefixes):
                prefix = next(p for p in defer_prefixes if target.startswith(p))
                deferred_items.setdefault(prefix, []).append((key, target, has_scale, scale_key))
                loaded.add(target)
                continue

            if tier == "ram" and in_memory:
                ram_jobs.append((key, target, has_scale, scale_key))
            else:
                began = time.perf_counter()
                store(key, target, tier, has_scale, scale_key)
                spent[tier] += time.perf_counter() - began
            loaded.add(target)

        if ram_jobs:                                        # one big read per tensor, several at a time, in the order of the file
            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max(cfg.ram_read_threads, 1)) as pool:
                reads = [pool.submit(_read_pair, ck, key, scale_key if has_scale else None) for key, _t, has_scale, scale_key in ram_jobs]
                for (key, target, has_scale, scale_key), read in zip(ram_jobs, reads):
                    pair = read.result()
                    ram_read["bytes"] += sum(t.numel() * t.element_size() for t in pair if t is not None)
                    store(key, target, "ram", has_scale, scale_key, preloaded=pair)
            ram_read["seconds"] = time.perf_counter() - began
            spent["ram"] += ram_read["seconds"]

    if cfg.profile:
        tiers = ", ".join(f"{tier} {seconds:.2f} s" for tier, seconds in spent.items())
        later = "; blocks loaded while the model runs are not counted" if deferred_items else ""
        print(f"trash-panda: stream_load took {time.perf_counter() - began_all:.2f} s ({tiers}{later})")
        if ram_read["bytes"]:
            gb = ram_read["bytes"] / 1e9
            print(
                f"trash-panda: read {gb:.2f} GB of streamed-block tensors into memory in {ram_read['seconds']:.2f} s "
                f"({gb / max(ram_read['seconds'], 1e-9):.2f} GB/s, {max(cfg.ram_read_threads, 1)} threads)"
            )

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
        plan=plan, device=device, compute_dtype=cdt, checkpoint=ck, disk_keys=disk_keys,
        deferred=DeferredLoad(deferred_items, store) if deferred_items else None,
    )
    return report
