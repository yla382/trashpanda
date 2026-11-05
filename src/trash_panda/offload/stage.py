"""Running one model of a pipeline: build, plan, load, run, release."""

from __future__ import annotations
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Dict
import torch
from . import config
from .hardware import detect_hardware
from .hooks import install_offload, release
from .loader import stream_load
from .measure import _peak_since, _start_peak
from .planner import Tier, parse_bytes, plan
from .report import fmt_bytes

_activations: Dict[str, int] = {}  # key -> measured activation bytes, kept for the life of the process

@contextmanager
def stage(build, path, *, key=None, key_map=None, skip_prefixes=(), order=None, order_file_destination=None, overlap=False,
          **plan_kwargs):
    """Load one model for the `with` block and give its memory back afterwards.
    `build` returns the model on the meta device. `path` is its checkpoint. `plan_kwargs` go to plan().
    The peak GPU memory of the block is remembered under `key` (default: the checkpoint path) and used as 
    `activation_vram` the next time the same key is planned, unless plan_kwargs set it. Nothing is remembered when the block raises.
    `order` names the blocks that run first (see install_offload()); the rest follow in the plan's order. When it is
    None, the order the blocks ran in is saved to `<checkpoint name>_order.json` (in `order_file_destination`, by
    default next to the checkpoint) and used the next time.
    With `overlap=True` the resident blocks are loaded on a worker thread while the model already runs: a block
    that starts before its weights are on the GPU waits for them. The peak memory is not remembered then, because
    the loading falls inside the measured part of the run.
    The block gets `(model, plan, handle)`; `handle.final_pass()` frees each block as it runs during the last call.
    The model cannot be used after the block."""
    k = key or str(path)
    model = build()
    plan_kwargs.setdefault("activation_vram", _activations.get(k))
    p = plan(model, path, key_map=key_map, skip_prefixes=skip_prefixes, **plan_kwargs)
    stream_load(model, path, plan=p, key_map=key_map, skip_prefixes=skip_prefixes,
                defer_prefixes=p.prefixes(Tier.GPU) if overlap else ())
    handle = install_offload(model, order=order, order_file_destination=order_file_destination)
    state = getattr(model, "_trash_panda_state", None)
    loading = state is not None and state.deferred is not None       # the weights are still arriving
    before = _start_peak() if torch.cuda.is_available() and not loading else None
    try:
        yield model, p, handle
        if before is not None:
            _activations[k] = _peak_since(before, p)
    finally:
        release(model, handle)


class PendingStage:
    """A model that is loaded in the background while another one still runs. Made by preload(); use it like stage():
    `with pending as (model, plan, handle):`.

    Its RAM tier is read on a thread of its own, into memory of its own. The first `with` waits for that thread, loads
    the GPU blocks (or starts the loader thread, with `overlap=True`) and installs the hooks, so the memory that the
    earlier model held on the GPU has to be free by then. A pending model that will not be used must be given up with
    cancel(). It can be entered only once."""

    def __init__(self, model, p, path, *, key, key_map, skip_prefixes, order, order_file_destination, overlap):
        self.model  = model
        self.plan   = p
        self.key    = key
        self.handle = None
        self._path, self._key_map, self._skip = path, key_map, tuple(skip_prefixes)
        self._order, self._order_dest, self._overlap = order, order_file_destination, overlap
        self._used   = False
        self._before = None
        self._error  = None
        self._read_s = 0.0
        self._done   = threading.Event()
        self._thread = threading.Thread(target=self._read, name="trash-panda-preload", daemon=True)
        self._thread.start()

    def ready(self) -> bool:
        """True once the background thread has finished reading (or has failed)."""
        return self._done.is_set()

    def _read(self) -> None:
        began = time.perf_counter()
        try:
            stream_load(self.model, self._path, plan=self.plan, key_map=self._key_map, skip_prefixes=self._skip,
                        defer_prefixes=self.plan.prefixes(Tier.GPU), ram_in_memory=True)
        except Exception as err:                           # raised again where the model is asked for
            self._error = err
        finally:
            self._read_s = time.perf_counter() - began
            self._done.set()

    def __enter__(self):
        if self._used:
            raise RuntimeError("this preloaded model was already used or cancelled")
        self._used = True
        began = time.perf_counter()
        self._done.wait()
        waited = time.perf_counter() - began
        if self._error is not None:
            release(self.model)
            raise self._error
        if config.current().profile:
            print(f"trash-panda: preload: the RAM tier was read in {self._read_s:.2f} s in the background; the model "
                  f"waited {waited:.2f} s for it")
        state = self.model._trash_panda_state
        if state.deferred is not None and not self._overlap:
            with state.checkpoint:                         # the same memory-mapped reads that stream_load() does
                for prefix in state.deferred.prefixes():
                    state.deferred.load(prefix)
            state.deferred = None
        self.handle = install_offload(self.model, order=self._order, order_file_destination=self._order_dest)
        loading = state.deferred is not None               # the weights are still arriving
        self._before = _start_peak() if torch.cuda.is_available() and not loading else None
        return self.model, self.plan, self.handle

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and self._before is not None:
                _activations[self.key] = _peak_since(self._before, self.plan)
        finally:
            release(self.model, self.handle)
        return False

    def cancel(self) -> None:
        """Give up a model that will not be used: wait for the reads to finish (they cannot be interrupted) and free
        everything they loaded. Does nothing if the model was already used."""
        if self._used:
            return
        self._used = True
        self._done.wait()
        release(self.model)


def preload(build, path, *, max_vram, max_ram, key=None, key_map=None, skip_prefixes=(), order=None,
            order_file_destination=None, overlap=False, **plan_kwargs) -> PendingStage:
    """Start loading the next model of a pipeline while the current one is still running.

    The model is planned now and its RAM tier is read on a background thread, so the drive works while the running model
    computes. Nothing is put on the GPU for its blocks until the `with` block of the returned object is entered, which
    must happen after the running model has been released:

        with stage(build_a, path_a) as (a, p, handle):
            pending = preload(build_b, path_b, max_vram="8GB", max_ram="6GB")
            run(a)
        with pending as (b, p, handle):
            run(b)

    `max_vram` is the GPU memory the model may use once the running one has been released: the GPU is in use now, so
    plan() cannot measure it. `max_ram` is the RAM the model's RAM tier may take while the running model still holds its
    own; it is cut down to the RAM that is free now (with a warning), and None means the RAM that is free now. The RAM
    tier is read into memory of its own whatever TRASH_PANDA_RAM_IN_MEMORY says, because a memory-mapped RAM tier is not
    read until it is used. The weights outside the blocks are put on the GPU as soon as they are read, so they need
    room there while the running model holds its own. The reads share the drive with whatever else is reading from it.
    The other arguments are those of stage(); `plan_kwargs` go to plan()."""
    if max_vram is None:
        raise ValueError("preload() needs max_vram: the GPU is in use by the running model, so the free VRAM now is not "
                         "what this model will have")
    k = key or str(path)
    hw = plan_kwargs.pop("hardware", None) or detect_hardware()
    cap = parse_bytes(max_ram)
    if hw.ram_available and (cap is None or cap > hw.ram_available):
        if cap is not None:
            warnings.warn(
                f"trash-panda: preload() was given max_ram={fmt_bytes(cap)} but only {fmt_bytes(hw.ram_available)} of RAM "
                "is available now, and the running model still holds its own; using the smaller figure.",
                RuntimeWarning,
            )
        cap = hw.ram_available
    model = build()
    plan_kwargs.setdefault("activation_vram", _activations.get(k))
    p = plan(model, path, key_map=key_map, skip_prefixes=skip_prefixes, max_vram=max_vram, max_ram=cap,
             hardware=hw, **plan_kwargs)
    return PendingStage(model, p, path, key=k, key_map=key_map, skip_prefixes=skip_prefixes, order=order,
                        order_file_destination=order_file_destination, overlap=overlap)
