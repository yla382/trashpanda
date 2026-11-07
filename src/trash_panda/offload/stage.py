"""Running one model of a pipeline: build, plan, load, run, release."""

from __future__ import annotations
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional
import torch
from . import config
from .blocks import find_blocks
from .hardware import detect_hardware
from .early import _EarlyLoader, _vram_available
from .hooks import _in_order, install_offload, release
from .loader import stream_load
from .measure import _peak_since, _start_peak
from .order_file import order_path, read_order, write_order
from .planner import Tier, parse_bytes, plan, ram_overhead, refit_gpu_blocks, vram_kept_back
from .report import fmt_bytes

_activations: Dict[str, int] = {}  # key -> measured activation bytes, kept for the life of the process


def _available_now(device):
    """The VRAM that a new allocation on `device` can use now, or None if it is not a CUDA device."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return _vram_available(torch.device(device))[0]


def _expected_after(running, hw, max_vram, max_ram):
    """The (VRAM, RAM) a model can count on once the running stage with plan `running` has been released. A figure the
    caller gave is used as it is. Otherwise it is what is free now plus what the running stage holds and will give back:
    its GPU blocks and the weights outside the blocks for VRAM, its RAM tier for RAM. The buffers and the activations of
    the running stage are not counted: if they are in use now they are not in the free figure, so it comes out too low,
    never too high. RAM can be too high, since a mapped RAM tier gives back only the pages that were touched."""
    vram = parse_bytes(max_vram)
    if vram is None:
        vram = hw.vram_free + running.other_bytes + sum(b.stored_bytes for b in running.blocks if b.tier == Tier.GPU)
    ram = parse_bytes(max_ram)
    if ram is None and running is not None:
        ram = hw.ram_available + sum(b.stored_bytes for b in running.blocks if b.tier == Tier.RAM)
    return vram, ram


def _in_run_order(p, path, order, order_file_destination):
    """The blocks of plan `p` in the order they will run: `order`, else the saved order file, else the plan's order."""
    if order is None:
        known = {b.prefix.rstrip(".") for b in p.blocks}
        order = read_order(order_path(path, order_file_destination), known)
    return p.blocks if order is None else _in_order(p.blocks, order)


def _early_ram_prefixes(p, path, hw, order, order_file_destination, min_free_ram=None):
    """The RAM-tier blocks that fit in the RAM that is free now, in the order they run: blocks are taken from the first
    one that runs until the next one does not fit. The RAM that plan() keeps back (`min_free_ram`, the pinned buffers)
    is not counted as free. When the free RAM is unknown, all the RAM-tier blocks are taken."""
    streamed = [b for b in _in_run_order(p, path, order, order_file_destination) if b.tier == Tier.RAM]
    if not hw.ram_available:
        return tuple(b.prefix for b in streamed)
    left = max(hw.ram_available - ram_overhead(p, hw, min_free_ram), 0)
    chosen = []
    for b in streamed:
        if b.stored_bytes > left:
            break
        left -= b.stored_bytes
        chosen.append(b.prefix)
    return tuple(chosen)


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

    def __init__(self, model, p, path, *, key, key_map, skip_prefixes, order, order_file_destination, overlap, after=None,
                 memory=None, vram_assumed=None):
        self.model  = model
        self.plan   = p
        self.key    = key
        self.handle = None
        self._path, self._key_map, self._skip = path, key_map, tuple(skip_prefixes)
        self._order, self._order_dest, self._overlap = order, order_file_destination, overlap
        self._used   = False
        self._after  = after
        self._memory = p.prefixes(Tier.RAM) if memory is None else tuple(memory)    # the RAM blocks read into memory now
        self._vram_assumed = vram_assumed                    # the VRAM the plan counted on; None: do not replan at entry
        self._early  = None
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
                        defer_prefixes=self.plan.prefixes(Tier.GPU), ram_in_memory_prefixes=self._memory)
            if self._after is not None and self.model._trash_panda_state.deferred is not None:
                self._early = self._make_early()
        except Exception as err:                           # raised again where the model is asked for
            self._error = err
        finally:
            self._read_s = time.perf_counter() - began
            self._done.set()
        if self._early is not None:
            self._early.run()                              # until it is done or the model is asked for

    def _make_early(self) -> _EarlyLoader:
        state = self.model._trash_panda_state
        blocks = _in_run_order(self.plan, state.checkpoint.path, self._order, self._order_dest)
        sizes = {b.prefix: b.stored_bytes for b in self.plan.blocks}
        return _EarlyLoader(state.deferred, state.checkpoint, [b.prefix for b in blocks if b.prefix in state.deferred.items],
                            sizes, self._after, state.device, keep=vram_kept_back(self.plan))

    def _stop_early(self) -> None:
        if self._early is not None:
            self._early.stop()
            self._thread.join()

    def _replan(self) -> None:
        """Fit the GPU part of the plan to the VRAM that is really free now, which may differ from the figure the plan was
        made with. A block planned for the GPU whose weights are not there yet and does not fit goes to the RAM tier;
        if there is room left, RAM-tier blocks that were read into memory move to the GPU. Blocks that are on the GPU
        already stay. Does nothing when the free VRAM is not known (no CUDA)."""
        state = self.model._trash_panda_state
        if self._vram_assumed is None:
            return
        available = _available_now(state.device)
        if available is None:
            return
        deferred = state.deferred
        in_gpu = set(self.plan.prefixes(Tier.GPU))
        pending = {p for p in in_gpu if deferred is not None and p in deferred.items and p not in deferred.loaded}
        promotable = set(self._memory) & set(self.plan.prefixes(Tier.RAM))
        if not pending and not promotable:
            return
        ordered = _in_run_order(self.plan, state.checkpoint.path, self._order, self._order_dest)
        promoted, demoted = refit_gpu_blocks(self.plan, ordered, available, self.plan.headroom_bytes, pending, promotable)
        tiers = {b.prefix: b for b in self.plan.blocks}
        with state.checkpoint:
            for prefix in demoted:
                if deferred.demote(prefix):
                    tiers[prefix].tier = Tier.RAM
        for prefix in promoted:
            self.model.get_submodule(prefix.rstrip(".")).to(state.device)
            tiers[prefix].tier = Tier.GPU
        if deferred is not None and not deferred.items:
            state.deferred = None
        if demoted:
            message = (f"preload(): {len(demoted)} blocks planned for the GPU were put in the RAM tier, because "
                       f"{fmt_bytes(available)} of VRAM was free when the model was entered and the plan counted on "
                       f"{fmt_bytes(self._vram_assumed)}")
            self.plan.warnings.append(message)
            warnings.warn("trash-panda: " + message, RuntimeWarning)
        if config.current().profile and (promoted or demoted):
            print(f"trash-panda: preload: replanned at entry with {fmt_bytes(available)} of VRAM free (the plan counted "
                  f"on {fmt_bytes(self._vram_assumed)}): {len(promoted)} blocks moved to the GPU, {len(demoted)} to RAM")

    def __enter__(self):
        if self._used:
            raise RuntimeError("this preloaded model was already used or cancelled")
        self._used = True
        began = time.perf_counter()
        self._done.wait()
        self._stop_early()
        waited = time.perf_counter() - began
        if self._error is not None:
            release(self.model)
            raise self._error
        if config.current().profile:
            print(f"trash-panda: preload: the RAM tier was read in {self._read_s:.2f} s in the background; the model "
                  f"waited {waited:.2f} s for it")
            mapped = len(self.plan.prefixes(Tier.RAM)) - len(self._memory)
            if mapped:
                print(f"trash-panda: preload: {len(self._memory)} RAM blocks were read into memory early; {mapped} stay "
                      "mapped from the file and are read when they first run")
            if self._early is not None:
                print(f"trash-panda: preload: {self._early.loaded} GPU blocks were loaded while the running model "
                      "freed its memory")
        state = self.model._trash_panda_state
        self._replan()
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
        self._stop_early()
        release(self.model)


def preload(build, path, *, max_vram=None, max_ram=None, running=None, key=None, key_map=None, skip_prefixes=(),
            order=None, order_file_destination=None, overlap=False, after=None, **plan_kwargs) -> PendingStage:
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
    plan() cannot measure it. `max_ram` is the RAM its RAM tier may take once the running model has been released, and
    the model is planned for it. Either can be left out when `running` is given: it is the plan of the running stage
    (the `p` in `with stage(...) as (a, p, handle)`), and the memory is then what is free now plus what that stage holds
    and will give back (its GPU blocks and weights outside the blocks, its RAM tier). The running stage's buffers and
    activations are not counted, so the VRAM figure is low, not high; a mapped RAM tier gives back only the pages it
    touched, so the RAM figure can be high. A figure that is given wins. Without `running`, `max_vram` has to be given,
    and a missing `max_ram` means the RAM that is free now.
    The prediction can be wrong, and when the model is entered the plan is fitted to the VRAM that is really free: a
    block planned for the GPU that is not there yet and does not fit goes to the RAM tier, with a warning; room that is
    left over takes RAM-tier blocks that were read into memory, in the order they run. Blocks already on the GPU stay.
    Only the RAM that is free now can be read now:
    the RAM-tier blocks are taken in the order they run, from the first one, until the next one does not fit in what is
    free (less `min_free_ram` and the pinned buffers). Those are read into memory of its own, whatever
    TRASH_PANDA_RAM_IN_MEMORY says, because a memory-mapped block is not read until it is used. The other RAM-tier
    blocks stay mapped from the file and are read when they first run, as in a model that was not preloaded. The weights
    outside the blocks are put on the GPU as soon as they are read, so they need room there while the running model
    holds its own. The reads share the drive with whatever else is reading from it.
    `after` is the queue that `handle.subscribe()` of the running stage gave. When it is given, the model's GPU blocks
    are loaded, in the order they will run, as the running model's final pass frees the same number of bytes of GPU
    memory, through two small pinned buffers on a copy stream of their own, so they do not wait for the running model to
    be released. The queue is read by the loader: do not read it elsewhere. The blocks that were not loaded early are
    loaded when the model is asked for, and a failure in the early loading only gives a warning.
    The other arguments are those of stage(); `plan_kwargs` go to plan()."""
    if max_vram is None and running is None:
        raise ValueError("preload() needs max_vram or running: the GPU is in use by the running model, so the free VRAM "
                         "now is not what this model will have")
    k = key or str(path)
    hw = plan_kwargs.pop("hardware", None) or detect_hardware()
    vram, ram = _expected_after(running, hw, max_vram, max_ram)
    if config.current().profile and running is not None:
        print(f"trash-panda: preload: planning for {fmt_bytes(vram)} of VRAM and "
              f"{fmt_bytes(ram) if ram is not None else 'the free RAM'} after the running stage is released "
              f"(free now: {fmt_bytes(hw.vram_free)} of VRAM, {fmt_bytes(hw.ram_available)} of RAM)")
    model = build()
    plan_kwargs.setdefault("activation_vram", _activations.get(k))
    p = plan(model, path, key_map=key_map, skip_prefixes=skip_prefixes, max_vram=vram, max_ram=ram,
             hardware=hw, **plan_kwargs)
    memory = _early_ram_prefixes(p, path, hw, order, order_file_destination, plan_kwargs.get("min_free_ram"))
    return PendingStage(model, p, path, key=k, key_map=key_map, skip_prefixes=skip_prefixes, order=order,
                        order_file_destination=order_file_destination, overlap=overlap, after=after, memory=memory,
                        vram_assumed=vram)


def _on_meta(value):
    """`value` with every tensor in it (also inside lists, tuples and dicts) moved to the meta device."""
    if isinstance(value, torch.Tensor):
        return value.to("meta")
    if isinstance(value, tuple) and hasattr(value, "_fields"):                  # a named tuple
        return type(value)(*(_on_meta(v) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_on_meta(v) for v in value)
    if isinstance(value, dict):
        return {k: _on_meta(v) for k, v in value.items()}
    return value


def record_order(build, path, *, args=(), kwargs=None, skip_prefixes=(), order_file_destination=None,
                 groups=None) -> Optional[List[str]]:
    """Find the order in which a model runs its blocks without loading it, and save it where stage() reads it.

    A model that has never run uses the order its blocks are defined in, which `forward()` can override, and the first
    run warns and is prefetched less well. This calls the model once on the meta device, so no weights are read and
    nothing runs on the GPU, and saves the blocks in the order they started to `<checkpoint name>_order.json` (in
    `order_file_destination`, by default next to the checkpoint), replacing a file that is already there. The next
    stage() or install_offload() of that checkpoint reads it, as it reads the file a real run leaves.

    `build` returns the model on the meta device, as for stage(). `args` and `kwargs` are the inputs of one call, with
    sizes as small as the model accepts; tensors in them are moved to the meta device. The order can depend on the
    inputs, so they should be like the ones the real run will use. `skip_prefixes` and `groups` are those of plan().
    Returns the saved block names, or None, with a warning, if the call failed on the meta device (some operations have
    no meta version, and code that reads a tensor's values cannot run there), no block ran, or the file could not be
    written. Nothing is saved then, and the first run goes on as it did without this call."""
    model = build()
    if any(not t.is_meta for t in model.parameters()):
        raise ValueError("record_order() needs the model on the meta device: build() returned a model with weights in "
                         "memory, and a call would run it for real")
    skip  = tuple(skip_prefixes)
    found = list(groups) if groups is not None else find_blocks(model)
    names = [pre.rstrip(".") for g in found for pre in g.prefixes if not (skip and pre.startswith(skip))]
    seen: List[str] = []

    def noting(name):
        def hook(_module, _inputs):
            if name not in seen:
                seen.append(name)
        return hook

    hooks = [model.get_submodule(n).register_forward_pre_hook(noting(n)) for n in names]
    if not isinstance(args, (tuple, list)):
        args = (args,)
    try:
        with torch.no_grad():
            model(*_on_meta(tuple(args)), **_on_meta(dict(kwargs or {})))
    except Exception as err:
        warnings.warn(f"trash-panda: record_order() could not call the model on the meta device ({type(err).__name__}: "
                      f"{err}); no order was saved.", RuntimeWarning)
        return None
    finally:
        for h in hooks:
            h.remove()
    if not seen:
        warnings.warn("trash-panda: record_order(): no block ran in the call (are the inputs the ones that reach the "
                      "blocks?); no order was saved.", RuntimeWarning)
        return None
    where = order_path(path, order_file_destination)
    if not write_order(where, seen):
        return None
    print(f"trash-panda: saved the order of {len(seen)} blocks to {where}")
    return seen
