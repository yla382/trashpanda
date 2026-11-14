"""Recording a run as a trace file (format 2, see docs/design/trace-format.md).

Off unless `TRASH_PANDA_TRACE` is on; the file is written to the folder `TRASH_PANDA_TRACE_DIR` (see `trash_panda.paths`). A recording is a *session*: `stage()` opens one if none is open (and writes the
file when its block ends), a `Coordinator` opens one for the whole `with` block, and every model that runs in it adds its
events. Every hook is a call on the object of its model (`Model`), and that object is None when nothing is recorded, so a run
that is not recorded pays for an `is not None` test.

Two clocks. The CPU's events are timed with `time.perf_counter`. The GPU's work (the blocks on the compute stream, the copies
on the copy streams) is timed with CUDA events and put on the CPU's axis at the end, from one event recorded when the session
began. A model run without CUDA has the CPU's events only."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

from .. import paths
from . import config

MIN_WAIT_US = 50                                   # a wait shorter than this is not worth an event
_lock = threading.RLock()
_session = None                                    # the recording in progress, if there is one
_local = threading.local()                         # the model whose weights this thread is loading (see `Loading`)


# -- the session

def open_session(coordinated: bool = False) -> bool:
    """Open a recording if TRASH_PANDA_TRACE is on and none is open. True if this call opened it: the caller closes it."""
    global _session
    with _lock:
        if not config.current().trace or _session is not None:
            return False
        _session = Session(coordinated)
        return True


def close_session(opened: bool) -> None:
    """Close the recording that `open_session` opened, and write the file."""
    global _session
    if not opened:
        return
    with _lock:
        session, _session = _session, None
    if session is not None:
        session.write()


def session():
    return _session


def add_model(key, path, plan, started_by="stage", overlap=False):
    """Add a model to the recording, or None when nothing is recorded. `started_by` is "stage", "preload" or "coordinator"."""
    s = _session
    if s is None:
        return None
    return s.add_model(key, path, plan, "coordinator" if s.coordinated else started_by, overlap)


def pool_changed(pool) -> None:
    """A pinned pool grew or was trimmed."""
    s = _session
    if s is not None:
        s.pool_changed(pool)


class Loading:
    """`with Loading(model):` while this thread loads the weights of `model`: stream_load() finds the model that way."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self):
        self.before = getattr(_local, "model", None)
        _local.model = self.model

    def __exit__(self, *exc):
        _local.model = self.before


def loading_model():
    return getattr(_local, "model", None)


class Session:
    def __init__(self, coordinated) -> None:
        self.when, self.coordinated = datetime.now(), coordinated        # the file is named by when the recording began
        self.began = time.perf_counter()
        self.lock = threading.RLock()
        self.tracks, self.pools, self.models, self.events = {}, {}, [], []
        self.gpu = []                                           # work that ran on the GPU: resolved when the file is written
        self.cuda = torch.cuda.is_available()
        self.ref = None
        if self.cuda:                                           # the point where the GPU's clock is tied to the CPU's
            self.ref = torch.cuda.Event(enable_timing=True)
            self.ref.record()
            self.ref.query()                                    # query() sends the queued commands: on Windows they wait to be batched
        self.ref_us = self.now()
        self.main = self.track("main", "main thread", "cpu", "main")
        self.compute = None

    def now(self) -> int:
        return int((time.perf_counter() - self.began) * 1_000_000)

    def track(self, key, name, kind, role, model=None) -> int:
        with self.lock:
            if key not in self.tracks:
                entry = {"id": len(self.tracks) + 1, "name": name, "kind": kind, "role": role}
                if model is not None:
                    entry["model"] = model
                self.tracks[key] = entry
            return self.tracks[key]["id"]

    def compute_track(self) -> int:
        if self.compute is None:
            self.compute = self.track("compute", "compute stream", "gpu", "compute_stream")
        return self.compute

    def slice(self, name, cat, tid, start, end, **args) -> None:
        with self.lock:
            self.events.append({"name": name, "cat": cat, "ph": "X", "pid": 1, "tid": tid, "ts": int(start),
                                "dur": max(int(end - start), 1), "args": args})

    def instant(self, name, cat, tid, at, **args) -> None:
        with self.lock:
            self.events.append({"name": name, "cat": cat, "ph": "i", "s": "t", "pid": 1, "tid": tid, "ts": int(at), "args": args})

    def gpu_event(self):
        """A CUDA event with timing, recorded now on the current stream."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        event.query()           # without it the GPU can stamp the event when the batch is sent, long after this moment
        return event

    def later(self, kind, *item) -> None:
        with self.lock:
            self.gpu.append((kind, item))

    def gpu_us(self, event) -> int:
        return self.ref_us + int(self.ref.elapsed_time(event) * 1000)

    def add_model(self, key, path, plan, started_by, overlap):
        stem = Path(str(key)).stem or "model"
        with self.lock:
            taken = {m.id for m in self.models}
            id, n = stem, 1
            while id in taken:
                n += 1
                id = f"{stem}-{n}"
            model = Model(self, id, stem, path, plan, started_by, overlap)
            self.models.append(model)
            return model

    def pool_id(self, pool, model) -> str:
        """The id of a pinned pool in the file: made when it is first used, and the model that uses it is noted. A pool that a
        Coordinator owns is "shared" (`trace_kind`), any other is the pool of one stage."""
        with self.lock:
            entry = self.pools.get(id(pool))
            if entry is None:
                entry = self.pools[id(pool)] = {"id": f"pool-{len(self.pools) + 1}", "kind": getattr(pool, "trace_kind", "stage"),
                                                "buffers": len(pool.buffers), "models": []}
            if model.id not in entry["models"]:
                entry["models"].append(model.id)
            entry["buffers"] = max(entry["buffers"], len(pool.buffers))
            if model.pool is None:
                model.pool = entry["id"]
            return entry["id"]

    def pool_changed(self, pool) -> None:
        with self.lock:
            entry = self.pools.get(id(pool))
            if entry is None:
                return
            entry["buffers"] = max(entry["buffers"], len(pool.buffers))
            self.instant(f"pool has {len(pool.buffers)} buffers", "pool", self.main, self.now(), pool=entry["id"], buffers=len(pool.buffers))

    # -- the file

    def write(self) -> None:
        if self.gpu:
            torch.cuda.synchronize()
        for kind, item in self.gpu:
            if kind == "slice":
                name, cat, tid, start, end, args = item
                self.slice(name, cat, tid, self.gpu_us(start), self.gpu_us(end), **args)
            elif kind == "instant":
                name, cat, tid, event, args = item
                self.instant(name, cat, tid, self.gpu_us(event), **args)
        phases, models = [], []
        for m in self.models:
            phases += m.phase_list()
            models.append(m.entry())
        last = max([e["ts"] + e.get("dur", 0) for e in self.events] + [p["end_us"] for p in phases] + [1])
        tracks = [t for t in sorted(self.tracks.values(), key=lambda t: t["id"]) if any(e["tid"] == t["id"] for e in self.events)]
        meta = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": t["id"], "args": {"name": t["name"]}} for t in tracks]
        doc = {
            "displayTimeUnit": "ms",
            "trashPanda": {"version": 2, "note": "recorded", "duration_us": last + 1, "models": models, "tracks": tracks,
                           "pools": list(self.pools.values()), "phases": phases},
            "traceEvents": meta + sorted(self.events, key=lambda e: e["ts"]),
        }
        out = paths.new_trace_path(paths.trace_dir(), self.when)
        out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"trash-panda: trace written to {out}")


def _dtype(dtype):
    return None if dtype is None else str(dtype).replace("torch.", "")


class Model:
    """What is recorded about one model. The methods take the times that `now()` gave, in microseconds."""

    def __init__(self, session, id, name, path, plan, started_by, overlap) -> None:
        self.s, self.id, self.name, self.path, self.plan = session, id, name, Path(path), plan
        self.started_by, self.overlap = started_by, overlap
        self.cfg = config.current()
        self.names = [b.prefix.rstrip(".") for b in plan.blocks]
        self.pool = None
        self.ram_source = {}                         # block -> "memory" or "mapped"
        self.calls, self.passes, self.phases = [], {}, []
        self.loads, self.reads = {}, {}              # block -> [start, end, bytes, track, source]
        self.unwrapped = {"seen": set(), "pass": 1}  # blocks that are not wrapped have no `learn`: a repeated name starts a pass

    def now(self) -> int:
        return self.s.now()

    def block(self, name):
        """The number of the block that a block name, or the name of one of its tensors, belongs to; None if none."""
        for i, block in enumerate(self.names):
            if name == block or name.startswith(block + "."):
                return i
        return None

    # -- tracks: what thread or stream an event is on

    def thread_track(self, source=None):
        """The track of the calling thread: the main thread, this model's loader, its early loader, or a reader."""
        if threading.current_thread() is threading.main_thread():
            return self.s.main
        if source == "chunked":
            return self.s.track(("early", self.id), f"{self.name} early loader", "cpu", "loader", self.id)
        if threading.current_thread().name.startswith("trash-panda-loader"):
            return self.s.track(("loader", self.id), f"{self.name} loader thread", "cpu", "loader", self.id)
        return self.s.track(("reader", self.id), f"{self.name} stream_load reader", "cpu", "reader", self.id)

    def role_track(self, role, label, kind="cpu"):
        return self.s.track((role, self.id), f"{self.name} {label}", kind, role, self.id)

    # -- the CPU

    def phase(self, name, start, end) -> None:
        self.phases.append((name, start, end))

    def call(self, name, pass_no, start, end) -> None:
        i = self.block(name)
        if i is None:
            return
        self.s.slice(f"call b{i} (pass {pass_no})", "call", self.s.main, start, end, model=self.id, block=i, **{"pass": pass_no})
        with self.s.lock:
            span = self.passes.setdefault(pass_no, [start, end])
            span[0], span[1] = min(span[0], start), max(span[1], end)

    def wait(self, reason, start, end, name=None) -> None:
        if end - start < MIN_WAIT_US:
            return
        args = {"model": self.id, "reason": reason}
        label = reason
        if name is not None and self.block(name) is not None:
            args["block"] = self.block(name)
            label = f"{reason} b{args['block']}"
        track = self.s.main if reason != "pinned buffer" else self.role_track("worker", "worker thread")
        self.s.slice(f"wait: {label}", "wait", track, start, end, **args)

    def load(self, target, nbytes, start, end, source) -> None:
        """One tensor of a kept block was loaded: the tensors of a block make one event."""
        i = self.block(target)
        if i is None:
            return
        with self.s.lock:
            item = self.loads.setdefault(i, [start, end, 0, self.thread_track(source), source])
            item[0], item[1], item[2] = min(item[0], start), max(item[1], end), item[2] + nbytes

    def read_ram(self, target, nbytes, start, end, track) -> None:
        """One tensor of a RAM-tier block was read into memory by stream_load, on the thread whose track is `track`."""
        i = self.block(target)
        if i is None:
            return
        with self.s.lock:
            item = self.reads.setdefault(i, [start, end, 0, track, "memory"])
            item[0], item[1], item[2] = min(item[0], start), max(item[1], end), item[2] + nbytes
            self.ram_source[i] = "memory"

    def mapped(self, target, nbytes) -> None:
        i = self.block(target)
        if i is None or i in self.ram_source:
            return
        self.ram_source[i] = "mapped"
        self.s.instant(f"mapped b{i}", "mapped", self.s.main, self.now(), model=self.id, block=i, bytes=nbytes)

    def preload(self, start, end) -> None:
        self.s.slice("preload", "preload", self.role_track("preload", "preload thread"), start, end, model=self.id)

    def replan(self, demoted, promoted) -> None:
        def ids(names):
            return [self.block(n) for n in names if self.block(n) is not None]
        self.s.instant("replan", "replan", self.s.main, self.now(), model=self.id, demoted=ids(demoted), promoted=ids(promoted))

    def cancel(self) -> None:
        self.s.instant("cancel", "cancel", self.s.main, self.now(), model=self.id)

    def lookahead(self, before, after) -> None:
        self.s.instant(f"look-ahead {after}", "lookahead", self.s.main, self.now(), model=self.id, **{"from": before, "to": after})

    def alloc(self, pool, buffer, nbytes, start, end) -> None:
        pid = self.s.pool_id(pool, self)
        self.s.slice("pin a buffer", "alloc", self.role_track("worker", "worker thread"), start, end, model=self.id, pool=pid,
                     buffer=buffer + 1, bytes=int(nbytes))

    def early_pool(self) -> None:
        """The two pinned buffers that the early loader copies through; they are its own, not the pool's."""
        with self.s.lock:
            key = ("early-pool", self.id)
            if key not in self.s.pools:
                self.s.pools[key] = {"id": f"early-{self.id}", "kind": "early_loader", "buffers": 2, "models": [self.id]}

    def pass_for_unwrapped(self, wrapped_pass, name) -> int:
        """The pass of a block that has no `learn` of its own: the pass of the wrapped blocks, or, with none, a count of repeats."""
        if wrapped_pass:
            return wrapped_pass
        u = self.unwrapped
        if name in u["seen"]:
            u["seen"].clear()
            u["pass"] += 1
        u["seen"].add(name)
        return u["pass"]

    def move(self, name, direction, nbytes, start, end) -> None:
        i = self.block(name)
        if i is None:
            return
        a, b = ("gpu", "ram") if direction == "release_gpu" else ("ram", "gpu")
        self.s.slice(f"{direction} b{i}", "move", self.s.main, start, end, model=self.id, block=i, bytes=int(nbytes), **{"from": a, "to": b})

    # -- the worker

    def job(self, pool, kind, name, buffer, nbytes, start, end, source=None) -> None:
        """A stage job (a RAM block copied into a pinned buffer) or a read job (a disk block read into one), on the worker thread."""
        i = self.block(name)
        if i is None:
            return
        pid = self.s.pool_id(pool, self)
        track = self.role_track("worker", "worker thread")
        args = {"model": self.id, "block": i, "bytes": int(nbytes), "pool": pid, "to": f"pinned:{buffer + 1}"}
        if kind == "stage":
            args.update({"from": "ram", "source": source or self.ram_source.get(i, "memory")})
        else:
            args["from"] = "disk"
        self.s.slice(f"{kind} b{i}", kind, track, start, end, **args)
        self.s.instant(f"acquire pinned:{buffer + 1}", "buffer", track, start, model=self.id, pool=pid, buffer=buffer + 1, block=i)

    # -- the GPU: CUDA events, put on the CPU's axis when the file is written

    def gpu_event(self):
        return self.s.gpu_event() if self.s.cuda else None

    def copy(self, pool, name, buffer, nbytes, begin, finish) -> None:
        """The copy of a block from a pinned buffer to the GPU, on the copy stream: `begin` and `finish` are CUDA events."""
        i = self.block(name)
        if i is None or begin is None:
            return
        finish.query()
        pid = self.s.pool_id(pool, self)
        track = self.role_track("copy_stream", "copy stream", "gpu")
        self.s.later("slice", f"copy b{i}", "copy", track, begin, finish,
                     {"model": self.id, "block": i, "bytes": int(nbytes), "pool": pid, "from": f"pinned:{buffer + 1}", "to": "staged"})
        self.s.later("instant", f"release pinned:{buffer + 1}", "buffer", track, finish,
                     {"model": self.id, "pool": pid, "buffer": buffer + 1, "block": i})

    def run_begin(self):
        return self.gpu_event()

    def run_end(self, begin, name, pass_no):
        """The block's kernels are queued: the event after them. Returns it (the block is free once the GPU reaches it)."""
        end = self.gpu_event()
        i = self.block(name)
        if begin is not None and end is not None and i is not None:
            self.s.later("slice", f"run b{i} (pass {pass_no})", "run", self.s.compute_track(), begin, end,
                         {"model": self.id, "block": i, "pass": pass_no})
        return end

    # -- release

    def free(self, name, tier, nbytes, when) -> None:
        """A block was freed after it ran in a final pass. `when` is a CUDA event for a block on the GPU, else None (now)."""
        i = self.block(name)
        if i is None:
            return
        args = {"model": self.id, "block": i, "tier": tier, "bytes": int(nbytes)}
        if when is not None:
            self.s.later("instant", f"free b{i}", "free", self.s.main, when, args)
        else:
            self.s.instant(f"free b{i}", "free", self.s.main, self.now(), **args)

    def drop(self, name, nbytes, start, end) -> None:
        """The RAM-tier tensors of a block were let go, on the drop thread; the block is free when that is done."""
        i = self.block(name)
        if i is None:
            return
        self.s.slice(f"drop b{i}", "drop", self.role_track("drop", "drop thread"), start, end, model=self.id, block=i, bytes=int(nbytes))
        self.s.instant(f"free b{i}", "free", self.s.main, end, model=self.id, block=i, tier="ram", bytes=int(nbytes))

    # -- what goes in the file

    def phase_list(self):
        out, floor = [], 0
        phases = list(self.phases) + [(f"pass {n}", s, e) for n, (s, e) in sorted(self.passes.items())]
        for name, start, end in sorted(phases, key=lambda p: p[1]):
            start = max(start, floor)
            end = max(end, start + 1)
            out.append({"model": self.id, "name": name, "start_us": start, "end_us": end})
            floor = end
        return out

    def entry(self):
        p, cfg = self.plan, self.cfg
        events = []
        for i, (start, end, nbytes, track, source) in sorted(self.loads.items()):
            events.append(("load", i, start, end, nbytes, track, {"source": source, "from": "disk", "via": "ram", "to": "gpu"}))
        for i, (start, end, nbytes, track, source) in sorted(self.reads.items()):
            events.append(("read", i, start, end, nbytes, track, {"from": "disk", "to": "ram"}))
        for cat, i, start, end, nbytes, track, extra in events:
            self.s.slice(f"{cat} b{i}", cat, track, start, end, model=self.id, block=i, bytes=nbytes, **extra)
        if any(item[4] == "chunked" for item in self.loads.values()):
            self.early_pool()
        try:
            size = self.path.stat().st_size if self.path.is_file() else sum(f.stat().st_size for f in self.path.glob("*.safetensors"))
        except OSError:
            size = 0
        return {
            "id": self.id, "name": self.name, "started_by": self.started_by,
            "checkpoint": {"path": str(self.path), "bytes": size},
            "dtypes": {"stored": _dtype(p.storage_dtype) or "as stored", "compute": _dtype(p.compute_dtype)},
            "blocks": [{"id": i, "name": b.prefix, "tier": b.tier.value, "stored_bytes": b.stored_bytes,
                        "expanded_bytes": b.expanded_bytes} for i, b in enumerate(p.blocks)],
            "outside_bytes": p.other_bytes,
            "plan": {"vram_budget": p.vram_budget, "ram_budget": p.ram_budget, "headroom_bytes": p.headroom_bytes,
                     "stream_buffer_bytes": p.stream_buffer_bytes, "staging_bytes": p.staging_bytes,
                     "lookahead": p.lookahead, "warnings": list(p.warnings)},
            "options": {"overlap": self.overlap, "ram_in_memory": cfg.ram_in_memory, "pinned_buffers": cfg.pinned_buffers,
                        "blocks_in_flight": cfg.blocks_in_flight, "adaptive_lookahead": cfg.adaptive_lookahead},
            "pool": self.pool,
        }
