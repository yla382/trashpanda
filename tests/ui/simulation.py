"""A simulation of what trash-panda does, for the test traces of the viewer (tests/ui/data/).

It does not run any model. It schedules the work the way the code does, and writes it as a trace in format 2
(docs/design/trace-format.md). Each rule below names the code it follows, so a trace can be checked against the code:

* The plan keeps the first blocks on the GPU and spreads the disk blocks among the others (`planner._spread_disk_blocks`).
* One worker thread does the stage jobs, one after the other (`_Prefetcher.worker`, `_stage_job`, `_stage_disk_job`). A job
  takes the next pinned buffer in turn, and waits for the last copy that read from it (`_PinnedPool.acquire`); a buffer is
  allocated, at the next power of two, the first time it is used.
* The copy to the GPU is started by the job and runs on a copy stream, one copy at a time (`_stage_job`).
* The main thread, for each block (`wrapped`): waits until at most `blocks_in_flight - 1` earlier blocks are still on the GPU
  (`wait_for_room`), waits until the worker has *issued* the copy (`take`: it does not wait for the copy), stages the next
  blocks (`stage_ahead`), and launches the block. The GPU waits for the copy itself (`compute.wait_event`).
* `stage_ahead` stages the next `lookahead` blocks that need a copy. From the second pass on the last block is linked to the
  first, so it reaches into the next pass, also at the end of the last pass, where the work is wasted.
* A kept block is loaded through ordinary memory: by `stream_load` on the main thread (memory map), by the loader thread with
  `overlap=True` (plain read), or by the early loader (chunks through its own pinned buffers).
* The RAM tier is read into memory by `stream_load` only with `ram_in_memory`; otherwise it stays mapped from the file and its
  first copy reads the file.
* A final pass frees each block after it runs; the RAM-tier tensors are dropped on the drop thread (`_Prefetcher.drop`).

Run `python -m tests.ui.simulation tests/ui/data` to write the files.
"""

import json
import sys
from pathlib import Path

MS = 1000
BLOCK = 25165824                                  # the stored size of a block: 24 MiB
CHUNK = 64 * 2 ** 20                              # what the early loader copies at a time


def pow2(n):
    return 1 << (n - 1).bit_length() if n > 0 else 0


def plan_tiers(n, resident, ram_blocks):
    """The tier of each block: the first `resident` on the GPU, and of the rest as few on disk as it takes for the others to fit
    in `ram_blocks`, spread evenly (`_spread_disk_blocks`, with blocks of one size)."""
    rest = n - resident
    k = max(0, rest - ram_blocks)
    chosen = {int((i + 0.5) * rest / k) for i in range(k)} if k else set()
    return ["gpu"] * resident + ["disk" if i in chosen else "ram" for i in range(rest)]


class Spec:
    """A model: its blocks, and how long its work takes."""

    def __init__(self, id, name, n, resident, ram_blocks=0, started_by="stage", overlap=False, ram_in_memory=True,
                 lookahead=2, nbuf=2, adaptive=False, run=70, launch=3, memcpy=30, pcie=40, disk=120, mapped_read=90,
                 load=50, early_load=70, ramread=40, in_flight=2, outside=BLOCK // 2):
        self.id, self.name, self.n = id, name, n
        self.tiers = plan_tiers(n, resident, ram_blocks)
        self.started_by, self.overlap, self.ram_in_memory = started_by, overlap, ram_in_memory
        self.lookahead, self.nbuf, self.adaptive = lookahead, nbuf, adaptive
        self.run, self.launch, self.memcpy, self.pcie, self.disk, self.mapped_read = run, launch, memcpy, pcie, disk, mapped_read
        self.load, self.early_load, self.ramread, self.in_flight, self.outside = load, early_load, ramread, in_flight, outside
        self.stored, self.expanded = BLOCK, 2 * BLOCK

    def streamed(self, b):
        return self.tiers[b] != "gpu"


class Pool:
    """Pinned buffers: taken in turn, each free again when the last copy that read from it is done."""

    def __init__(self, sim, id, kind, nbuf, models):
        self.sim, self.id, self.kind = sim, id, kind
        self.cursor, self.free_at, self.size, self.models = 0, [0.0] * nbuf, [0] * nbuf, models
        sim.pools[id] = {"id": id, "kind": kind, "buffers": nbuf, "models": list(models)}

    def acquire(self, t, nbytes, model, track):
        """The buffer taken next, and when the worker has it (after waiting for its last copy, and allocating it)."""
        i = self.cursor
        self.cursor = (i + 1) % len(self.free_at)
        start = t
        if self.free_at[i] > t:
            self.sim.slice("wait for a pinned buffer", "wait", track, t, self.free_at[i], model=model, reason="pinned buffer")
            start = self.free_at[i]
        if self.size[i] < nbytes:
            self.size[i] = pow2(nbytes)
            self.sim.slice("pin a buffer", "alloc", track, start, start + 6, model=model, pool=self.id, buffer=i + 1, bytes=self.size[i])
            start += 6
        return i, start

    def grow(self, n, at):
        while len(self.free_at) < n:
            self.free_at.append(0.0)
            self.size.append(0)
        self.sim.pools[self.id]["buffers"] = len(self.free_at)
        self.sim.instant(f"pool grew to {n}", "pool", self.sim.main, at, pool=self.id, buffers=n)


class Sim:
    def __init__(self, note):
        self.note, self.events, self.tracks, self.pools, self.models, self.phases = note, [], {}, {}, [], []
        self.phase_end = {}
        self.main = self.track("main", "main thread", "cpu", "main")
        self.compute = self.track("compute", "compute stream", "gpu", "compute_stream")

    def track(self, key, name, kind, role, model=None):
        if key not in self.tracks:
            self.tracks[key] = {"id": len(self.tracks) + 1, "name": name, "kind": kind, "role": role}
            if model:
                self.tracks[key]["model"] = model
        return self.tracks[key]["id"]

    def slice(self, name, cat, tid, s, e, **args):
        self.events.append({"name": name, "cat": cat, "ph": "X", "pid": 1, "tid": tid, "ts": round(s * MS), "dur": max(round((e - s) * MS), 1), "args": args})

    def instant(self, name, cat, tid, t, **args):
        self.events.append({"name": name, "cat": cat, "ph": "i", "s": "t", "pid": 1, "tid": tid, "ts": round(t * MS), "args": args})

    def phase(self, model, name, s, e):
        start = max(round(s * MS), self.phase_end.get(model, 0))               # the phases of a model follow each other
        end = max(round(e * MS), start + 1)
        self.phase_end[model] = end
        self.phases.append({"model": model, "name": name, "start_us": start, "end_us": end})

    def add_model(self, spec, pool, checkpoint_blocks=None):
        n_disk = spec.tiers.count("disk")
        plan = {"vram_budget": 40 * BLOCK, "ram_budget": 12 * BLOCK, "headroom_bytes": BLOCK, "stream_buffer_bytes": int(1.3 * spec.expanded),
                "staging_bytes": spec.lookahead * spec.stored, "lookahead": spec.lookahead, "warnings": []}
        if n_disk:
            plan["warnings"].append(f"{n_disk} blocks will be read from disk on every pass; this is slow unless the drive is fast (NVMe).")
        self.models.append({
            "id": spec.id, "name": spec.name, "started_by": spec.started_by,
            "checkpoint": {"path": f"{spec.name}.safetensors", "bytes": spec.n * spec.stored},
            "dtypes": {"stored": "fp16", "compute": "bf16"},
            "blocks": [{"id": i, "name": f"blocks.{i}.", "tier": t, "stored_bytes": spec.stored, "expanded_bytes": spec.expanded} for i, t in enumerate(spec.tiers)],
            "outside_bytes": spec.outside, "plan": plan,
            "options": {"overlap": spec.overlap, "ram_in_memory": spec.ram_in_memory, "pinned_buffers": spec.nbuf, "blocks_in_flight": spec.in_flight,
                        "adaptive_lookahead": spec.adaptive},
            "pool": pool.id if pool else None,
        })

    def document(self):
        last = max([e["ts"] + e.get("dur", 0) for e in self.events] + [p["end_us"] for p in self.phases])
        tracks = [dict(t) for t in sorted(self.tracks.values(), key=lambda t: t["id"]) if any(e["tid"] == t["id"] for e in self.events)]
        meta = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": t["id"], "args": {"name": t["name"]}} for t in tracks]
        return {
            "displayTimeUnit": "ms",
            "trashPanda": {"version": 2, "note": self.note, "duration_us": last + 1000, "models": self.models, "tracks": tracks,
                           "pools": list(self.pools.values()), "phases": self.phases},
            "traceEvents": meta + sorted(self.events, key=lambda e: e["ts"]),
        }


class Stage:
    """One model run by stage(): its load, then its passes."""

    def __init__(self, sim, spec, pool, t0):
        self.sim, self.s, self.pool = sim, spec, pool
        self.main_free = self.gpu_free = self.worker_free = self.copy_free = t0
        self.in_flight, self.staged, self.ready, self.warm = [], {}, {}, set()
        self.lookahead, self.pass_no, self.passes_total = spec.lookahead, 0, 0
        self.worker = sim.track(f"worker-{spec.id}", f"{spec.name} worker thread", "cpu", "worker", spec.id)
        self.copy = sim.track(f"copy-{spec.id}", f"{spec.name} copy stream", "gpu", "copy_stream", spec.id)
        self.final = False
        self.last_end = t0

    # -- loading

    def stream_load(self, t, kept_early=None, only_ram=False):
        """`stream_load`: kept blocks on the main thread (or left to the loader thread with overlap), the RAM tier read or mapped.
        Returns when the model can start to run."""
        s, sim = self.s, self.sim
        kept_early = kept_early or {}
        reads = []
        for b, tier in enumerate(s.tiers):
            if tier == "gpu" and b in kept_early:
                self.ready[b] = kept_early[b]
            elif tier == "gpu" and not s.overlap and not only_ram:
                sim.slice(f"load b{b}", "load", sim.main, t, t + s.load, model=s.id, block=b, bytes=s.stored, source="mmap",
                          **{"from": "disk", "via": "ram", "to": "gpu"})
                t += s.load
                self.ready[b] = t
            elif tier == "ram" and not s.ram_in_memory:
                sim.instant(f"mapped b{b}", "mapped", sim.main, t, model=s.id, block=b, bytes=s.stored)
            elif tier == "ram":
                reads.append(b)
        if reads:                                    # the RAM tier is read last, by the reader thread (`ram_read_threads`)
            reader = sim.track(f"reader-{s.id}", f"{s.name} stream_load reader", "cpu", "reader", s.id)
            r = t
            for b in reads:
                sim.slice(f"read b{b}", "read", reader, r, r + s.ramread, model=s.id, block=b, bytes=s.stored, **{"from": "disk", "to": "ram"})
                r += s.ramread
            t = r
        return t

    def start_loader(self, t):
        """`overlap=True`: the loader thread loads the kept blocks, one after the other, while the model already runs."""
        s, sim = self.s, self.sim
        loader = sim.track(f"loader-{s.id}", f"{s.name} loader thread", "cpu", "loader", s.id)
        for b, tier in enumerate(s.tiers):
            if tier == "gpu" and b not in self.ready:
                sim.slice(f"load b{b}", "load", loader, t, t + s.load, model=s.id, block=b, bytes=s.stored, source="read",
                          **{"from": "disk", "via": "ram", "to": "gpu"})
                t += s.load
                self.ready[b] = t

    # -- the worker

    def schedule(self, b, k, request):
        """A stage job: `_stage_job` for a RAM block, `_stage_disk_job` for a disk block."""
        s, sim = self.s, self.sim
        begin = max(self.worker_free, request)
        i, begin = self.pool.acquire(begin, s.stored, s.id, self.worker)
        sim.instant(f"acquire pinned:{i + 1}", "buffer", self.worker, begin, model=s.id, pool=self.pool.id, buffer=i + 1, block=b)
        if s.tiers[b] == "ram":
            cached = s.ram_in_memory or b in self.warm
            dur, cat, source = s.memcpy + (0 if cached else s.mapped_read), "stage", "memory" if s.ram_in_memory else "mapped"
            self.warm.add(b)
            args = {"from": "ram", "to": f"pinned:{i + 1}", "source": source}
        else:
            dur, cat, args = s.disk, "read", {"from": "disk", "to": f"pinned:{i + 1}"}
        end = begin + dur
        sim.slice(f"{cat} b{b} (pass {k})", cat, self.worker, begin, end, model=s.id, block=b, bytes=s.stored, pool=self.pool.id, **args)
        copy_start = max(end, self.copy_free)
        copy_end = copy_start + s.pcie
        sim.slice(f"copy b{b} (pass {k})", "copy", self.copy, copy_start, copy_end, model=s.id, block=b, bytes=s.stored,
                  pool=self.pool.id, **{"from": f"pinned:{i + 1}", "to": "staged"})
        sim.instant(f"release pinned:{i + 1}", "buffer", self.copy, copy_end, model=s.id, pool=self.pool.id, buffer=i + 1, block=b)
        self.pool.free_at[i] = copy_end
        self.worker_free, self.copy_free = end, copy_end
        self.staged[(b, k)] = (end, copy_end)

    def following(self, b, k):
        """The blocks after (b, k) in the order they run; the last block is linked to the first from the second pass on."""
        for nb in range(b + 1, self.s.n):
            yield nb, k
        if k >= 2:
            for nb in range(b + 1):
                yield nb, k + 1

    def stage_ahead(self, b, k, t):
        """`stage_ahead`: the next `lookahead` blocks that need a copy; a block that is staged already still counts."""
        wanted = 0
        for nb, nk in self.following(b, k):
            if wanted >= self.lookahead:
                break
            if self.s.streamed(nb):
                if (nb, nk) not in self.staged:
                    self.schedule(nb, nk, t)
                wanted += 1

    # -- the main thread

    def run_block(self, b, k, final=False):
        s, sim = self.s, self.sim
        t = self.main_free
        if not s.streamed(b) and self.ready.get(b, 0) > t:                      # the gate of a block that is still loading
            sim.slice(f"wait for b{b} to load", "wait", sim.main, t, self.ready[b], model=s.id, block=b, reason="loader")
            t = self.ready[b]
        while len(self.in_flight) >= s.in_flight:                              # wait_for_room
            end = self.in_flight.pop(0)
            if end > t:
                sim.slice("wait for the GPU", "wait", sim.main, t, end, model=s.id, reason="gpu")
                t = end
        copy_end = 0.0
        if s.streamed(b):                                                      # take: until the worker has issued the copy
            if (b, k) not in self.staged:
                self.schedule(b, k, t)
            job_end, copy_end = self.staged[(b, k)]
            if job_end > t:
                sim.slice(f"wait for b{b}", "wait", sim.main, t, job_end, model=s.id, block=b, reason="staged")
                t = job_end
        self.stage_ahead(b, k, t)
        launch_end = t + s.launch
        sim.slice(f"call b{b} (pass {k})", "call", sim.main, t, launch_end, model=s.id, block=b, **{"pass": k})
        gpu_start = max(self.gpu_free, t + s.launch / 4, copy_end)              # the GPU waits for the copy, the main thread does not
        gpu_end = gpu_start + s.run
        sim.slice(f"run b{b} (pass {k})", "run", sim.compute, gpu_start, gpu_end, model=s.id, block=b, **{"pass": k})
        self.gpu_free = gpu_end
        self.in_flight.append(gpu_end)
        self.main_free = launch_end
        if final:
            self.free_block(b, launch_end, gpu_end)
        return t, launch_end

    def free_block(self, b, launch_end, gpu_end):
        s, sim = self.s, self.sim
        tier = s.tiers[b]
        if tier == "gpu":
            sim.instant(f"free b{b}", "free", sim.main, gpu_end, model=s.id, block=b, tier=tier, bytes=s.stored)
        elif tier == "ram":
            drop = sim.track(f"drop-{s.id}", f"{s.name} drop thread", "cpu", "drop", s.id)
            sim.slice(f"drop b{b}", "drop", drop, launch_end, launch_end + 5, model=s.id, block=b, bytes=s.stored)
            sim.instant(f"free b{b}", "free", sim.main, launch_end + 5, model=s.id, block=b, tier=tier, bytes=s.stored)
        else:
            sim.instant(f"free b{b}", "free", sim.main, launch_end, model=s.id, block=b, tier=tier, bytes=0)

    def run_pass(self, final=False):
        s = self.s
        self.pass_no += 1
        k, start = self.pass_no, self.main_free
        last = start
        for b in range(s.n):
            _, last = self.run_block(b, k, final)
        self.sim.phase(s.id, f"pass {k}", start, last)
        self.last_end = last
        return last

    def sync_and_gap(self, gap):
        """The code of the pipeline between two steps: it needs the result, so it waits for the GPU."""
        self.main_free = max(self.main_free, self.gpu_free) + gap

    def finish(self):
        """`release`: the worker finishes the jobs it has, and the stage ends."""
        self.main_free = max(self.main_free, self.gpu_free, self.worker_free, self.copy_free)
        return self.main_free


def one_stage(sim, spec, t0, passes, pool=None, gap=0, steps=None, guidance=1, final_last=False, early=None, pre=None):
    """A model run by stage(): a load, then `passes` passes (in `steps` steps of `guidance` passes each, with `gap` between the
    steps), and the release. Returns the time it ends."""
    pool = pool or (Pool(sim, f"pool-{spec.id}", "stage", spec.nbuf, [spec.id]) if any(spec.streamed(b) for b in range(spec.n)) else None)
    st = Stage(sim, spec, pool, t0)
    sim.add_model(spec, pool)
    t = st.stream_load(t0, kept_early=early)
    sim.phase(spec.id, "load", t0, t)
    st.main_free = t
    if spec.overlap:
        st.start_loader(t)
    steps = steps or passes
    for step in range(steps):
        for g in range(guidance):
            final = final_last and step == steps - 1 and g == guidance - 1
            st.run_pass(final)
        if step < steps - 1:
            st.sync_and_gap(gap)
    return st.finish(), st


def write(sim, path):
    Path(path).write_text(json.dumps(sim.document(), indent=1), encoding="utf-8")


# -- the scenarios

def single_model():
    sim = Sim("one model, plain stage(): kept blocks loaded by stream_load, one pass")
    spec = Spec("m", "model", n=8, resident=3, ram_blocks=3)
    one_stage(sim, spec, 0, passes=1)
    return sim


def overlap():
    sim = Sim("stage(overlap=True): the loader thread loads the kept blocks while the model already runs; two passes")
    spec = Spec("m", "model", n=8, resident=3, ram_blocks=3, overlap=True)
    one_stage(sim, spec, 0, passes=2)
    return sim


def sampling_steps():
    sim = Sim("sampling steps: three steps of two passes (guidance), the look-ahead reaching into the next pass")
    spec = Spec("m", "diffusion model", n=10, resident=4, ram_blocks=4)
    one_stage(sim, spec, 0, passes=6, steps=3, guidance=2, gap=25)
    return sim


def mapped_ram():
    sim = Sim("a mapped RAM tier (ram_in_memory off): nothing is read at load, the first copy of each block reads the file")
    spec = Spec("m", "model", n=8, resident=3, ram_blocks=3, ram_in_memory=False)
    one_stage(sim, spec, 0, passes=3)
    return sim


def sequential_stages():
    sim = Sim("three stage() calls in a row, no Coordinator: each model has its own pinned pool and worker")
    t = 0.0
    for spec, passes in ((Spec("text", "text encoder", n=4, resident=4), 1),
                         (Spec("unet", "unet", n=10, resident=4, ram_blocks=4), 4),
                         (Spec("vae", "vae", n=4, resident=3, ram_blocks=1), 1)):
        t, _ = one_stage(sim, spec, t, passes, gap=25)
        t += 40                                                               # the code between the stages: release, build the next
    return sim


def adaptive_lookahead():
    sim = Sim("adaptive look-ahead: after the warm-up passes the look-ahead grows from 2 to 3, and the pool with it")
    spec = Spec("m", "model", n=10, resident=2, ram_blocks=8, adaptive=True, disk=60)
    pool = Pool(sim, "pool-m", "stage", spec.nbuf, [spec.id])
    sim.add_model(spec, pool)
    st = Stage(sim, spec, pool, 0)
    t = st.stream_load(0)
    sim.phase(spec.id, "load", 0, t)
    st.main_free = t
    for k in range(1, 9):
        if k == 5:                                                            # learn() at the start of pass 5: the tuner has seen 4 passes
            st.lookahead = 3
            pool.grow(3, st.main_free)
            sim.instant("look-ahead 3", "lookahead", sim.main, st.main_free, model=spec.id, **{"from": 2, "to": 3})
        st.run_pass()
        st.sync_and_gap(20)
    st.finish()
    return sim


def release_gpu():
    sim = Sim("release_gpu() and restore_gpu() between passes: the kept blocks leave the GPU and come back")
    spec = Spec("m", "model", n=8, resident=3, ram_blocks=3)
    pool = Pool(sim, "pool-m", "stage", spec.nbuf, [spec.id])
    sim.add_model(spec, pool)
    st = Stage(sim, spec, pool, 0)
    t = st.stream_load(0)
    sim.phase(spec.id, "load", 0, t)
    st.main_free = t
    st.run_pass()
    st.run_pass()
    t = st.finish()                                                            # quiesce: nothing staged, nothing copying, no block running
    for direction, a, b in (("release_gpu", "gpu", "ram"), ("restore_gpu", "ram", "gpu")):
        for blk in range(3):
            sim.slice(f"{direction} b{blk}", "move", sim.main, t, t + 20, model=spec.id, block=blk, bytes=spec.stored, **{"from": a, "to": b})
            t += 20
        t += 300 if direction == "release_gpu" else 0                         # the GPU is used by something else meanwhile
    st.staged.clear()
    st.pass_no = 2
    st.main_free = st.gpu_free = st.worker_free = st.copy_free = t
    st.run_pass()
    st.finish()
    return sim


def coordinator_pipeline():
    sim = Sim("a Coordinator: text, unet and vae with one shared pinned pool; each next model is preloaded while the one before it runs")
    text = Spec("text", "text encoder", n=4, resident=4, started_by="coordinator")
    unet = Spec("unet", "unet", n=10, resident=4, ram_blocks=4, started_by="coordinator", ramread=30)
    vae = Spec("vae", "vae", n=4, resident=3, ram_blocks=1, started_by="coordinator", ramread=30)
    shared = Pool(sim, "pool-shared", "shared", 2, ["text", "unet", "vae"])
    Pool(sim, "early-unet", "early_loader", 2, ["unet"])
    Pool(sim, "early-vae", "early_loader", 2, ["vae"])

    def preload(spec, start, releases):
        """preload(): the RAM tier is read in the background; then the early loader loads the kept blocks one at a time, the first
        into the VRAM that is free already and the others into the memory the running model frees in its final pass."""
        thread = sim.track(f"preload-{spec.id}", f"{spec.name} preload thread", "cpu", "preload", spec.id)
        early = sim.track(f"early-{spec.id}", f"{spec.name} early loader", "cpu", "loader", spec.id)
        done = Stage(sim, spec, shared, start).stream_load(start, only_ram=True)
        sim.slice("preload", "preload", thread, start, done, model=spec.id)
        plan, t = [], done
        for j, b in enumerate(b for b, tier in enumerate(spec.tiers) if tier == "gpu"):
            room = t if j < 1 else max(t, releases[min(j - 1, len(releases) - 1)])
            plan.append((b, room, room + spec.early_load))
            t = room + spec.early_load
        return {"ram_done": done, "plan": plan, "track": early}

    def enter(spec, pre, enter_at, replan_pending):
        """Entering a preloaded model: it waits for the preload thread, then stops the early loader (which finishes the block it is
        loading), replans for the VRAM that is free, and loads what is still not on the GPU. Returns the stage, ready to run."""
        ready = max(enter_at, pre["ram_done"])
        if ready > enter_at:
            sim.slice("wait for the preloaded model", "wait", sim.main, enter_at, ready, model=spec.id, reason="preload")
        loaded = {}
        for b, s, e in pre["plan"]:
            if s >= ready:                                                     # the early loader had not started this block
                break
            sim.slice(f"load b{b}", "load", pre["track"], s, e, model=spec.id, block=b, bytes=spec.stored, source="chunked",
                      **{"from": "disk", "via": "ram", "to": "gpu"})
            loaded[b] = e
        ready = max([ready] + list(loaded.values()))
        pending = [b for b, tier in enumerate(spec.tiers) if tier == "gpu" and b not in loaded]
        demoted = pending[-1:] if replan_pending else []
        if demoted:
            sim.instant("replan", "replan", sim.main, ready, model=spec.id, demoted=demoted, promoted=[])
        st = Stage(sim, spec, shared, ready)
        t = ready
        for b in pending:
            if b in demoted:
                continue
            sim.slice(f"load b{b}", "load", sim.main, t, t + spec.load, model=spec.id, block=b, bytes=spec.stored, source="mmap",
                      **{"from": "disk", "via": "ram", "to": "gpu"})
            t += spec.load
            loaded[b] = t
        st.ready = loaded
        sim.phase(spec.id, "load", enter_at, t)
        st.main_free = t
        return st

    # text: a stage; its final pass frees the memory that the unet's early loader fills
    sim.add_model(text, shared)
    st = Stage(sim, text, shared, 0)
    t = st.stream_load(0)
    sim.phase("text", "load", 0, t)
    st.main_free = t
    releases = [t + (j + 1) * (text.launch + text.run) for j in range(4)]       # the final pass frees a block as each one ends
    pre_unet = preload(unet, t, releases)
    st.run_pass(final=True)
    text_end = st.finish()

    # unet: entered after the text encoder was released
    sim.add_model(unet, shared)
    st = enter(unet, pre_unet, text_end + 20, replan_pending=True)
    passes = 4
    rel_start = st.main_free + (passes - 1) * (10 * (unet.launch + unet.run) + 60)
    pre_vae = preload(vae, st.main_free, [rel_start + (j + 1) * (unet.launch + unet.run) for j in range(10)])
    for p in range(passes):
        st.run_pass(final=(p == passes - 1))
        if p < passes - 1:
            st.sync_and_gap(25)
    unet_end = st.finish()

    sim.add_model(vae, shared)
    st = enter(vae, pre_vae, unet_end + 20, replan_pending=False)
    st.run_pass()
    st.finish()
    return sim


SCENARIOS = {
    "single_model": single_model,
    "overlap": overlap,
    "sampling_steps": sampling_steps,
    "mapped_ram": mapped_ram,
    "sequential_stages": sequential_stages,
    "adaptive_lookahead": adaptive_lookahead,
    "release_gpu": release_gpu,
    "coordinator_pipeline": coordinator_pipeline,
}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for name, make in SCENARIOS.items():
        sim = make()
        write(sim, out / f"scenario_{name}.json")
        doc = sim.document()["trashPanda"]
        print(f"scenario_{name}.json: {len(sim.events)} events, {doc['duration_us'] / 1000:.0f} ms, models {[m['id'] for m in doc['models']]}")
