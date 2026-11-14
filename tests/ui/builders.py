"""Small traces in format 2, for the tests of the checker. `trace()` is valid as it comes; a test spoils one thing."""

import copy

MS = 1000                                           # microseconds in a millisecond

MAIN, WORKER, COPY, LOADER, COMPUTE, DROP, PRELOAD = 1, 2, 3, 4, 5, 6, 7


def block(i, tier):
    return {"id": i, "name": f"blocks.{i}.", "tier": tier, "stored_bytes": 100, "expanded_bytes": 200}


def model(id="m", tiers=("gpu", "ram", "disk"), pool="p", started_by="stage"):
    return {
        "id": id, "name": id, "started_by": started_by,
        "checkpoint": {"path": f"{id}.safetensors", "bytes": 300},
        "dtypes": {"stored": "fp16", "compute": "bf16"},
        "blocks": [block(i, t) for i, t in enumerate(tiers)],
        "outside_bytes": 0,
        "plan": {"vram_budget": 1000, "ram_budget": 1000, "headroom_bytes": 0, "stream_buffer_bytes": 260,
                 "staging_bytes": 200, "lookahead": 2, "warnings": []},
        "options": {"overlap": False, "ram_in_memory": False, "ram_preload": True, "vram_preload": True,
                    "pinned_buffers": 2, "blocks_in_flight": 2, "adaptive_lookahead": False},
        "pool": pool,
    }


def track(id, name, kind="cpu", role="main", model=None):
    out = {"id": id, "name": name, "kind": kind, "role": role}
    if model is not None:
        out["model"] = model
    return out


def tracks():
    return [
        track(MAIN, "main thread"),
        track(WORKER, "worker thread", role="worker", model="m"),
        track(COPY, "copy stream", kind="gpu", role="copy_stream", model="m"),
        track(LOADER, "loader thread", role="loader", model="m"),
        track(COMPUTE, "compute stream", kind="gpu", role="compute_stream"),
        track(DROP, "drop thread", role="drop", model="m"),
        track(PRELOAD, "preload thread", role="preload", model="m"),
    ]


def pool(id="p", kind="stage", buffers=2, models=("m",)):
    return {"id": id, "kind": kind, "buffers": buffers, "models": list(models)}


def sl(name, cat, tid, start_ms, end_ms, **args):
    return {"name": name, "cat": cat, "ph": "X", "pid": 1, "tid": tid, "ts": start_ms * MS, "dur": (end_ms - start_ms) * MS, "args": args}


def inst(name, cat, tid, at_ms, **args):
    return {"name": name, "cat": cat, "ph": "i", "s": "t", "pid": 1, "tid": tid, "ts": at_ms * MS, "args": args}


# One valid event of each category, for the model "m" (blocks 0, 1 and 2 on the GPU, in RAM and on disk) and the pool "p".
SAMPLES = {
    "call": lambda: sl("call b1", "call", MAIN, 1, 3, model="m", block=1, **{"pass": 1}),
    "run": lambda: sl("run b1", "run", COMPUTE, 1, 3, model="m", block=1, **{"pass": 1}),
    "wait": lambda: sl("wait for b1", "wait", MAIN, 1, 3, model="m", block=1, reason="staged"),
    "load": lambda: sl("load b0", "load", LOADER, 1, 3, model="m", block=0, bytes=100, source="read", **{"from": "disk", "via": "ram", "to": "gpu"}),
    "read": lambda: sl("read b2", "read", WORKER, 1, 3, model="m", block=2, bytes=100, pool="p", **{"from": "disk", "to": "pinned:1"}),
    "stage": lambda: sl("stage b1", "stage", WORKER, 1, 3, model="m", block=1, bytes=100, pool="p", source="memory", **{"from": "ram", "to": "pinned:2"}),
    "copy": lambda: sl("copy b1", "copy", COPY, 1, 3, model="m", block=1, bytes=100, pool="p", **{"from": "pinned:2", "to": "staged"}),
    "alloc": lambda: sl("pin", "alloc", WORKER, 1, 3, model="m", pool="p", buffer=1, bytes=128),
    "buffer": lambda: inst("acquire pinned:1", "buffer", WORKER, 1, model="m", pool="p", buffer=1, block=1),
    "mapped": lambda: inst("mapped b1", "mapped", MAIN, 1, model="m", block=1, bytes=100),
    "move": lambda: sl("release_gpu b0", "move", MAIN, 1, 3, model="m", block=0, bytes=100, **{"from": "gpu", "to": "ram"}),
    "free": lambda: inst("free b0", "free", MAIN, 1, model="m", block=0, tier="gpu", bytes=100),
    "drop": lambda: sl("drop b1", "drop", DROP, 1, 3, model="m", block=1, bytes=100),
    "preload": lambda: sl("preload", "preload", PRELOAD, 1, 3, model="m"),
    "replan": lambda: inst("replan", "replan", MAIN, 1, model="m", demoted=[0], promoted=[]),
    "cancel": lambda: inst("cancel", "cancel", MAIN, 1, model="m"),
    "lookahead": lambda: inst("look-ahead 3", "lookahead", MAIN, 1, model="m", **{"from": 2, "to": 3}),
    "pool": lambda: inst("pool grew", "pool", MAIN, 1, pool="p", buffers=3),
}


def trace(events=None, models=None, track_list=None, pools=None, phases=None, duration_ms=100, **plan):
    """A valid trace of one model, unless the arguments say otherwise."""
    doc = {
        "displayTimeUnit": "ms",
        "trashPanda": {
            "version": 2,
            "duration_us": duration_ms * MS,
            "models": [model()] if models is None else models,
            "tracks": tracks() if track_list is None else track_list,
            "pools": [pool()] if pools is None else pools,
            "phases": [{"model": "m", "name": "pass 1", "start_us": 0, "end_us": duration_ms * MS}] if phases is None else phases,
            **plan,
        },
        "traceEvents": [SAMPLES["run"]()] if events is None else events,
    }
    return copy.deepcopy(doc)
