"""Reading and checking a trace file. The format is described in docs/design/trace-format.md."""

import json
from pathlib import Path

STARTED_BY = ("stage", "preload", "coordinator")
TIERS = ("gpu", "ram", "disk")
TRACK_KINDS = ("cpu", "gpu")
ROLES = ("main", "worker", "loader", "drop", "preload", "reader", "compute_stream", "copy_stream", "early_copy_stream")
POOL_KINDS = ("stage", "shared", "early_loader")
WAIT_REASONS = ("gpu", "staged", "loader", "pinned buffer", "preload")
SOURCES = {"load": ("mmap", "read", "chunked"), "stage": ("memory", "mapped")}
PLACES = ("disk", "ram", "gpu", "staged")                  # and pinned:N, a buffer of the event's pool
COUNTERS = ("vram", "ram", "pinned")

# category -> (the kind of event it is, the args it needs besides the ones that name a model)
CATEGORIES = {
    "call": ("X", ("block", "pass")),
    "run": ("X", ("block", "pass")),
    "wait": ("X", ("reason",)),
    "load": ("X", ("block", "from", "via", "to", "bytes", "source")),
    "read": ("X", ("block", "from", "to", "bytes")),
    "stage": ("X", ("block", "from", "to", "pool", "bytes", "source")),
    "copy": ("X", ("block", "from", "to", "pool", "bytes")),
    "alloc": ("X", ("pool", "buffer", "bytes")),
    "buffer": ("i", ("pool", "buffer", "block")),
    "mapped": ("i", ("block", "bytes")),
    "move": ("X", ("block", "from", "to", "bytes")),
    "free": ("i", ("block", "tier", "bytes")),
    "drop": ("X", ("block", "bytes")),
    "preload": ("X", ()),
    "replan": ("i", ("demoted", "promoted")),
    "cancel": ("i", ()),
    "lookahead": ("i", ("from", "to")),
    "pool": ("i", ("pool", "buffers")),                     # a pool event has no model
}
NO_MODEL = ("pool",)


class TraceError(ValueError):
    """The file is not a usable trace."""


def check_trace(data):
    """Raise TraceError unless `data` has what the viewer needs: the events and the plan they ran under."""
    if not isinstance(data, dict) or not isinstance(data.get("traceEvents"), list):
        raise TraceError("it is not a trace file: there is no 'traceEvents' list")
    plan = data.get("trashPanda")
    if not isinstance(plan, dict):
        raise TraceError("it has no 'trashPanda' section, which holds the plan the replay starts from")
    version = plan.get("version", 1)
    if version == 1:
        _check_v1(data, plan)
    elif version == 2:
        _check_v2(data, plan)
    else:
        raise TraceError(f"the trace format {version!r} is not known (this viewer reads 1 and 2)")


def _check_v1(data, plan):
    for key in ("blocks", "pinned_buffers", "duration_us"):
        if key not in plan:
            raise TraceError(f"the 'trashPanda' section has no '{key}'")
    for phase in plan.get("phases", []):
        if not isinstance(phase, dict) or not {"name", "start_us", "end_us"} <= set(phase) or phase["start_us"] >= phase["end_us"]:
            raise TraceError("each of the 'phases' needs a name, and a start_us before its end_us")
    for event in data["traceEvents"]:
        if not isinstance(event, dict):
            raise TraceError("'traceEvents' must be a list of objects")
        if event.get("ph") == "X":
            for key in ("ts", "dur", "tid", "cat"):
                if key not in event:
                    raise TraceError(f"the event {event.get('name')!r} has no '{key}'")
            if "block" not in (event.get("args") or {}):
                raise TraceError(f"the event {event.get('name')!r} has no 'block' in its args")


def _need(obj, keys, what):
    if not isinstance(obj, dict):
        raise TraceError(f"{what} must be an object")
    for key in keys:
        if key not in obj:
            raise TraceError(f"{what} has no '{key}'")


def _unique(items, key, what):
    seen = set()
    for item in items:
        if item[key] in seen:
            raise TraceError(f"{what} '{item[key]}' appears twice")
        seen.add(item[key])
    return seen


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_v2(data, plan):
    for key in ("duration_us", "models", "tracks", "pools"):
        if key not in plan:
            raise TraceError(f"the 'trashPanda' section has no '{key}'")
    duration = plan["duration_us"]
    if not _number(duration) or duration <= 0:
        raise TraceError("'duration_us' must be a positive number")
    for name in ("models", "tracks", "pools"):
        if not isinstance(plan[name], list):
            raise TraceError(f"'{name}' must be a list")
    if not plan["models"]:
        raise TraceError("'models' is empty: a trace is of at least one model")

    models = {}
    for model in plan["models"]:
        _need(model, ("id", "name", "started_by", "blocks", "options", "plan"), "a model")
        if model["started_by"] not in STARTED_BY:
            raise TraceError(f"model '{model['id']}': 'started_by' is {model['started_by']!r}, not one of {STARTED_BY}")
        for block in model["blocks"]:
            _need(block, ("id", "name", "tier"), f"a block of model '{model['id']}'")
            if block["tier"] not in TIERS:
                raise TraceError(f"block {block['id']} of model '{model['id']}': tier {block['tier']!r} is not one of {TIERS}")
        _unique(model["blocks"], "id", f"a block number of model '{model['id']}'")
        models[model["id"]] = {b["id"] for b in model["blocks"]}
    _unique(plan["models"], "id", "the model")

    for track in plan["tracks"]:
        _need(track, ("id", "name", "kind", "role"), "a track")
        if track["kind"] not in TRACK_KINDS:
            raise TraceError(f"track '{track['name']}': 'kind' must be one of {TRACK_KINDS}")
        if track["role"] not in ROLES:
            raise TraceError(f"track '{track['name']}': 'role' {track['role']!r} is not one of {ROLES}")
        if track.get("model") is not None and track["model"] not in models:
            raise TraceError(f"track '{track['name']}' is of the model '{track['model']}', which is not in 'models'")
    tracks = _unique(plan["tracks"], "id", "the track")

    pools = {}
    for pool in plan["pools"]:
        _need(pool, ("id", "kind", "buffers"), "a pool")
        if pool["kind"] not in POOL_KINDS:
            raise TraceError(f"pool '{pool['id']}': 'kind' must be one of {POOL_KINDS}")
        if not isinstance(pool["buffers"], int) or pool["buffers"] < 1:
            raise TraceError(f"pool '{pool['id']}': 'buffers' must be a whole number from 1")
        pools[pool["id"]] = pool["buffers"]
    _unique(plan["pools"], "id", "the pool")
    for model in plan["models"]:
        if model.get("pool") is not None and model["pool"] not in pools:
            raise TraceError(f"model '{model['id']}' uses the pool '{model['pool']}', which is not in 'pools'")

    last_end = {}
    for phase in plan.get("phases", []):
        _need(phase, ("model", "name", "start_us", "end_us"), "a phase")
        if phase["model"] not in models:
            raise TraceError(f"the phase '{phase['name']}' is of the model '{phase['model']}', which is not in 'models'")
        if not (_number(phase["start_us"]) and _number(phase["end_us"])) or phase["start_us"] >= phase["end_us"]:
            raise TraceError(f"the phase '{phase['name']}' needs a start_us before its end_us")
        if phase["end_us"] > duration:
            raise TraceError(f"the phase '{phase['name']}' ends after the recording")
        if phase["start_us"] < last_end.get(phase["model"], 0):
            raise TraceError(f"the phases of model '{phase['model']}' overlap, at '{phase['name']}'")
        last_end[phase["model"]] = phase["end_us"]

    for event in data["traceEvents"]:
        if not isinstance(event, dict):
            raise TraceError("'traceEvents' must be a list of objects")
        ph = event.get("ph")
        if ph == "M":
            continue
        if ph == "C":
            _check_counter(event, duration, models)
            continue
        _check_event(event, ph, duration, models, tracks, pools)


def _check_counter(event, duration, models):
    if event.get("name") not in COUNTERS:
        raise TraceError(f"a counter must be one of {COUNTERS}, not {event.get('name')!r}")
    if not _number(event.get("ts")) or not 0 <= event["ts"] <= duration:
        raise TraceError(f"the counter '{event['name']}' has a time outside the recording")
    args = event.get("args")
    if not isinstance(args, dict) or not all(_number(v) for v in args.values()):
        raise TraceError(f"the counter '{event['name']}' needs numbers in its args, one for each model")
    for model in args:
        if model not in models:
            raise TraceError(f"the counter '{event['name']}' names the model '{model}', which is not in 'models'")


def _check_event(event, ph, duration, models, tracks, pools):
    name = event.get("name")
    cat = event.get("cat")
    if cat not in CATEGORIES:
        raise TraceError(f"the event {name!r} has the category {cat!r}, which is not known")
    kind, needs = CATEGORIES[cat]
    if ph != kind:
        raise TraceError(f"the event {name!r} ({cat}) must be {'a slice' if kind == 'X' else 'an instant'}")
    for key in ("ts", "tid"):
        if key not in event:
            raise TraceError(f"the event {name!r} has no '{key}'")
    if event["tid"] not in tracks:
        raise TraceError(f"the event {name!r} is on the track {event['tid']!r}, which is not in 'tracks'")
    if not _number(event["ts"]) or event["ts"] < 0:
        raise TraceError(f"the event {name!r} starts before the recording")
    end = event["ts"] + (event.get("dur") or 0)
    if kind == "X":
        if not _number(event.get("dur")) or event["dur"] < 0:
            raise TraceError(f"the slice {name!r} has no valid 'dur'")
    if end > duration:
        raise TraceError(f"the event {name!r} ends after the recording")

    args = event.get("args")
    if not isinstance(args, dict):
        raise TraceError(f"the event {name!r} has no 'args'")
    required = needs if cat in NO_MODEL else ("model",) + needs
    for key in required:
        if key not in args:
            raise TraceError(f"the event {name!r} ({cat}) has no '{key}' in its args")
    model = args.get("model")
    if "model" in args and model not in models:
        raise TraceError(f"the event {name!r} is of the model {model!r}, which is not in 'models'")
    if "block" in args and args["block"] not in models[model]:
        raise TraceError(f"the event {name!r} is of block {args['block']!r}, which model '{model}' does not have")
    if "pool" in args and args["pool"] not in pools:
        raise TraceError(f"the event {name!r} uses the pool {args['pool']!r}, which is not in 'pools'")
    for key in ("from", "to", "via"):
        if key in args and cat != "lookahead":
            _check_place(name, key, args[key], pools.get(args.get("pool")))
    if "tier" in args and args["tier"] not in TIERS:
        raise TraceError(f"the event {name!r} has the tier {args['tier']!r}, which is not one of {TIERS}")
    if cat == "wait" and args["reason"] not in WAIT_REASONS:
        raise TraceError(f"the wait {name!r} has the reason {args['reason']!r}, which is not one of {WAIT_REASONS}")
    if cat in SOURCES and args["source"] not in SOURCES[cat]:
        raise TraceError(f"the event {name!r} has the source {args['source']!r}, which is not one of {SOURCES[cat]}")


def _check_place(name, key, place, buffers):
    if place in PLACES:
        return
    if isinstance(place, str) and place.startswith("pinned:") and place[7:].isdigit():
        if buffers is None:
            raise TraceError(f"the event {name!r} has '{key}' {place!r} but no 'pool' that has buffers")
        if not 1 <= int(place[7:]) <= buffers:
            raise TraceError(f"the event {name!r}: '{key}' {place!r} is beyond the {buffers} buffers of its pool")
        return
    raise TraceError(f"the event {name!r} has '{key}' {place!r}, which is not a place ({', '.join(PLACES)}, pinned:N)")


def load_trace(path):
    """Read a trace file and check it."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceError(f"it is not valid JSON ({exc})") from None
    check_trace(data)
    return data
