"""Reading and checking a trace file."""

import json
from pathlib import Path

DEMO = Path(__file__).with_name("dummy_trace.json")      # made-up data to work on the viewer with, not a real run


class TraceError(ValueError):
    """The file is not a usable trace."""


def check_trace(data):
    """Raise TraceError unless `data` has what the viewer needs: the events and the plan they ran under."""
    if not isinstance(data, dict) or not isinstance(data.get("traceEvents"), list):
        raise TraceError("it is not a trace file: there is no 'traceEvents' list")
    plan = data.get("trashPanda")
    if not isinstance(plan, dict):
        raise TraceError("it has no 'trashPanda' section, which holds the plan the replay starts from")
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


def load_trace(path):
    """Read a trace file and check it."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceError(f"it is not valid JSON ({exc})") from None
    check_trace(data)
    return data
