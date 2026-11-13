import json

import pytest

from trash_panda.ui.load import DEMO, TraceError, check_trace, load_trace


def write(tmp_path, data):
    path = tmp_path / "t.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def valid():
    return json.loads(DEMO.read_text(encoding="utf-8"))


def test_the_demo_trace_is_a_valid_trace():
    data = load_trace(DEMO)
    assert len(data["trashPanda"]["blocks"]) == 8
    assert data["trashPanda"]["pinned_buffers"] == 2


def phase(data, name):
    return next(p for p in data["trashPanda"]["phases"] if p["name"] == name)


def test_the_demo_phases_follow_each_other_and_fill_the_whole_run():
    phases = valid()["trashPanda"]["phases"]
    assert phases[0]["start_us"] == 0
    for before, after in zip(phases, phases[1:]):
        assert before["end_us"] == after["start_us"]
    assert phases[-1]["end_us"] == valid()["trashPanda"]["duration_us"]


def test_the_demo_runs_and_waits_fill_the_whole_pass():
    data = valid()
    events = [e for e in data["traceEvents"] if e["ph"] == "X" and e["cat"] in ("run", "wait")]
    pass_ = phase(data, "pass 1")
    assert sum(e["dur"] for e in events) == pass_["end_us"] - pass_["start_us"]


def test_the_demo_load_puts_every_kept_block_where_it_is_kept():
    data = valid()
    load = phase(data, "load")
    events = [e for e in data["traceEvents"] if e["ph"] == "X" and e["args"].get("phase") == "load"]
    assert all(load["start_us"] <= e["ts"] and e["ts"] + e["dur"] <= load["end_us"] for e in events)
    for block in data["trashPanda"]["blocks"]:
        reads = [e for e in events if e["cat"] == "read" and e["args"]["block"] == block["id"]]
        copies = [e for e in events if e["cat"] == "copy" and e["args"]["block"] == block["id"]]
        assert all(e["args"]["from"] == "disk" for e in reads)
        if block["tier"] == "disk":
            assert not reads and not copies                     # these stay in the file
        else:
            assert len(reads) == 1
            assert len(copies) == (1 if block["tier"] == "gpu" else 0)
            assert (reads[0]["args"]["to"] == "ram") == (block["tier"] == "ram")


def test_nothing_runs_during_the_load():
    data = valid()
    load = phase(data, "load")
    runs = [e for e in data["traceEvents"] if e.get("cat") == "run"]
    assert all(e["ts"] >= load["end_us"] for e in runs)


def test_every_demo_event_is_inside_the_pass():
    data = valid()
    end = data["trashPanda"]["duration_us"]
    for e in data["traceEvents"]:
        if e["ph"] in ("X", "i"):
            assert 0 <= e["ts"] and e["ts"] + e.get("dur", 0) <= end, e["name"]


def test_every_streamed_block_is_freed_once():
    data = valid()
    tiers = {b["id"]: b["tier"] for b in data["trashPanda"]["blocks"]}
    freed = [e["args"]["block"] for e in data["traceEvents"] if e.get("cat") == "free"]
    assert sorted(freed) == sorted(i for i, tier in tiers.items() if tier != "gpu")


def test_a_pinned_buffer_is_never_held_by_two_blocks_at_once():
    data = valid()
    held = {}
    buffer_events = [e for e in data["traceEvents"] if e.get("cat") == "buffer"]
    for e in sorted(buffer_events, key=lambda e: (e["ts"], e["name"].startswith("acquire"))):     # a release before an acquire at the same time
        buffer = e["args"]["buffer"]
        if e["name"].startswith("acquire"):
            assert buffer not in held, f"pinned:{buffer} taken again at {e['ts']}"
            held[buffer] = e["args"]["block"]
        else:
            assert held.pop(buffer) == e["args"]["block"]
    assert not held


def test_a_file_that_is_not_json_is_an_error(tmp_path):
    with pytest.raises(TraceError, match="not valid JSON"):
        load_trace(write(tmp_path, "{nope"))


def test_a_file_without_events_is_an_error(tmp_path):
    with pytest.raises(TraceError, match="traceEvents"):
        load_trace(write(tmp_path, {"hello": 1}))


def test_a_file_without_the_plan_is_an_error(tmp_path):
    with pytest.raises(TraceError, match="trashPanda"):
        load_trace(write(tmp_path, {"traceEvents": []}))


@pytest.mark.parametrize("key", ["blocks", "pinned_buffers", "duration_us"])
def test_the_plan_must_have_each_of_its_fields(key):
    data = valid()
    del data["trashPanda"][key]
    with pytest.raises(TraceError, match=key):
        check_trace(data)


def test_a_phase_must_have_its_fields_and_run_forwards():
    data = valid()
    data["trashPanda"]["phases"][0]["end_us"] = 0
    with pytest.raises(TraceError, match="phases"):
        check_trace(data)
    data = valid()
    del data["trashPanda"]["phases"][0]["name"]
    with pytest.raises(TraceError, match="phases"):
        check_trace(data)


def test_a_trace_without_phases_is_fine():
    data = valid()
    del data["trashPanda"]["phases"]
    check_trace(data)


@pytest.mark.parametrize("key", ["ts", "dur", "tid", "cat"])
def test_a_slice_must_have_its_fields(key):
    data = valid()
    del next(e for e in data["traceEvents"] if e["ph"] == "X")[key]
    with pytest.raises(TraceError, match=key):
        check_trace(data)


def test_a_slice_must_name_its_block():
    data = valid()
    next(e for e in data["traceEvents"] if e["ph"] == "X")["args"] = {}
    with pytest.raises(TraceError, match="block"):
        check_trace(data)


def test_events_must_be_objects():
    data = valid()
    data["traceEvents"].append(3)
    with pytest.raises(TraceError, match="objects"):
        check_trace(data)
