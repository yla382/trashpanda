import json

import pytest

from trash_panda.ui.load import TraceError, check_trace, load_trace

from .helpers import SIMPLE


def write(tmp_path, data):
    path = tmp_path / "t.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def valid():
    return json.loads(SIMPLE.read_text(encoding="utf-8"))


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
