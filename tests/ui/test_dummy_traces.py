"""Checks on the made-up trace files in `data/`, which the viewer is worked on with: each follows the rules the code of trash-panda follows
(one worker thread, a pool of pinned buffers, a look-ahead, a loader thread for the kept blocks), so a viewer that is
right for them is right for a run that behaves the same way."""

import json
import re
import pytest

from trash_panda.ui.load import load_trace

from .helpers import SIMPLE, STEPS

BOTH = [SIMPLE, STEPS]


def slices(data, cat=None, tid=None):
    return [e for e in data["traceEvents"] if e["ph"] == "X" and (cat is None or e["cat"] == cat) and (tid is None or e["tid"] == tid)]


def end(e):
    return e["ts"] + e.get("dur", 0)


def thread(data, name):
    return next(e["tid"] for e in data["traceEvents"] if e["ph"] == "M" and e["name"] == "thread_name" and name in e["args"]["name"])


def phases(data, prefix=""):
    return [p for p in data["trashPanda"]["phases"] if p["name"].startswith(prefix)]


@pytest.fixture(scope="module")
def steps():
    return load_trace(STEPS)


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_every_trace_loads_and_stays_inside_the_run(path):
    data = load_trace(path)
    last = data["trashPanda"]["duration_us"]
    for e in data["traceEvents"]:
        if e["ph"] in ("X", "i"):
            assert 0 <= e["ts"] <= last and end(e) <= last, e["name"]


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_the_main_thread_does_one_thing_at_a_time(path):
    data = load_trace(path)
    main = min(slices(data, "run"), key=lambda e: e["ts"])["tid"]
    own = sorted((e for e in slices(data, tid=main) if e["cat"] in ("run", "wait")), key=lambda e: e["ts"])
    for before, after in zip(own, own[1:]):
        assert end(before) <= after["ts"], f"{before['name']} overlaps {after['name']}"


@pytest.mark.parametrize("path", BOTH, ids=lambda p: p.name)
def test_a_pinned_buffer_is_never_held_by_two_blocks_at_once(path):
    data = load_trace(path)
    held = {}
    buffer_events = [e for e in data["traceEvents"] if e.get("cat") == "buffer"]
    for e in sorted(buffer_events, key=lambda e: (e["ts"], e["name"].startswith("acquire"))):      # a release first when both happen at the same time
        buffer = e["args"]["buffer"]
        if e["name"].startswith("acquire"):
            assert buffer not in held, f"pinned:{buffer} taken again at {e['ts']}"
            held[buffer] = e["args"]["block"]
        else:
            assert held.pop(buffer) == e["args"]["block"]
    assert not held


def test_the_steps_trace_has_overlap_and_four_passes(steps):
    plan = steps["trashPanda"]
    assert plan["overlap"] is True and plan["lookahead"] == 2 and plan["pinned_buffers"] == 2
    assert [p["name"] for p in phases(steps)] == ["load", "pass 1", "pass 2", "pass 3", "pass 4"]


def test_the_passes_follow_each_other_with_a_gap_between_them(steps):
    passes = phases(steps, "pass")
    for before, after in zip(passes, passes[1:]):
        assert before["end_us"] < after["start_us"]             # the scheduler and the rest of the pipeline run in between
    assert phases(steps, "load")[0]["end_us"] <= passes[0]["start_us"]
    assert passes[-1]["end_us"] == steps["trashPanda"]["duration_us"]


def test_each_block_runs_once_in_every_pass(steps):
    runs = slices(steps, "run")
    for p in phases(steps, "pass"):
        mine = [e for e in runs if p["start_us"] <= e["ts"] < p["end_us"]]
        assert [e["args"]["block"] for e in sorted(mine, key=lambda e: e["ts"])] == list(range(len(steps["trashPanda"]["blocks"])))


def test_the_loader_thread_loads_each_kept_block_once_while_the_model_runs(steps):
    loader = thread(steps, "loader")
    loads = slices(steps, "load", loader)
    kept = [b["id"] for b in steps["trashPanda"]["blocks"] if b["tier"] == "gpu"]
    assert [e["args"]["block"] for e in sorted(loads, key=lambda e: e["ts"])] == kept
    runs = slices(steps, "run")
    assert any(r["ts"] < end(load) and load["ts"] < end(r) for load in loads for r in runs), "no load overlaps a run"
    in_order = sorted(loads, key=lambda e: e["ts"])
    for before, after in zip(in_order, in_order[1:]):
        assert end(before) <= after["ts"]                       # one thread, one block at a time


def test_the_model_waits_for_the_loader_only_in_the_first_pass(steps):
    waits = [e for e in slices(steps, "wait") if e["args"]["reason"] == "loader"]
    assert waits
    first = phases(steps, "pass")[0]
    assert all(first["start_us"] <= e["ts"] < first["end_us"] for e in waits)
    for w in waits:                                             # and it ends when the block is loaded
        load = next(e for e in slices(steps, "load") if e["args"]["block"] == w["args"]["block"])
        assert end(w) == end(load)


def test_the_worker_does_one_stage_job_at_a_time(steps):
    worker = thread(steps, "worker")
    own = sorted(slices(steps, tid=worker), key=lambda e: e["ts"])
    for before, after in zip(own, own[1:]):
        assert end(before) <= after["ts"], f"{before['name']} overlaps {after['name']}"


def test_the_copy_stream_does_one_copy_at_a_time(steps):
    own = sorted(slices(steps, "copy"), key=lambda e: e["ts"])
    for before, after in zip(own, own[1:]):
        assert end(before) <= after["ts"]


def test_a_block_is_read_and_copied_before_it_runs(steps):
    streamed = [b["id"] for b in steps["trashPanda"]["blocks"] if b["tier"] != "gpu"]
    for block in streamed:
        runs = sorted((e for e in slices(steps, "run") if e["args"]["block"] == block), key=lambda e: e["ts"])
        copies = sorted((e for e in slices(steps, "copy") if e["args"]["block"] == block), key=lambda e: e["ts"])
        first = sorted((e for e in slices(steps) if e["args"]["block"] == block and e["cat"] in ("stage", "read")
                        and e["args"]["to"].startswith("pinned:")), key=lambda e: e["ts"])
        assert len(runs) == len(copies) == len(first) == 4
        for run, copy, before in zip(runs, copies, first):
            assert before["ts"] < copy["ts"] and end(before) <= copy["ts"]
            assert end(copy) <= run["ts"]


def test_no_more_blocks_wait_on_the_gpu_than_the_look_ahead(steps):
    waiting = []
    for run in slices(steps, "run"):
        block = run["args"]["block"]
        if steps["trashPanda"]["blocks"][block]["tier"] != "gpu":
            copy = max((e for e in slices(steps, "copy") if e["args"]["block"] == block and end(e) <= run["ts"]), key=end)
            waiting.append((end(copy), run["ts"]))
    for at, _ in waiting:
        assert sum(1 for done, started in waiting if done <= at < started) <= steps["trashPanda"]["lookahead"]


def test_the_look_ahead_reaches_into_the_next_pass_from_the_second_pass_on(steps):
    passes = phases(steps, "pass")
    first_streamed = next(b["id"] for b in steps["trashPanda"]["blocks"] if b["tier"] != "gpu")

    def started(k):
        copy = next(e for e in slices(steps, "copy") if e["args"]["block"] == first_streamed and f"(pass {k})" in e["name"])
        return copy["ts"]

    assert started(2) >= passes[0]["end_us"]                    # the first pass is not linked to the second yet
    assert started(3) < passes[1]["end_us"]                     # from the second pass on, it is
    assert started(4) < passes[2]["end_us"]


def test_the_disk_blocks_are_read_again_in_every_pass(steps):
    disk = [b["id"] for b in steps["trashPanda"]["blocks"] if b["tier"] == "disk"]
    for block in disk:
        reads = [e for e in slices(steps, "read") if e["args"]["block"] == block]
        assert len(reads) == 4 and all(e["args"]["to"].startswith("pinned:") for e in reads)


def test_the_ram_tier_is_read_into_memory_once_before_the_run(steps):
    load = phases(steps, "load")[0]
    ram = [b["id"] for b in steps["trashPanda"]["blocks"] if b["tier"] == "ram"]
    reads = [e for e in slices(steps, "read") if e["args"]["to"] == "ram"]
    assert sorted(e["args"]["block"] for e in reads) == ram
    assert all(load["start_us"] <= e["ts"] and end(e) <= load["end_us"] for e in reads)


def test_the_time_on_the_main_thread_is_less_than_the_run(steps):
    main = min(slices(steps, "run"), key=lambda e: e["ts"])["tid"]
    busy = sum(e["dur"] for e in slices(steps, tid=main))
    load = phases(steps, "load")[0]
    assert busy <= steps["trashPanda"]["duration_us"] - (load["end_us"] - load["start_us"])


def test_the_steps_trace_file_is_the_one_the_generator_wrote():
    json.loads(STEPS.read_text(encoding="utf-8"))               # plain JSON, nothing else in the file
    assert re.search(r'"lookahead": 2', STEPS.read_text(encoding="utf-8"))


@pytest.fixture(params=BOTH, ids=lambda p: p.name)
def both(request):
    return load_trace(request.param)


def test_the_phases_come_in_order_and_cover_the_run(both):
    found = both["trashPanda"]["phases"]
    assert found[0]["start_us"] == 0
    for before, after in zip(found, found[1:]):
        assert before["end_us"] <= after["start_us"]
    assert found[-1]["end_us"] == both["trashPanda"]["duration_us"]


def test_the_load_phase_comes_first_and_the_blocks_run_after_it_has_started(both):
    load = phases(both, "load")[0]
    first_run = min(e["ts"] for e in slices(both, "run"))
    assert load["start_us"] == 0 and load["end_us"] <= first_run


def test_nothing_goes_from_the_file_to_the_gpu_in_one_step(both):
    for e in slices(both):
        args = e["args"]
        if args.get("from") == "disk":
            assert args.get("to") != "gpu" or args.get("via") == "ram", e["name"]


def test_a_kept_block_is_loaded_through_ordinary_memory_and_never_through_a_pinned_buffer(both):
    for block in both["trashPanda"]["blocks"]:
        if block["tier"] != "gpu":
            continue
        mine = [e for e in slices(both) if e["args"]["block"] == block["id"] and e["cat"] != "run" and e["cat"] != "wait"]
        assert len(mine) == 1, mine
        load = mine[0]
        assert load["cat"] == "load" and load["args"]["from"] == "disk"
        assert load["args"]["via"] == "ram" and load["args"]["to"] == "gpu"


def test_a_ram_block_is_read_from_the_file_into_ordinary_memory_once_before_the_run(both):
    load = phases(both, "load")[0]
    ram = [b["id"] for b in both["trashPanda"]["blocks"] if b["tier"] == "ram"]
    reads = [e for e in slices(both, "read") if e["args"]["to"] == "ram"]
    assert sorted(e["args"]["block"] for e in reads) == ram
    assert all(load["start_us"] <= e["ts"] and end(e) <= load["end_us"] for e in reads)
    assert all(e["args"]["from"] == "disk" for e in reads)


def test_a_disk_block_is_not_read_before_the_run(both):
    disk = [b["id"] for b in both["trashPanda"]["blocks"] if b["tier"] == "disk"]
    load = phases(both, "load")[0]
    for e in slices(both):
        if e["args"]["block"] in disk and e["cat"] in ("read", "stage", "copy"):
            assert e["ts"] >= load["end_us"], e["name"]


def test_the_simple_trace_loads_before_the_model_runs_and_nothing_waits_for_the_loader():
    simple = load_trace(SIMPLE)
    assert simple["trashPanda"]["overlap"] is False
    last_load = max(end(e) for e in slices(simple, "load"))
    assert last_load <= min(e["ts"] for e in slices(simple, "run"))
    assert not [e for e in slices(simple, "wait") if e["args"]["reason"] == "loader"]
