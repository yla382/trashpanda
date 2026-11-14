"""The traces of several models in tests/ui/data/: three made-up models, and the file the recorder writes."""

import json

import pytest

from trash_panda.ui.load import load_trace

from .helpers import DATA

THREE = DATA / "three_models.json"
RECORDED = DATA / "recorded_tiny.json"


def end(e):
    return e["ts"] + e.get("dur", 0)


@pytest.fixture(scope="module")
def three():
    return load_trace(THREE)


def test_three_models_are_in_the_file_and_each_was_started_in_its_own_way(three):
    models = three["trashPanda"]["models"]
    assert [m["id"] for m in models] == ["text", "unet", "vae"]
    assert [m["started_by"] for m in models] == ["stage", "preload", "coordinator"]


def test_every_model_has_its_own_blocks_and_the_numbers_start_at_0_in_each(three):
    for m in three["trashPanda"]["models"]:
        assert [b["id"] for b in m["blocks"]] == list(range(len(m["blocks"])))


def test_the_models_overlap_in_time(three):
    phases = three["trashPanda"]["phases"]
    span = {m: (min(p["start_us"] for p in phases if p["model"] == m), max(p["end_us"] for p in phases if p["model"] == m))
            for m in ("text", "unet", "vae")}
    assert span["unet"][0] < span["text"][1]            # the unet is preloaded while the text encoder runs
    assert span["vae"][0] < span["unet"][1]             # the vae is loaded while the unet's last pass runs


def test_the_main_thread_does_one_thing_at_a_time_for_all_the_models(three):
    main = next(t["id"] for t in three["trashPanda"]["tracks"] if t["role"] == "main")
    own = sorted((e for e in three["traceEvents"] if e["ph"] == "X" and e["tid"] == main), key=lambda e: e["ts"])
    for before, after in zip(own, own[1:]):
        assert end(before) <= after["ts"], f"{before['name']} overlaps {after['name']}"


def test_every_model_runs_in_each_of_its_passes(three):
    calls = [e for e in three["traceEvents"] if e.get("cat") == "call"]
    for m in three["trashPanda"]["models"]:
        passes = [p for p in three["trashPanda"]["phases"] if p["model"] == m["id"] and p["name"].startswith("pass")]
        mine = [e for e in calls if e["args"]["model"] == m["id"]]
        assert len(mine) == len(passes) * len(m["blocks"])


def test_the_recorded_file_is_a_valid_trace_of_one_model():
    doc = load_trace(RECORDED)
    plan = doc["trashPanda"]
    assert plan["version"] == 2 and len(plan["models"]) == 1
    assert [p["name"] for p in plan["phases"]] == ["load", "pass 1", "pass 2", "pass 3"]
    assert len([e for e in doc["traceEvents"] if e.get("cat") == "call"]) == 3 * len(plan["models"][0]["blocks"])


def test_the_recorded_file_is_plain_json():
    json.loads(RECORDED.read_text(encoding="utf-8"))


def test_the_vae_has_an_early_loader_with_buffers_of_its_own_and_the_unet_a_pool_of_its_own(three):
    pools = {p["id"]: p for p in three["trashPanda"]["pools"]}
    assert pools["early-vae"]["kind"] == "early_loader" and pools["early-vae"]["models"] == ["vae"]
    assert pools["pool-unet"]["kind"] == "stage" and pools["pool-unet"]["models"] == ["unet"]


def test_the_early_loader_loads_the_vae_in_chunks_while_the_unet_still_runs(three):
    loads = [e for e in three["traceEvents"] if e.get("cat") == "load" and e["args"]["model"] == "vae"]
    assert loads and all(e["args"]["source"] == "chunked" for e in loads)
    unet_end = max(p["end_us"] for p in three["trashPanda"]["phases"] if p["model"] == "unet")
    assert min(e["ts"] for e in loads) < unet_end


def test_the_final_pass_of_the_unet_frees_each_block_once_after_it_ran(three):
    frees = [e for e in three["traceEvents"] if e.get("cat") == "free"]
    assert sorted(e["args"]["block"] for e in frees) == list(range(10))
    last = max((p for p in three["trashPanda"]["phases"] if p["model"] == "unet" and p["name"].startswith("pass")),
               key=lambda p: p["start_us"])
    runs = {e["args"]["block"]: end(e) for e in three["traceEvents"]
            if e.get("cat") == "call" and e["args"]["model"] == "unet" and e["args"]["pass"] == 4}
    for e in frees:
        assert e["ts"] == runs[e["args"]["block"]] and e["ts"] >= last["start_us"]


def test_a_replan_puts_a_vae_block_in_ram_and_that_block_is_not_loaded(three):
    replans = [e for e in three["traceEvents"] if e.get("cat") == "replan"]
    assert [(e["args"]["model"], e["args"]["demoted"]) for e in replans] == [("vae", [3])]
    loaded = {e["args"]["block"] for e in three["traceEvents"] if e.get("cat") == "load" and e["args"]["model"] == "vae"}
    assert 3 not in loaded and loaded == {0, 1, 2}


def compute_runs(doc):
    stream = next(t["id"] for t in doc["trashPanda"]["tracks"] if t["role"] == "compute_stream")
    return stream, sorted((e for e in doc["traceEvents"] if e.get("cat") == "run" and e["tid"] == stream), key=lambda e: e["ts"])


def test_the_gpu_has_a_compute_stream_on_its_own_clock_with_one_run_for_each_call(three):
    stream, runs = compute_runs(three)
    track = next(t for t in three["trashPanda"]["tracks"] if t["id"] == stream)
    assert track["kind"] == "gpu"
    assert len(runs) == len([e for e in three["traceEvents"] if e.get("cat") == "call"])


def test_the_compute_stream_runs_one_block_at_a_time_and_a_block_runs_after_it_was_launched(three):
    _, runs = compute_runs(three)
    for before, after in zip(runs, runs[1:]):
        assert end(before) <= after["ts"]
    launched = {(e["args"]["model"], e["args"]["pass"], e["args"]["block"]): e["ts"] for e in three["traceEvents"] if e.get("cat") == "call"}
    for r in runs:
        assert r["ts"] >= launched[(r["args"]["model"], r["args"]["pass"], r["args"]["block"])]


def test_the_gpu_is_idle_while_the_unet_waits_for_a_copy(three):
    _, runs = compute_runs(three)
    gaps = [(a, b) for a, b in zip(runs, runs[1:])
            if a["args"]["model"] == b["args"]["model"] == "unet" and a["args"]["pass"] == b["args"]["pass"] and b["ts"] - end(a) >= 500]
    assert gaps, "no idle gap of the GPU within a pass"
