"""Recording a run as a trace file (format 2, docs/design/trace-format.md): `TRASH_PANDA_TRACE=1` makes stage() write one into TRASH_PANDA_TRACE_DIR.

These tests only look at the file that stage() leaves, so they hold for any recorder that writes format 2."""

import importlib
import json
import re
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trash_panda.offload.trace")          # until the recorder exists, these tests are skipped

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import preload, stage  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402
from trash_panda.ui.load import check_trace  # noqa: E402

stage_module = importlib.import_module("trash_panda.offload.stage")

HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
PLAN = dict(compute_dtype="bf16", hardware=HARDWARE, min_free_ram=0, resident_blocks=2)     # bf16: the float32 blocks are wrapped


def X():
    return torch.randn(3, 16, dtype=torch.bfloat16)                  # the weights outside the blocks are expanded to bf16 too


@pytest.fixture(autouse=True)
def on_the_cpu(monkeypatch):
    """Run on the CPU whatever the machine has, and nothing remembered at the start."""
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), nn=torch.nn))
    monkeypatch.setattr(stage_module, "_activations", {})


def record(path, checkpoint, passes=2, **more):
    x = X()
    with stage(meta_model, checkpoint, **{**PLAN, **more}) as (model, _p, _handle):
        outputs = [model(x) for _ in range(passes)]
    return outputs, json.loads(path.read_text(encoding="utf-8"))


def events(doc, cat):
    return sorted((e for e in doc["traceEvents"] if e.get("cat") == cat), key=lambda e: e["ts"])


def phase(doc, name):
    return next(p for p in doc["trashPanda"]["phases"] if p["name"] == name)


def test_nothing_is_written_unless_asked_for(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.delenv("TRASH_PANDA_TRACE", raising=False)
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path / "traces"))
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (model, _p, _handle):
        model(X())
    assert not (tmp_path / "traces").exists()


def test_the_file_is_a_valid_trace_in_format_2(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path)
    check_trace(doc)
    assert doc["trashPanda"]["version"] == 2
    assert [m["started_by"] for m in doc["trashPanda"]["models"]] == ["stage"]


def test_the_file_is_written_when_the_block_ends_and_not_before(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (model, _p, _handle):
        model(X())
        assert not trace_file.exists()
    assert trace_file.exists()


def test_a_path_in_the_variable_is_not_a_switch_that_names_a_file(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_TRACE", "0")
    monkeypatch.setenv("TRASH_PANDA_TRACE_DIR", str(tmp_path / "traces"))
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (model, _p, _handle):
        model(X())
    assert not (tmp_path / "traces").exists()


def test_each_recording_is_a_new_file_named_by_the_time_in_the_trace_folder(tiny_checkpoint, trace_file, capsys):
    _, path = tiny_checkpoint
    record(trace_file, path, passes=1)
    record(trace_file, path, passes=1)
    files = trace_file.files()
    assert len(files) == 2 and files[0].parent == trace_file.directory
    assert all(re.fullmatch(r"\d{8}-\d{6}-\d{3}(-\d+)?\.json", f.name) for f in files)
    out = capsys.readouterr().out
    assert out.count("trash-panda: trace written to ") == 2 and str(files[0]) in out and str(files[1]) in out


def test_the_file_is_written_when_the_block_raises(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    with pytest.raises(RuntimeError):
        with stage(meta_model, path, **PLAN) as (model, _p, _handle):
            model(X())
            raise RuntimeError("the caller's error")
    check_trace(json.loads(trace_file.read_text(encoding="utf-8")))


def test_recording_does_not_change_the_result(tiny_checkpoint, trace_file, monkeypatch):
    _, path = tiny_checkpoint
    torch.manual_seed(1)
    recorded, _ = record(trace_file, path, passes=1)
    monkeypatch.delenv("TRASH_PANDA_TRACE")
    torch.manual_seed(1)
    x = X()
    with stage(meta_model, path, **PLAN) as (model, _p, _handle):
        plain = model(x)
    assert torch.equal(recorded[0], plain)


def test_the_model_has_the_blocks_of_the_plan_with_their_tiers_and_sizes(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (model, p, _handle):
        model(X())
    block_plans = [(b.prefix, b.tier.value, b.stored_bytes, b.expanded_bytes) for b in p.blocks]
    doc = json.loads(trace_file.read_text(encoding="utf-8"))
    entry = doc["trashPanda"]["models"][0]
    assert [(b["name"], b["tier"], b["stored_bytes"], b["expanded_bytes"]) for b in entry["blocks"]] == block_plans
    assert [b["tier"] for b in entry["blocks"]] == ["gpu", "gpu", "ram", "ram"]
    assert [b["id"] for b in entry["blocks"]] == [0, 1, 2, 3]
    assert entry["plan"]["lookahead"] == p.lookahead


def test_every_pass_has_one_call_for_each_block_in_order(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path, passes=3)
    calls = events(doc, "call")
    assert [(e["args"]["pass"], e["args"]["block"]) for e in calls] == [(n, b) for n in (1, 2, 3) for b in range(4)]
    assert all(e["dur"] > 0 for e in calls)
    main = doc["trashPanda"]["tracks"][0]
    assert main["role"] == "main" and main["kind"] == "cpu"
    assert all(e["tid"] == main["id"] for e in calls)


def test_the_calls_of_a_pass_do_not_overlap(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path, passes=2)
    calls = events(doc, "call")
    for before, after in zip(calls, calls[1:]):
        assert before["ts"] + before["dur"] <= after["ts"]


def test_the_phases_are_a_load_and_one_for_each_pass(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path, passes=3)
    phases = doc["trashPanda"]["phases"]
    assert [p["name"] for p in phases] == ["load", "pass 1", "pass 2", "pass 3"]
    for before, after in zip(phases, phases[1:]):
        assert before["end_us"] <= after["start_us"]


def test_each_call_is_inside_the_phase_of_its_pass(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path, passes=2)
    for e in events(doc, "call"):
        p = phase(doc, f"pass {e['args']['pass']}")
        assert p["start_us"] <= e["ts"] and e["ts"] + e["dur"] <= p["end_us"]


def test_each_kept_block_has_one_load_through_ordinary_memory_and_the_others_have_none(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path)
    entry = doc["trashPanda"]["models"][0]
    loads = events(doc, "load")
    assert [e["args"]["block"] for e in loads] == [0, 1]                  # resident_blocks=2
    for e in loads:
        args = e["args"]
        assert (args["from"], args["via"], args["to"], args["source"]) == ("disk", "ram", "gpu", "mmap")
        assert args["bytes"] == entry["blocks"][args["block"]]["stored_bytes"]


def test_the_loads_are_inside_the_load_phase(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path)
    load = phase(doc, "load")
    for e in events(doc, "load"):
        assert load["start_us"] <= e["ts"] and e["ts"] + e["dur"] <= load["end_us"]


def test_the_duration_covers_the_whole_recording(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path)
    last = max(e["ts"] + e.get("dur", 0) for e in doc["traceEvents"] if e["ph"] in ("X", "i"))
    assert doc["trashPanda"]["duration_us"] >= last


def test_a_second_stage_writes_its_own_file_and_the_first_is_not_mixed_in(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    record(trace_file, path, passes=3)
    _, second = record(trace_file, path, passes=1)
    assert len(events(second, "call")) == 4
    assert [p["name"] for p in second["trashPanda"]["phases"]] == ["load", "pass 1"]


def test_with_overlap_the_loads_are_on_the_loader_thread_and_were_read_without_a_memory_map(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path, overlap=True)
    check_trace(doc)
    tracks = {t["id"]: t for t in doc["trashPanda"]["tracks"]}
    loads = events(doc, "load")
    assert [e["args"]["block"] for e in loads] == [0, 1]
    assert all(tracks[e["tid"]]["role"] == "loader" for e in loads)
    assert all(e["args"]["source"] == "read" for e in loads)
    assert doc["trashPanda"]["models"][0]["options"]["overlap"] is True


def test_without_overlap_the_loads_are_on_the_main_thread_through_a_memory_map(tiny_checkpoint, trace_file):
    _, path = tiny_checkpoint
    _, doc = record(trace_file, path)
    tracks = {t["id"]: t for t in doc["trashPanda"]["tracks"]}
    assert all(tracks[e["tid"]]["role"] == "main" for e in events(doc, "load"))
    assert doc["trashPanda"]["models"][0]["options"]["overlap"] is False


def test_the_load_phase_of_a_preloaded_model_starts_when_it_is_entered_and_not_when_its_preload_began(tiny_checkpoint, trace_file):
    """The preload has a slice of its own. A model that waits for its turn is not loading while it waits: its load phase is the time from
    the entry until it is ready (for a model that comes after a long one, the wait is most of the time of the run)."""
    _, path = tiny_checkpoint
    with stage(meta_model, path, **PLAN) as (first, _p, _h):
        pending = preload(meta_model, path, **{**PLAN, "max_vram": "1GB", "max_ram": "1GB"})
        first(X())
        time.sleep(0.4)                                            # the preloaded model waits for its turn
        with pending as (second, _q, _g):
            second(X())
    doc = json.loads(trace_file.read_text(encoding="utf-8"))
    check_trace(doc)
    assert [m["started_by"] for m in doc["trashPanda"]["models"]] == ["stage", "preload"]
    second_id = doc["trashPanda"]["models"][1]["id"]
    [slice_] = [e for e in doc["traceEvents"] if e.get("cat") == "preload" and e["args"]["model"] == second_id]
    load = next(p for p in doc["trashPanda"]["phases"] if p["model"] == second_id and p["name"] == "load")
    assert load["start_us"] >= slice_["ts"] + slice_["dur"]                  # after the preload thread was done
    assert load["start_us"] - slice_["ts"] >= 350_000                        # and after the wait
    assert load["end_us"] - load["start_us"] < 300_000                       # the entry itself is short


def test_blocks_that_are_not_wrapped_and_load_while_the_model_runs_are_recorded_in_their_first_call_too(tiny_checkpoint, trace_file):
    """Blocks that are already in compute dtype on the device have no wrapper. With overlap their weights arrive after the model starts, and
    the first call of each block goes through a gate that puts the hooks in. A model that runs one pass (a decoder) calls each block once:
    that call has to be recorded there, or the model has no calls at all."""
    _, path = tiny_checkpoint
    with stage(meta_model, path, compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0, resident_blocks=4, overlap=True) as (model, _p, handle):
        handle.final_pass()
        model(torch.randn(3, 16))
    doc = json.loads(trace_file.read_text(encoding="utf-8"))
    check_trace(doc)
    calls = events(doc, "call")
    assert sorted(c["args"]["block"] for c in calls) == [0, 1, 2, 3] and {c["args"]["pass"] for c in calls} == {1}
    assert sorted(f["args"]["block"] for f in events(doc, "free")) == [0, 1, 2, 3]       # the final pass freed each block after it ran
    assert [p["name"] for p in doc["trashPanda"]["phases"] if p["name"].startswith("pass")] == ["pass 1"]
