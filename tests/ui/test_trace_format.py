"""The checker of format 2 (docs/design/trace-format.md): what it accepts and what it rejects."""

import json

import pytest

from trash_panda.ui.load import CATEGORIES, NO_MODEL, TraceError, check_trace, load_trace

from . import builders as b
from .helpers import SIMPLE


def rejected(doc, match):
    with pytest.raises(TraceError, match=match):
        check_trace(doc)


def with_event(event, **more):
    return b.trace(events=[event], **more)


# -- what is accepted

def test_a_trace_of_one_model_is_accepted():
    check_trace(b.trace())


def test_every_category_is_accepted_and_every_category_has_a_sample():
    assert set(b.SAMPLES) == set(CATEGORIES)
    check_trace(b.trace(events=[make() for make in b.SAMPLES.values()]))


def test_metadata_is_left_alone():
    meta = {"name": "thread_name", "ph": "M", "pid": 1, "tid": 99, "args": {"name": "anything"}}
    check_trace(b.trace(events=[meta, b.SAMPLES["run"]()]))


def test_two_models_with_their_own_pools_are_accepted():
    second = b.model("v", tiers=("gpu", "ram"), pool="q", started_by="preload")
    mine = b.tracks() + [b.track(20, "worker of v", role="worker", model="v")]
    pools = [b.pool(), b.pool("q", models=("v",)), b.pool("early-v", kind="early_loader", models=("v",))]
    phases = [{"model": "m", "name": "pass 1", "start_us": 0, "end_us": 60000},
              {"model": "v", "name": "load", "start_us": 30000, "end_us": 90000}]       # the phases of two models may overlap
    event = b.sl("stage v0", "stage", 20, 1, 3, model="v", block=1, bytes=100, pool="q", source="memory", **{"from": "ram", "to": "pinned:1"})
    check_trace(b.trace(events=[event], models=[b.model(), second], track_list=mine, pools=pools, phases=phases))


def test_counters_with_the_bytes_of_each_model_are_accepted():
    counter = {"name": "vram", "ph": "C", "pid": 1, "ts": 5000, "args": {"m": 1234}}
    check_trace(b.trace(events=[counter]))


def test_the_trace_files_of_format_1_are_still_accepted():
    check_trace(json.loads(SIMPLE.read_text(encoding="utf-8")))


# -- the file as a whole

def test_an_unknown_version_is_rejected():
    doc = b.trace()
    doc["trashPanda"]["version"] = 3
    rejected(doc, "not known")


@pytest.mark.parametrize("key", ["duration_us", "models", "tracks", "pools"])
def test_the_plan_must_have_each_of_its_sections(key):
    doc = b.trace()
    del doc["trashPanda"][key]
    rejected(doc, key)


def test_the_duration_must_be_positive():
    rejected(b.trace(duration_ms=0), "duration_us")


def test_a_trace_without_a_model_is_rejected():
    rejected(b.trace(models=[], events=[]), "models")


# -- models, tracks, pools, phases

def test_two_models_cannot_have_the_same_id():
    rejected(b.trace(models=[b.model(), b.model()]), "appears twice")


def test_started_by_must_be_known():
    rejected(b.trace(models=[{**b.model(), "started_by": "magic"}]), "started_by")


@pytest.mark.parametrize("key", ["id", "name", "started_by", "blocks", "options", "plan"])
def test_a_model_must_have_each_of_its_fields(key):
    broken = b.model()
    del broken[key]
    rejected(b.trace(models=[broken]), key)


def test_a_block_has_a_known_tier_and_a_number_that_is_unique_in_its_model():
    broken = b.model()
    broken["blocks"][0]["tier"] = "cloud"
    rejected(b.trace(models=[broken]), "tier")
    twice = b.model()
    twice["blocks"][1]["id"] = 0
    rejected(b.trace(models=[twice]), "appears twice")


def test_two_models_may_use_the_same_block_numbers():
    check_trace(b.trace(models=[b.model(), b.model("v", pool="p")]))


def test_the_pool_of_a_model_must_exist():
    rejected(b.trace(models=[b.model(pool="nope")]), "nope")


def test_a_model_without_a_pool_is_accepted():
    check_trace(b.trace(models=[b.model(pool=None)], pools=[]))


def test_a_track_must_be_of_a_known_kind_and_role_and_model():
    items = b.tracks()
    items[0]["kind"] = "fpga"
    rejected(b.trace(track_list=items), "kind")
    items = b.tracks()
    items[0]["role"] = "boss"
    rejected(b.trace(track_list=items), "role")
    items = b.tracks()
    items[1]["model"] = "ghost"
    rejected(b.trace(track_list=items), "ghost")


def test_two_tracks_cannot_have_the_same_id():
    items = b.tracks()
    items[1]["id"] = items[0]["id"]
    rejected(b.trace(track_list=items), "appears twice")


def test_a_pool_has_a_known_kind_and_at_least_one_buffer():
    rejected(b.trace(pools=[{**b.pool(), "kind": "magic"}]), "kind")
    rejected(b.trace(pools=[{**b.pool(), "buffers": 0}]), "buffers")
    rejected(b.trace(pools=[b.pool(), b.pool()]), "appears twice")


def test_a_phase_is_of_a_model_and_runs_forwards_inside_the_recording():
    rejected(b.trace(phases=[{"model": "ghost", "name": "x", "start_us": 0, "end_us": 10}]), "ghost")
    rejected(b.trace(phases=[{"model": "m", "name": "x", "start_us": 10, "end_us": 10}]), "start_us")
    rejected(b.trace(phases=[{"model": "m", "name": "x", "start_us": 0, "end_us": 10 ** 9}]), "after the recording")


def test_the_phases_of_one_model_cannot_overlap():
    phases = [{"model": "m", "name": "load", "start_us": 0, "end_us": 50000},
              {"model": "m", "name": "pass 1", "start_us": 40000, "end_us": 90000}]
    rejected(b.trace(phases=phases), "overlap")


# -- events

@pytest.mark.parametrize("cat", sorted(CATEGORIES))
def test_each_category_is_accepted_alone(cat):
    check_trace(with_event(b.SAMPLES[cat]()))


def required(cat):
    kind, needs = CATEGORIES[cat]
    return needs if cat in NO_MODEL else ("model",) + needs


@pytest.mark.parametrize("cat,key", [(cat, key) for cat in sorted(CATEGORIES) for key in required(cat)])
def test_each_required_argument_is_needed(cat, key):
    event = b.SAMPLES[cat]()
    del event["args"][key]
    rejected(with_event(event), key)


@pytest.mark.parametrize("cat", sorted(CATEGORIES))
def test_a_slice_is_not_an_instant_and_the_other_way_round(cat):
    event = b.SAMPLES[cat]()
    flipped = "i" if event["ph"] == "X" else "X"
    event["ph"] = flipped
    event.setdefault("dur", 1000)
    rejected(with_event(event), "must be")


def test_an_unknown_category_is_rejected():
    event = b.SAMPLES["run"]()
    event["cat"] = "dance"
    rejected(with_event(event), "dance")


def test_an_event_on_an_unknown_track_is_rejected():
    event = b.SAMPLES["run"]()
    event["tid"] = 99
    rejected(with_event(event), "99")


def test_an_event_of_an_unknown_model_or_block_or_pool_is_rejected():
    event = b.SAMPLES["run"]()
    event["args"]["model"] = "ghost"
    rejected(with_event(event), "ghost")
    event = b.SAMPLES["run"]()
    event["args"]["block"] = 77
    rejected(with_event(event), "block 77|77")
    event = b.SAMPLES["stage"]()
    event["args"]["pool"] = "nope"
    rejected(with_event(event), "nope")


def test_a_block_number_is_checked_against_the_model_of_the_event():
    other = b.model("v", tiers=("gpu",), pool="p")                       # model "v" has no block 2
    event = b.SAMPLES["run"]()
    event["args"].update(model="v", block=2)
    rejected(with_event(event, models=[b.model(), other]), "does not have")


def test_a_pinned_buffer_must_be_one_of_its_pool():
    event = b.SAMPLES["stage"]()
    event["args"]["to"] = "pinned:3"                                      # the pool has two buffers
    rejected(with_event(event), "beyond")
    event["args"]["to"] = "pinned:0"
    rejected(with_event(event), "beyond")


def test_a_pinned_buffer_needs_a_pool():
    event = b.SAMPLES["read"]()
    del event["args"]["pool"]
    rejected(with_event(event), "pool")


def test_a_place_must_be_a_known_one():
    event = b.SAMPLES["move"]()
    event["args"]["to"] = "the moon"
    rejected(with_event(event), "not a place")


def test_a_wait_has_a_known_reason():
    event = b.SAMPLES["wait"]()
    event["args"]["reason"] = "boredom"
    rejected(with_event(event), "reason")


@pytest.mark.parametrize("cat", ["load", "stage"])
def test_a_source_must_be_a_known_one(cat):
    event = b.SAMPLES[cat]()
    event["args"]["source"] = "teleport"
    rejected(with_event(event), "source")


def test_a_freed_block_has_a_known_tier():
    event = b.SAMPLES["free"]()
    event["args"]["tier"] = "cloud"
    rejected(with_event(event), "tier")


def test_an_event_must_be_inside_the_recording():
    early = b.SAMPLES["run"]()
    early["ts"] = -1
    rejected(with_event(early), "before")
    late = b.SAMPLES["run"]()
    late["ts"] = 99 * b.MS
    late["dur"] = 5 * b.MS
    rejected(with_event(late), "after the recording")


def test_a_slice_needs_a_duration():
    event = b.SAMPLES["run"]()
    del event["dur"]
    rejected(with_event(event), "dur")


def test_an_event_needs_args():
    event = b.SAMPLES["run"]()
    del event["args"]
    rejected(with_event(event), "args")


def test_events_must_be_objects():
    rejected(b.trace(events=[3]), "objects")


# -- counters

@pytest.mark.parametrize("counter,match", [
    ({"name": "disk", "ph": "C", "ts": 1, "args": {"m": 1}}, "must be one of"),
    ({"name": "vram", "ph": "C", "ts": 1, "args": {"m": "lots"}}, "numbers"),
    ({"name": "vram", "ph": "C", "ts": 1, "args": {"ghost": 1}}, "ghost"),
    ({"name": "vram", "ph": "C", "ts": 10 ** 9, "args": {"m": 1}}, "outside the recording"),
])
def test_a_counter_must_be_known_numeric_and_inside_the_recording(counter, match):
    rejected(b.trace(events=[counter]), match)


# -- from a file

def test_a_file_in_format_2_loads(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(b.trace()), encoding="utf-8")
    assert load_trace(path)["trashPanda"]["version"] == 2
