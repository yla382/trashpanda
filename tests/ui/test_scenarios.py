"""The scenario traces (tests/ui/data/scenario_*.json, made by tests/ui/simulation.py): each follows the rules of the code that
the simulation names, and these tests check that it does. What a rule is, and where it is in the code, is in the docstring of
the simulation."""

import re
from collections import defaultdict

import pytest

from trash_panda.ui.load import load_trace

from .helpers import DATA
from .simulation import SCENARIOS

NAMES = sorted(SCENARIOS)


def end(e):
    return e["ts"] + e.get("dur", 0)


class Trace:
    """A scenario, loaded, with the questions the tests ask."""

    def __init__(self, name):
        self.doc = load_trace(DATA / f"scenario_{name}.json")
        self.plan = self.doc["trashPanda"]
        self.events = self.doc["traceEvents"]
        self.tracks = {t["id"]: t for t in self.plan["tracks"]}

    def of(self, cat, model=None):
        return sorted((e for e in self.events if e.get("cat") == cat and (model is None or e["args"].get("model") == model)),
                      key=lambda e: e["ts"])

    def on_role(self, role):
        return [t["id"] for t in self.plan["tracks"] if t["role"] == role]

    def model(self, id):
        return next(m for m in self.plan["models"] if m["id"] == id)

    def tiers(self, id):
        return [b["tier"] for b in self.model(id)["blocks"]]

    def phases(self, model, prefix=""):
        return [p for p in self.plan["phases"] if p["model"] == model and p["name"].startswith(prefix)]

    def by_key(self, cat, model):
        """(block, pass) -> the event of that category, for events named "... b<N> (pass <K>)"."""
        out = {}
        for e in self.of(cat, model):
            out[(e["args"]["block"], int(re.search(r"pass (\d+)", e["name"]).group(1)))] = e
        return out


@pytest.fixture(scope="module", params=NAMES)
def trace(request):
    t = Trace(request.param)
    t.name = request.param
    return t


@pytest.fixture(scope="module")
def traces():
    return {name: Trace(name) for name in NAMES}


def call_end(trace, model, pass_no, block):
    c = next(e for e in trace.of("call", model) if e["args"]["pass"] == pass_no and e["args"]["block"] == block)
    return end(c)


def non_overlapping(slices):
    ordered = sorted(slices, key=lambda e: e["ts"])
    for before, after in zip(ordered, ordered[1:]):
        assert end(before) <= after["ts"], f"{before['name']} overlaps {after['name']}"


# -- every scenario

def test_every_scenario_is_a_valid_trace_and_has_every_phase_in_order(trace):
    for m in trace.plan["models"]:
        phases = trace.phases(m["id"])
        assert phases[0]["name"] == "load"
        for before, after in zip(phases, phases[1:]):
            assert before["end_us"] <= after["start_us"]


def test_each_thread_and_stream_does_one_thing_at_a_time(trace):
    by_track = defaultdict(list)
    for e in trace.events:
        if e["ph"] == "X":
            by_track[e["tid"]].append(e)
    for tid, slices in by_track.items():
        non_overlapping(slices)


def test_every_block_runs_on_the_gpu_after_it_was_launched_and_after_its_copy(trace):
    calls = {(e["args"]["model"], e["args"]["block"], e["args"]["pass"]): e for e in trace.of("call")}
    copies = {(e["args"]["model"], e["args"]["block"], int(re.search(r"pass (\d+)", e["name"]).group(1))): e for e in trace.of("copy")}
    runs = trace.of("run")
    assert len(runs) == len(calls)
    for r in runs:
        key = (r["args"]["model"], r["args"]["block"], r["args"]["pass"])
        assert r["ts"] >= calls[key]["ts"]
        if key in copies:
            assert r["ts"] >= end(copies[key]), "a block ran before its copy was done"


def test_the_main_thread_waits_for_the_gpu_so_that_a_block_starts_after_the_one_two_before_it_ended(trace):
    for m in trace.plan["models"]:
        limit = m["options"]["blocks_in_flight"]
        calls = trace.of("call", m["id"])
        runs = {(e["args"]["block"], e["args"]["pass"]): e for e in trace.of("run", m["id"])}
        for i, c in enumerate(calls):
            if i >= limit:
                back = calls[i - limit]
                assert c["ts"] >= end(runs[(back["args"]["block"], back["args"]["pass"])]) - 1, (m["id"], i)


def test_the_main_thread_waits_for_a_staged_block_only_until_the_worker_has_issued_its_copy(trace):
    jobs = defaultdict(list)
    for e in trace.of("stage") + trace.of("read"):
        if str(e["args"]["to"]).startswith("pinned"):
            jobs[(e["args"]["model"], e["args"]["block"])].append(end(e))
    for w in (e for e in trace.of("wait") if e["args"]["reason"] == "staged"):
        assert end(w) in jobs[(w["args"]["model"], w["args"]["block"])], w["name"]


def test_a_pinned_buffer_holds_one_block_from_when_the_job_takes_it_until_its_copy_is_done(trace):
    held = {}
    events = sorted((e for e in trace.of("buffer")), key=lambda e: (e["ts"], e["name"].startswith("acquire")))
    for e in events:
        key = (e["args"]["pool"], e["args"]["buffer"])
        if e["name"].startswith("acquire"):
            assert key not in held, f"{key} taken twice"
            held[key] = e["args"]["block"]
        else:
            assert held.pop(key) == e["args"]["block"]
    assert not held


def test_the_pinned_buffers_are_taken_in_turn(trace):
    by_pool = defaultdict(list)
    for e in trace.of("stage") + trace.of("read"):
        if str(e["args"]["to"]).startswith("pinned"):
            by_pool[e["args"]["pool"]].append(e)
    for pool, jobs in by_pool.items():
        size = next(p["buffers"] for p in trace.plan["pools"] if p["id"] == pool)
        used = [int(e["args"]["to"].split(":")[1]) for e in sorted(jobs, key=lambda e: e["ts"])]
        grown = any(e["args"]["pool"] == pool for e in trace.of("pool"))
        if not grown:
            assert used == [(i % size) + 1 for i in range(len(used))], (pool, used[:8])


def test_a_buffer_is_pinned_at_the_next_power_of_two_the_first_time_it_is_used(trace):
    for e in trace.of("alloc"):
        assert e["args"]["bytes"] & (e["args"]["bytes"] - 1) == 0


def test_the_disk_blocks_are_spread_among_the_others_and_not_put_last(trace):
    for m in trace.plan["models"]:
        tiers = [b["tier"] for b in m["blocks"]]
        rest = [t for t in tiers if t != "gpu"]
        if "disk" in rest and "ram" in rest:
            first_disk = rest.index("disk")
            assert "ram" in rest[first_disk:], f"{m['id']}: the disk blocks come last: {tiers}"


def test_a_disk_block_is_read_in_every_pass_and_a_ram_block_is_copied_in_every_pass(trace):
    for m in trace.plan["models"]:
        passes = {int(re.search(r"pass (\d+)", e["name"]).group(1)) for e in trace.of("call", m["id"])
                  for _ in [0]} if False else {e["args"]["pass"] for e in trace.of("call", m["id"])}
        for block, tier in enumerate(trace.tiers(m["id"])):
            cat = {"disk": "read", "ram": "stage"}.get(tier)
            if cat is None:
                continue
            seen = {int(re.search(r"pass (\d+)", e["name"]).group(1)) for e in trace.of(cat, m["id"]) if e["args"]["block"] == block}
            assert passes <= seen, (m["id"], block)


def test_a_kept_block_is_loaded_through_ordinary_memory_once_and_never_through_a_pinned_buffer(trace):
    for m in trace.plan["models"]:
        kept = [i for i, t in enumerate(trace.tiers(m["id"])) if t == "gpu"]
        loads = [e["args"]["block"] for e in trace.of("load", m["id"])]
        assert sorted(loads) == sorted(set(loads))
        assert set(loads) <= set(kept)
        for e in trace.of("load", m["id"]):
            assert e["args"]["via"] == "ram" and e["args"]["from"] == "disk" and e["args"]["to"] == "gpu"


def test_nothing_is_read_from_the_file_straight_to_the_gpu(trace):
    for e in trace.events:
        if e["ph"] == "X" and e.get("args", {}).get("from") == "disk":
            assert e["args"]["to"] != "gpu" or e["args"].get("via") == "ram"


# -- one model, plain stage()

def test_single_model_has_its_disk_blocks_spread_and_loads_the_kept_blocks_on_the_main_thread_through_a_map(traces):
    t = traces["single_model"]
    assert t.tiers("m") == ["gpu", "gpu", "gpu", "ram", "disk", "ram", "disk", "ram"]
    main = t.on_role("main")[0]
    loads = t.of("load")
    assert [e["args"]["block"] for e in loads] == [0, 1, 2]
    assert all(e["tid"] == main and e["args"]["source"] == "mmap" for e in loads)
    for before, after in zip(loads, loads[1:]):
        assert end(before) == after["ts"]                                       # one block after the other, no gap


def test_single_model_reads_the_ram_tier_into_memory_after_the_kept_blocks_and_starts_when_the_load_is_done(traces):
    t = traces["single_model"]
    reads = [e for e in t.of("read") if e["args"]["to"] == "ram"]
    assert [e["args"]["block"] for e in reads] == [3, 5, 7]
    assert reads[0]["ts"] >= end(t.of("load")[-1])
    assert all(t.tracks[e["tid"]]["role"] == "reader" for e in reads)
    load = t.phases("m", "load")[0]
    assert load["end_us"] >= end(reads[-1])
    assert t.of("call")[0]["ts"] >= load["end_us"]
    assert not [e for e in t.of("wait") if e["args"]["reason"] == "loader"]


# -- overlap

def test_with_overlap_the_loader_thread_loads_the_kept_blocks_while_the_model_runs_and_the_first_pass_waits_for_them(traces):
    t = traces["overlap"]
    loads = t.of("load")
    assert all(t.tracks[e["tid"]]["role"] == "loader" and e["args"]["source"] == "read" for e in loads)
    first_pass = t.phases("m", "pass 1")[0]
    assert all(e["ts"] >= first_pass["start_us"] for e in loads)                # they start when the model does
    waits = [e for e in t.of("wait") if e["args"]["reason"] == "loader"]
    assert waits
    by_block = {e["args"]["block"]: e for e in loads}
    for w in waits:
        assert end(w) == end(by_block[w["args"]["block"]])                      # until the block is loaded
        assert w["ts"] < first_pass["end_us"]
    second = t.phases("m", "pass 2")[0]
    assert not [w for w in waits if w["ts"] >= second["start_us"]]              # and only in the first pass


# -- sampling steps

def test_sampling_steps_have_six_passes_with_a_gap_between_steps_and_none_between_the_two_passes_of_a_step(traces):
    t = traces["sampling_steps"]
    passes = t.phases("m", "pass")
    assert len(passes) == 6
    gaps = [(b["start_us"] - a["end_us"]) / 1000 for a, b in zip(passes, passes[1:])]
    assert [g < 1 for g in gaps] == [True, False, True, False, True]            # guidance: back to back; the step code in between
    assert all(g >= 25 for g in gaps[1::2])


def test_the_look_ahead_reaches_into_the_next_pass_from_the_second_pass_on_and_is_wasted_at_the_very_end(traces):
    t = traces["sampling_steps"]
    jobs = t.by_key("stage", "m")
    passes = {i + 1: p for i, p in enumerate(t.phases("m", "pass"))}
    first_streamed = next(b for b, tier in enumerate(t.tiers("m")) if tier == "ram")
    assert jobs[(first_streamed, 2)]["ts"] >= passes[1]["end_us"] - 3000        # pass 1 is not linked to pass 2 yet
    for k in range(2, 6):
        assert jobs[(first_streamed, k + 1)]["ts"] < passes[k]["end_us"]       # from pass 2 on it is
    assert (first_streamed, 7) in jobs                                          # work for a pass that never comes
    assert not [c for c in t.of("call") if c["args"]["pass"] == 7]


# -- a mapped RAM tier

def test_a_mapped_ram_tier_is_not_read_at_load_and_the_first_copy_of_each_block_reads_the_file(traces):
    t = traces["mapped_ram"]
    assert len(t.of("mapped")) == 3 and not [e for e in t.of("read") if e["args"]["to"] == "ram"]
    jobs = t.by_key("stage", "m")
    assert all(e["args"]["source"] == "mapped" for e in jobs.values())
    for block in (3, 5, 7):
        first, second, third = (jobs[(block, k)]["dur"] for k in (1, 2, 3))
        assert first > second == third                                          # the pages are in the file cache after the first read


# -- several stage() calls, no Coordinator

def test_sequential_stages_follow_each_other_with_pools_of_their_own_and_no_shared_pool(traces):
    t = traces["sequential_stages"]
    assert [m["started_by"] for m in t.plan["models"]] == ["stage"] * 3
    kinds = {p["id"]: p["kind"] for p in t.plan["pools"]}
    assert set(kinds.values()) == {"stage"} and set(kinds) == {"pool-unet", "pool-vae"}
    span = {m["id"]: (min(p["start_us"] for p in t.phases(m["id"])), max(p["end_us"] for p in t.phases(m["id"]))) for m in t.plan["models"]}
    assert span["text"][1] <= span["unet"][0] and span["unet"][1] <= span["vae"][0]
    workers = {e["tid"] for e in t.of("stage") + t.of("read") if str(e["args"].get("to", "")).startswith("pinned")}
    assert len(workers) == 2                                                    # a worker thread for each model that streams


# -- an adaptive look-ahead

def test_the_adaptive_look_ahead_grows_after_the_warm_up_passes_and_the_pool_grows_with_it(traces):
    t = traces["adaptive_lookahead"]
    [change] = t.of("lookahead")
    assert (change["args"]["from"], change["args"]["to"]) == (2, 3)
    fifth = t.phases("m", "pass 5")[0]
    assert change["ts"] <= fifth["start_us"] and change["ts"] >= t.phases("m", "pass 4")[0]["end_us"]
    [grew] = t.of("pool")
    assert grew["args"]["buffers"] == 3 and grew["ts"] == change["ts"]
    pinned = [e for e in t.of("stage") + t.of("read") if str(e["args"]["to"]).startswith("pinned")]
    buffers_before = {e["args"]["to"] for e in pinned if e["ts"] < change["ts"]}
    buffers_after = {e["args"]["to"] for e in pinned if e["ts"] > change["ts"] + 60000}
    assert buffers_before == {"pinned:1", "pinned:2"} and "pinned:3" in buffers_after
    assert [e["args"]["buffer"] for e in t.of("alloc") if e["ts"] > change["ts"]] == [3]    # the third buffer is pinned when first used


# -- release_gpu and restore_gpu

def test_release_gpu_moves_the_kept_blocks_to_ram_and_restore_gpu_brings_them_back_between_passes(traces):
    t = traces["release_gpu"]
    moves = t.of("move")
    assert [(e["name"].split()[0], e["args"]["from"], e["args"]["to"]) for e in moves] == (
        [("release_gpu", "gpu", "ram")] * 3 + [("restore_gpu", "ram", "gpu")] * 3)
    assert sorted(e["args"]["block"] for e in moves[:3]) == [0, 1, 2]
    calls = t.of("call")
    inside = [c for c in calls if moves[0]["ts"] <= c["ts"] < end(moves[-1])]
    assert not inside                                                            # nothing runs while the weights are away
    assert max(c["args"]["pass"] for c in calls if c["ts"] < moves[0]["ts"]) == 2
    assert min(c["args"]["pass"] for c in calls if c["ts"] >= end(moves[-1])) == 3


# -- a Coordinator

def test_a_coordinator_shares_one_pinned_pool_among_the_models_and_each_next_model_has_an_early_loader_of_its_own(traces):
    t = traces["coordinator_pipeline"]
    assert [m["started_by"] for m in t.plan["models"]] == ["coordinator"] * 3
    assert {m["pool"] for m in t.plan["models"]} == {"pool-shared"}
    pools = {p["id"]: p for p in t.plan["pools"]}
    assert pools["pool-shared"]["kind"] == "shared" and pools["pool-shared"]["models"] == ["text", "unet", "vae"]
    assert pools["early-unet"]["kind"] == pools["early-vae"]["kind"] == "early_loader"
    assert {e["args"]["pool"] for e in t.of("stage") + t.of("read") if "pool" in e["args"]} == {"pool-shared"}
    chunked = [e for e in t.of("load") if e["args"]["source"] == "chunked"]
    assert {e["args"]["model"] for e in chunked} == {"unet", "vae"}
    assert all(t.tracks[e["tid"]]["role"] == "loader" and t.tracks[e["tid"]].get("model") == e["args"]["model"] for e in chunked)


def test_the_next_model_is_preloaded_while_the_one_before_it_runs(traces):
    t = traces["coordinator_pipeline"]
    spans = {m["id"]: (min(p["start_us"] for p in t.phases(m["id"])), max(p["end_us"] for p in t.phases(m["id"]))) for m in t.plan["models"]}
    preloads = {e["args"]["model"]: e for e in t.of("preload")}
    assert preloads["unet"]["ts"] < spans["text"][1] and preloads["vae"]["ts"] < spans["unet"][1]
    early_vae = [e for e in t.of("load") if e["args"]["model"] == "vae" and e["args"]["source"] == "chunked"]
    assert early_vae and min(e["ts"] for e in early_vae) < spans["unet"][1]     # the vae is loading while the unet's last pass runs
    assert max(e["ts"] for e in t.of("call", "unet")) > min(e["ts"] for e in early_vae)


def test_a_replan_demotes_a_kept_block_that_the_early_loader_had_not_loaded_and_it_is_never_loaded(traces):
    t = traces["coordinator_pipeline"]
    [replan] = t.of("replan")
    [block] = replan["args"]["demoted"]
    assert replan["args"]["model"] == "unet" and t.tiers("unet")[block] == "gpu"
    loads = {e["args"]["block"]: e for e in t.of("load", "unet")}
    assert block not in loads
    assert set(loads) == {b for b, tier in enumerate(t.tiers("unet")) if tier == "gpu"} - {block}
    early = [e for e in loads.values() if e["args"]["source"] == "chunked"]
    assert early and all(e["ts"] < replan["ts"] for e in early)                  # what was loaded early was loaded before the entry
    entry = [e for e in loads.values() if e["args"]["source"] == "mmap"]
    assert all(e["ts"] >= replan["ts"] for e in entry)                           # the rest, after the replan, on the main thread


def test_the_early_loader_is_stopped_when_the_model_is_entered_after_the_block_it_is_loading(traces):
    t = traces["coordinator_pipeline"]
    [replan] = t.of("replan")
    last_early = max(end(e) for e in t.of("load", "unet") if e["args"]["source"] == "chunked")
    assert replan["ts"] >= last_early                                             # entry waited for the block in progress


def test_the_final_pass_frees_each_block_after_it_ran_and_drops_the_ram_tier_on_the_drop_thread(traces):
    t = traces["coordinator_pipeline"]
    for model, last_pass in (("text", 1), ("unet", 4)):
        frees = {e["args"]["block"]: e for e in t.of("free", model)}
        assert sorted(frees) == list(range(len(t.tiers(model))))
        runs = {e["args"]["block"]: e for e in t.of("run", model) if e["args"]["pass"] == last_pass}
        for block, tier in enumerate(t.tiers(model)):
            f = frees[block]
            stored = t.model(model)["blocks"][block]["stored_bytes"]
            assert f["args"]["tier"] == tier and f["args"]["bytes"] == (0 if tier == "disk" else stored)
            run = runs[block]
            assert f["ts"] >= (end(run) if tier == "gpu" else call_end(t, model, last_pass, block))   # a GPU block is free once the GPU is done with it
    frees = {e["args"]["block"]: e for e in t.of("free", "unet")}
    for e in t.of("drop", "unet"):
        assert end(e) == frees[e["args"]["block"]]["ts"]                         # a RAM block is announced when its memory is gone
    assert not t.of("free", "vae")                                              # the vae did not use final_pass
