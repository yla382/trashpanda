"""The memory in use that the page works out from the events (static/usage.js), run with Node on the test traces."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from . import builders as b
from .helpers import DATA

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="Node is needed to run the page's code")
SCRIPT = Path(__file__).parent / "js" / "usage_check.mjs"
THREE = DATA / "three_models.json"

BLOCK = 25165824                         # the size of a block in three_models.json
POW2 = 33554432                          # the buffer for a block of that size: the next power of two


def usage(path, *times):
    out = subprocess.run([node, str(SCRIPT), str(path), *map(str, times)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def row(result, t):
    return next(r for r in result["rows"] if r["t"] == t)


def kinds(models, model):
    return models.get(model, {"total": 0, "kinds": {}})["kinds"]


@pytest.fixture(scope="module")
def three():
    return usage(THREE, 0, 100, 250, 1000, 1500, 3200, 3600, 3900)


def test_the_models_use_nothing_before_they_start_and_the_total_is_the_sum_of_the_models(three):
    first = row(three, 0)
    assert first["vram"]["total"] == sum(v["total"] for v in first["vram"]["models"].values())
    for r in three["rows"]:
        for resource in ("vram", "ram", "pinned"):
            assert r[resource]["total"] == sum(v["total"] for v in r[resource]["models"].values())
    assert "vae" not in row(three, 1000)["vram"]["models"] or row(three, 1000)["vram"]["models"]["vae"]["total"] == 0


def test_the_weights_outside_the_blocks_count_while_a_model_lives_and_the_kept_blocks_once_they_are_loaded(three):
    at100 = kinds(row(three, 100)["vram"]["models"], "text")
    assert at100["outside"] == BLOCK // 2
    assert at100["kept"] == 3 * BLOCK                   # the loads finish at 30, 60 and 90 ms; the fourth ends at 120
    at250 = kinds(row(three, 250)["vram"]["models"], "text")
    assert at250["kept"] == 4 * BLOCK


def test_a_block_that_runs_is_expanded_while_it_runs_and_only_one_block_of_a_model_runs_at_a_time(three):
    for t in (250, 1500, 3200):
        for model, entry in row(three, t)["vram"]["models"].items():
            assert entry["kinds"].get("running", 0) in (0, 2 * BLOCK), (t, model)    # an expanded block is twice the stored size


def test_the_staged_blocks_of_the_unet_are_at_most_its_look_ahead(three):
    for t in (1000, 1500):
        assert kinds(row(three, t)["vram"]["models"], "unet").get("staged", 0) <= 2 * BLOCK


def test_the_final_pass_frees_the_kept_blocks_of_the_unet(three):
    assert kinds(row(three, 1000)["vram"]["models"], "unet").get("kept") == 4 * BLOCK
    after = kinds(row(three, 3600)["vram"]["models"], "unet")
    assert after.get("kept", 0) == 0


def test_the_ram_tier_of_the_unet_counts_once_it_was_read_and_until_it_is_freed(three):
    assert kinds(row(three, 1000)["ram"]["models"], "unet") == {"tier": 4 * BLOCK}     # blocks 4 to 7 were read into memory
    assert row(three, 3900)["ram"]["models"].get("unet", {"total": 0})["total"] == 0


def test_the_pinned_buffers_are_the_next_power_of_two_for_each_buffer_of_a_stage_pool_and_for_the_early_loader(three):
    assert kinds(row(three, 1000)["pinned"]["models"], "unet") == {"pool:pool-unet": 2 * POW2}
    early = kinds(row(three, 3200)["pinned"]["models"], "vae")
    assert early == {"pool:early-vae": 2 * 8388608}      # blocks of 6.3 MB, so 8 MB each, two buffers


def test_the_peak_covers_every_sampled_time(three):
    assert three["peaks"]["vram"] >= max(r["vram"]["total"] for r in three["rows"])
    assert three["peaks"]["ram"] >= max(r["ram"]["total"] for r in three["rows"])
    assert max(three["series"]) <= three["peaks"]["vram"]


def test_a_demoted_block_is_never_counted_on_the_gpu_and_a_promoted_one_is_counted_from_the_replan(tmp_path):
    ev = [b.inst("replan", "replan", b.MAIN, 20, model="m", demoted=[0], promoted=[1])]       # block 0 was planned for the GPU, block 1 for RAM
    path = tmp_path / "replan.json"
    path.write_text(json.dumps(b.trace(events=ev, phases=[{"model": "m", "name": "load", "start_us": 0, "end_us": 50000}])), encoding="utf-8")
    r = usage(path, 10, 30)
    assert kinds(row(r, 10)["vram"]["models"], "m").get("kept", 0) == 0           # nothing is loaded yet, and block 0 never will be
    assert kinds(row(r, 30)["vram"]["models"], "m").get("kept") == 100             # block 1 is on the GPU since the replan; block 0 is not


def test_counters_in_a_trace_are_used_instead_of_the_working_out(tmp_path):
    counters = [{"name": "vram", "ph": "C", "pid": 1, "ts": 5000, "args": {"m": 700}},
                {"name": "vram", "ph": "C", "pid": 1, "ts": 40000, "args": {"m": 900}}]
    path = tmp_path / "counters.json"
    path.write_text(json.dumps(b.trace(events=[b.SAMPLES["run"]()] + counters)), encoding="utf-8")
    r = usage(path, 1, 10, 50)
    assert r["measured"]["vram"] is True
    assert row(r, 1)["vram"]["total"] == 0               # before the first counter
    assert row(r, 10)["vram"]["total"] == 700 and row(r, 50)["vram"]["total"] == 900
    assert kinds(row(r, 10)["vram"]["models"], "m") == {"measured": 700}


def test_a_trace_of_format_1_still_gets_the_memory_of_its_kept_blocks():
    r = usage(DATA / "dummy_trace_steps.json", 1000)
    assert r["has"]["vram"] is True
    assert kinds(row(r, 1000)["vram"]["models"], "model").get("kept", 0) > 0
