"""Recording a run on a real GPU: the CUDA events of the GPU work are put on the same axis as the CPU's clock."""

import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import stage  # noqa: E402
from trash_panda.ui.load import check_trace  # noqa: E402


@pytest.fixture
def recorded(tiny_checkpoint, trace_file):
    _, checkpoint = tiny_checkpoint
    with stage(meta_model, checkpoint, compute_dtype="bf16", resident_blocks=1, headroom_vram="64GB", min_free_ram=0) as (model, _p, _h):
        x = torch.randn(3, 16, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
    doc = json.loads(trace_file.read_text(encoding="utf-8"))
    check_trace(doc)
    return doc


def of(doc, cat):
    return sorted((e for e in doc["traceEvents"] if e.get("cat") == cat), key=lambda e: e["ts"])


def test_every_call_has_a_run_on_the_gpu_that_starts_when_the_call_starts(recorded):
    calls, runs = of(recorded, "call"), of(recorded, "run")
    assert len(calls) == len(runs) == 12                                 # 4 blocks, 3 passes
    key = lambda e: (e["args"]["pass"], e["args"]["block"])             # noqa: E731
    by_run = {key(r): r for r in runs}
    for c in calls:
        r = by_run[key(c)]
        assert abs(r["ts"] - c["ts"]) < 5_000                           # the two clocks agree to within a few ms


def test_a_streamed_block_is_copied_to_the_gpu_before_it_runs(recorded):
    copies = of(recorded, "copy")
    first_run = {}
    for r in of(recorded, "run"):
        first_run.setdefault(r["args"]["block"], r)
    assert copies and all(c["ts"] < first_run[c["args"]["block"]]["ts"] + first_run[c["args"]["block"]]["dur"] for c in copies[:3])
    first = {}
    for c in copies:
        first.setdefault(c["args"]["block"], c)
    for block, c in first.items():
        assert c["ts"] + c["dur"] <= first_run[block]["ts"] + first_run[block]["dur"]        # a block's first copy is done by the end of its run
