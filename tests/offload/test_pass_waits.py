"""The prefetcher measures, with no profile, how long the main thread waited for staged blocks in each pass."""
import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402


def _plan():
    """bf16 compute from an fp32 file: the blocks are expanded when they run, so they are wrapped even on the CPU."""
    return Plan.manual(compute_dtype="bf16", resident=["blocks.0."], streamed=["blocks.1.", "blocks.2.", "blocks.3."])


def _run(path, device, passes):
    model = meta_model()
    stream_load(model, path, plan=_plan(), device=device)
    handle = install_offload(model, order=["blocks.0", "blocks.1", "blocks.2", "blocks.3"], profile=False)
    x = torch.randn(3, 16, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(passes):
            model(x)
    prefetcher = handle._prefetcher
    handle.remove()
    return prefetcher


def test_a_finished_pass_is_logged_when_the_next_one_starts(tiny_checkpoint):
    _, path = tiny_checkpoint
    prefetcher = _run(path, "cpu", passes=3)
    assert len(prefetcher.pass_waits) == 2                          # the third pass is still in progress
    for took, waited in prefetcher.pass_waits:
        assert took > 0 and waited == 0                             # nothing is staged on the CPU, so nothing is waited for


def test_one_pass_has_nothing_to_log_yet(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert len(_run(path, "cpu", passes=1).pass_waits) == 0


def test_the_profile_stays_off(tiny_checkpoint):
    _, path = tiny_checkpoint
    prefetcher = _run(path, "cpu", passes=2)
    assert prefetcher.profile is False and not prefetcher.stats     # no profile figures were gathered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_wait_for_staged_blocks_is_measured_without_the_profile(tiny_checkpoint):
    _, path = tiny_checkpoint
    prefetcher = _run(path, "cuda", passes=3)
    assert prefetcher.profile is False and not prefetcher.stats
    assert len(prefetcher.pass_waits) == 2
    assert all(took > 0 and waited >= 0 for took, waited in prefetcher.pass_waits)
    assert prefetcher.waited >= sum(waited for _took, waited in prefetcher.pass_waits)
    assert prefetcher.waited > 0                                    # blocks 1 to 3 are staged, and taking them costs some time
