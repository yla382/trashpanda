"""release_gpu() / restore_gpu(): the resident blocks leave the GPU and come back while the hooks stay."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from .test_offload import _fp8_checkpoint, needs_fp8  # noqa: E402
from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402
from trash_panda.offload.hooks import OffloadHandle, _Prefetcher, _defer_block  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
BLOCKS = [f"blocks.{i}." for i in range(4)]


def _cpu_handle(path):
    """A handle of a model on the CPU, where no block stays on a GPU, so nothing is registered."""
    model = meta_model()
    plan = Plan.manual(compute_dtype="bf16", resident=BLOCKS[:1], streamed=BLOCKS[1:])
    stream_load(model, path, plan=plan, device="cpu")
    return install_offload(model, order=[b.rstrip(".") for b in BLOCKS], profile=False)


# -- bookkeeping, on the CPU -------------------------------------------------------------------------------------------

def test_nothing_is_registered_on_the_cpu(tiny_checkpoint):
    _, path = tiny_checkpoint
    handle = _cpu_handle(path)
    assert handle._resident == []
    assert handle.release_gpu() == 0 and handle.restore_gpu() == 0
    handle.remove()


def test_release_and_restore_report_the_bytes_and_do_nothing_the_second_time(tiny_checkpoint):
    _, path = tiny_checkpoint
    handle = _cpu_handle(path)
    moves = []
    handle._resident.append((lambda: moves.append("out") or 5, lambda: moves.append("back") or 7))
    handle._resident.append((lambda: moves.append("out") or 1, lambda: moves.append("back") or 2))
    assert handle.release_gpu() == 6 and handle.release_gpu() == 0
    assert handle.restore_gpu() == 9 and handle.restore_gpu() == 0
    assert moves == ["out", "out", "back", "back"]
    handle.remove()


def test_a_failed_release_can_be_tried_again(tiny_checkpoint):
    _, path = tiny_checkpoint
    handle = _cpu_handle(path)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return 4

    handle._resident.append((flaky, lambda: 4))
    with pytest.raises(RuntimeError, match="boom"):
        handle.release_gpu()
    assert handle.restore_gpu() == 0                    # it never counted as released
    assert handle.release_gpu() == 4
    handle.remove()


def test_release_leaves_the_prefetcher_at_rest(tiny_checkpoint):
    _, path = tiny_checkpoint
    handle = _cpu_handle(path)
    pre = handle._prefetcher
    staged = Future()
    entry = SimpleNamespace(staged=staged)
    pre.window = [entry]
    pre.began = 12.5                                    # a pass is "in progress"
    handle.release_gpu()
    assert pre.window == [] and entry.staged is None
    assert pre.began is None                            # the pause is not part of the next pass
    handle.remove()


def test_the_pause_does_not_show_up_as_a_pass(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    plan = Plan.manual(compute_dtype="bf16", resident=BLOCKS[:1], streamed=BLOCKS[1:])
    stream_load(model, path, plan=plan, device="cpu")
    handle = install_offload(model, order=[b.rstrip(".") for b in BLOCKS], profile=False)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        model(x)
        model(x)
        logged = len(handle._prefetcher.pass_waits)
        handle.release_gpu()
        handle.restore_gpu()
        model(x)
    assert len(handle._prefetcher.pass_waits) == logged     # the pass before the pause was never closed
    handle.remove()


# -- blocks whose weights are loaded while the model runs ---------------------------------------------------------------

def test_a_deferred_block_joins_the_registry_when_it_is_installed():
    pre = _Prefetcher(torch.device("cpu"))
    handle = OffloadHandle()
    handle._prefetcher = pre
    block = nn.Linear(4, 4)
    pre.hold(["b"])
    pre.mark_ready("b")
    inner = OffloadHandle()
    inner._resident.append((lambda: 3, lambda: 3))
    _defer_block(block, "b", pre, handle, lambda: inner)

    assert handle._deferred == 1 and handle._resident == []
    with pytest.raises(RuntimeError, match="not been installed"):
        handle.release_gpu()
    block(torch.randn(2, 4))                            # the first call installs the hooks
    assert handle._deferred == 0 and len(handle._resident) == 1
    assert handle.release_gpu() == 3 and handle.restore_gpu() == 3


# -- on the GPU ---------------------------------------------------------------------------------------------------------

def _gpu_model(path, compute_dtype):
    model = meta_model()
    plan = Plan.manual(compute_dtype=compute_dtype, resident=BLOCKS)
    stream_load(model, path, plan=plan, device="cuda")
    handle = install_offload(model, order=[b.rstrip(".") for b in BLOCKS], profile=False)
    return model, handle


def _devices(model):
    return {t.device.type for b in model.blocks for t in [*b.parameters(), *b.buffers()]}


@needs_gpu
@pytest.mark.parametrize("compute_dtype, wrapped", [("fp32", False), ("bf16", True)])
def test_resident_blocks_leave_the_gpu_and_come_back_unchanged(tiny_checkpoint, compute_dtype, wrapped):
    _, path = tiny_checkpoint
    model, handle = _gpu_model(path, compute_dtype)
    assert len(handle._resident) == 4                   # all four blocks stay on the GPU, wrapped or not
    x = torch.randn(3, 16, device="cuda", dtype=torch.float32 if compute_dtype == "fp32" else torch.bfloat16)
    with torch.no_grad():
        before = model(x).clone()
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated()

        moved = handle.release_gpu()
        assert moved > 0 and _devices(model) == {"cpu"}
        assert used - torch.cuda.memory_allocated() >= moved        # the memory was really given back
        if wrapped:                                     # a wrapped block runs from the CPU copy, as a RAM-tier block would
            assert torch.equal(model(x), before)

        assert handle.restore_gpu() == moved and _devices(model) == {"cuda"}
        assert torch.equal(model(x), before)
    handle.remove()


@needs_gpu
@needs_fp8
def test_fp8_blocks_with_scales_make_the_round_trip(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", resident=BLOCKS), device="cuda")
    handle = install_offload(model, order=[b.rstrip(".") for b in BLOCKS], profile=False)
    assert len(handle._resident) == 4
    x = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        before = model(x).clone()
        moved = handle.release_gpu()
        assert moved > 0 and _devices(model) == {"cpu"}          # the scales moved with the weights
        assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn
        assert torch.equal(model(x), before)
        assert handle.restore_gpu() == moved and _devices(model) == {"cuda"}
        assert torch.equal(model(x), before)
    handle.remove()
