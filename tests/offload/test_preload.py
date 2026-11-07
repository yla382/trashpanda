import importlib
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import PendingStage, Plan, parse_bytes, preload, stage, stream_load  # noqa: E402
from trash_panda.offload.checkpoint import Checkpoint  # noqa: E402
from trash_panda.offload.hardware import Hardware  # noqa: E402

stage_module = importlib.import_module("trash_panda.offload.stage")

HARDWARE = Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=10 * 10 ** 9)
PLAN = dict(compute_dtype="fp32", hardware=HARDWARE, min_free_ram=0)
PRE = dict(PLAN, max_vram="1GB", max_ram="1GB", resident_blocks=2)        # blocks 0 and 1 on the GPU, 2 and 3 in RAM


def _cuda(available):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available), nn=torch.nn)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Run on the CPU whatever the machine has, with no CUDA seen by stage(), and nothing remembered at the start."""
    real = stage_module.stream_load
    monkeypatch.setattr(stage_module, "stream_load", lambda model, path, plan, **kw: real(model, path, plan=plan, device="cpu", **kw))
    monkeypatch.setattr(stage_module, "torch", _cuda(False))
    monkeypatch.setattr(stage_module, "_activations", {})


@pytest.fixture
def reads(monkeypatch):
    """The names of the threads that read tensors with Checkpoint.read_tensor(), in order. A test can hold the reads back
    by clearing `gate`."""
    seen, gate = [], threading.Event()
    gate.set()
    real = Checkpoint.read_tensor

    def read_tensor(self, key):
        assert gate.wait(10), "the test never let the reads go on"
        seen.append(threading.current_thread().name)
        return real(self, key)

    monkeypatch.setattr(Checkpoint, "read_tensor", read_tensor)
    return SimpleNamespace(threads=seen, gate=gate)


def test_a_preloaded_model_gives_the_same_output_as_a_staged_one(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    pending = preload(meta_model, path, **PRE)
    assert isinstance(pending, PendingStage)
    with pending as (model, p, handle):
        assert p is pending.plan and handle is pending.handle
        assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert all(t.is_meta for t in model.parameters())


def test_the_ram_tier_is_read_on_other_threads_before_the_model_is_asked_for(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    assert pending._done.wait(10)
    assert reads.threads and threading.current_thread().name not in reads.threads     # none of them on this thread
    model = pending.model
    assert not model.blocks[2].lin1.weight.is_meta and not model.blocks[3].lin2.weight.is_meta     # the RAM tier is in
    assert model.blocks[0].lin1.weight.is_meta and model.blocks[1].lin2.weight.is_meta             # the GPU blocks are not
    pending.cancel()


def test_preload_does_not_wait_for_the_reads(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    reads.gate.clear()
    pending = preload(meta_model, path, **PRE)                      # would hang here if the reads ran on this thread
    assert not pending.ready()
    reads.gate.set()
    with pending as (model, _p, _handle):
        assert pending.ready()
        model(torch.randn(1, 16))


def test_the_gpu_blocks_are_loaded_when_the_model_is_asked_for(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    pending._done.wait(10)
    assert pending.model.blocks[0].lin1.weight.is_meta
    with pending as (model, _p, _handle):
        assert not any(t.is_meta for t in model.parameters())
        assert model._trash_panda_state.deferred is None


def test_with_overlap_the_gpu_blocks_arrive_while_the_model_runs(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    with preload(meta_model, path, overlap=True, **PRE) as (model, p, _handle):
        assert model._trash_panda_state.deferred.prefixes() == list(p.resident)
        assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert all(t.is_meta for t in model.parameters())


def test_an_error_in_the_reads_is_raised_where_the_model_is_asked_for(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint

    def broken(self, key):
        raise OSError("the drive went away")

    monkeypatch.setattr(Checkpoint, "read_tensor", broken)
    pending = preload(meta_model, path, **PRE)                      # nothing is raised here
    with pytest.raises(OSError, match="the drive went away"):
        with pending:
            pytest.fail("the block must not run")
    assert all(t.is_meta for t in pending.model.parameters())


def test_max_vram_or_the_running_plan_has_to_be_given(tiny_checkpoint):
    _, path = tiny_checkpoint
    with pytest.raises(ValueError, match="max_vram or running"):
        preload(meta_model, path, max_ram="1GB", **PLAN)
    with pytest.raises(ValueError, match="max_vram or running"):
        preload(meta_model, path, max_vram=None, max_ram="1GB", **PLAN)


def test_the_plan_is_made_with_the_memory_given(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    assert pending.plan.ram_budget == parse_bytes("1GB")
    assert pending.plan.vram_budget >= 0
    pending.cancel()


def test_the_model_is_planned_for_the_max_ram_given_even_if_more_than_is_free_now(tiny_checkpoint, recwarn):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **{**PRE, "max_ram": "50GB"})
    assert pending.plan.ram_budget == parse_bytes("50GB")
    assert not [w for w in recwarn if "available" in str(w.message)]         # it is not cut down, and there is no warning
    pending.cancel()


def test_max_ram_none_means_the_ram_that_is_free_now(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **{**PRE, "max_ram": None})
    assert pending.plan.ram_budget == HARDWARE.ram_available
    pending.cancel()


def test_cancel_waits_for_the_reads_and_frees_everything(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    reads.gate.clear()
    pending = preload(meta_model, path, **PRE)
    threading.Timer(0.1, reads.gate.set).start()
    pending.cancel()                                                # returns only after the reads are done
    assert pending.ready()
    assert all(t.is_meta for t in pending.model.parameters())


def test_a_pending_model_can_be_used_only_once(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    with pending:
        pass
    with pytest.raises(RuntimeError, match="already"):
        with pending:
            pass
    pending.cancel()                                                # does nothing, and does not raise


def test_a_cancelled_model_cannot_be_entered(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **PRE)
    pending.cancel()
    with pytest.raises(RuntimeError, match="already"):
        with pending:
            pass


def test_the_next_model_is_read_while_the_running_one_is_open_and_entered_after_it_is_released(tiny_checkpoint):
    ref, path = tiny_checkpoint
    x = torch.randn(3, 16)
    with stage(meta_model, path, **PLAN) as (first, _p, _handle):
        pending = preload(meta_model, path, **PRE)
        assert pending._done.wait(10)                               # read while the first model is still loaded
        assert not any(t.is_meta for t in first.parameters())
        assert torch.allclose(first(x), ref(x), atol=1e-6)
    assert all(t.is_meta for t in first.parameters())               # released
    with pending as (second, _p, _handle):
        assert torch.allclose(second(x), ref(x), atol=1e-6)


def test_the_peak_is_remembered_under_the_key_and_used_by_the_next_plan(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setattr(stage_module, "torch", _cuda(True))
    monkeypatch.setattr(stage_module, "_start_peak", lambda device=None: 7)
    monkeypatch.setattr(stage_module, "_peak_since", lambda before, plan=None, device=None: 123)
    seen = []
    real = stage_module.plan
    monkeypatch.setattr(stage_module, "plan", lambda model, path, **kw: (seen.append(kw.get("activation_vram")), real(model, path, **kw))[1])
    with preload(meta_model, path, key="b", **PRE):
        pass
    with preload(meta_model, path, key="b", **PRE):
        pass
    assert seen == [None, 123]


def test_nothing_is_remembered_when_the_block_raises(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setattr(stage_module, "torch", _cuda(True))
    monkeypatch.setattr(stage_module, "_start_peak", lambda device=None: 7)
    monkeypatch.setattr(stage_module, "_peak_since", lambda before, plan=None, device=None: 123)
    with pytest.raises(RuntimeError, match="boom"):
        with preload(meta_model, path, **PRE):
            raise RuntimeError("boom")
    assert stage_module._activations == {}


def test_stream_load_can_be_told_to_read_into_memory_for_one_call(tiny_checkpoint, monkeypatch, reads):
    _, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_RAM_IN_MEMORY", raising=False)
    plan = Plan.manual(compute_dtype="fp32", streamed=["blocks.2.", "blocks.3."], resident=["blocks.0.", "blocks.1."])
    stream_load(meta_model(), path, plan=plan, device="cpu")
    assert reads.threads == []                                      # mapped: nothing was read
    stream_load(meta_model(), path, plan=plan, device="cpu", ram_in_memory=True)
    assert len(reads.threads) > 0
    reads.threads.clear()
    monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1")
    stream_load(meta_model(), path, plan=plan, device="cpu", ram_in_memory=False)
    assert reads.threads == []                                      # the argument wins over the environment


# -- reading only the part of the RAM tier that fits in the RAM free now
# One block of the tiny model is 4224 bytes in float32. resident_blocks=1 leaves blocks 1, 2 and 3 in the RAM tier.

BLOCK = 4224
BLOCK_TENSORS = 4                                    # norm weight and bias, lin1 and lin2
SMALL = dict(PRE, resident_blocks=1)


def _free(nbytes):
    return Hardware(vram_total=0, vram_free=0, ram_total=16 * 10 ** 9, ram_available=nbytes)


def test_the_blocks_that_fit_in_the_free_ram_are_read_now_and_the_rest_stay_mapped(tiny_checkpoint, reads, monkeypatch):
    ref, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_RAM_IN_MEMORY", raising=False)
    x = torch.randn(3, 16)
    pending = preload(meta_model, path, **{**SMALL, "hardware": _free(2 * BLOCK + 100)})
    assert pending.plan.prefixes("ram") == ("blocks.1.", "blocks.2.", "blocks.3.")        # planned for the whole budget
    assert pending._memory == ("blocks.1.", "blocks.2.")
    assert pending._done.wait(10)
    assert len(reads.threads) == 2 * BLOCK_TENSORS                                         # only those two were read
    with pending as (model, _p, _handle):
        assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_everything_is_read_now_when_the_whole_ram_tier_fits(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **SMALL)
    assert pending._memory == ("blocks.1.", "blocks.2.", "blocks.3.")
    assert pending._done.wait(10)
    assert len(reads.threads) == 3 * BLOCK_TENSORS
    pending.cancel()


def test_the_blocks_read_now_are_the_ones_that_run_first(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, order=["blocks.3", "blocks.2", "blocks.1"],
                      **{**SMALL, "hardware": _free(2 * BLOCK + 100)})
    assert pending._memory == ("blocks.3.", "blocks.2.")
    pending.cancel()


def test_the_blocks_read_now_stop_at_the_first_one_that_does_not_fit(tiny_checkpoint, reads):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **{**SMALL, "hardware": _free(BLOCK - 1)})
    assert pending._memory == ()
    assert pending._done.wait(10)
    assert reads.threads == []                                      # all three stay mapped
    pending.cancel()


def test_min_free_ram_is_not_counted_as_free(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **{**SMALL, "hardware": _free(9000), "min_free_ram": 5000})
    assert pending._memory == ()                                    # 9000 - 5000 is less than one block
    pending.cancel()
    pending = preload(meta_model, path, **{**SMALL, "hardware": _free(9000), "min_free_ram": 400})
    assert pending._memory == ("blocks.1.", "blocks.2.")
    pending.cancel()


def test_everything_is_read_now_when_the_free_ram_is_not_known(tiny_checkpoint):
    _, path = tiny_checkpoint
    pending = preload(meta_model, path, **{**SMALL, "hardware": _free(0)})
    assert pending._memory == ("blocks.1.", "blocks.2.", "blocks.3.")
    pending.cancel()


def test_stream_load_reads_only_the_named_blocks_into_memory(tiny_checkpoint, monkeypatch, reads):
    ref, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_RAM_IN_MEMORY", raising=False)
    plan = Plan.manual(compute_dtype="fp32", streamed=["blocks.2.", "blocks.3."], resident=["blocks.0.", "blocks.1."])
    model = meta_model()
    stream_load(model, path, plan=plan, device="cpu", ram_in_memory_prefixes=("blocks.2.",))
    assert len(reads.threads) == BLOCK_TENSORS                      # block 3 is mapped, nothing is read for it
    for name in ("blocks.2.lin1.weight", "blocks.3.lin1.weight"):
        assert torch.equal(model.get_parameter(name), ref.get_parameter(name))
    reads.threads.clear()
    stream_load(meta_model(), path, plan=plan, device="cpu", ram_in_memory=True, ram_in_memory_prefixes=())
    assert reads.threads == []                                      # the prefixes win over the flag


def test_ram_overhead_is_the_reserve_and_the_pinned_buffers(tiny_checkpoint):
    from trash_panda.offload import config
    from trash_panda.offload.planner import _pinned_size, ram_overhead
    plan = Plan.manual(compute_dtype="fp32", streamed=["blocks.1."], resident=["blocks.0."])
    plan.blocks[1].stored_bytes = 1000
    gpu = Hardware(vram_total=10 ** 9, vram_free=10 ** 9, ram_total=16 * 10 ** 9, ram_available=10 ** 10)
    pinned = config.current().pinned_buffers * _pinned_size(1000)
    assert ram_overhead(plan, gpu, 500) == 500 + pinned
    assert ram_overhead(plan, HARDWARE, 500) == 500                 # no GPU, no pinned buffers
