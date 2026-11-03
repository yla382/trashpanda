import threading
import time
import warnings

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


def _model(path, dtype="fp32", defer=ALL, **tiers):
    """A tiny model whose resident blocks in `defer` are still on the meta device."""
    model = meta_model()
    plan = Plan.manual(compute_dtype=dtype, **(tiers or dict(resident=ALL)))
    stream_load(model, path, plan=plan, defer_prefixes=defer, device="cpu")
    return model


def _deferred(model):
    return model._trash_panda_state.deferred


def _meta(model, i):
    return model.blocks[i].lin1.weight.is_meta


def _x(dtype=torch.float32):
    torch.manual_seed(1)
    return torch.randn(3, 16).to(dtype)


def test_deferred_blocks_stay_on_the_meta_device_until_they_are_loaded(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = _model(path, defer=ALL[:2])
    assert [_meta(model, i) for i in range(4)] == [True, True, False, False]
    assert _deferred(model).prefixes() == ALL[:2]
    _deferred(model).load("blocks.0.")
    assert not _meta(model, 0)
    assert torch.equal(model.blocks[0].lin1.weight, ref.blocks[0].lin1.weight)


def test_nothing_is_deferred_by_default(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), device="cpu")
    assert model._trash_panda_state.deferred is None
    assert not any(p.is_meta for p in model.parameters())


def test_only_resident_blocks_can_be_deferred(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path, defer=ALL, resident=ALL[:2], streamed=ALL[2:])
    assert _deferred(model).prefixes() == ALL[:2]
    assert not _meta(model, 2) and not _meta(model, 3)


def test_the_result_is_the_same_when_the_blocks_load_while_the_model_runs(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = _model(path)
    handle = install_offload(model)
    assert torch.allclose(model(_x()), ref(_x()), atol=1e-6)
    assert torch.allclose(model(_x()), ref(_x()), atol=1e-6)        # and again, after the blocks are loaded
    handle.remove()


def test_blocks_that_are_expanded_while_they_run_give_the_same_result(tiny_checkpoint):
    _, path = tiny_checkpoint
    plain = meta_model()
    stream_load(plain, path, plan=Plan.manual(compute_dtype="bf16", resident=ALL), device="cpu")
    plain_handle = install_offload(plain)
    model = _model(path, dtype="bf16")
    handle = install_offload(model)
    assert all("forward" in b.__dict__ for b in plain.blocks)             # stored as fp32, run as bf16: wrapped
    x = _x(torch.bfloat16)
    assert torch.equal(model(x), plain(x))
    assert torch.equal(model(x), plain(x))
    handle.remove()
    plain_handle.remove()


def test_the_first_call_waits_for_a_block_that_is_not_loaded_yet(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = _model(path)
    deferred, real, gate = _deferred(model), _deferred(model).load, threading.Event()
    reached = []

    def slow(prefix):
        if prefix == "blocks.2.":
            gate.wait(10)
        real(prefix)
        reached.append(prefix)

    deferred.load = slow
    handle = install_offload(model)
    out = []
    caller = threading.Thread(target=lambda: out.append(model(_x())))
    caller.start()
    time.sleep(0.3)
    assert caller.is_alive()                                         # blocks 0 and 1 ran, block 2 is not loaded
    assert reached == ["blocks.0.", "blocks.1."]
    gate.set()
    caller.join(10)
    assert not caller.is_alive()
    assert torch.allclose(out[0], ref(_x()), atol=1e-6)
    handle.remove()


def test_blocks_are_loaded_in_the_order_given(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = _model(path)
    deferred, real, loaded = _deferred(model), _deferred(model).load, []
    deferred.load = lambda prefix: (real(prefix), loaded.append(prefix))
    handle = install_offload(model, order=["blocks.2", "blocks.1"])
    assert torch.allclose(model(_x()), ref(_x()), atol=1e-6)
    assert loaded == ["blocks.2.", "blocks.1.", "blocks.0.", "blocks.3."]
    handle.remove()


def test_a_load_that_fails_raises_in_the_call_that_needs_the_block(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path)
    deferred, real = _deferred(model), _deferred(model).load

    def broken(prefix):
        if prefix == "blocks.1.":
            raise OSError("the drive went away")
        real(prefix)

    deferred.load = broken
    with pytest.warns(UserWarning, match="loading the weights failed"):
        handle = install_offload(model)
        with pytest.raises(RuntimeError, match="blocks.1 were not loaded"):
            model(_x())
    handle.remove()


def test_removing_the_handle_stops_the_loader_after_the_block_it_is_on(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path)
    deferred, real, gate, inside = _deferred(model), _deferred(model).load, threading.Event(), threading.Event()

    def slow(prefix):
        if prefix == "blocks.1.":
            inside.set()
            gate.wait(10)
        real(prefix)

    deferred.load = slow
    handle = install_offload(model)
    assert inside.wait(10)                                            # the loader is in the middle of block 1
    threading.Timer(0.3, gate.set).start()
    handle.remove()                                                   # joins the thread once block 1 is done
    assert not any(t.name == "trash-panda-loader" for t in threading.enumerate())
    assert not _meta(model, 0) and not _meta(model, 1)
    assert _meta(model, 3)                                            # the loader did not go on to the last block
    assert all("forward" not in b.__dict__ for b in model.blocks)     # and the gates are gone


def test_removing_the_handle_before_anything_ran_leaves_the_blocks_as_they_were(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path)
    handle = install_offload(model)
    handle.remove()
    assert all("forward" not in b.__dict__ for b in model.blocks)
    assert not any(t.name == "trash-panda-loader" for t in threading.enumerate())


def test_a_final_pass_set_before_the_only_call_frees_blocks_that_were_loaded_late(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path)
    handle = install_offload(model)
    handle.final_pass()
    model(_x())
    assert all(p.is_meta for p in model.blocks.parameters())
    handle.remove()


def test_a_final_pass_after_an_earlier_call_frees_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path)
    handle = install_offload(model)
    model(_x())
    handle.final_pass()
    model(_x())
    assert all(p.is_meta for p in model.blocks.parameters())
    handle.remove()


def test_a_final_pass_frees_late_blocks_that_are_expanded_while_they_run(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _model(path, dtype="bf16")
    handle = install_offload(model)
    handle.final_pass()
    model(_x(torch.bfloat16))
    assert all(p.is_meta for p in model.blocks.parameters())
    handle.remove()


def test_blocks_loaded_late_and_blocks_loaded_at_once_do_not_trigger_the_order_warning(tiny_checkpoint):
    _, path = tiny_checkpoint
    for defer in (ALL[:2], ALL[1:2], ALL[2:], ALL):
        model = _model(path, dtype="bf16", defer=defer)
        handle = install_offload(model, order=["blocks.0", "blocks.1", "blocks.2", "blocks.3"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model(_x(torch.bfloat16))
            model(_x(torch.bfloat16))
        assert [str(w.message) for w in caught if "block order differs" in str(w.message)] == [], defer
        handle.remove()


def test_the_order_file_lists_the_blocks_that_were_loaded_late(tiny_checkpoint):
    import json
    _, path = tiny_checkpoint
    model = _model(path, dtype="bf16", defer=ALL[:2])
    handle = install_offload(model)
    model(_x(torch.bfloat16))
    handle.remove()
    saved = json.loads((path.parent / "tiny_order.json").read_text(encoding="utf-8"))["order"]
    assert saved == ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
