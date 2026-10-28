import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, release, stream_load  # noqa: E402
from trash_panda.offload import hooks  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]
MIXED = dict(resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:])


def _loaded(path):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", **MIXED), device="cpu")
    return model, install_offload(model)


def test_release_leaves_every_weight_on_the_meta_device(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    model.register_buffer("extra", torch.ones(3))
    release(model, handle)
    assert all(p.is_meta for p in model.parameters())
    assert all(b.is_meta for b in model.buffers())


def test_release_forgets_what_the_loader_remembered(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    assert "_trash_panda_state" in model.__dict__
    release(model, handle)
    assert "_trash_panda_state" not in model.__dict__


def test_release_removes_the_hooks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    assert "forward" in model.blocks[3].__dict__                # the disk block was wrapped
    release(model, handle)
    assert all("forward" not in block.__dict__ for block in model.blocks)


def test_release_lets_go_of_the_weights_it_held(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    held = [weakref.ref(model.blocks[0].lin1.weight), weakref.ref(model.blocks[1].lin1.weight)]   # resident, streamed
    assert all(ref() is not None for ref in held)
    release(model, handle)
    assert all(ref() is None for ref in held)                   # nothing else points at them any more


def test_release_works_after_the_model_has_run(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    model(torch.randn(3, 16))
    release(model, handle)
    assert all(p.is_meta for p in model.parameters())


def test_release_closes_the_prefetcher_once(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    closed = []
    monkeypatch.setattr(hooks._Prefetcher, "close", lambda self: closed.append(self))   # before install: the handle keeps the method
    model, handle = _loaded(path)
    release(model, handle)
    assert len(closed) == 1


def test_release_twice_is_harmless(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    release(model, handle)
    release(model, handle)
    release(model)
    assert all(p.is_meta for p in model.parameters())


def test_release_without_a_handle_still_empties_the_model(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, _handle = _loaded(path)
    release(model)
    assert all(p.is_meta for p in model.parameters())


def test_release_collects_garbage_once_and_not_once_per_module(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    calls = []
    real = gc.collect
    monkeypatch.setattr(hooks.gc, "collect", lambda *a: calls.append(1) or real(*a))
    release(model, handle)
    assert calls == [1]


@pytest.mark.parametrize("cuda_is_there, expected", [(True, [1]), (False, [])], ids=["cuda", "no-cuda"])
def test_release_empties_the_gpu_and_pinned_caches_only_when_cuda_is_there(
        tiny_checkpoint, monkeypatch, cuda_is_there, expected):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    gpu, pinned = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_is_there)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: gpu.append(1))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: pinned.append(1), raising=False)
    release(model, handle)
    assert gpu == expected
    assert pinned == expected


def test_release_works_when_torch_has_no_pinned_cache_call(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.delattr(torch._C, "_host_emptyCache", raising=False)
    release(model, handle)
    assert all(p.is_meta for p in model.parameters())
