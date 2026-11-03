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


def _wrapped(path):
    """Every block is stored as fp32 and runs as bf16, so every block is wrapped and has an entry that holds its tensors."""
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", streamed=ALL), device="cpu")
    return model, install_offload(model)


@pytest.mark.parametrize("ran", [False, True], ids=["never-ran", "ran"])
def test_release_lets_go_of_the_tensors_that_wrapped_blocks_held(tiny_checkpoint, ran):
    _, path = tiny_checkpoint
    model, handle = _wrapped(path)
    assert all("forward" in block.__dict__ for block in model.blocks)
    if ran:
        model(torch.randn(3, 16).bfloat16())
    held = [weakref.ref(p) for block in model.blocks for p in block.parameters()]
    assert all(ref() is not None for ref in held)
    release(model, handle)                                       # the caller still has `handle`
    assert all(ref() is None for ref in held)


def test_release_after_a_final_pass_lets_go_of_the_tensors_too(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _wrapped(path)
    held = [weakref.ref(p) for block in model.blocks for p in block.parameters()]
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    release(model, handle)
    assert all(ref() is None for ref in held)


def _fake_ram(monkeypatch, free_gb, tensors=()):
    """Make the memory reads give `free_gb` in turn (and nothing else), and return the list of RAM reads made."""
    reads = []
    values = iter(free_gb)

    def read():
        reads.append(1)
        free = next(values)
        return 16 * 10 ** 9, int(free * 1e9), int((free + 10) * 1e9)

    monkeypatch.setattr(hooks, "_read_ram", read)
    monkeypatch.setattr(hooks, "_process_memory", lambda: (9 * 10 ** 9, 8 * 10 ** 9))
    monkeypatch.setattr(hooks, "_cpu_tensors", lambda: list(tensors))
    return reads


def test_the_profile_traces_the_memory_while_releasing(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    _fake_ram(monkeypatch, [3.0, 3.1, 3.2, 10.5, 10.6])
    release(model, handle)
    out = capsys.readouterr().out
    assert "memory while releasing" in out
    for what, free in [("start", "3.00"), ("hooks removed", "3.10"), ("weights dropped", "3.20"), ("gc", "10.50"),
                       ("caches emptied", "10.60")]:
        assert f"  {what}" in out and f"{free} GB free, process 9.00 / 8.00 GB, 0 tensors of 0.00 GB" in out
    assert "commit headroom 13.00 -> 20.60 GB" in out
    assert "still alive" not in out


def test_nothing_is_traced_without_the_profile(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    monkeypatch.delenv("TRASH_PANDA_PROFILE", raising=False)
    reads = _fake_ram(monkeypatch, [1.0] * 10)
    monkeypatch.setattr(hooks, "_cpu_tensors", lambda: pytest.fail("the tensors were scanned"))
    release(model, handle)
    assert reads == [] and "memory while releasing" not in capsys.readouterr().out


def test_the_trace_leaves_out_what_the_system_cannot_tell(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    monkeypatch.setattr(hooks, "_read_ram", lambda: (16 * 10 ** 9, 5 * 10 ** 9, None))
    monkeypatch.setattr(hooks, "_process_memory", lambda: None)
    release(model, handle)
    out = capsys.readouterr().out
    assert "5.00 GB free, 0 tensors of 0.00 GB" in out
    assert ", process" not in out and "commit headroom" not in out


def test_the_trace_names_what_holds_a_big_tensor_that_is_still_alive(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    big = torch.empty(3 * 10 ** 7, dtype=torch.uint8)                # 30 MB, so more than 0.1 GB of them
    holder = {"weights": [big]}
    tensors = [(big.numel() * 4, weakref.ref(big))]
    _fake_ram(monkeypatch, [1.0] * 5, tensors=tensors)
    release(model, handle)
    out = capsys.readouterr().out
    assert "1 tensors of 0.12 GB" in out
    assert "still alive at the end, the biggest tensor is held by:" in out
    assert "list of 1" in out and "dict with keys 'weights'" in out
    assert holder


def test_cpu_tensors_finds_each_big_storage_once():
    big = torch.empty(2 ** 21)                                      # 8 MB
    view = big[:10]
    small = torch.empty(100)
    meta = torch.empty(2 ** 21, device="meta")
    found = hooks._cpu_tensors()
    sizes = [size for size, _ref in found]
    assert sizes.count(8 * 2 ** 20) == 1
    assert 400 not in sizes and sum(1 for size in sizes if size == 8 * 2 ** 20) == 1
    assert any(ref() is big or ref() is view for _size, ref in found)
    assert small is not None and meta is not None


def test_cpu_tensors_forgets_a_tensor_that_was_freed():
    big = torch.empty(2 ** 21)
    ref = next(ref for _size, ref in hooks._cpu_tensors() if ref() is big)
    del big
    gc.collect()
    assert ref() is None


def test_holders_say_what_refers_to_an_object():
    target = torch.empty(4)
    box = {"model": [target]}
    text = " | ".join(hooks._holders(target))
    assert "list of 1" in text and "dict with keys 'model'" in text
    assert box


def test_process_memory_is_readable_here():
    memory = hooks._process_memory()
    if memory is None:
        pytest.skip("this machine cannot say how much memory the process holds")
    working_set, private = memory
    assert working_set > 10 ** 7 and private > 10 ** 7
