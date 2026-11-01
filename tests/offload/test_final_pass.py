import weakref

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, offload_blocks, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]
MIXED = dict(resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:])


def _loaded(path, dtype="fp32", **tiers):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype=dtype, **(tiers or MIXED)), device="cpu")
    return model, install_offload(model)


def test_final_pass_reaches_the_prefetcher_from_install_offload(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path)
    assert handle._prefetcher.final is False
    handle.final_pass()
    assert handle._prefetcher.final is True


def test_final_pass_reaches_the_prefetcher_from_offload_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    handle = offload_blocks(model.blocks, device="cpu", compute_dtype=torch.float32)
    assert handle._prefetcher.final is False
    handle.final_pass()
    assert handle._prefetcher.final is True


def test_a_pass_before_final_pass_frees_nothing(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model, _handle = _loaded(path, resident=ALL[:2], streamed=ALL[2:])
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert not any(p.is_meta for p in model.blocks.parameters())
    assert torch.allclose(model(x), ref(x), atol=1e-6)         # and it can run again


def test_the_final_pass_still_computes_the_right_result(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model, handle = _loaded(path)
    x = torch.randn(3, 16)
    handle.final_pass()
    assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_the_final_pass_frees_every_block_in_every_tier(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    handle.final_pass()
    model(torch.randn(3, 16))
    assert all(p.is_meta for p in model.blocks.parameters())
    assert all(b.is_meta for b in model.blocks.buffers())


def test_the_final_pass_leaves_the_weights_outside_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    handle.final_pass()
    model(torch.randn(3, 16))
    assert not model.embed.weight.is_meta
    assert not model.head.weight.is_meta


def test_blocks_are_freed_one_at_a_time_as_they_run(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL[:2], streamed=ALL[2:])
    seen = []
    for block in model.blocks:
        block.register_forward_pre_hook(lambda *_: seen.append([b.lin1.weight.is_meta for b in model.blocks]))
    handle.final_pass()
    model(torch.randn(3, 16))
    # when block k starts, the blocks before it are gone and it and the ones after it are still there
    assert seen == [[j < k for j in range(4)] for k in range(4)]


def test_the_final_pass_lets_go_of_the_weights_it_held(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL[:2], streamed=ALL[2:])
    held = [weakref.ref(block.lin1.weight) for block in model.blocks]
    handle.final_pass()
    model(torch.randn(3, 16))
    assert all(ref() is None for ref in held)


def test_blocks_that_are_expanded_while_they_run_are_freed_too(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, dtype="bf16", streamed=ALL)   # stored as fp32, so every block is wrapped
    assert all("forward" in block.__dict__ for block in model.blocks)
    held = [weakref.ref(block.lin1.weight) for block in model.blocks]
    handle.final_pass()
    model(torch.randn(3, 16).bfloat16())
    assert all(p.is_meta for p in model.blocks.parameters())
    assert all(ref() is None for ref in held)


def test_remove_after_a_final_pass_is_harmless(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    handle.final_pass()
    model(torch.randn(3, 16))
    handle.remove()
    assert all("forward" not in block.__dict__ for block in model.blocks)
