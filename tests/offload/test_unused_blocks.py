import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, module_nbytes, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


def _loaded(path, dtype="fp32", **tiers):
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype=dtype, **tiers), device="cpu")
    return model, install_offload(model)


def _run_blocks(model, count, dtype=torch.float32):
    x = torch.randn(3, 16).to(dtype)
    for block in list(model.blocks)[:count]:
        x = block(x)


def test_blocks_that_never_ran_are_listed_with_their_size(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, resident=ALL)                    # nothing is wrapped
    left_over = module_nbytes(model.blocks[2]) + module_nbytes(model.blocks[3])
    _run_blocks(model, 2)
    text = handle._prefetcher.report()
    assert "blocks that never ran: blocks.2, blocks.3" in text
    assert f"({left_over / 1e9:.2f} GB)" in text
    assert "skip_prefixes" in text


def test_wrapped_blocks_that_never_ran_are_listed_too(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")      # stored as fp32, so every block is wrapped
    _run_blocks(model, 1, torch.bfloat16)
    text = handle._prefetcher.report()
    assert "blocks that never ran: blocks.1, blocks.2, blocks.3" in text


def test_nothing_is_said_when_every_block_ran(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, streamed=ALL, dtype="bf16")
    _run_blocks(model, 4, torch.bfloat16)
    assert "never ran" not in handle._prefetcher.report()


def test_nothing_is_said_before_any_block_has_run(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    _, handle = _loaded(path, resident=ALL)
    assert handle._prefetcher.report() == ""


def test_without_the_profile_nothing_is_kept_or_said(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, resident=ALL)
    _run_blocks(model, 2)
    assert handle._prefetcher.known == {}
    assert handle._prefetcher.report() == ""


def test_a_long_list_is_cut_short(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model, handle = _loaded(path, resident=ALL)
    pre = handle._prefetcher
    pre.known = {f"extra.{i}": 1000 for i in range(8)}
    pre.seen = {"extra.0": None}
    text = pre.report()
    assert "extra.1, extra.2, extra.3, extra.4, extra.5, extra.6 and 1 more" in text


def test_blocks_whose_weights_load_later_are_listed_when_they_never_run(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), defer_prefixes=ALL, device="cpu")
    handle = install_offload(model)                                # a loader thread now loads the four blocks
    try:
        _run_blocks(model, 3)
        assert "blocks that never ran: blocks.3" in handle._prefetcher.report()
    finally:
        handle.remove()
