import warnings

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]
IN_ORDER = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]


def _loaded(path, order=None, **tiers):
    """Every block is stored as fp32 and run as bf16, so every block is wrapped (and has an entry) on the CPU."""
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", **(tiers or dict(streamed=ALL))), device="cpu")
    return model, install_offload(model, order=order)


def _entries(handle):
    """The wrapped blocks by name, found by following the links from the first block given."""
    found, entry = {}, handle._prefetcher.head
    while entry is not None and entry.name not in found:
        found[entry.name] = entry
        entry = entry.next
    return found


def _chain(handle):
    """The names in the order the links lead from the first block given, before any pass has run."""
    names, entry = [], handle._prefetcher.head
    while entry is not None and entry.name not in names:
        names.append(entry.name)
        entry = entry.next
    return names


def _order_warnings(caught):
    return [str(w.message) for w in caught if "block order differs" in str(w.message)]


def _run(model, times=1):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(times):
            model(torch.randn(3, 16).bfloat16())
    return _order_warnings(caught)


def test_the_links_follow_the_plan_order_before_any_call(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path)
    assert _chain(handle) == IN_ORDER


def test_blocks_that_need_no_wrapping_have_no_link(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL[:3], disk=ALL[3:]), device="cpu")
    handle = install_offload(model)
    assert _chain(handle) == ["blocks.3"]                      # on the CPU in fp32 only the disk block is wrapped


def test_order_puts_the_named_blocks_first(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, order=["blocks.2", "blocks.3"])
    assert _chain(handle) == ["blocks.2", "blocks.3", "blocks.0", "blocks.1"]


def test_order_takes_prefixes_with_or_without_the_dot(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, order=["blocks.3.", "blocks.1"])
    assert _chain(handle) == ["blocks.3", "blocks.1", "blocks.0", "blocks.2"]


def test_a_name_given_twice_counts_once(tiny_checkpoint):
    _, path = tiny_checkpoint
    _, handle = _loaded(path, order=["blocks.1", "blocks.1"])
    assert _chain(handle) == ["blocks.1", "blocks.0", "blocks.2", "blocks.3"]


def test_a_name_that_is_not_in_the_plan_raises(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", streamed=ALL), device="cpu")
    with pytest.raises(ValueError, match="blocks.9"):
        install_offload(model, order=["blocks.9"])


def test_a_run_in_the_order_given_does_not_warn(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, _ = _loaded(path)
    assert _run(model, times=2) == []


def test_a_wrong_first_block_warns_once_and_names_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, _ = _loaded(path, order=["blocks.3"])               # the hint starts with 3, the model starts with 0
    caught = _run(model, times=3)
    assert len(caught) == 1
    assert "blocks.0 ran first, but blocks.3 was expected first" in caught[0]
    assert "order=[" in caught[0]


def test_a_wrong_link_in_the_middle_warns_and_names_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, _ = _loaded(path, order=["blocks.0", "blocks.2", "blocks.1", "blocks.3"])
    caught = _run(model, times=2)
    assert len(caught) == 1
    assert "blocks.1 ran after blocks.0, but blocks.2 was expected" in caught[0]


def test_the_first_pass_corrects_the_links_behind_it(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, order=["blocks.3"])
    _run(model)
    entries = _entries(handle)
    assert entries["blocks.0"].next is entries["blocks.1"]
    assert entries["blocks.1"].next is entries["blocks.2"]
    assert entries["blocks.2"].next is entries["blocks.3"]


def test_a_wrong_order_gives_the_same_result(tiny_checkpoint):
    _, path = tiny_checkpoint
    x = torch.randn(3, 16).bfloat16()
    right, _ = _loaded(path)
    wrong, _ = _loaded(path, order=["blocks.3", "blocks.1"])
    with pytest.warns(UserWarning, match="block order differs"):
        out = wrong(x)
    assert torch.equal(out, right(x))
