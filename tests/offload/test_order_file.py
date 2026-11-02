import json
import warnings

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402
from trash_panda.offload.order_file import order_path, read_order, write_order  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]
IN_ORDER = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
REVERSED = IN_ORDER[::-1]


def _loaded(path, order=None, destination=None, **tiers):
    """Every block is stored as fp32 and run as bf16, so every block is wrapped (and has an entry) on the CPU."""
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", **(tiers or dict(streamed=ALL))), device="cpu")
    return model, install_offload(model, order=order, order_file_destination=destination)


def _chain(handle):
    names, entry = [], handle._prefetcher.head
    while entry is not None and entry.name not in names:
        names.append(entry.name)
        entry = entry.next
    return names


def _run(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.randn(3, 16).bfloat16())
    return [str(w.message) for w in caught]


def _file(path):
    return path.parent / "tiny_order.json"


def _saved(file):
    return json.loads(file.read_text(encoding="utf-8"))["order"]


def test_the_file_is_named_after_the_checkpoint_and_sits_next_to_it(tmp_path):
    assert order_path(tmp_path / "model.safetensors") == tmp_path / "model_order.json"


def test_the_file_goes_to_the_folder_given(tmp_path):
    assert order_path(tmp_path / "model.safetensors", tmp_path / "orders") == tmp_path / "orders" / "model_order.json"


def test_a_checkpoint_folder_keeps_the_file_inside_it(tmp_path):
    folder = tmp_path / "vae"
    folder.mkdir()
    assert order_path(folder) == folder / "vae_order.json"


def test_read_order_gives_none_for_a_missing_broken_or_unknown_file(tmp_path):
    known = set(IN_ORDER)
    assert read_order(tmp_path / "none.json", known) is None
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert read_order(tmp_path / "broken.json", known) is None
    write_order(tmp_path / "unknown.json", ["blocks.9"])
    assert read_order(tmp_path / "unknown.json", known) is None
    write_order(tmp_path / "fine.json", REVERSED)
    assert read_order(tmp_path / "fine.json", known) == REVERSED


def test_a_run_saves_the_order_when_the_handle_is_removed(tiny_checkpoint, capsys):
    _, path = tiny_checkpoint
    model, handle = _loaded(path)
    _run(model)
    assert not _file(path).exists()                            # nothing is written while the model is in use
    handle.remove()
    assert _saved(_file(path)) == IN_ORDER
    assert "saved the order of 4 blocks" in capsys.readouterr().out


def test_the_next_install_uses_the_saved_order(tiny_checkpoint, capsys):
    _, path = tiny_checkpoint
    write_order(_file(path), REVERSED)
    _, handle = _loaded(path)
    assert _chain(handle) == REVERSED
    assert "using the block order saved" in capsys.readouterr().out


def test_a_file_that_was_right_is_not_written_again(tiny_checkpoint):
    _, path = tiny_checkpoint
    _file(path).write_text(json.dumps({"order": IN_ORDER}, indent=4), encoding="utf-8")   # the library writes indent 2
    before = _file(path).read_text(encoding="utf-8")
    model, handle = _loaded(path)
    assert _run(model) == []
    handle.remove()
    assert _file(path).read_text(encoding="utf-8") == before


def test_a_file_that_was_wrong_is_corrected(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(_file(path), REVERSED)
    model, handle = _loaded(path)
    assert any("block order differs" in m for m in _run(model))
    handle.remove()
    assert _saved(_file(path)) == IN_ORDER


def test_a_file_that_names_a_block_the_model_lacks_is_ignored_and_replaced(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(_file(path), ["blocks.9"])
    model, handle = _loaded(path)
    assert _chain(handle) == IN_ORDER                          # the plan order
    _run(model)
    handle.remove()
    assert _saved(_file(path)) == IN_ORDER


def test_an_explicit_order_neither_reads_nor_writes_the_file(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(_file(path), REVERSED)
    before = _file(path).read_text(encoding="utf-8")
    model, handle = _loaded(path, order=["blocks.2"])
    assert _chain(handle) == ["blocks.2", "blocks.0", "blocks.1", "blocks.3"]
    _run(model)
    handle.remove()
    assert _file(path).read_text(encoding="utf-8") == before


def test_an_explicit_order_does_not_create_a_file(tiny_checkpoint):
    _, path = tiny_checkpoint
    model, handle = _loaded(path, order=IN_ORDER)
    _run(model)
    handle.remove()
    assert not _file(path).exists()


def test_the_destination_folder_is_created_and_used(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    folder = tmp_path / "orders" / "here"
    model, handle = _loaded(path, destination=folder)
    _run(model)
    handle.remove()
    assert _saved(folder / "tiny_order.json") == IN_ORDER
    assert not _file(path).exists()


def test_a_destination_that_cannot_be_written_warns_and_the_run_goes_on(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    model, handle = _loaded(path, destination=blocker)
    assert model(torch.randn(3, 16).bfloat16()).shape == (3, 4)
    with pytest.warns(UserWarning, match="could not save the block order"):
        handle.remove()


def test_blocks_that_are_not_wrapped_are_saved_too(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), device="cpu")
    handle = install_offload(model)
    assert all("forward" not in block.__dict__ for block in model.blocks)     # none of them is wrapped
    model(torch.randn(3, 16))
    handle.remove()
    assert _saved(_file(path)) == IN_ORDER


def test_wrapped_and_unwrapped_blocks_are_saved_in_the_order_they_ran(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL[:2], disk=ALL[2:]), device="cpu")
    handle = install_offload(model)
    model(torch.randn(3, 16))
    handle.remove()
    assert _saved(_file(path)) == IN_ORDER


def test_only_the_blocks_that_ran_are_saved(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), device="cpu")
    handle = install_offload(model)
    x = model.embed(torch.randn(3, 16))
    for i in (2, 0):                                                          # a call that runs two blocks, out of order
        x = model.blocks[i](x)
    handle.remove()
    assert _saved(_file(path)) == ["blocks.2", "blocks.0"]


def test_removing_the_handle_removes_the_note_hooks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), device="cpu")
    handle = install_offload(model)
    handle.remove()
    assert all(not block._forward_pre_hooks for block in model.blocks)

