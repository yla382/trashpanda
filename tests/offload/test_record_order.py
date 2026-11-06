import collections
import json
import warnings

import pytest

torch = pytest.importorskip("torch")

from .tiny_model import Tiny  # noqa: E402
from trash_panda.offload import Plan, install_offload, record_order, stream_load  # noqa: E402
from trash_panda.offload.order_file import write_order  # noqa: E402
from trash_panda.offload.stage import _on_meta  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]
IN_ORDER = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
REVERSED = IN_ORDER[::-1]


class Backwards(Tiny):
    """Defines its blocks in one order and runs them in the other, like a model whose forward() overrides the definition."""

    def forward(self, x):
        x = self.embed(x)
        for i in (3, 2, 1, 0):
            x = self.blocks[i](x)
        return self.head(x)


class Picky(Tiny):
    """Takes a dict and a keyword, and runs the blocks backwards when `flip` is set."""

    def forward(self, x, *, extra=None, flip=False):
        x = self.embed(x["x"] + extra[0])
        for i in ((3, 2, 1, 0) if flip else (0, 1, 2, 3)):
            x = self.blocks[i](x)
        return self.head(x)


class Silent(Tiny):
    def forward(self, x):
        return self.head(self.embed(x))


class Reader(Tiny):
    def forward(self, x):
        if x.sum().item() > 0:                                     # reads a value: there is none on the meta device
            x = x + 1
        return super().forward(x)


def _builder(cls):
    def build():
        with torch.device("meta"):
            return cls().eval()
    return build


def _file(path):
    return path.parent / "tiny_order.json"


def _saved(file):
    return json.loads(file.read_text(encoding="utf-8"))["order"]


def _x():
    return torch.randn(3, 16)


def test_the_order_the_blocks_run_in_is_saved_next_to_the_checkpoint(tiny_checkpoint, capsys):
    _, path = tiny_checkpoint
    assert record_order(_builder(Backwards), path, args=(_x(),)) == REVERSED
    assert _saved(_file(path)) == REVERSED
    assert "saved the order of 4 blocks" in capsys.readouterr().out


def test_a_model_defined_in_the_order_it_runs_gives_that_order(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert record_order(_builder(Tiny), path, args=(_x(),)) == IN_ORDER
    assert _saved(_file(path)) == IN_ORDER


def test_the_next_run_uses_the_saved_order_and_does_not_warn(tiny_checkpoint, capsys):
    _, path = tiny_checkpoint
    record_order(_builder(Backwards), path, args=(_x(),))
    capsys.readouterr()
    model = _builder(Backwards)()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", streamed=ALL), device="cpu")
    handle = install_offload(model)
    assert "using the block order saved" in capsys.readouterr().out
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(_x().bfloat16())
    assert [str(w.message) for w in caught if "block order differs" in str(w.message)] == []
    handle.remove()
    assert _saved(_file(path)) == REVERSED                         # a file that was right is left alone


def test_the_first_run_without_the_call_warns_about_that_model(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = _builder(Backwards)()
    stream_load(model, path, plan=Plan.manual(compute_dtype="bf16", streamed=ALL), device="cpu")
    handle = install_offload(model)
    with pytest.warns(UserWarning, match="block order differs"):
        model(_x().bfloat16())
    handle.remove()


def test_tensors_in_the_inputs_are_moved_to_the_meta_device(tiny_checkpoint):
    _, path = tiny_checkpoint
    got = record_order(_builder(Picky), path, args=({"x": _x()},), kwargs={"extra": [torch.ones(16)], "flip": True})
    assert got == REVERSED
    assert record_order(_builder(Picky), path, args=({"x": _x()},), kwargs={"extra": [torch.ones(16)]}) == IN_ORDER


def test_one_tensor_can_be_given_without_a_tuple(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert record_order(_builder(Backwards), path, args=_x()) == REVERSED


def test_the_inputs_of_the_caller_are_not_changed(tiny_checkpoint):
    _, path = tiny_checkpoint
    x = _x()
    record_order(_builder(Tiny), path, args=(x,))
    assert x.device.type == "cpu" and not x.is_meta


def test_on_meta_keeps_the_structure_and_the_other_values():
    Pair = collections.namedtuple("Pair", "a b")
    out = _on_meta({"t": torch.ones(2), "l": [torch.ones(1), 3], "p": Pair(torch.ones(1), "s"), "n": None})
    assert out["t"].is_meta and out["l"][0].is_meta and out["l"][1] == 3
    assert isinstance(out["p"], Pair) and out["p"].a.is_meta and out["p"].b == "s" and out["n"] is None


def test_the_hooks_are_gone_after_the_call(tiny_checkpoint):
    _, path = tiny_checkpoint
    made = []

    def build():
        made.append(_builder(Backwards)())
        return made[-1]

    record_order(build, path, args=(_x(),))
    assert all(not block._forward_pre_hooks for block in made[0].blocks)


def test_a_model_with_weights_in_memory_is_refused(tiny_checkpoint):
    _, path = tiny_checkpoint
    with pytest.raises(ValueError, match="meta device"):
        record_order(lambda: Tiny().eval(), path, args=(_x(),))
    assert not _file(path).exists()


def test_a_call_that_fails_on_the_meta_device_warns_and_saves_nothing(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(_file(path), REVERSED)
    before = _file(path).read_text(encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="could not call the model on the meta device"):
        assert record_order(_builder(Reader), path, args=(_x(),)) is None
    assert _file(path).read_text(encoding="utf-8") == before


def test_a_call_in_which_no_block_runs_warns_and_saves_nothing(tiny_checkpoint):
    _, path = tiny_checkpoint
    with pytest.warns(RuntimeWarning, match="no block ran"):
        assert record_order(_builder(Silent), path, args=(_x(),)) is None
    assert not _file(path).exists()


def test_skipped_blocks_are_left_out(tiny_checkpoint):
    _, path = tiny_checkpoint
    assert record_order(_builder(Backwards), path, args=(_x(),), skip_prefixes=("blocks.3.",)) == REVERSED[1:]
    assert _saved(_file(path)) == REVERSED[1:]


def test_a_file_that_is_already_there_is_replaced(tiny_checkpoint):
    _, path = tiny_checkpoint
    write_order(_file(path), IN_ORDER)
    record_order(_builder(Backwards), path, args=(_x(),))
    assert _saved(_file(path)) == REVERSED


def test_the_destination_folder_is_created_and_used(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    folder = tmp_path / "orders" / "here"
    record_order(_builder(Backwards), path, args=(_x(),), order_file_destination=folder)
    assert _saved(folder / "tiny_order.json") == REVERSED
    assert not _file(path).exists()


def test_a_destination_that_cannot_be_written_warns_and_gives_none(tiny_checkpoint, tmp_path):
    _, path = tiny_checkpoint
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    with pytest.warns(UserWarning, match="could not save the block order"):
        assert record_order(_builder(Backwards), path, args=(_x(),), order_file_destination=blocker) is None


def test_the_checkpoint_is_not_opened(tmp_path):
    path = tmp_path / "missing.safetensors"                         # nothing to read: only the file name is used
    assert record_order(_builder(Backwards), path, args=(_x(),)) == REVERSED
    assert json.loads((tmp_path / "missing_order.json").read_text(encoding="utf-8"))["order"] == REVERSED
