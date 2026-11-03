import threading
import time

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from .test_offload import _fp8_checkpoint, needs_fp8  # noqa: E402
from .tiny_model import Tiny, meta_model  # noqa: E402
from trash_panda.offload import Plan, install_offload, stream_load  # noqa: E402
from trash_panda.offload.checkpoint import Checkpoint  # noqa: E402

ALL = [f"blocks.{i}." for i in range(4)]


@pytest.fixture
def in_memory(monkeypatch):
    monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1")


def test_read_tensor_gives_the_same_values_as_get_tensor(tiny_checkpoint):
    ref, path = tiny_checkpoint
    with Checkpoint(path) as ck:
        for key, expected in ref.state_dict().items():
            read = ck.read_tensor(key)
            assert read.dtype == expected.dtype and read.shape == expected.shape
            assert torch.equal(read, expected)


def test_read_tensor_works_outside_a_with_block_too(tiny_checkpoint):
    ref, path = tiny_checkpoint
    assert torch.equal(Checkpoint(path).read_tensor("head.weight"), ref.head.weight)


def test_a_tensor_that_was_read_is_not_backed_by_the_file(tiny_checkpoint):
    _, path = tiny_checkpoint
    ck = Checkpoint(path)
    first = ck.read_tensor("head.weight")
    first.zero_()                                                    # changing it must not touch what the next read sees
    assert ck.read_tensor("head.weight").abs().sum() > 0


def test_a_tensor_with_no_elements_can_be_read(tmp_path):
    path = tmp_path / "empty.safetensors"
    save_file({"a": torch.zeros(0, 4), "b": torch.ones(2)}, str(path))
    ck = Checkpoint(path)
    assert ck.read_tensor("a").shape == (0, 4)
    assert torch.equal(ck.read_tensor("b"), torch.ones(2))


def test_streamed_blocks_are_read_into_memory_of_their_own(tiny_checkpoint, in_memory):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_only_the_tensors_of_streamed_blocks_are_read_that_way(tiny_checkpoint, in_memory, monkeypatch):
    _, path = tiny_checkpoint
    read = []
    real = Checkpoint.read_tensor
    monkeypatch.setattr(Checkpoint, "read_tensor", lambda self, key: (read.append(key), real(self, key))[1])
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", resident=ALL[:2], streamed=ALL[2:]), device="cpu")
    assert read and all(key.startswith(("blocks.2.", "blocks.3.")) for key in read)


def test_the_profile_says_how_much_was_read_and_how_fast(tiny_checkpoint, in_memory, monkeypatch, capsys):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert "of streamed-block tensors into memory in" in capsys.readouterr().out


def test_nothing_is_said_without_the_profile(tiny_checkpoint, in_memory, monkeypatch, capsys):
    _, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_PROFILE", raising=False)
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert "into memory" not in capsys.readouterr().out


def _output(path, monkeypatch, own, x, dtype="fp32", n=4, d=16, **tiers):
    """What the model gives for x after loading with the RAM tier mapped (own=False) or read into memory (own=True)."""
    monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1" if own else "0")
    model = meta_model(d=d, n=n)
    stream_load(model, path, plan=Plan.manual(compute_dtype=dtype, **(tiers or dict(streamed=ALL))), device="cpu")
    handle = install_offload(model)
    out = model(x), model(x)
    handle.remove()
    return out


def test_blocks_that_are_expanded_while_they_run_give_the_same_result(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    x = torch.randn(3, 16).bfloat16()                                 # stored as fp32, run as bf16: wrapped by the hooks
    mapped = _output(path, monkeypatch, False, x, dtype="bf16")
    own = _output(path, monkeypatch, True, x, dtype="bf16")
    assert torch.equal(mapped[0], own[0]) and torch.equal(own[0], own[1])


@needs_fp8
def test_scaled_fp8_blocks_are_read_with_their_scales(tmp_path, monkeypatch):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1")
    model = meta_model()
    report = stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert report.kinds["fp8_scaled"] == 8
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn   # still compact
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-5)
    handle.remove()


@needs_fp8
def test_plain_fp8_blocks_are_read_without_scales(tmp_path, monkeypatch):
    _, path = _fp8_checkpoint(tmp_path, with_scale=False)
    x = torch.randn(3, 16)
    mapped = _output(path, monkeypatch, False, x)
    own = _output(path, monkeypatch, True, x)
    assert torch.equal(mapped[0], own[0])


@needs_fp8
def test_weights_that_are_quantized_while_loading_give_the_same_result(tmp_path, monkeypatch):
    torch.manual_seed(2)
    ref = Tiny(d=128, n=2).eval()
    path = tmp_path / "big.safetensors"
    from safetensors.torch import save_file as save
    save({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    x = torch.randn(3, 128)
    kwargs = dict(n=2, d=128, storage_dtype="fp8", streamed=["blocks.0.", "blocks.1."])

    def run(own):
        monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1" if own else "0")
        model = meta_model(d=128, n=2)
        stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", storage_dtype="fp8", streamed=kwargs["streamed"]), device="cpu")
        handle = install_offload(model)
        out = model(x)
        handle.remove()
        return out

    assert torch.equal(run(False), run(True))


def _wipe_the_blocks_in_the_file(path):
    """Overwrite the bytes of every tensor of the blocks inside the checkpoint file, in place."""
    ck = Checkpoint(path)
    with open(path, "r+b") as fh:
        for key in ck.keys():
            if key.startswith("blocks."):
                info = ck.info(key)
                fh.seek(info.start)
                fh.write(bytes(info.end - info.start))


def test_the_model_does_not_notice_when_the_file_changes_after_loading(tiny_checkpoint, in_memory):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    _wipe_the_blocks_in_the_file(path)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_a_mapped_model_does_notice_which_is_what_the_test_above_rules_out(tiny_checkpoint, monkeypatch):
    ref, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_RAM_IN_MEMORY", raising=False)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    _wipe_the_blocks_in_the_file(path)
    x = torch.randn(3, 16)
    assert not torch.allclose(model(x), ref(x), atol=1e-6)


def test_a_checkpoint_in_several_files_is_read_the_same_way(tiny_checkpoint, tmp_path, monkeypatch):
    ref, _ = tiny_checkpoint
    folder = tmp_path / "shards"
    folder.mkdir()
    state = {k: v.contiguous() for k, v in ref.state_dict().items()}
    first = {k: v for k, v in state.items() if k.startswith(("blocks.0.", "blocks.1."))}
    save_file(first, str(folder / "part1.safetensors"))
    save_file({k: v for k, v in state.items() if k not in first}, str(folder / "part2.safetensors"))
    x = torch.randn(3, 16)
    monkeypatch.setenv("TRASH_PANDA_RAM_IN_MEMORY", "1")
    model = meta_model()
    stream_load(model, folder, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_it_works_together_with_blocks_that_are_loaded_while_the_model_runs(tiny_checkpoint, in_memory):
    ref, path = tiny_checkpoint
    model = meta_model()
    plan = Plan.manual(compute_dtype="fp32", resident=ALL[:2], streamed=ALL[2:])
    stream_load(model, path, plan=plan, defer_prefixes=ALL[:2], device="cpu")
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    handle.remove()


def _slow_reads(monkeypatch, seconds=0.02):
    """Make every read take a while and remember which thread did it. Returns that list of thread names."""
    names = []
    real = Checkpoint.read_tensor

    def read(self, key):
        names.append(threading.current_thread().name)
        time.sleep(seconds)
        return real(self, key)

    monkeypatch.setattr(Checkpoint, "read_tensor", read)
    return names


def test_the_reading_is_spread_over_the_threads_asked_for(tiny_checkpoint, in_memory, monkeypatch):
    ref, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_RAM_READ_THREADS", "3")
    names = _slow_reads(monkeypatch)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert len(set(names)) == 3
    assert threading.current_thread().name not in names
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)


def test_one_thread_reads_in_the_order_of_the_file(tiny_checkpoint, in_memory, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_RAM_READ_THREADS", "1")
    read = []
    real = Checkpoint.read_tensor
    monkeypatch.setattr(Checkpoint, "read_tensor", lambda self, key: (read.append(key), real(self, key))[1])
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    offsets = [Checkpoint(path).info(key).start for key in read]
    assert offsets == sorted(offsets)


def test_a_read_that_fails_stops_the_load(tiny_checkpoint, in_memory, monkeypatch):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_RAM_READ_THREADS", "2")
    real = Checkpoint.read_tensor

    def read(self, key):
        if key.startswith("blocks.2."):
            raise OSError("the drive went away")
        return real(self, key)

    monkeypatch.setattr(Checkpoint, "read_tensor", read)
    with pytest.raises(OSError, match="the drive went away"):
        stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")


def test_every_streamed_tensor_is_loaded_whatever_the_thread_count(tiny_checkpoint, in_memory, monkeypatch):
    ref, path = tiny_checkpoint
    for threads in ("1", "2", "8"):
        monkeypatch.setenv("TRASH_PANDA_RAM_READ_THREADS", threads)
        model = meta_model()
        stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
        assert not any(t.is_meta for t in model.parameters()), threads
        for name, expected in ref.state_dict().items():
            assert torch.equal(model.get_parameter(name), expected), (threads, name)


def test_the_profile_splits_the_load_time_by_tier(tiny_checkpoint, in_memory, monkeypatch, capsys):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    monkeypatch.setenv("TRASH_PANDA_RAM_READ_THREADS", "2")
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", resident=ALL[:2], streamed=ALL[2:]), device="cpu")
    out = capsys.readouterr().out
    assert "stream_load took" in out and "ram " in out and "gpu_raw " in out
    assert "2 threads" in out


def test_the_profile_times_the_load_without_the_new_setting_too(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    monkeypatch.delenv("TRASH_PANDA_RAM_IN_MEMORY", raising=False)
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    out = capsys.readouterr().out
    assert "stream_load took" in out and "into memory" not in out


def test_the_profile_says_when_some_blocks_are_loaded_later(tiny_checkpoint, monkeypatch, capsys):
    _, path = tiny_checkpoint
    monkeypatch.setenv("TRASH_PANDA_PROFILE", "1")
    stream_load(meta_model(), path, plan=Plan.manual(compute_dtype="fp32", resident=ALL), defer_prefixes=ALL, device="cpu")
    assert "not counted" in capsys.readouterr().out


def test_a_tensor_is_not_allocated_by_torch(tiny_checkpoint, monkeypatch):
    # PyTorch's CPU allocator keeps freed blocks on Windows, so a model released with GBs read this way held on to them
    ref, path = tiny_checkpoint
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("torch.empty was used"))
    ck = Checkpoint(path)
    read = ck.read_tensor("head.weight")
    monkeypatch.undo()
    assert torch.equal(read, ref.head.weight)
    read += 1                                                          # and it is a tensor that can be written to
