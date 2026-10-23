import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.checkpoint import Checkpoint  # noqa: E402


def test_read_into_matches_the_reference_weights(tiny_checkpoint):
    ref, path = tiny_checkpoint
    ck = Checkpoint(path)
    for key, value in ref.state_dict().items():
        info = ck.info(key)
        out = torch.empty(info.nbytes, dtype=torch.uint8)
        ck.read_into(key, out)
        assert torch.equal(out.view(info.torch_dtype).reshape(info.shape), value)


def test_read_into_a_slice_of_a_bigger_buffer(tiny_checkpoint):
    ref, path = tiny_checkpoint
    ck = Checkpoint(path)
    key = "blocks.0.lin1.weight"
    info = ck.info(key)
    big = torch.zeros(info.nbytes + 512, dtype=torch.uint8)
    window = big[256:256 + info.nbytes]
    ck.read_into(key, window)
    assert torch.equal(window.view(info.torch_dtype).reshape(info.shape), ref.state_dict()[key])
    assert big[:256].eq(0).all() and big[256 + info.nbytes:].eq(0).all()      # nothing outside the slice was touched


def test_read_into_works_while_the_checkpoint_is_open(tiny_checkpoint):
    ref, path = tiny_checkpoint
    key = "head.weight"
    with Checkpoint(path) as ck:
        info = ck.info(key)
        out = torch.empty(info.nbytes, dtype=torch.uint8)
        ck.read_into(key, out)
    assert torch.equal(out.view(info.torch_dtype).reshape(info.shape), ref.state_dict()[key])


def test_read_into_rejects_a_wrong_buffer(tiny_checkpoint):
    _, path = tiny_checkpoint
    ck = Checkpoint(path)
    key = "blocks.0.lin1.weight"
    size = ck.info(key).nbytes
    with pytest.raises(ValueError):
        ck.read_into(key, torch.empty(size, dtype=torch.float32))            # not uint8
    with pytest.raises(ValueError):
        ck.read_into(key, torch.empty(size - 1, dtype=torch.uint8))          # too small
    with pytest.raises(ValueError):
        ck.read_into(key, torch.empty(size * 2, dtype=torch.uint8)[::2])     # not contiguous
