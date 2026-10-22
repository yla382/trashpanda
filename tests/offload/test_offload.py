import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from .tiny_model import Tiny, meta_model  # noqa: E402
from trash_panda.offload import (  # noqa: E402
    Plan,
    install_offload,
    offload_blocks,
    quantize_fp8,
    stream_load,
)
from trash_panda.offload import hooks  # noqa: E402
from trash_panda.offload.formats import expand  # noqa: E402

needs_fp8 = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="this torch has no float8")
ALL = [f"blocks.{i}." for i in range(4)]


@pytest.mark.parametrize(
    "tiers",
    [
        dict(resident=ALL),
        dict(streamed=ALL),
        dict(disk=ALL),
        dict(resident=ALL[:1], streamed=ALL[1:3], disk=ALL[3:]),
    ],
    ids=["all-gpu", "all-ram", "all-disk", "mixed"],
)
def test_outputs_match_the_reference_model(tiny_checkpoint, tiers):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", **tiers), device="cpu")
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    assert torch.allclose(model(x), ref(x), atol=1e-6)        # a second pass works too
    handle.remove()


def test_low_level_api_without_a_plan(tiny_checkpoint):
    ref, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    handle = offload_blocks(model.blocks, device="cpu", compute_dtype=torch.float32)
    x = torch.randn(2, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-6)
    handle.remove()


def test_failed_forward_puts_the_weights_back(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    install_offload(model)
    with pytest.raises(RuntimeError):
        model.blocks[0](torch.randn(3, 99))                    # wrong width: fails inside the block
    assert model.blocks[0].lin1.weight.is_meta                 # the compact (here: on-disk) placeholder is back


def test_remove_restores_the_blocks(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    handle = install_offload(model)
    assert "forward" in model.blocks[0].__dict__
    handle.remove()
    assert "forward" not in model.blocks[0].__dict__


def test_report_lists_unexpected_keys(tiny_checkpoint, tmp_path):
    ref, _ = tiny_checkpoint
    state = {k: v.contiguous() for k, v in ref.state_dict().items()}
    state["extra.weight"] = torch.zeros(2, 2)
    path = tmp_path / "extra.safetensors"
    save_file(state, str(path))
    report = stream_load(meta_model(), path, compute_dtype="fp32", device="cpu")
    assert report.unexpected == ["extra.weight"]
    assert "extra.weight" in str(report)


def test_missing_tensors_are_an_error(tiny_checkpoint, tmp_path):
    ref, _ = tiny_checkpoint
    state = {k: v.contiguous() for k, v in ref.state_dict().items() if k != "head.bias"}
    path = tmp_path / "missing.safetensors"
    save_file(state, str(path))
    with pytest.raises(RuntimeError, match="head.bias"):
        stream_load(meta_model(), path, compute_dtype="fp32", device="cpu")


def test_install_offload_needs_a_plan(tiny_checkpoint):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    with pytest.raises(ValueError, match="needs a Plan"):
        install_offload(model)


def _fp8_checkpoint(tmp_path, with_scale):
    """A checkpoint whose block weights are fp8, plus the matching reference model (dequantized weights)."""
    torch.manual_seed(1)
    ref = Tiny().eval()
    tensors = {}
    for k, v in ref.state_dict().items():
        if k.startswith("blocks.") and k.endswith(".weight") and v.dim() == 2:
            q, scale = quantize_fp8(v)
            tensors[k] = q
            if with_scale:
                tensors[k + "_scale"] = scale
            v.copy_(expand(q, scale if with_scale else None, torch.float32, "cpu"))
        else:
            tensors[k] = v.clone()
    path = tmp_path / ("fp8_scaled.safetensors" if with_scale else "fp8_plain.safetensors")
    try:
        save_file(tensors, str(path))
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"this safetensors cannot save float8: {exc}")
    return ref, path


@needs_fp8
def test_scaled_fp8_blocks_stay_compact_and_match(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    report = stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert report.kinds["fp8_scaled"] == 8                      # 4 blocks x 2 linear weights
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn
    handle = install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-5)
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn    # back to compact form afterwards
    handle.remove()


@needs_fp8
def test_plain_fp8_without_scale_is_supported_and_flagged(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=False)
    model = meta_model()
    report = stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    assert report.kinds["fp8_plain"] == 8
    assert any("no scale" in w for w in report.warnings)
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn   # still 1 byte per weight in RAM


@needs_fp8
def test_fp8_blocks_can_stream_from_disk(tmp_path):
    ref, path = _fp8_checkpoint(tmp_path, with_scale=True)
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", disk=ALL), device="cpu")
    install_offload(model)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), ref(x), atol=1e-5)


@needs_fp8
def test_quantize_on_load_makes_big_weights_fp8(tmp_path):
    torch.manual_seed(2)
    ref = Tiny(d=128, n=2).eval()
    path = tmp_path / "big.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    model = meta_model(d=128, n=2)
    plan = Plan.manual(compute_dtype="fp32", storage_dtype="fp8", streamed=["blocks.0.", "blocks.1."])
    report = stream_load(model, path, plan=plan, device="cpu")
    assert report.kinds["fp8_scaled"] == 4                      # 2 blocks x 2 linear weights
    assert model.blocks[0].lin1.weight.dtype == torch.float8_e4m3fn
    assert model.blocks[0].norm.weight.dtype == torch.float32   # small tensors are left alone
    install_offload(model)
    x = torch.randn(3, 128)
    out, expected = model(x), ref(x)
    assert (out - expected).abs().max() < 0.2 * expected.abs().max()   # lossy, but close


def test_prefetcher_close_is_safe_without_cuda_and_can_repeat():
    prefetcher = hooks._Prefetcher(torch.device("cpu"))
    assert prefetcher.stream is None and prefetcher.worker is None and prefetcher.pool is None
    prefetcher.close()
    prefetcher.close()                                          # closing twice does nothing
    assert prefetcher.stream is None and prefetcher.worker is None and prefetcher.pool is None


def test_remove_closes_the_shared_prefetcher_once_and_last(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    model = meta_model()
    stream_load(model, path, plan=Plan.manual(compute_dtype="fp32", streamed=ALL), device="cpu")
    closed = []

    def fake_close(self):
        closed.append("forward" in model.blocks[0].__dict__)    # was the block still wrapped when close ran?

    monkeypatch.setattr(hooks._Prefetcher, "close", fake_close)
    handle = install_offload(model)
    handle.remove()
    assert closed == [False]                                    # closed exactly once, after the blocks were restored


def test_offload_blocks_closes_only_a_prefetcher_it_created(tiny_checkpoint, monkeypatch):
    _, path = tiny_checkpoint
    closed = []
    monkeypatch.setattr(hooks._Prefetcher, "close", lambda self: closed.append(self))

    own = meta_model()
    stream_load(own, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    offload_blocks(own.blocks, device="cpu", compute_dtype=torch.float32).remove()
    assert len(closed) == 1                                     # it made its own prefetcher, so it closes it

    closed.clear()
    shared = meta_model()
    stream_load(shared, path, offload_prefixes=["blocks."], compute_dtype="fp32", device="cpu")
    given = hooks._Prefetcher(torch.device("cpu"))
    offload_blocks(shared.blocks, device="cpu", compute_dtype=torch.float32, prefetcher=given).remove()
    assert closed == []                                         # a prefetcher passed in is left to its owner
