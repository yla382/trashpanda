import pytest

torch = pytest.importorskip("torch")

from trash_panda.offload.formats import (  # noqa: E402
    Kind,
    classify,
    expand,
    parse_dtype,
    quantize_fp8,
    should_quantize,
    stored_nbytes,
)

needs_fp8 = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="this torch has no float8")


def test_parse_dtype():
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("fp16") is torch.float16
    assert parse_dtype("float32") is torch.float32
    assert parse_dtype("auto") is None
    assert parse_dtype(None) is None
    assert parse_dtype(torch.float16) is torch.float16
    with pytest.raises(ValueError):
        parse_dtype("fp7")


@needs_fp8
def test_classify():
    assert classify(torch.bfloat16, has_scale=False) == Kind.CAST
    assert classify(torch.float8_e4m3fn, has_scale=True) == Kind.FP8_SCALED
    assert classify(torch.float8_e4m3fn, has_scale=False) == Kind.FP8_PLAIN


@needs_fp8
def test_fp8_roundtrip_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    q, scale = quantize_fp8(w)
    assert q.dtype == torch.float8_e4m3fn
    back = expand(q, scale, torch.float32, "cpu")
    assert (back - w).abs().max() < 0.1 * w.abs().max()


def test_expand_casts_plain_tensors():
    x = torch.randn(4, 4)
    y = expand(x, None, torch.float16, "cpu")
    assert y.dtype == torch.float16
    assert torch.allclose(y.float(), x, atol=1e-2)


def test_expand_leaves_integer_tensors_alone():
    x = torch.arange(6)
    assert expand(x, None, torch.bfloat16, "cpu").dtype == torch.int64


def test_should_quantize_only_large_2d_weights():
    assert should_quantize("blocks.0.lin.weight", (128, 128))
    assert not should_quantize("blocks.0.lin.bias", (128,))
    assert not should_quantize("blocks.0.norm.weight", (128,))
    assert not should_quantize("blocks.0.lin.weight", (8, 8))


@needs_fp8
def test_stored_nbytes_estimates():
    key, shape = "blocks.0.lin.weight", (128, 128)
    n = 128 * 128
    assert stored_nbytes(key, shape, torch.float32, n * 4, None) == n * 4              # as stored
    assert stored_nbytes(key, shape, torch.float32, n * 4, torch.bfloat16) == n * 2     # cast down
    assert stored_nbytes(key, shape, torch.float32, n * 4, torch.float8_e4m3fn) == n + 4  # quantized
    assert stored_nbytes("x.bias", (128,), torch.float32, 512, torch.float8_e4m3fn) == 512  # not eligible
