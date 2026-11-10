# Loading

Reading a checkpoint into a meta-device model, straight into the right tiers.

::: trash_panda.offload.loader.stream_load
    options:
      show_docstring_description: false

Fills a meta-device model from a checkpoint. Pass a `plan`, or give the prefixes directly. Returns a
[`LoadReport`](#loadreport) and remembers the settings on the model for `install_offload()`.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | | the model on the meta device |
| `path` | | a file, a folder of shards, or an index json |
| `plan` | `None` | a `Plan` from `plan()` or `Plan.manual()` |
| `offload_prefixes` | `()` | tensors with these prefixes wait in RAM, in compact form |
| `resident_prefixes` | `()` | ... wait on the compute device, in compact form (wins over the above) |
| `disk_prefixes` | `()` | ... are read from the file when needed (wins over both) |
| `skip_prefixes` | `()` | entries that are neither loaded nor required |
| `defer_prefixes` | `()` | resident blocks left on the meta device, to be loaded later |
| `ram_in_memory` | `None` | read the RAM tier into memory instead of mapping it (default: the setting) |
| `ram_in_memory_prefixes` | `None` | read only the RAM-tier blocks with these prefixes into memory |
| `compute_dtype` | `"bf16"` | the dtype everything else is expanded to |
| `storage_dtype` | `"auto"` | the dtype the weights wait in |
| `device` | `None` | the compute device (default: CUDA if available, else CPU) |
| `key_map` | `None` | checkpoint name to model name |

```python
p = plan(model, "model.safetensors")
print(stream_load(model, "model.safetensors", plan=p))
```

## LoadReport

What `stream_load()` returns; `print(report)` shows a summary. Fields: `counts` and `nbytes` (per placement: `"normal"`,
`"gpu_raw"`, `"ram"`, `"disk"`), `kinds` (tensors per `Kind`), `unexpected` (checkpoint keys that are not in the model),
`skipped`, `aux_keys` and `warnings`.

## Checkpoint

`Checkpoint(path)` reads the headers of a (possibly sharded) safetensors checkpoint; tensors are read on demand. Use it
as a context manager (`with Checkpoint(path) as ck:`) to keep the files open while reading many tensors.

| Method | Returns |
|---|---|
| `keys()` | the tensor names, as a list |
| `info(key)` | the [`TensorInfo`](#tensorinfo) of a tensor |
| `keys_with_prefix(prefix)` | the names that start with `prefix` |
| `total_bytes()` | the size of all tensors in bytes |
| `get_tensor(key)` | a tensor (memory-mapped inside `with`) |
| `read_tensor(key)` | a tensor read into memory of its own, with one sequential read |
| `read_into(key, out)` | `None`; reads a tensor's bytes into a contiguous uint8 CPU tensor |
| `read(keys)` | a dict of tensors, opening each file once |

`key in ck` and `len(ck)` work too.

## TensorInfo

The header entry of one tensor (immutable): `key`, `dtype` (a safetensors string such as `"BF16"`), `shape`, `file`,
`start` and `end` (byte offsets), and the properties `numel`, `nbytes` and `torch_dtype`.

## Kind

How a tensor is stored. An enum (and a `str`): `Kind.CAST` (`"cast"`, an ordinary float dtype), `Kind.FP8_SCALED`
(`"fp8_scaled"`, fp8 with a scale tensor) and `Kind.FP8_PLAIN` (`"fp8_plain"`, fp8 without a scale).

## parse_dtype

`parse_dtype(value) -> Optional[torch.dtype]`

Turns `"bf16"`, `"fp16"`, `"fp32"`, `"fp8"` or a torch dtype into a torch dtype. `None` and `"auto"` give `None` (keep as
stored). An unknown name raises a `ValueError`.

## expand

`expand(raw, scale, compute_dtype, device) -> torch.Tensor`

Expands a stored tensor to `compute_dtype` on `device`. The compact tensor is moved first and converted there, so fewer
bytes cross the PCIe bus. `scale` is the tensor's fp8 scale, or `None`.

## quantize_fp8

`quantize_fp8(w, dtype=None) -> (q, scale)`

Per-tensor fp8 quantization (default `torch.float8_e4m3fn`), with `w` approximately `q * scale`. Lossy.
