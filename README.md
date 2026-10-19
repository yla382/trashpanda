# trash-panda

Run models bigger than your GPU: stream weights layer by layer across GPU, RAM and disk.

> **Status: early (Phase 1).** The API may change. The package name is temporary.

## Why

Loading a big checkpoint the usual way needs two full copies in RAM, and a model that does not fit in VRAM
simply fails. trash-panda builds the model without allocating weights (meta device), reads the checkpoint
tensor by tensor, and keeps each block of the model wherever it fits best:

| Tier | Where the block waits | Cost per forward pass |
|---|---|---|
| `gpu`  | on the GPU, in compact form (for example fp8) | only the expansion to bf16 |
| `ram`  | in system RAM, in compact form | a transfer over PCIe |
| `disk` | in the checkpoint file | a file read |

Only one block at a time is expanded to the compute dtype, so peak VRAM is roughly one block plus your
activations, whatever the model size.

It works with any floating-point checkpoint: fp32, fp16 and bf16 are streamed as they are (no quantization
needed), and fp8 (with or without scale tensors) is kept compact so more fits.

## Install

Install PyTorch first (the build that matches your GPU, see https://pytorch.org), then:

```
pip install dumpsterpanda
```

The package is imported as `trash_panda`:

```python
import trash_panda
```

To try the latest development version straight from GitHub instead:

```
pip install git+https://github.com/yla382/trashpanda.git
```

### Why is it called `dumpsterpanda` on PyPI? 🦝😢

The project is called **trash-panda**, but the name [`trashpanda`](https://pypi.org/project/trashpanda/#description)
was already taken on PyPI, and PyPI wouldn't let `trash-panda` sit that close to it either. So the raccoon moved
one dumpster over. The import is still `trash_panda`.

## Use

```python
import torch
from trash_panda import plan, stream_load, install_offload

with torch.device("meta"):                      # the structure only, no weights in memory
    model = MyModel(config).eval()

p = plan(model, "model.safetensors")            # what fits where, with warnings
print(p)
print(stream_load(model, "model.safetensors", plan=p))   # lazy load straight into the right tiers
install_offload(model)                          # hooks that stream blocks during forward

out = model(x)
```

### Automatic or manual

`plan()` decides from the checkpoint's sizes and the free memory on your machine. Everything it decides can be
overridden:

```python
from trash_panda import Plan, Tier, plan

p = plan(model, path, resident_blocks=12, max_vram="9GB", min_free_ram="4GB", compute_dtype="bf16")
p.assign(Tier.DISK, indices=[27])               # edit it afterwards

p = Plan.manual(resident=["blocks.0.", "blocks.1."], streamed=["blocks.2.", "blocks.3."])   # or write it by hand
```

Or skip plans entirely and give prefixes directly (the low-level API):

```python
stream_load(model, path, offload_prefixes=["blocks."], resident_prefixes=["blocks.0."], compute_dtype="bf16")
offload_blocks(model.blocks, device="cuda", compute_dtype=torch.bfloat16)
```

### Storage dtype

By default weights are kept exactly as stored in the file. Optionally, `storage_dtype="bf16"` casts waiting
weights down, and `storage_dtype="fp8"` quantizes large 2-D weights to fp8 on load (lossy; small tensors such
as norms are left alone).

## What it does not do (yet)

* prefetching the next block while the current one computes
* int8 / 4-bit formats (fp8 and ordinary float dtypes only)
* LoRA/LoKr adapters, multi-GPU, model definitions

## Development

```
pip install pytest psutil
pytest
```

Tests run on the CPU with a tiny fake model.

## License

Apache-2.0
