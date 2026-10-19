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

* coordinating memory between several models that are loaded at the same time (for example a text encoder
  and a diffusion model). Each model is planned on its own from the memory that is free when you plan it, so
  plan them one after the other, or free the first model before loading the next
* one-call cleanup after a model is done (a `release(model)` helper and a pipeline `stage()` helper that remove
  the hooks, drop the weights, empty the GPU cache and report memory before and after). For now, free memory
  yourself: `handle.remove()`, `del model`, `gc.collect()`, `torch.cuda.empty_cache()`; for the strictest cleanup
  (system RAM in particular), run heavy stages in their own process
* prefetching the next block while the current one computes
* int8 / 4-bit formats (fp8 and ordinary float dtypes only)
* LoRA/LoKr adapters, multi-GPU, model definitions

## Known issues

* **Automatic plans change from run to run.** `plan()` uses the RAM that is free at the moment you call it, and
  free RAM varies. On a 16 GB machine the same script gave a RAM budget of 6.2 GB in one run and 8.0 GB in
  another, which moved several blocks between RAM and disk
* **Load other models before you plan.** If a second model (a text encoder, a VAE) is loaded after `plan()`, the
  RAM the plan counted on is taken, Windows pages the weights out, and the first step is very slow. Create the other
  models first so the plan sees the memory that is really left
* **The default RAM reserve is conservative.** `min_free_ram="3GB"` keeps 3 GB free, which on a 16 GB machine can
  send blocks to disk that would have fit. Try `"500MB"` to `"1GB"`
* **The default VRAM headroom is a fixed guess.** `headroom_vram="1.5GB"` does not know how much memory your
  activations need (it grows with image size). If too many blocks are resident, Windows silently spills GPU memory
  into shared system memory and every step becomes several times slower, with no error. Lower `resident_blocks`, or
  set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control Panel so it fails loudly
* **Disk blocks are read one after another with the compute.** Nothing overlaps, so each disk block adds its full
  read time to every step. This is slow on a SATA SSD
* **Tested on one setup so far:** a 12B diffusion model (fp8 with scales) on Windows with an RTX 3060 (12 GB) and
  16 GB of RAM. Other models, GPUs and operating systems are not covered yet

## Next steps

* planner: replace the fixed 3 GB RAM reserve with one that scales with the machine
* planner: fill the GPU from a measured activation size instead of a fixed headroom
* planner: take memory held by other loaded models into account, or at least warn about it
* prefetch the next block (read from disk, copy from RAM) while the current block computes, and use pinned memory
* a cost model that chooses between GPU, RAM and disk from measured disk and PCIe speed
* `release()` / `stage()` helpers for cleaning up between models

## Development

```
pip install pytest psutil
pytest
```

Tests run on the CPU with a tiny fake model.

## License

Apache-2.0
