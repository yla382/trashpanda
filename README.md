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

### Fitting the GPU to your workload

How many blocks can stay on the GPU depends on how much memory your activations need, which grows with image
size and cannot be known before the model runs. By default the planner keeps a fixed `headroom_vram="1.5GB"` free.
To use a measured value instead, run your real call once and read the peak:

```python
from trash_panda import install_offload, measure_activation_vram, plan, stream_load

p = plan(model, path)
stream_load(model, path, plan=p)
install_offload(model)

need = measure_activation_vram(lambda: run_my_generation(), p)   # include everything that runs, such as a VAE decode
print(f"activations need {need / 1e9:.2f} GB")

p = plan(model, path, activation_vram="2.6GB")                   # next time: fill the rest of the GPU
```

The value is per model and per input size, so measure again if you change the resolution.

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
* full overlap of copying and computing: blocks waiting in RAM are prefetched through pinned buffers, but the
  copy into those buffers still runs on the main thread, and disk blocks are not prefetched at all
* int8 / 4-bit formats (fp8 and ordinary float dtypes only)
* LoRA/LoKr adapters, multi-GPU, model definitions

## Known issues

* **Automatic plans change from run to run.** `plan()` uses the RAM that is free at the moment you call it, and
  free RAM varies. On a 16 GB machine the same script gave a RAM budget of 6.2 GB in one run and 8.0 GB in
  another, which moved several blocks between RAM and disk
* **Load other models before you plan.** If a second model (a text encoder, a VAE) is loaded after `plan()`, the
  memory the plan counted on is taken. In RAM, Windows pages the weights out and the first step is very slow. On the
  GPU, too many blocks stay resident and the overflow spills into shared memory, which can make every step several
  times slower. Create the other models first so the plan sees the memory that is really left
* **The default RAM reserve is a rule of thumb.** `min_free_ram="auto"` keeps 8% of the machine's RAM free (between
  1 and 4 GB, so about 1.3 GB on a 16 GB machine). If your other programs use a lot of memory, or very little, set
  `min_free_ram` yourself, for example `"500MB"` or `0`
* **The default VRAM headroom is a fixed guess.** `headroom_vram="1.5GB"` does not know how much memory your
  activations need (it grows with image size), and measuring it is a manual step for now (see "Fitting the GPU to
  your workload"). If too many blocks are resident, Windows silently spills GPU memory into shared system memory
  and every step becomes several times slower, with no error. Pass a measured `activation_vram`, lower
  `resident_blocks`, or set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control
  Panel so it fails loudly
* **Disk blocks are read one after another with the compute.** Nothing overlaps, so each disk block adds its full
  read time to every step. This is slow on a SATA SSD: with 9 blocks (3.2 GB) on disk, a step took 17.2 s instead of
  10.9 s, which is about the time to read 3.2 GB at 500 MB/s
* **Tested on one setup so far:** a 12B diffusion model (fp8 with scales) on Windows with an RTX 3060 (12 GB) and
  16 GB of RAM. Other models, GPUs and operating systems are not covered yet

## Roadmap

Roughly in the order they are planned. Nothing here is promised or scheduled.

**Speed**
* prefetching, step 2 (rest): move the copy into the pinned buffers to a background thread, and count the pinned
  buffers (about 1 GB) in the planner's RAM budget
* prefetching, step 3: read disk blocks ahead of time into the pinned buffers, with a look-ahead of more than one
  block for slow drives
* a per-tier timer (copy, wait and compute time) behind a debug flag, to see where each step's time goes
* cheaper expansion: reuse preallocated buffers and fuse the fp8 to bf16 conversion and scaling, instead of
  creating new tensors for every weight on every pass
* a cost model that chooses between GPU, RAM and disk from measured disk and PCIe speed

**Planner**
* measure the activation size automatically (and remember it per model and input size) instead of asking you to run
  `measure_activation_vram` once
* take memory held by other loaded models into account, or at least warn about it
* plan a pipeline in stages (encode, denoise, decode) so memory used by an earlier stage can be released
* warn when a plan fills almost all of the GPU, since the overflow is silent and slow
* a KV-cache budget for language models, from the context length you intend to use

**Formats**
* 4-bit and 8-bit storage (for example NF4 with `bitsandbytes`, or our own implementation), to fit more of the
  model in RAM and on the GPU and halve the traffic
* fp8 checkpoints with block-wise scales (the `weight_scale_inv` layout used on the Hugging Face Hub)
* GGUF models, possibly through `llama-cpp-python` as a separate backend

**Compatibility**
* `from_pretrained`-style helper for `transformers` models: build on the meta device with real buffers, find the
  weight files (Hub or local), translate old checkpoint names, plan, load and install the hooks in one call
* ignore buffers that are not in the checkpoint (rotary tables, position ids) instead of failing
* make `model.dtype`, `model.device` and `model.to(...)` behave on offloaded models, so `generate()` and
  `pipeline()` work
* `diffusers` pipelines: build the large components with `stream_load` and hand them to the pipeline class, so
  `from_pretrained` does not load everything into RAM first
* LoRA and other adapters on offloaded blocks
* compare speed and memory against the offloading built into `diffusers` and `accelerate`

**Larger ideas**
* compute the RAM tier on the CPU instead of copying it, as llama.cpp does, for token-by-token generation
* batching for language models, so one weight copy serves many sequences
* multi-GPU support

**Reliability and project**
* `release()` / `stage()` helpers for cleaning up between models
* tests on tiny `transformers` and `diffusers` models on the CPU, in CI
* tests on more models, GPUs and operating systems (Linux in particular), with benchmarks published here
* a documentation site (GitHub Pages) with guides and an API reference
* a final project name, and a contributor agreement before outside code is accepted

## Development

```
pip install pytest psutil
pytest
```

Tests run on the CPU with a tiny fake model.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
