# trash-panda

<p align="left">
  <img src="https://raw.githubusercontent.com/yla382/trashpanda/main/assets/raccoon_1.png" width="260" alt="A raccoon holding a circuit board in a trash can">
</p>

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

### Releasing a model

When a model is done, `release()` gives its memory back, so the next stage can be planned and loaded with it. Pass the
handle that `install_offload()` (or `offload_blocks()`) returned:

```python
from trash_panda import install_offload, plan, release, stream_load

p = plan(encoder, "encoder.safetensors")
stream_load(encoder, "encoder.safetensors", plan=p)
handle = install_offload(encoder)
embeddings = encoder(tokens)

release(encoder, handle)                         # its RAM and VRAM are free again

p = plan(unet, "unet.safetensors")               # this plan sees the memory the encoder gave back
```

`release()` removes the hooks, replaces every weight with an empty tensor on the meta device, and empties the GPU cache
and PyTorch's cache of pinned memory. The model cannot be run afterwards. `handle.remove()` alone does not do this: it
restores the blocks, but the weights stay in memory.

### Running a stage

`stage()` does the whole life of one model for you: it builds the model, plans, loads it, installs the hooks, and
releases it when the `with` block ends, also if the block raises:

```python
import torch
from trash_panda import stage

def build_unet():
    with torch.device("meta"):
        return MyUNet(config)

with stage(build_unet, "unet.safetensors", resident_blocks=12) as (unet, p):
    print(p)
    latents = denoise(unet, ...)
                                                 # here the UNet's RAM and VRAM are free again
```

The keyword arguments go to `plan()`, except `key_map` and `skip_prefixes`, which go to both `plan()` and
`stream_load()`. Use them when the checkpoint's names differ from the model's, or when part of the model should not be
loaded (for example the vision tower of a text encoder). The peak GPU memory of the block is remembered under `key` (by
default the checkpoint path) and used as `activation_vram` the next time the same key is planned in the same process.
The model cannot be used after the block, and adapters are not applied.

### Storage dtype

By default weights are kept exactly as stored in the file. Optionally, `storage_dtype="bf16"` casts waiting
weights down, and `storage_dtype="fp8"` quantizes large 2-D weights to fp8 on load (lossy; small tensors such
as norms are left alone).

## Benchmarks

Measurements on one machine, and a script to reproduce them with SDXL, are in [BENCHMARK.md](BENCHMARK.md) and
[examples/](examples/). They are single runs on a Windows desktop whose free memory changes from run to run, so they
are indicative only, and they do not show that trash-panda is faster than the other tools.

## What it does not do (yet)

* coordinating memory between several models that are loaded at the same time (for example a text encoder
  and a diffusion model). Each model is planned on its own from the memory that is free when you plan it, so
  plan them one after the other, or free the first model before loading the next
* a helper that runs the stages of a pipeline in order and reports memory before and after each one. `stage()` handles
  one model at a time, so call it once per model (see "Running a stage")
* full overlap of reading and computing for disk blocks: blocks waiting in RAM and on disk are prefetched through
  pinned buffers by a background thread, two blocks ahead by default, but the drive's speed sets a floor that no
  amount of prefetching removes (see "Known issues")
* Macs and other non-NVIDIA GPUs: the prefetching (a second CUDA stream, events and pinned buffers) is CUDA-only.
  On a Mac the library should fall back to the CPU, with plain copies and no prefetching, but this has not been tested
  on a Mac yet, and the Apple GPU (`mps`) has not been tried at all. A Mac's CPU and GPU also share one memory pool,
  so splitting a model between "GPU" and "RAM" does not free anything there
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
* **Disk blocks are limited by the speed of the drive.** Blocks on disk are read ahead of time (two blocks ahead by
  default, `lookahead=`), which hid about a third of the read time in testing, but a read takes much longer than
  one block's compute on a slow drive. On the test machine's SATA SSD (about 400 MB/s measured, with 1, 2, 4 or 8
  reader threads alike) a plan with 11 blocks (4.1 GB) on disk took about 16 s per step, against about 9.3 s with
  every streamed block in RAM. A deeper look-ahead (4) was no better and uses more VRAM. These are single runs
* **The first use of a RAM-tier block may be slow.** The RAM tier is mapped from the checkpoint file, so Windows may
  read a block from disk the first time it is used. In one profiled run a few blocks made the main thread wait
  1 to 3 s each, and a text encoder that runs once copied its blocks at about disk speed. The cause is not
  confirmed yet (see the roadmap)
* **Tested on one setup so far:** a 12B diffusion model (fp8 with scales) on Windows with an RTX 3060 (12 GB) and
  16 GB of RAM. Other models, GPUs and operating systems are not covered yet. The unit tests run on the CPU, and
  `examples/sdxl_benchmark.py` has a path for machines without CUDA (the Apple GPU or the CPU) that is untested

## Out of scope for now

The current direction is sequential model chaining: each model of a pipeline is loaded, run and released before the
next one is planned (for example text encoder, then diffusion model, then VAE). The following are not part of that
direction, or come after it, and are not planned for now:

* **ControlNet**: an extra network that steers a diffusion model with a guide image (edges, pose, depth). It runs
  alongside the diffusion model at every denoising step, so both models must be loaded at the same time
* **Speculative decoding**: a small draft model proposes several tokens and the large language model verifies them in
  one pass. Both models run throughout the generation, so both must be loaded at the same time
* **GGUF models**, possibly through `llama-cpp-python` as a separate backend
* **Computing the RAM tier on the CPU** instead of copying it to the GPU, as llama.cpp does, for token-by-token
  generation
* **Batching for language models**, so one weight copy serves many sequences
* **Multi-GPU support**
* **A KV-cache budget for language models**, from the context length you intend to use
* **4-bit and 8-bit storage** (for example NF4 with `bitsandbytes`, or our own implementation), to fit more of the
  model in RAM and on the GPU and halve the traffic
* **fp8 checkpoints with block-wise scales** (the `weight_scale_inv` layout used on the Hugging Face Hub), with one
  scale per 128 x 128 block of weights. It touches the scale lookup, the expansion and the staging of the scales
* **LoRA and other adapters** on offloaded blocks
* **A `from_pretrained`-style helper for `transformers` models** (find the weight files, build on the meta device with
  real buffers, plan, load and install the hooks in one call), ignoring buffers that are not in the checkpoint (rotary
  tables, position ids) instead of failing, and making `model.dtype`, `model.device` and `model.to(...)` behave on
  offloaded models so `generate()` and `pipeline()` work
* **A memory-mapped disk tier** as an experiment: map the checkpoint once and let the operating system keep recently
  used blocks cached, instead of an explicit split between RAM and disk. To be measured against explicit read-ahead,
  because a least-recently-used cache can thrash when the model is read in a cycle
* **Cheaper expansion**: reuse preallocated buffers and fuse the fp8 to bf16 conversion and scaling, instead of
  creating new tensors for every weight on every pass
* **A cost model** that chooses between GPU, RAM and disk from measured disk and PCIe speed
* **A command-line interface**, for example `trash-panda plan model.safetensors` to print a plan without writing a
  script, and a hardware report for bug reports
* **Visualizing the data flow**: a static report or trace file, built from the profile data, that shows each block
  moving between tiers, the overlap of copying and compute, and the stalls. A live interface would be a later step
* **Logging**: an optional log through Python's `logging` module, and an optional structured file (for example
  `TRASH_PANDA_LOG=run.jsonl`) with the plan, memory and timings of a run. The visualization and a comparison of two
  runs could read it

These may be considered in the future. They are not part of the immediate plan.

## Roadmap

The current focus is model chaining: running the models of a pipeline one after another while the memory moves from
one to the next as it frees up. The memory a stage no longer needs is released while the computation continues, and the
next model is loaded into that space as it appears, instead of waiting for a stage to finish and release everything
before the next one starts loading. `release()` does the whole-model version by hand today. The items below serve
that focus, roughly in the order they are planned. Nothing here is promised or scheduled.

**Speed**
* read the RAM tier into memory ahead of use. In the profile of an all-RAM plan, 21 of the 140 streamed blocks were
  late by about 1 s each, which is roughly the time to read one block from the drive, and each of the slowest was late
  in only 1 of 7 passes. That suggests nearly all the waiting is in the first pass, when the file-backed RAM tier is
  read from disk on first use. If so, steady steps may be much faster than the average the progress bar shows (about
  6 s against 9.3 s) and the first step much slower. This is an inference, not a measurement: the profile now lists
  the time and the waiting of every pass (`TRASH_PANDA_PROFILE=1`), which will confirm or reject it. If it holds, a
  background thread would touch the RAM-tier pages once when the hooks are installed, keeping the tier file-backed
* prefetch models that run a single pass (a text encoder, a VAE) from their first block: the prefetcher learns the
  block order during the first pass, so a one-pass model is not prefetched at all. Hand it the order of the blocks
  when the hooks are installed, and let the order seen at run time correct it
* the number of pinned buffers as a tested option (`TRASH_PANDA_PINNED_BUFFERS`, 3 or 4 instead of 2), if the
  profile's "worker waited for a free pinned buffer" line shows the worker is held up by them
* overlap the stages of a pipeline: while the text encoder runs, load as much of the diffusion model as there is
  spare VRAM and RAM for, then release the encoder and start sampling with what is already loaded while the rest keeps
  loading; the same for the VAE during sampling. It needs the staged planning above, a plan that predicts the memory
  that will be free after a stage, and blocks that can change from streamed to resident while running. Preloading
  would also compete with the encoder for the drive when the encoder reads from disk. An idea, not a design

**Planner**
In this order, because the silent slowdowns seen so far (GPU memory spilling into shared memory at about 11.6 to
11.8 GB, a headroom that was too thin, an overflow after sampling at 2K) all come from the planner knowing too little:
* plan a pipeline in stages (encode, denoise, decode) so memory used by an earlier stage can be released.
  `release()` does the releasing by hand now. In single runs, releasing the text encoder before the diffusion model was
  planned gave the planner several GB more RAM, and releasing the diffusion model before the VAE was built stopped the
  decode of a 2048 x 1024 image from spilling into shared GPU memory (watched by eye, not logged). `stage()` builds,
  plans and loads one model around your code, releases it afterwards, and records its activation size. It runs the
  text encoder (a `transformers` model), the diffusion model and the VAE of the Krea2 pipeline today. Left is a smaller
  step such as `release_gpu()` / `restore_gpu()` on the handle, which moves the resident blocks off the GPU and back
  while keeping the hooks. The cost of putting them back for each image, and the RAM they need meanwhile, have not been
  measured

**Compatibility**
* `diffusers` pipelines: build the large components with `stream_load` and hand them to the pipeline class, so
  `from_pretrained` does not load everything into RAM first

**Reliability and project**
* investigate sudden exits: rarely, and only when RAM is already high (about 12 GB in use on a 16 GB machine)
  before a large model is loaded, the process ends with no Python error, usually while the weights are being
  loaded. It is probably the operating system ending the process or a crash in native code, but this has not been
  confirmed. Next time, record the exit code and run with `PYTHONFAULTHANDLER=1`, and check whether it also happens
  from a clean memory state
* find out where a sudden exit goes wrong: when a run ends without a Python error, check the exit code right
  afterwards (`$LASTEXITCODE` in PowerShell) and run with `PYTHONFAULTHANDLER=1`. A large negative code such as
  `-1073741819` is a crash in native code, while a normal code points to the process being ended by the system. Also
  print the memory in use and the available commit (RAM plus page file) before loading, so a failed run shows what
  the machine had left
* tests on tiny `transformers` and `diffusers` models on the CPU, in CI
* tests on more models, GPUs and operating systems (Linux in particular), with benchmarks published here. A first
  test on a Mac is planned: the unit tests and `examples/sdxl_benchmark.py --size 512 --steps 10` in the baseline and
  trash-panda modes, to see whether it installs and runs on the CPU or the Apple GPU. Real Mac support, if there is
  demand, would be a separate piece of work (`mps`, unified memory, fp8 on the Apple GPU)
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
