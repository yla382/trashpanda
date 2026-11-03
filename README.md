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

with stage(build_unet, "unet.safetensors", resident_blocks=12) as (unet, p, _):
    print(p)
    latents = denoise(unet, ...)
                                                 # here the UNet's RAM and VRAM are free again
```

The block gets the model, its plan and the handle that `install_offload()` returns. The keyword arguments go to
`plan()`, except `key_map` and `skip_prefixes`, which go to both `plan()` and
`stream_load()`. Use them when the checkpoint's names differ from the model's, or when part of the model should not be
loaded (for example the vision tower of a text encoder). The peak GPU memory of the block is remembered under `key` (by
default the checkpoint path) and used as `activation_vram` the next time the same key is planned in the same process.
The model cannot be used after the block, and adapters are not applied.

The prefetcher stages the next blocks ahead of the one that runs, so it has to know the order the blocks run in. It
used to learn that during the first pass, which left the first pass without prefetching. Now the order is known
before the first call:

* With `order=None` (the default), `stage()` looks for `<checkpoint name>_order.json` next to the checkpoint, or in the
  folder `order_file_destination`, and uses the order in it. If there is no file, it uses the order the model defines
  its blocks in. When the model is released, the order the blocks ran in, every block that ran, is written to that
  file. This happens when there was no file, or when the order in it turned out wrong, and the next run starts with
  the real order. A file that names a block the model does not have is ignored and replaced, and a folder that cannot
  be written gives one warning.
* `forward()` decides the real order, and it can differ from the defined one: a diffusion model can define its four
  text blocks after its 28 main blocks, and run them first. When the first blocks that run do not follow the order
  given, a warning appears once. A run without a file would warn on the first run only.
* `order=["txtfusion.layerwise_blocks.0", ...]` names the blocks that run first, in that order, and the others follow
  in the plan's order. With it, the file is neither read nor written.

The file is named after the checkpoint file, so two checkpoints with the same file name (diffusers models are
usually all `diffusion_pytorch_model.safetensors`) share a name if `order_file_destination` is one folder, and then
overwrite each other's file. With the default location, next to each checkpoint, this does not happen.

By default `stage()` loads every block before the first call. With `overlap=True` it does not wait for the resident
blocks: they are loaded one at a time, in the order the blocks run, on a worker thread, and the model starts as soon as
its first block is on the GPU. A block that starts before its weights have arrived waits for them. The weights outside
the blocks, the blocks that wait in RAM and the blocks on disk are handled as before. If the loading fails, the call
that needs the missing block raises an error that names it. The peak GPU memory is not remembered for the next plan
when `overlap=True`, because the load falls inside the measured part of the run. It is off by default. On the test
machine it gave no measurable gain (see "Settings and environment variables"), and it is groundwork for running one
model's loads behind another model's compute rather than a speed-up on its own.

### Storage dtype

By default weights are kept exactly as stored in the file. Optionally, `storage_dtype="bf16"` casts waiting
weights down, and `storage_dtype="fp8"` quantizes large 2-D weights to fp8 on load (lossy; small tensors such
as norms are left alone).

## Settings and environment variables

Every tuning setting is kept, with its default, in `trash_panda.offload.config`. Each can be changed with an
environment variable named `TRASH_PANDA_<SETTING>` in capitals, for example `TRASH_PANDA_LOOKAHEAD=0`. The variable is
read when a function runs, not when the library is imported. An argument passed to a function wins over the
environment, and the environment wins over the default. Switches take `1` / `0`, `true` / `false`, `yes` / `no` or
`on` / `off`.

In PowerShell a variable lasts for that window only:

```
$env:TRASH_PANDA_PROFILE = "1"
Remove-Item Env:TRASH_PANDA_PROFILE
```

In bash, use `export TRASH_PANDA_PROFILE=1` and `unset TRASH_PANDA_PROFILE`.

| Variable | Default | What it does |
|---|---|---|
| `TRASH_PANDA_PROFILE` | `0` | Print where loading and prefetching spent their time (see "Reading the profile") |
| `TRASH_PANDA_LOOKAHEAD` | `2` | Blocks ahead whose weights are copied while one computes. `0` turns prefetching off |
| `TRASH_PANDA_PINNED_BUFFERS` | `2` | Page-locked host buffers that blocks are copied through |
| `TRASH_PANDA_BLOCKS_IN_FLIGHT` | `2` | The GPU may run at most this many blocks behind the copies |
| `TRASH_PANDA_RAM_IN_MEMORY` | `0` | Read the tensors of the blocks that wait in RAM into memory of their own while loading, instead of keeping them mapped from the checkpoint file |
| `TRASH_PANDA_RAM_READ_THREADS` | `1` | Threads that do that reading. Only used with `TRASH_PANDA_RAM_IN_MEMORY` |
| `TRASH_PANDA_MIN_FREE_RAM` | `auto` | RAM kept free for the system: `auto` (the three settings below) or a size such as `2GB` |
| `TRASH_PANDA_RAM_RESERVE_FRACTION` | `0.08` | With `auto`, the share of the machine's RAM kept free for the system. The result stays between the minimum and the maximum below |
| `TRASH_PANDA_RAM_RESERVE_MIN_GB` | `1.0` | With `auto`, the least RAM kept free, in GiB, however small the share above comes out |
| `TRASH_PANDA_RAM_RESERVE_MAX_GB` | `4.0` | With `auto`, the most RAM kept free, in GiB, however large the share above comes out |
| `TRASH_PANDA_RAM_RESERVE_UNKNOWN_GB` | `2.0` | The reserve when the machine's RAM cannot be read |
| `TRASH_PANDA_COMMIT_WARN_FRACTION` | `0.9` | Warn when the blocks in RAM need more than this share of the Windows commit limit |
| `TRASH_PANDA_HEADROOM_VRAM` | `1.5GB` | VRAM kept free for activations when none was measured |
| `TRASH_PANDA_STREAM_BUFFER_FACTOR` | `1.3` | Working area for a streamed block, as a multiple of its expanded size |
| `TRASH_PANDA_ACTIVATION_MARGIN_FRACTION` | `0.05` | Safety margin added to a measured activation size, as a share of the VRAM. It is never smaller than the minimum below |
| `TRASH_PANDA_ACTIVATION_MARGIN_MIN_GB` | `0.5` | The smallest safety margin added to a measured activation size, in GiB |
| `TRASH_PANDA_VRAM_FILL_WARN_FRACTION` | `0.9` | Warn when the planned GPU use is above this share of the free VRAM |
| `TRASH_PANDA_FIND_BLOCKS_MIN_FRACTION` | `0.01` | A group of layers must hold at least this share of the model's bytes to count as the blocks |
| `TRASH_PANDA_FIND_BLOCKS_MIN_BLOCKS` | `2` | The least number of blocks a group of layers must have to count as the blocks |

Two options are arguments of `stage()`, not environment variables: `overlap=True` (see "Running a stage") and `order=`
with `order_file_destination=`.

### What we measured

All numbers come from one machine (Windows 11, an RTX 3060 with 12 GB, 16 GB of RAM and a SATA SSD) and one pipeline:
a text encoder, a 12B diffusion model with fp8 weights and scale tensors, and a VAE, making one 1024 x 1024 image in 7
steps. The diffusion model's plan was 12 blocks (5.21 GB) on the GPU and 20 blocks (7.29 GB) in RAM, and the RAM in use
at the start was 5.5 to 6.5 GB. They are single runs. Identical runs differed by 2 to 3 s in total time, so a
difference smaller than that is not a result.

* **The drive sets the pace of every cold read.** The text encoder's blocks, the diffusion model's resident blocks and
  its RAM tier all ran at 0.36 to 0.42 GB/s the first time they were read, however they were read (the memory-mapped
  copy, one reader thread, two reader threads). A run reads about 16 GB cold, which is roughly 40 s of a total of about
  95 s. The first step of the diffusion model is the first read of its RAM tier: 20 to 22 s, against 6.3 s for every
  later step, and 12.5 to 15 s of that difference is the main thread waiting for blocks. About 2.8 s of the first step
  is one-time warm-up in its first three blocks.
* **Defaults:** 94.9, 95.6, 96.5 and 98.3 s in four runs.
* **`TRASH_PANDA_RAM_IN_MEMORY=1`:** the first step takes 9.0 s and the sampling 47 s instead of 61 s, because the read
  of the RAM tier has moved to the load: 7.29 GB in 20 to 23 s (0.31 GB/s with one thread, 0.34 to 0.36 with two). The
  total was 104 to 108 s, which is 8 to 12 s longer than with the defaults. The read is slower than the memory-mapped
  copy, and it no longer overlaps with the compute of the first step. Two threads read 16% faster than one; more were
  not tried, because both are close to what the drive delivers. The tensors are private memory, so they count against
  the Windows commit limit while the model is loaded.
* **`stage(overlap=True)`:** 96.5 to 98.8 s in four runs, against 94.9 to 98.3 s without it, and 109.3 s together with
  `TRASH_PANDA_RAM_IN_MEMORY=1` against 105.0 s. It cannot help much here. The compute of a model is small next to the
  load of its resident blocks: a resident block of the diffusion model takes about 1.1 s to load and 0.2 s to run, so
  only about 3 s of compute can overlap with the load, and the text encoder computes for 0.2 s against an 11.7 s load.
  The loader thread also reads with plain seek-and-read, which ran at 0.25 to 0.28 GB/s against 0.41 GB/s for the
  memory-mapped reads that `stream_load()` does for the same blocks.
* **`TRASH_PANDA_LOOKAHEAD`:** a depth of 4 was no better than 2 for disk blocks and uses more VRAM.
* **A read-ahead of the RAM tier** through ordinary file reads, ahead of the copy, made the first step slower (34.7 s
  against 21 to 22 s) and was removed: the pages it read did not serve the memory-mapped copy.
* **Memory after a release.** On Windows, PyTorch's own CPU allocator keeps blocks after they are freed (3 GB of
  tensors freed, working set unchanged), so memory read into `torch.empty` tensors still counted as taken when the next
  model was planned (a RAM budget of 2.0 to 2.6 GB, against about 9 GB). `TRASH_PANDA_RAM_IN_MEMORY` allocates through
  Python's allocator instead, which gives the memory back at once, and after the fix the next plan saw 8.2 to 9.3 GB.

### Current recommendation

* Leave every setting at its default. The defaults were the fastest total in these measurements.
* Leave `TRASH_PANDA_RAM_IN_MEMORY` off unless you want steady steps from the first one, and have the RAM and the commit
  headroom for it (about 7 GB of private memory on the test pipeline). It does not make a single run shorter.
* Leave `overlap` off. It is groundwork for starting the next model's loads behind compute that is long enough to hide
  them (see the roadmap), not a speed-up for one pipeline.
* A slow first step is the first read of the RAM tier from the drive. The profile shows it as a first pass with a low
  GB/s in the RAM copies and a large "waited". Only a faster drive shortens it.
* Set `TRASH_PANDA_PROFILE=1` when you want to see where the time goes.

### Reading the profile

With `TRASH_PANDA_PROFILE=1`, each stage prints a report when its hooks are removed:

* `stream_load took X s (...)`: the time of the load, by tier. With `overlap=True` the blocks that are loaded while the
  model runs are not counted.
* `read N GB of streamed-block tensors into memory in X s (Y GB/s, T threads)`: only with `TRASH_PANDA_RAM_IN_MEMORY`.
* `per pass`: for each pass through the blocks, its time, the time the main thread waited for staged blocks, and the GB
  and GB/s of RAM copies and disk reads done meanwhile. A first pass much slower than the others, with a low GB/s, is
  cold reads.
* `first pass, main thread time by block`: the six slowest blocks of the first pass, with their time split into waiting
  for the copy (`staged`), waiting for the GPU (`gpu`) and the rest (launching work, one-time warm-up).
* `loader thread`: with `overlap=True`, how long the resident blocks took to load and how long the main thread waited
  for them.
* `memory while releasing`: at five points of `release()`, the RAM the machine has free, the working set and the private
  bytes of the process, and the CPU tensors of 1 MB or more that are still alive. If more than 0.1 GB stays alive it
  also names what holds the biggest. Read the working set, not the private bytes: the private bytes include GPU memory
  that Windows charges to the process. The profile scans the live objects at these five points, which costs a little
  time.

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
* **The first step of a model with a RAM tier is slow.** The RAM tier is mapped from the checkpoint file, so each block
  is read from the drive the first time it is used. On the test machine that was 7.3 GB at about 0.4 GB/s: the first
  step of a diffusion model took 20 to 22 s against 6.3 s for the others, and 12.5 to 15 s of the difference was the
  main thread waiting for blocks. Every cold read on that machine ran at 0.36 to 0.42 GB/s, however it was read, so
  only a faster drive shortens it. `TRASH_PANDA_RAM_IN_MEMORY=1` moves the read to the load, before the first step, but
  the total time is not shorter (see "Settings and environment variables")
* **Checkpoints with the same file name share an order file name.** The order file is named after the checkpoint file
  plus `_order.json`. Diffusers models are usually all called `diffusion_pytorch_model.safetensors`, so with one shared
  `order_file_destination` they overwrite each other's file. With the default location, next to the checkpoint, this
  does not happen. A file that names a block the model does not have is ignored, so nothing breaks, but the file is
  rewritten on every run
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
* **A ComfyUI custom node** that loads and runs a model through trash-panda inside a ComfyUI workflow
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
* a better first order for a model that has never run. The first run uses the order the model defines its blocks in,
  which `forward()` can override (a diffusion model can define its four text blocks after its 28 main blocks and
  run them first), so that run warns and is prefetched less well. The effect measured so far is small: the first step
  of that model went from 25 s to 22 s, which is within the variation between runs. Two ways to know the real order
  before the first call: a helper that records it without a real run (a call on the meta device with small inputs
  that the caller gives), or reading the source of `forward()` with the `ast` module, which fails on code that is not
  plain
* the loader thread of `stage(overlap=True)` reads with plain seek-and-read, which ran at 0.25 to 0.28 GB/s against
  0.41 GB/s for the memory-mapped reads that `stream_load()` does for the same blocks (12 blocks, 5.21 GB: 18.5 to
  21.2 s against 12.7 s). Memory-mapped reads in the loader would remove most of that gap. The loader would have to open
  and keep its own mapping for the whole load, because mapping the whole file for every read charges the Windows commit
  limit (see the comment in `checkpoint.py`). What `overlap` can save on the test pipeline stays bounded by the compute
  that can overlap with the load, about 3 s, so this matters mostly for chaining. Two other parts of `overlap` are open:
  * the peak GPU memory is not remembered with `overlap=True`, because the load falls inside the measured part of the
    run. It has to be measured after the load, or with the weights subtracted
  * the order of the weights outside the blocks (embeddings, input and output layers) in the order file, if they
    should be loaded in the order they are needed; today they are loaded before the first call
* the number of pinned buffers as a tested option (`TRASH_PANDA_PINNED_BUFFERS`, 3 or 4 instead of 2), if the
  profile's "worker waited for a free pinned buffer" line shows the worker is held up by them

**Planner**
`stage()` and `release()` run the models of a pipeline one at a time today (see "Running a stage"), and `stage()` runs
the text encoder, the diffusion model and the VAE of the Krea2 pipeline. In single runs, releasing the text encoder
before the diffusion model was planned gave the planner several GB more RAM, and releasing the diffusion model before
the VAE was built stopped the decode of a 2048 x 1024 image from spilling into shared GPU memory (watched by eye, not
logged). What is left is to let memory move between stages while they run. The steps below are small, each can be
tested on its own, and they are listed in the order they depend on one another:
* announce each release as an event on a queue: which block, how many bytes, in which tier, and a CUDA event that
  marks when the memory can be reused (a tensor freed in Python may still be read by queued GPU work). The hook only
  puts the event on the queue, so the compute thread never waits for whoever reacts to it. Whether the allocator
  reuses memory freed on one stream for copies on another needs a test
* find out whether a RAM-tier block can be given back before the model is released. The RAM tier is the checkpoint
  file mapped into memory, and it is not known whether part of the mapping can be dropped, or whether the blocks would
  have to be separate tensors. A small script that maps a checkpoint, touches the blocks, drops them one at a time and
  watches the free RAM would settle it
* start the next model's reads behind compute that is long enough to hide them. The measurements show that reads and
  compute only overlap when something computes while the drive works: on the test pipeline the text encoder computes for
  0.2 s against an 11.7 s load, and the diffusion model's reads come before any of its compute, so there is nothing to
  hide them behind. A pipeline with a language model that rewrites the prompt, or one that makes several images, has
  stages that compute for a long time while the drive is idle. `TRASH_PANDA_RAM_IN_MEMORY` is the memory model for it (a
  model read ahead before it starts cannot have its pages dropped by the system), and `stage(overlap=True)` (a loader
  thread, ready marks, the order file) is the groundwork
* load the next model's blocks into the VRAM that a final pass has freed. A block would move in its stored form (fp8,
  for example) and be expanded in the stream buffer only when it runs. Compare the next stage's time with its time
  when it is loaded after the release. A background thread would take the events from the queue and load the next
  planned blocks, waiting for each event's CUDA event first. Loading early would compete with the running stage for the
  drive when that stage reads from disk
* plan a stage with the memory that is expected to be free when it starts. `plan()` uses the memory that is free when
  it runs, so it needs a way to be given the memory that the running stage will release. The stage would be planned
  once for that end state, with the activation headroom from an earlier measurement, and the loading above would only
  fill the plan in as memory frees up; if the free memory at the start differs from the prediction, it is planned once
  more then. Planning again on every event is the alternative if this proves too coarse
* let the depth of the staging follow the profile: a look-ahead above 2 only where the main thread waits for blocks,
  and the spare VRAM to the next model where it does not
* one pinned pool for the whole pipeline, sized for the largest block of any stage and kept until the pipeline ends,
  instead of one pool per model
* `release_gpu()` / `restore_gpu()` on the handle, which move the resident blocks off the GPU and back while keeping
  the hooks. The cost of putting them back for each image, and the RAM they need meanwhile, have not been measured
* something that holds these together: it runs the stages of a pipeline in order and owns the memory budgets that move
  between them. Its interface is not decided; it is better settled after the steps above have measurements

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
* an option to keep chosen blocks or weights when `final_pass()` frees a model, if a model turns out to need it. By
  default every block is freed after it runs and the weights outside the blocks are left alone. A block that runs more
  than once in a pass (weight sharing as in ALBERT, or a caller that runs the model twice in one step) would be freed
  after its first run, and a weight tied between a block and an outside layer would be shared with a layer that still
  needs it. A list of name prefixes, like `skip_prefixes`, would leave those alone. It may not be needed
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
