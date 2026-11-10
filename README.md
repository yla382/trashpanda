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
The model cannot be used after the block, and adapters are not applied. `pool=` is a pinned pool that outlives the block
(a `Coordinator` passes one, see "Running a pipeline"); the plan then counts the pinned memory the pool holds already.

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
* `record_order(build, path, args=(...), kwargs={...})` finds the order of a model that has never run, without a real
  run: it calls the model once on the meta device, so no weights are read and nothing runs on the GPU, and writes the
  file that the first `stage()` then reads. `args` and `kwargs` are the inputs of one call, as small as the model
  accepts; tensors in them are moved to the meta device. The order can depend on the inputs, so give ones like the real
  run's. It replaces a file that is already there. If the call cannot run on the meta device (an operation without a
  meta version, or code that reads a tensor's values), it warns, returns `None` and saves nothing, and the first run
  goes on as it did without the call. `skip_prefixes` and `order_file_destination` are the same as for `stage()`.

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

A model can free its blocks as it runs. `handle.final_pass()`, called before the last call of the model, frees each block
right after it runs, so the VRAM and RAM of the first blocks are free while the last ones still compute. The model
cannot run again. `handle.subscribe()` returns a queue that gets a `Release` for each block freed this way: its name,
its tier (`"gpu"`, `"ram"` or `"disk"`), the bytes it held in that tier (0 for a disk block), a CUDA event that marks
when the GPU memory it held can be reused, and the time it was queued. The block is already freed when its `Release` is
queued, and nothing is queued until `subscribe()` is called:

```python
with stage(build_encoder, "encoder.safetensors") as (encoder, p, handle):
    releases = handle.subscribe()
    handle.final_pass()
    embeddings = encode(encoder, ...)
    while not releases.empty():
        r = releases.get_nowait()
        print(r.name, r.tier, f"{r.nbytes / 1e9:.2f} GB")
```

On a GPU, the tensors of a RAM-tier block are dropped on a thread of their own. With a memory-mapped RAM tier the last
tensor of the checkpoint to go takes the whole mapping with it (1.1 s for 6.95 GB of touched pages), and that would hold
up the model's last call. The `Release` of such a block is queued when its memory is really gone, so it can arrive up to
about a second after the last call returns, and a queue read right after that call may not have the last one yet.
`release()` and leaving `stage()` wait for the thread.

Blocks that a model never runs, such as the encoder half of a VAE that only decodes, are named by the profile (see
"Reading the profile"). `skip_prefixes` leaves them out.

A model can also give up its GPU memory between two calls and take it back, with the hooks staying in.
`handle.release_gpu()` moves the weights of the blocks that stay on the GPU (the resident blocks) to ordinary CPU memory
and returns the bytes it moved; `handle.restore_gpu()` puts them back and returns the bytes. Call them between two calls
of the model, not during one. Calling either one twice in a row does nothing the second time.

```python
with stage(build_diffusion, "diffusion.safetensors") as (model, p, handle):
    for prompt in prompts:
        latents = sample(model, prompt)
        handle.release_gpu()          # the VRAM is free while something else runs
        image = decode(latents)
        handle.restore_gpu()          # before the model runs again
```

While the weights are released they take as much RAM as their stored size, in ordinary (not pinned) memory. A resident
block that is stored in a compact format (fp8 with scales) still runs while it is released, from its CPU copy, as a
RAM-tier block does, but slower; a resident block that is stored in the compute dtype does not, so restore before the
model runs. The weights outside the blocks stay on the GPU. With `overlap=True` the hooks of a block go in when the block
first runs, and `release_gpu()` raises a `RuntimeError` until every block has run once. On the test machine the
weights moved at 3 to 5 GB/s (see "What we measured").

### Loading the next model while one runs

`preload()` starts loading the next model of a pipeline while the current one is still running, so the drive works
during the running model's compute:

```python
from trash_panda import preload, stage

with stage(build_first, "first.safetensors") as (first, p, _):
    pending = preload(build_second, "second.safetensors", max_vram="11GB", max_ram="8GB")
    result = run_long_computation(first, ...)    # a long computation; the second model's RAM tier is read meanwhile
with pending as (second, p, _):
    output = run(second, result)
```

It plans the next model now, for the memory it will have once the running one has been released. `max_vram` is that
GPU memory (the GPU is in use now, so it cannot be measured) and `max_ram` is the RAM its RAM tier may take; `None` for
`max_ram` means the RAM that is free now. Either can be left out when `running=p` is given, the plan of the running
stage (the `p` in `with stage(...) as (first, p, handle)`): the memory is then what is free now plus what that stage
holds and will give back, which is its GPU blocks and the weights outside the blocks for VRAM, and its RAM tier for RAM.
The running stage's buffers and activations are not counted, so the VRAM figure comes out low, never high; a
memory-mapped RAM tier gives back only the pages that were touched, so the RAM figure can be high. A figure that is
given wins, and without `running` `max_vram` is required.

Only the RAM that is free now can be read now. The RAM-tier blocks are taken in the order they run, from the first one,
until the next one does not fit in the free RAM (less `min_free_ram` and the pinned buffers). Those are read on a
background thread into memory of its own, whatever `TRASH_PANDA_RAM_IN_MEMORY` says: a memory-mapped RAM tier is not
read until it is used, so there would be nothing to read ahead. The other RAM-tier blocks stay mapped from the file and
are read when they first run, as in a model that was not preloaded. The GPU blocks are loaded when the `with` block is
entered, which has to happen after the running model was released; with `overlap=True` they are loaded on the worker
thread, as `stage()` does. The `with` block waits for the reads, and an error in them is raised there. A pending model
that will not be used has to be given up with `cancel()`, which waits for the reads and frees what they loaded. It can
be entered once. The other arguments are those of `stage()`.

`ram_preload=False` reads none of the RAM tier early: all of it stays mapped from the file and is read when its blocks
first run, as in a model that was not preloaded, and no RAM-tier block moves to the GPU at entry. On the test machine
reading the RAM tier into memory ran slower per byte than the mapped read (0.33 GB/s against 0.38 to 0.42), and it only
pays off if the running stage is long enough to hide the read (see "What we measured").

The prediction can be wrong (another program took VRAM, the running model kept a tensor, the earlier measurement of the
activations was low), so when the `with` block is entered the plan is fitted to the VRAM that is really free: what the
driver reports plus what PyTorch holds cached and unused, less the headroom the plan kept for activations and the room
for streaming blocks. A block planned for the GPU that is not there yet and does not fit goes to the RAM tier, mapped
from the file, with a warning; blocks already on the GPU stay. If there is room left, RAM-tier blocks that were read
into memory move to the GPU, in the order they run, up to the first that does not fit. Blocks that stayed mapped are not
moved, because that would be a cold read at entry. Nothing is changed when the free VRAM is not known (no CUDA).

The weights outside the blocks (640 MB for the diffusion model of the test pipeline) are put on the GPU as soon as they
are read, so they need room there while the running model holds its own. The reads share the drive with whatever else is
reading from it. With `TRASH_PANDA_PROFILE=1` it prints how long the background read took and how long the model waited
for it, how many RAM blocks were read early and how many stay mapped, what it planned for when `running` was given, and
what the replan at entry moved.

With `after=`, the GPU blocks of the next model are loaded early too, into VRAM that is free already and into the VRAM
that the running model's final pass frees:

```python
with stage(build_first, "first.safetensors") as (first, p, handle):
    releases = handle.subscribe()
    pending = preload(build_second, "second.safetensors", max_vram="11GB", max_ram="8GB", after=releases)
    handle.final_pass()
    result = run(first, ...)        # the second model's GPU blocks load as the first one frees its own
with pending as (second, p, _):
    output = run(second, result)
```

`after` is the queue that `handle.subscribe()` returns. The loader reads it, so nothing else should. It loads the
blocks of the next model in the order they will run, in two cases. A block goes into VRAM that is free already when it
fits there and what is left still covers the margin below and what the running stage keeps free for itself (its plan's
headroom and streaming room, or the configured `headroom_vram` when only a budget was given as `max_vram`). Otherwise it
waits until the GPU blocks announced on the queue add up to more than what has been loaded that way so far plus the
block, and loads when the GPU really has room for it. Room means that the memory the driver reports free plus what
PyTorch holds cached and unused covers the block and leaves free the larger of 5% of the GPU's memory (at least 256 MB)
and what the new model's plan counted on keeping free. Without a GPU there is no free VRAM to fill, and only the
announced releases count. Each block goes through two small pinned buffers (64 MB each, separate from the running
model's) and a copy stream of its own. The destination is allocated on the compute stream, so it can reuse the memory
the running model freed there, and the copy waits for an event recorded right after that allocation, which covers
everything the running model has queued so far. The allocator can hand out memory that was freed on the CPU side while
kernels that use it are still queued, because the CPU runs ahead of the GPU. A copy that waited only for the event of
the last block that was announced could land before those kernels: on the test pipeline a VAE loaded that way decoded
to a black image. The cost is that a copy starts when the GPU has caught up with the point the CPU had reached. Copying
straight from the memory-mapped file instead held up the running model's compute, also on a separate stream. The
blocks that were not loaded early load when the `with` block is entered, and a failure in the early loading only gives
a warning. It loads GPU blocks only, and a block larger than the free memory waits for room. The loader reads from the
drive, so it keeps pace with the running model only when the next model's file is in the file cache or the model
computes for long enough.

With `running=` (the plan of the running stage) the early loader also gets the staging that the running stage does not
use. A stage plans room for `lookahead` staged blocks; with `TRASH_PANDA_ADAPTIVE_LOOKAHEAD=1` it lowers the
look-ahead to 1 where the main thread does not wait for blocks, and the room of the blocks it no longer stages
(`plan.spare_staging_bytes`) counts as memory that is free already: the next model's blocks may load that many bytes
before the running model frees anything, as long as the GPU really has room for them. The figure is read each time, so a
later decision of the tuner counts.

Reading early moves reads, it does not remove them, so it saves time only where the drive would otherwise be idle or
the reads overlap work that does not use it; compare the total time of the whole pipeline. On the test pipeline (text
encoder, diffusion model and VAE, on a SATA SSD) preparing the diffusion model while the encoder runs did not shorten
the run: the encoder computes for 2.7 s and the diffusion model's RAM tier takes 24 s to read, so with `ram_preload`
the run was 9 s longer than with three plain `stage()` calls, and with only the GPU blocks loaded early it was within
the noise. Preparing the VAE while the diffusion model runs did pay, by 2 to 3 s, because that model computes for about
50 s and the VAE is small (see "What we measured").

### Running a pipeline

A `Coordinator` runs the models of a pipeline one after another. It owns one pinned pool for all of them and prepares
each next model while the running one computes:

```python
from trash_panda import Coordinator, Step

co = Coordinator(
    Step("text", lambda: encoder.qwen, encoder.weights_path, skip_prefixes=SKIP_PREFIXES),
    Step("unet", build_unet, "unet.safetensors", resident_blocks=12),
    Step("vae", lambda: ae.ae, ae.weights_path, skip_prefixes=("encoder.",)),
)
with co:
    with co.stage("text", ram_preload=False) as (text, p, handle):
        cond = encode(text, ...)
    with co.stage("unet") as (unet, p, handle):
        latents = denoise(unet, cond, before_last=handle.final_pass)
    with co.stage("vae") as (vae, p, handle):
        image = decode(vae, latents)
```

A `Step` is a name and the arguments of `stage()` for one model (`build`, `path`, `key`, `key_map`, `skip_prefixes`,
`order`, `overlap`, `max_vram`, `max_ram`, and the arguments of `plan()`). `pool`, `running` and `after` are set by the
`Coordinator` and cannot be given. `co.stage(name)` yields what `stage()` yields, `(model, plan, handle)`, and releases
the model when the block ends. It must be used inside `with co:`, and a stage cannot start inside another stage of the
same `Coordinator`. A step can run again (a loop over images). The order of the `Step`s in the constructor is the
order used for "next", not the order of the `co.stage()` calls.

* **One pinned pool.** `with co:` makes one `_PinnedPool` that every model uses, instead of each model pinning its own
  buffers again, and it is dropped, with its pinned memory given back, when the block ends. Its buffers grow to the
  largest block of any model and do not shrink. Each plan counts the buffers the pool has already (`pinned_held`, the
  size of each buffer), so only what the buffers still have to grow by is kept back from the RAM tier. Buffers that a
  model added to the pool (the look-ahead tuner does) are dropped when its stage ends, so the next model cycles through
  the configured number of buffers. `stage()` and `preload()` take `pool=` too.
* **Preparing the next step.** While a stage runs, the next model is planned for the memory the running one will give
  back (`preload(running=p)`). Three arguments of `co.stage()` say what is prepared. `then` names the step: by default
  the next one in the order of the `Coordinator`, a step's name for another one, `None` when there is none (nothing
  follows the last step). `ram_preload=True` (the default) reads the next model's RAM tier into memory on a background
  thread; `False` leaves it mapped from the file. `vram_preload=True` (the default) loads its GPU blocks early, into VRAM
  that is free and, if the running model calls `handle.final_pass()`, into the memory the final pass frees; `False`
  loads them when the step starts. With both `False` nothing is prepared and the next step loads like `stage()` does.
  The queue of `handle.subscribe()` belongs to the loader then, so do not read it in a stage that prepares a next step.
* **A different step than the prepared one** gives the prepared model up (its reads finish first) and loads the asked
  one as `stage()` does. Leaving `with co:` gives up a prepared model nobody asked for.

On the test machine's SATA drive, pass `ram_preload=False` for a stage that is shorter than the read of the next
model's RAM tier (see "What we measured" and "Current recommendation").

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
| `TRASH_PANDA_ADAPTIVE_LOOKAHEAD` | `0` | After the first passes, try a deeper look-ahead (up to `TRASH_PANDA_LOOKAHEAD_MAX`) where the main thread waits, and a look-ahead of 1 where it does not, and keep the change only if the passes do not get slower (see "What we measured"). The staging that is no longer used is lent to the next model (see "Loading the next model while one runs"). Off by default |
| `TRASH_PANDA_LOOKAHEAD_MAX` | `4` | The deepest look-ahead that the setting above tries. The pinned buffers grow to match, and a depth is only taken if the driver reports free VRAM for the extra staged blocks |
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
* **`TRASH_PANDA_LOOKAHEAD`:** a depth of 4 was no better than 2 for disk blocks and uses more VRAM. With stand-in
  models (12 blocks of 0.07 to 0.54 GB, the file in the file cache, single runs) a deeper look-ahead shortened a pass by
  3% to 8%, only where the copies and the compute took about the same time (20 ms of compute per block), and only with
  as many pinned buffers as the look-ahead (4 with 4, not 4 with 2). Where the copies set the pace (no compute) or the
  compute did (100 ms per block) it made no difference beyond the noise of about 5% between runs.
  `TRASH_PANDA_ADAPTIVE_LOOKAHEAD=1` tries the depth by itself. In a check on the GPU its output was identical to the
  fixed look-ahead and it cost nothing where compute set the pace, but it found the gain in one of nine cases (disk
  blocks, 20 ms: 5.3% faster) and missed it in the others, so it is off by default. Where the compute sets the pace
  (100 ms per block) it lowers the look-ahead to 1, which was 0.4% to 2% slower in the grid and 0.8% in the check on the
  GPU, and frees one block of staging for the next model.
* **A read-ahead of the RAM tier** through ordinary file reads, ahead of the copy, made the first step slower (34.7 s
  against 21 to 22 s) and was removed: the pages it read did not serve the memory-mapped copy.
* **Memory after a release.** On Windows, PyTorch's own CPU allocator keeps blocks after they are freed (3 GB of
  tensors freed, working set unchanged), so memory read into `torch.empty` tensors still counted as taken when the next
  model was planned (a RAM budget of 2.0 to 2.6 GB, against about 9 GB). `TRASH_PANDA_RAM_IN_MEMORY` allocates through
  Python's allocator instead, which gives the memory back at once, and after the fix the next plan saw 8.2 to 9.3 GB.
* **Taking the resident blocks off the GPU and back** (`release_gpu()` / `restore_gpu()`) cost about 0.25 s per GB to
  release and 0.3 s per GB to restore, for blocks in the compute dtype and for fp8 blocks with scales alike. Stand-in
  blocks of two 4096 x 4096 linear layers, built on the GPU, median of 5 cycles: 1 GB took 0.22 s and 0.27 s, 2 GB took
  0.45 s and 0.58 s, 4 GB took 1.04 s and 1.27 s (fp8: 1.32 s), which is 3.9 to 4.6 GB/s out and 3.0 to 3.7 GB/s back.
  The VRAM they held was free afterwards, and the first pass after the restore took as long as any other (within the
  noise). A plain tensor moves to the GPU at 5.1 GB/s from ordinary memory and 12.9 GB/s from pinned memory, so pinned
  copies could make the restore up to 2.5 times faster, at the price of pinned RAM as large as the weights. Nothing
  above 4 GB was run, because the free RAM was between 2.6 and 9.7 GB during the measurement; with 2.6 GB free the copy
  to the CPU fell to 1.2 GB/s.
* **The Coordinator on the test pipeline** (one 1024 x 1024 image, 7 steps, no overlap, single runs; the start RAM and
  the state of the file cache differed from run to run, which moved the totals by several seconds). Three plain
  `stage()` calls took 98.4 s (start RAM 9.7 GB). A `Coordinator` with the defaults (both preload arguments on, every
  stage) took 107.3 s (9.5 GB), 111.7 s and 113.9 s (6.5 to 7.1 GB). At matched RAM the whole 8.9 s gap between the
  first two is in the diffusion model's phase (76.1 s against 84.9 s). Preparing it during the encoder read its RAM tier
  into memory, 7.29 GB in 21.6 s (0.33 GB/s), of which the 2.7 s encoder hid 3.6 s, so the model waited 18 to 20 s at
  entry; the first denoise step then took 9.7 s instead of 22.3 s, which was not enough to make up for it. The mapped
  read overlaps the cold read with the compute of the first step. Preparing the VAE while the diffusion model ran
  loaded 5 or 6 of its 6 blocks early and shortened the steps after the diffusion model by 2 to 3 s.
* **`ram_preload=False` for the encoder, `vram_preload` left on** took 99.5 s (start RAM 8.2 GB): 2 of the diffusion
  model's 12 GPU blocks (0.87 GB) loaded during the encoder, its load took 12.2 s instead of 15.7 s, and the encoder
  phase took about 3 s longer (the prompt was encoded in 3.9 s instead of 1.1 s). The window is the encoder's 3 to 4 s
  and the drive reads 0.4 GB/s, so it can hide about 1.5 GB at most. The cost in the encoder was not traced; one
  possibility is that loading into free VRAM makes the allocator ask the driver for new memory while the encoder runs.
  One run with the diffusion model's weights in the file cache (the encoder loaded in 2.7 s instead of 12 s) took
  81.3 s, but it is not comparable with the others.
* **Where the time goes** (the 107.3 s run, from the time between the stages): loading weights took 48.3 s (the
  encoder 12.0 s, the diffusion model 36.3 s, 45% of the run), the compute 52.6 s (49%), the rest about 6 s. A run reads
  about 16 GB through the drive. On a drive that reads 3 GB/s that part would take a few seconds, and preparing the
  next model while the encoder runs should pay off then.
* **`overlap=True` in a `Coordinator`** with the defaults took 105.1 s (start RAM 6.5 GB, no table), against 107.3 to
  113.9 s without it. The encoder's blocks load on a worker thread while the diffusion model's RAM tier is read, so the
  model waited 1.2 s at entry instead of 18 s, at the price of a longer first step. The difference from the runs
  without it is within the spread of those runs.

### Current recommendation

* Leave every setting at its default. The defaults were the fastest total in these measurements.
* Leave `TRASH_PANDA_RAM_IN_MEMORY` off unless you want steady steps from the first one, and have the RAM and the commit
  headroom for it (about 7 GB of private memory on the test pipeline). It does not make a single run shorter.
* Leave `overlap` off. It is groundwork for starting the next model's loads behind compute that is long enough to hide
  them (see the roadmap), not a speed-up for one pipeline.
* A slow first step is the first read of the RAM tier from the drive. The profile shows it as a first pass with a low
  GB/s in the RAM copies and a large "waited". Only a faster drive shortens it.
* Set `TRASH_PANDA_PROFILE=1` when you want to see where the time goes.
* In a `Coordinator`, pass `ram_preload=False` for a stage that is shorter than the read of the next model's RAM tier
  (the encoder before a diffusion model on a SATA drive), or `then=None` to prepare nothing. Preparing the VAE while the
  diffusion model runs is worth keeping on.

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
* `final pass freed N blocks over X s (...)`: after `handle.final_pass()`, the blocks and GB freed in each tier and the
  time from the first block freed to the last. With a subscriber (`handle.subscribe()`) it also gives the most
  `Release` objects the queue held at once and how many were unread when the profile was printed.
* `blocks that never ran`: named blocks that were loaded but that no call used, with their size. If the model never uses
  them, leave them out with `skip_prefixes`. It is printed once some block has run.
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
  plan them one after the other, or free the first model before loading the next. Models that run one after another
  are what `Coordinator` is for (see "Running a pipeline")
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
* **A PyTorch warning with the profile on.** With `TRASH_PANDA_PROFILE=1`, a release may print
  `Deallocating Tensor that still has live PyObject references`. The memory trace of the
  profile keeps a weak reference to every large tensor it finds, and PyTorch warns when one of them is freed. It is
  harmless, nothing reads those tensors afterwards, and it appears from run to run only by chance. Without the profile
  it should not appear
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
* **Starting the next model's reads behind compute in a pipeline that runs the same stages for many inputs in a row**:
  the first model of the next input would be read while the last stage of the current one computes. It goes beyond
  `preload()`, which covers one run through the stages
* **Reading the order of the blocks from the source of `forward()`** with the `ast` module, to know the order of a
  model that has never run without calling it. It fails on code that is not plain (helper methods, dynamic
  dispatch, branches that depend on the inputs), and a wrong order is prefetched with confidence, which is worse
  than the definition order. `record_order()` calls the model on the meta device instead and sees the real order
* **Cheaper expansion**: reuse preallocated buffers and fuse the fp8 to bf16 conversion and scaling, instead of
  creating new tensors for every weight on every pass
* **A cost model** that chooses between GPU, RAM and disk from measured disk and PCIe speed
* **A command-line interface**, for example `trash-panda plan model.safetensors` to print a plan without writing a
  script, and a hardware report for bug reports
* **Visualizing the data flow, with a trace of a run**: the events the profile adds up today (block copies from RAM or
  disk, waits for a copy, waits for the GPU, block runs, loader-thread loads, release steps) would be recorded with
  their start and end times and written to a file in the Chrome trace-event format, which Perfetto and
  `chrome://tracing` open as a timeline with a track per thread and stream. It would show each block moving between
  tiers, the overlap of copying and compute, the stalls, the time each block of a final pass was freed, and blocks that
  never start, which show as a gap in the timeline. It would be switched on by its own environment variable and be off
  by default. A web interface of our own would only be needed for what a timeline cannot show, such as a view of the
  plan, and a live interface would be a later step
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
