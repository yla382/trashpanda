# Loading the next model early

How `preload()` and early loading work, and what they cost. The arguments are in the [reference](../reference/stages.md#trash_panda.offload.stage.preload).

## Loading the next model while one runs

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
