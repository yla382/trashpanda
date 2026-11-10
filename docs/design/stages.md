# Stages in detail

How a model is loaded, run and released, and how the order of its blocks is found. The short version is in the [guide](../guides/running-a-stage.md); the arguments are in the [reference](../reference/stages.md).

## Releasing a model

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

## Running a stage

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
