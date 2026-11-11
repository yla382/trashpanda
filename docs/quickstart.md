# Quick start

## Install

Install PyTorch first (the build that matches your GPU, see [pytorch.org](https://pytorch.org)), then:

```
pip install dumpsterpanda
```

The package is imported as `trash_panda`. The weights have to be in safetensors format: one file, a folder of shards,
or the index json of a sharded checkpoint.

For the command line (`trash-panda plan` and `trash-panda info`), install `dumpsterpanda[cli]` instead.

## One model: `stage()`

`stage()` builds the model, plans where each block should live, loads it, installs the hooks, and gives all of its
memory back when the `with` block ends:

```python
import torch
from trash_panda import stage

def build():
    with torch.device("meta"):                  # the structure only: no weights are allocated
        return MyModel(config).eval()

with stage(build, "model.safetensors") as (model, p, handle):
    print(p)                                    # where each block lives
    out = model(x)
                                                # here the model's RAM and VRAM are free again
```

Inside the block, `model` is an ordinary `torch.nn.Module`. Call it as you would any model; the blocks are streamed to
the GPU as they run. The model cannot be used after the block.

## Read the plan

`print(p)` shows what the planner decided, from the checkpoint's sizes and the memory that is free on your machine.
This is the plan of a 12B diffusion model with fp8 weights on a 12 GB GPU:

```
Plan (auto): 32 blocks, compute dtype bfloat16
   12 blocks resident on the GPU: 5.21 GB
   20 blocks streamed from RAM: 7.29 GB
  weights outside blocks (on the device): 640.3 MB
  room reserved for one streamed block: 1.13 GB
  room reserved for blocks staged ahead (look-ahead 2): 868.4 MB
  VRAM budget: 10.19 GB   RAM budget: 8.41 GB
```

Each block is in one of three tiers:

| Tier | Where the block waits | Cost per forward pass |
|---|---|---|
| `gpu` | on the GPU, in compact form | only the expansion to the compute dtype |
| `ram` | in system RAM, in compact form | a transfer over PCIe |
| `disk` | in the checkpoint file | a file read |

Only one block at a time is expanded to the compute dtype, so the peak VRAM is roughly one block plus your
activations. A line that starts with `WARNING:` says what to look at, for example blocks that are read from disk on
every pass.

## Steer the plan

The keyword arguments of `stage()` go to [`plan()`](reference/planning.md#trash_panda.offload.planner.plan). Anything
the planner decides can be set by hand:

```python
with stage(build, "model.safetensors",
           resident_blocks=12,          # how many blocks stay on the GPU
           max_vram="9GB",              # VRAM the model may use
           min_free_ram="4GB",          # RAM kept free for the system
           compute_dtype="bf16") as (model, p, handle):
    ...
```

`skip_prefixes=("encoder.",)` leaves part of a checkpoint out (for example the encoder half of a VAE that only
decodes), and `key_map` maps checkpoint names to the model's names when they differ.

## The same in three calls

`stage()` is a wrapper around three functions, which you can call yourself for more control:

```python
import torch
from trash_panda import plan, stream_load, install_offload, release

with torch.device("meta"):
    model = MyModel(config).eval()

p = plan(model, "model.safetensors")                         # what fits where, with warnings
print(stream_load(model, "model.safetensors", plan=p))       # lazy load straight into the right tiers
handle = install_offload(model)                              # hooks that stream blocks during forward

out = model(x)

release(model, handle)                                       # RAM and VRAM are free again
```

## Look at a checkpoint first

The command line shows a plan without writing a script or loading anything:

```
trash-panda plan model.safetensors
trash-panda info model.safetensors
```

`plan` prints where each block would live on this machine and how the GPU and RAM budgets are used. `info` prints what
is in the file. See the [command line guide](guides/command-line.md).

## Several models, one after another

Run each model in its own `stage()` and the memory of the first is free when the second is planned:

```python
with stage(build_text, "text.safetensors") as (text, p, _):
    cond = encode(text, ...)
with stage(build_unet, "unet.safetensors") as (unet, p, _):
    latents = denoise(unet, cond, ...)
```

A [`Coordinator`](guides/running-a-pipeline.md) does this for a whole pipeline, with one pinned pool for all the models
and the next model loading while the current one computes.

## See where the time goes

Set `TRASH_PANDA_PROFILE=1` before you start. At each release, trash-panda then prints how long the prefetching waited
for blocks, per pass, and what each tier cost.

```powershell
$env:TRASH_PANDA_PROFILE = "1"
```

The other settings are listed on the [settings](reference/settings.md) page.

## Good to know

* A model with blocks in RAM or on disk is slow on its first pass, because the blocks are read from the drive the first
  time they run. Later passes are as fast as the compute allows.
* Adapters (LoRA and similar) are not applied, and only fp8 and ordinary float dtypes are supported.
* It has been tested on one setup so far (see [known issues](project/known-issues.md)).

## Next

* [Running a stage](guides/running-a-stage.md) and [Running a pipeline](guides/running-a-pipeline.md)
* [Command line](guides/command-line.md)
* [API reference](reference/index.md)
