# Pipelines in detail

How a `Coordinator` shares one pinned pool and prepares the next model. The short version is in the [guide](../guides/running-a-pipeline.md); the arguments are in the [reference](../reference/stages.md#coordinator).

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
