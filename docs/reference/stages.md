# Stages and pipelines

Run one model with `stage()`, load the next one in the background with `preload()`, or run a whole pipeline with a
`Coordinator`.

::: trash_panda.offload.stage.stage
    options:
      show_docstring_description: false

Loads one model for the `with` block and gives its memory back afterwards. Yields `(model, plan, handle)`; the model
cannot be used after the block.

| Parameter | Default | Meaning |
|---|---|---|
| `build` | | a function that returns the model on the meta device |
| `path` | | the checkpoint: a file, a folder of shards, or an index json |
| `key` | `None` | the name the peak GPU memory is remembered under (default: the path) |
| `key_map` | `None` | checkpoint name to model name, a dict or a function |
| `skip_prefixes` | `()` | model entries with these prefixes are left out |
| `order` | `None` | names of the blocks that run first; `None` reads and writes an order file |
| `order_file_destination` | `None` | the folder of the order file (default: next to the checkpoint) |
| `overlap` | `False` | load the resident blocks on a worker thread while the model already runs |
| `pool` | `None` | a pinned pool that outlives the block (a `Coordinator` passes one) |
| `**plan_kwargs` | | passed to [`plan()`](planning.md#trash_panda.offload.planner.plan), for example `resident_blocks=12` |

```python
with stage(build_unet, "unet.safetensors", resident_blocks=12) as (unet, p, handle):
    latents = denoise(unet, ...)
```

Details: [stages in detail](../design/stages.md).

::: trash_panda.offload.stage.preload
    options:
      show_docstring_description: false

Starts loading the next model while the current one still runs. Returns a [`PendingStage`](#pendingstage). It needs
`max_vram` or `running`.

| Parameter | Default | Meaning |
|---|---|---|
| `build`, `path` | | as for `stage()` |
| `max_vram` | `None` | the GPU memory the model may use once the running one is released |
| `max_ram` | `None` | the RAM its RAM tier may take (default: the RAM free now) |
| `running` | `None` | the plan of the running stage; replaces `max_vram` and `max_ram` |
| `after` | `None` | the queue from the running stage's `handle.subscribe()`; with it the GPU blocks load early |
| `ram_preload` | `True` | read the RAM tier into memory now; `False` leaves it mapped from the file |
| `pool` | `None` | the pinned pool the model will use |
| `key`, `key_map`, `skip_prefixes`, `order`, `order_file_destination`, `overlap` | | as for `stage()` |
| `**plan_kwargs` | | passed to `plan()` |

```python
with stage(build_first, "first.safetensors") as (first, p, handle):
    pending = preload(build_second, "second.safetensors", running=p, after=handle.subscribe())
    handle.final_pass()
    result = run(first, ...)
with pending as (second, p, _):
    output = run(second, result)
```

Details: [loading the next model early](../design/loading-the-next-model.md).

## PendingStage

What `preload()` returns. Use it like `stage()`: `with pending as (model, plan, handle):`. It can be entered once.

| Method | Returns | Meaning |
|---|---|---|
| `ready()` | `bool` | `True` once the background reading has finished |
| `cancel()` | `None` | gives up a model that will not be used and frees what was loaded |

::: trash_panda.offload.stage.record_order
    options:
      show_docstring_description: false

Finds the order in which a model runs its blocks, without loading it, and saves it where `stage()` reads it. Returns
the block names, or `None` (with a warning) if the call failed on the meta device.

| Parameter | Default | Meaning |
|---|---|---|
| `build`, `path` | | as for `stage()` |
| `args` | `()` | the positional inputs of one call, as small as the model accepts |
| `kwargs` | `None` | the keyword inputs of that call |
| `skip_prefixes` | `()` | entries to leave out |
| `order_file_destination` | `None` | the folder of the order file |
| `groups` | `None` | the blocks to use instead of finding them |

```python
record_order(build_unet, "unet.safetensors", args=(img, context, t, pos, mask))
```

## Coordinator

Runs the models of a pipeline one after another with one pinned pool. `Coordinator(*steps)` takes `Step`s; use it as
`with co:` and run each model with `co.stage(name)`.

::: trash_panda.offload.coordinator.Coordinator.stage
    options:
      show_docstring_description: false
      heading_level: 3

Runs one step and yields `(model, plan, handle)`; the model is released when the block ends. Must be inside `with co:`.

| Parameter | Default | Meaning |
|---|---|---|
| `name` | | the name of the step to run |
| `then` | the next step | the step to prepare while this one runs; `None` for none |
| `ram_preload` | `True` | read the next step's RAM tier into memory in the background; `False` leaves it mapped |
| `vram_preload` | `True` | load the next step's GPU blocks early; `False` loads them when the step starts |

```python
co = Coordinator(Step("text", build_text, text_path), Step("unet", build_unet, unet_path))
with co:
    with co.stage("text", ram_preload=False) as (text, p, handle):
        cond = encode(text, ...)
    with co.stage("unet") as (unet, p, handle):
        latents = denoise(unet, cond, ...)
```

Guide: [running a pipeline](../guides/running-a-pipeline.md).

::: trash_panda.offload.coordinator.Step
    options:
      show_docstring_description: false

One model of a pipeline: a name, a build function, a path, and the other arguments of `stage()` as keyword arguments.
`pool`, `running` and `after` are set by the `Coordinator` and cannot be given.

```python
Step("unet", build_unet, "unet.safetensors", resident_blocks=12, overlap=True)
```
