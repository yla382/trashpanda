# Running a pipeline

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

A `Step` is a name and the arguments of `stage()` for one model. `co.stage(name)` yields what `stage()` yields, and
releases the model when the block ends.

Three arguments of `co.stage()` say what is prepared for the next step:

| Argument | Default | Meaning |
|---|---|---|
| `then` | the next step | the step to prepare; `None` for none |
| `ram_preload` | `True` | read its RAM tier into memory in the background (`False`: it stays mapped from the file) |
| `vram_preload` | `True` | load its GPU blocks early (`False`: they load when the step starts) |

With both flags `False`, nothing is prepared and the next step loads like `stage()` does.

!!! note
    On a SATA drive, `ram_preload=False` is the better choice for a stage that is shorter than the read of the next
    model's RAM tier. See [what we measured](../design/measurements.md) and [pipelines in
    detail](../design/pipelines.md).

See [`Coordinator`](../reference/stages.md#coordinator) in the API reference.
