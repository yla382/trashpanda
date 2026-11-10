# Running a stage

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
`plan()`, except `key_map` and `skip_prefixes`, which go to both `plan()` and `stream_load()`. Use them when the
checkpoint's names differ from the model's, or when part of the model should not be loaded (for example the vision
tower of a text encoder). The model cannot be used after the block, and adapters are not applied.

See [`stage`](../reference/stages.md#trash_panda.offload.stage.stage) in the API reference for every argument.
