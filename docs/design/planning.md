# Planning in practice

How plans are made and edited, and how to size the VRAM headroom. The arguments are in the [reference](../reference/planning.md).

## Automatic or manual

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

## Fitting the GPU to your workload

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
