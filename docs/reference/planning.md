# Planning

Deciding where each block of a model lives, from the checkpoint's sizes and the memory that is free.

::: trash_panda.offload.planner.plan
    options:
      show_docstring_description: false

Decides where each block lives. Every argument that is set overrides the automatic choice. Returns a [`Plan`](#plan-class);
`print(plan)` shows it with its warnings.

| Parameter | Default | Meaning |
|---|---|---|
| `model` | | the model on the meta device |
| `checkpoint` | | the checkpoint path (or a `Checkpoint`) |
| `compute_dtype` | `"bf16"` | the dtype a block is expanded to when it runs |
| `storage_dtype` | `"auto"` | the dtype the weights wait in; `"auto"` keeps them as stored |
| `resident_blocks` | `None` | force this many blocks (from the start) onto the GPU; `None` is as many as fit |
| `max_vram` | `None` | VRAM to use (default: what is free now) |
| `headroom_vram` | `None` | VRAM kept free for activations (setting `headroom_vram`, `"1.5GB"`) |
| `activation_vram` | `None` | measured activation memory; replaces `headroom_vram` |
| `max_ram` | `None` | RAM to use for streamed blocks (default: what is available now) |
| `min_free_ram` | `None` | RAM kept free for the system (setting `min_free_ram`, `"auto"`) |
| `allow_disk` | `True` | blocks that do not fit in RAM stay in the file instead (slow) |
| `lookahead` | `None` | blocks staged ahead on the GPU (setting `lookahead`, `2`; `0` is off) |
| `spread_disk` | `True` | put the disk blocks evenly among the others instead of at the end |
| `hardware` | `None` | a [`Hardware`](#hardware) to plan for (default: detect this machine) |
| `groups` | `None` | the blocks to use instead of finding them |
| `key_map` | `None` | checkpoint name to model name |
| `skip_prefixes` | `()` | model entries to leave out |
| `pinned_held` | `()` | sizes of the buffers of a pinned pool shared with earlier models (a `Coordinator` sets it) |

```python
p = plan(model, "model.safetensors", resident_blocks=12, max_vram="9GB", min_free_ram="4GB")
print(p)
```

## Plan {#plan-class}

The result of `plan()`, or of `Plan.manual()`. The main fields:

| Field | Meaning |
|---|---|
| `blocks` | a list of [`BlockPlan`](#blockplan) |
| `compute_dtype`, `storage_dtype` | the dtype blocks are expanded to, and the one they wait in |
| `vram_budget`, `ram_budget` | the VRAM and RAM the plan was made for, in bytes |
| `warnings` | the warnings, as a list of strings |
| `source` | `"auto"` or `"manual"` |

The other fields (`other_bytes`, `stream_buffer_bytes`, `staging_bytes`, `lookahead`, `headroom_bytes`,
`spare_staging_bytes`) are the memory the plan reserved; `print(plan)` lists them.

| Method | Returns |
|---|---|
| `Plan.manual(resident=(), streamed=(), disk=(), compute_dtype="bf16", storage_dtype="auto")` | a plan built from lists of block prefixes |
| `plan.assign(tier, *, indices=None, prefixes=None)` | the plan itself, with those blocks moved to `tier` |
| `plan.prefixes(tier)` | the prefixes of the blocks in a tier, as a tuple |
| `plan.resident`, `plan.streamed`, `plan.disk` | the prefixes in the GPU, RAM and disk tier |
| `plan.summary()` | the text that `print(plan)` shows |

```python
p = Plan.manual(resident=["blocks.0.", "blocks.1."], streamed=["blocks.2.", "blocks.3."])
p.assign(Tier.DISK, indices=[3])
```

## BlockPlan

Where one block lives: `prefix` (for example `"blocks.3."`), `tier` (a `Tier`, default `Tier.RAM`), `stored_bytes` (its
size while it waits) and `expanded_bytes` (its size once expanded to the compute dtype).

## Tier

An enum (and a `str`) with three members: `Tier.GPU` (`"gpu"`, resident on the compute device), `Tier.RAM` (`"ram"`,
waits in system RAM) and `Tier.DISK` (`"disk"`, stays in the checkpoint file).

## Hardware

A snapshot of the machine: `vram_total`, `vram_free`, `ram_total`, `ram_available` (bytes; the VRAM fields are 0
without a CUDA GPU), `commit_available` (Windows, else `None`), `gpu_name`, `compute_capability`, `bf16_supported`
and `native_fp8`. `summary()` returns the GPU and RAM lines as text.

## detect_hardware

`detect_hardware(device_index: int = 0) -> Hardware`

Reads the GPU with that index and the system RAM. `plan()` calls it unless you pass `hardware=`.

## parse_bytes

`parse_bytes(value) -> Optional[int]`

Turns `"9GB"`, `"8GiB"` or a number into bytes; `None` stays `None`.

```python
parse_bytes("9GB")     # 9000000000
parse_bytes("8GiB")    # 8589934592
```

::: trash_panda.offload.measure.measure_activation_vram
    options:
      show_docstring_description: false

Runs `fn()` once and returns the peak GPU memory it needed beyond what was already allocated, in bytes, less the
streaming buffer that the plan already counts. Pass the result to `plan(activation_vram=...)`.

```python
need = measure_activation_vram(lambda: run_my_generation(), p)
p = plan(model, path, activation_vram=need)
```

::: trash_panda.offload.blocks.find_blocks
    options:
      show_docstring_description: false

Finds groups of repeated blocks (`ModuleList` or `Sequential` containers). Returns a list of
[`BlockGroup`](#blockgroup). `min_fraction` and `min_blocks` default to the settings `find_blocks_min_fraction` and
`find_blocks_min_blocks`.

## module_nbytes

`module_nbytes(module) -> int`

The bytes of a module's parameters and buffers. It works on the meta device.

## BlockGroup

A group of blocks that `find_blocks()` found: `name` (the container's module path, for example `"blocks"`), `prefixes`
(one parameter-name prefix per block) and `nbytes` (the size of the whole group in the model's own dtype).
