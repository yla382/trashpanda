# Hooks and handles

The hooks that stream blocks during a forward pass, the handle that controls them, and how to give a model's memory back.

::: trash_panda.offload.hooks.install_offload
    options:
      show_docstring_description: false

Installs the hooks for every block in the plan of a loaded model. Returns an [`OffloadHandle`](#offloadhandle).

| Parameter | Default | Meaning |
|---|---|---|
| `model` | | a model loaded with `stream_load()` |
| `plan` | `None` | the plan (default: the one `stream_load()` used) |
| `device` | `None` | the compute device (default: the one it loaded to) |
| `compute_dtype` | `None` | the dtype blocks are expanded to (default: the plan's) |
| `lookahead` | `None` | blocks staged ahead (default: the plan's) |
| `profile` | `None` | print where the time goes (default: the `profile` setting) |
| `order` | `None` | names of the blocks that run first; `None` reads and writes an order file |
| `order_file_destination` | `None` | the folder of the order file |
| `pool` | `None` | a pinned pool that outlives the handle |

```python
stream_load(model, path, plan=p)
handle = install_offload(model)
out = model(x)
```

::: trash_panda.offload.hooks.offload_blocks
    options:
      show_docstring_description: false

The low-level version: wraps each block so its weights are expanded on `device` only while it runs. Blocks whose weights
are already on the device in the compute dtype are left alone. Returns an `OffloadHandle`.

| Parameter | Default | Meaning |
|---|---|---|
| `blocks` | | the modules to wrap |
| `device` | `None` | the compute device (default: CUDA if available) |
| `compute_dtype` | `torch.bfloat16` | the dtype blocks are expanded to |
| `lookahead` | `None` | blocks staged ahead (`0` turns prefetching off) |
| `names` | `None` | a name per block, for the order file and for messages |
| `profile` | `None` | print where the time goes |
| `disk`, `prefetcher`, `hint` | | advanced |

```python
offload_blocks(model.blocks, device="cuda", compute_dtype=torch.bfloat16)
```

## OffloadHandle

Returned by the install functions. Its methods:

### final_pass

`handle.final_pass() -> None`

From now on each block is freed right after it runs. Call it before the last call of the model; the model cannot run
again.

### subscribe

`handle.subscribe() -> queue.SimpleQueue[Release]`

A queue that gets a [`Release`](#release-class) for every block the final pass frees. Nothing is queued until you call it.

### release_gpu

`handle.release_gpu() -> int`

Moves the weights of the blocks that stay on the GPU to CPU memory and keeps the hooks. Returns the bytes moved. Call it
between two calls of the model.

### restore_gpu

`handle.restore_gpu() -> int`

Puts them back. Returns the bytes moved.

### remove

`handle.remove() -> None`

Takes the hooks off and restores the blocks. The weights stay in memory; use `release()` to free them.

```python
handle.release_gpu()      # the VRAM is free while something else runs
...
handle.restore_gpu()      # before the model runs again
```

## Release {#release-class}

What `subscribe()` queues: `name`, `tier` (`"gpu"`, `"ram"` or `"disk"`), `nbytes` (what the block held in that tier; 0
for disk), `event` (a CUDA event that marks when the GPU memory can be reused, or `None`) and `queued` (the time it was
queued). Import it from `trash_panda.offload.hooks`.

::: trash_panda.offload.hooks.release
    options:
      show_docstring_description: false

Gives back the memory a model holds: removes the hooks, drops every weight (the model ends up on the meta device) and
empties the GPU cache and PyTorch's cache of pinned memory. Pass the handle from `install_offload()`. Safe to call twice;
the model cannot run afterwards.

```python
release(model, handle)
```

## DiskReader

Reads one block's tensors from the checkpoint file. Used internally for the disk tier; you do not need it.
