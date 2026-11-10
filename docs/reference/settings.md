# Settings

The tuning settings live in one place, `trash_panda.offload.config`. Each one can be set with an environment variable,
`TRASH_PANDA_` followed by the name in capitals, for example `TRASH_PANDA_LOOKAHEAD=0`.

**Precedence:** an argument passed to a function, then the environment, then the default below. The environment is read
when a function runs, not when the library is imported. A boolean takes `1`/`0`, `true`/`false`, `yes`/`no` or
`on`/`off`; an invalid value raises a `ValueError` that names the variable.

In PowerShell a variable lasts for that window only:

```powershell
$env:TRASH_PANDA_PROFILE = "1"
Remove-Item Env:TRASH_PANDA_PROFILE
```

In bash, use `export TRASH_PANDA_PROFILE=1` and `unset TRASH_PANDA_PROFILE`.

Two options are arguments of `stage()`, not settings: `overlap=True`, and `order=` with `order_file_destination=`.

## Prefetching

| Setting | Default | Meaning |
|---|---|---|
| `lookahead` | `2` | blocks ahead whose weights are copied while one computes (`0` turns prefetching off) |
| `pinned_buffers` | `2` | page-locked host buffers that blocks are copied through |
| `blocks_in_flight` | `2` | the GPU may run at most this many blocks behind the copies |
| `adaptive_lookahead` | `false` | try a deeper look-ahead after the first passes, and keep it only if the passes get faster |
| `lookahead_max` | `4` | the deepest look-ahead the adaptive setting tries (the pinned buffers grow to match) |

## VRAM

| Setting | Default | Meaning |
|---|---|---|
| `headroom_vram` | `"1.5GB"` | VRAM kept free for activations when none was measured |
| `stream_buffer_factor` | `1.3` | working area for a streamed block, as a multiple of its expanded size |
| `activation_margin_fraction` | `0.05` | margin added to a measured activation size: this share of the VRAM ... |
| `activation_margin_min_gb` | `0.5` | ... but at least this many GiB |
| `vram_fill_warn_fraction` | `0.9` | warn when the planned GPU use is above this share of the VRAM the plan was made for |

## RAM

| Setting | Default | Meaning |
|---|---|---|
| `ram_in_memory` | `false` | read the tensors of streamed blocks into memory of their own while loading, not mapped from the file |
| `ram_read_threads` | `1` | threads that do that reading |
| `min_free_ram` | `"auto"` | RAM kept free for the system: `"auto"` or a size such as `"2GB"` |
| `ram_reserve_fraction` | `0.08` | `"auto"` keeps this share of the machine's RAM free ... |
| `ram_reserve_min_gb` | `1.0` | ... but at least this many GiB |
| `ram_reserve_max_gb` | `4.0` | ... and at most this many |
| `ram_reserve_unknown_gb` | `2.0` | the reserve when the machine's RAM cannot be read |
| `commit_warn_fraction` | `0.9` | warn when streamed blocks need more than this share of the commit limit |

## Finding the blocks of a model

| Setting | Default | Meaning |
|---|---|---|
| `find_blocks_min_fraction` | `0.01` | a group of repeated blocks must hold at least this share of the model's bytes |
| `find_blocks_min_blocks` | `2` | ... and have at least this many blocks |

## Diagnostics

| Setting | Default | Meaning |
|---|---|---|
| `profile` | `false` | print where the prefetching spent its time |

In code, `trash_panda.offload.config.DEFAULTS` holds the shipped values and `config.current()` returns them with the
environment applied.
