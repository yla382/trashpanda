# API reference

Every public name of `trash_panda`, by area: its signature, what it does in a sentence, its parameters with defaults,
what it returns, and an example. All of them can be imported from the top level, for example
`from trash_panda import stage`. How things work, and why, is in the [Design](../design/index.md) section.

| Area | Names |
|---|---|
| [Stages and pipelines](stages.md) | `stage`, `preload`, `PendingStage`, `record_order`, `Coordinator`, `Step` |
| [Planning](planning.md) | `plan`, `Plan`, `BlockPlan`, `Tier`, `Hardware`, `detect_hardware`, `parse_bytes`, `measure_activation_vram`, `find_blocks`, `BlockGroup`, `module_nbytes` |
| [Loading](loading.md) | `stream_load`, `LoadReport`, `Checkpoint`, `TensorInfo`, `Kind`, `parse_dtype`, `expand`, `quantize_fp8` |
| [Hooks and handles](handles.md) | `install_offload`, `offload_blocks`, `OffloadHandle`, `Release`, `DiskReader`, `release` |
| [Settings](settings.md) | the tuning settings and their `TRASH_PANDA_*` environment variables |

Start with `stage()` for one model, or a `Coordinator` for several. `plan()`, `stream_load()` and `install_offload()` are
the three calls that `stage()` makes, for when you want to do it yourself.

`Release` is not exported from the top level; import it from `trash_panda.offload.hooks`.
