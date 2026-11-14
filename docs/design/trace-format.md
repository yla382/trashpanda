# The trace file (format 2)

The file that `trash-panda trace` shows. It is a Chrome trace-event JSON file, so Perfetto and `chrome://tracing` open it
as a timeline, with one extra top-level section, `trashPanda`, that holds what the trash-panda viewer needs: the models,
their plans, the tracks and the phases. What the viewer has to show, and why, is in [the trace viewer](trace-viewer.md).

Status: the format, its checker (`trash_panda.ui.load.check_trace`), the recorder (`trash_panda/offload/trace.py`) and the
viewer exist. The recorder writes format 2; the viewer plays formats 1 and 2 (see "Versions").

## How a file is made

Recording is off unless `TRASH_PANDA_TRACE` is on (`1`, `true`, `yes` or `on`). A recording is a session: `stage()` opens one
if none is open, a `Coordinator` opens one for its whole `with` block, and every model that runs while it is open adds its
events. The call that opened the session closes it, and the file is written, when its block ends, also when the block
raises. One recording is open at a time.

The file is written to `TRASH_PANDA_TRACE_DIR` (by default the `traces` folder of `TRASH_PANDA_HOME`, by default
`~/.cache/trash_panda`). It is named by the time the recording began, to the millisecond (`20261114-143310-869.json`; `-2`,
`-3` follow the name if it is taken), and its path is printed. The hooks reach the recording through one object for each
model, which is `None` when nothing is recorded, so a run that is not recorded pays for an `is not None` test.

## Layout

```json
{
  "displayTimeUnit": "ms",
  "trashPanda": { "version": 2, "duration_us": 0, "models": [], "tracks": [], "pools": [], "phases": [] },
  "traceEvents": []
}
```

All times are microseconds from the start of the recording, as in the Chrome format. `duration_us` is the length of the
recording; no event ends after it.

## `trashPanda`

| Field | Meaning |
|---|---|
| `version` | `2` |
| `note` | optional text shown under the title |
| `duration_us` | the length of the recording |
| `models` | the models in the recording, in the order they started (below) |
| `tracks` | the threads and CUDA streams that events are on (below) |
| `pools` | the pinned pools (below); may be empty, for example without CUDA |
| `phases` | optional stretches of a model's life (below) |

### `models`

One entry per model: what one `stage()`, or one `Step` of a `Coordinator`, loads, runs and releases.

| Field | Meaning |
|---|---|
| `id` | a name that is unique in the file, used by every event (`args.model`) |
| `name` | the name to show: the step name, or the checkpoint's name |
| `tag` | optional short label for its blocks (`U`); the viewer makes one if it is missing |
| `started_by` | `stage`, `preload` (by hand) or `coordinator` |
| `checkpoint` | `{path, bytes}` |
| `dtypes` | `{stored, compute}`, for example `fp16` and `bf16` |
| `blocks` | every block: `{id, name, tier, stored_bytes, expanded_bytes}`; `id` is a number, unique in the model; `tier` is where the plan puts it: `gpu`, `ram` or `disk` |
| `outside_bytes` | the weights outside the blocks, kept expanded on the GPU |
| `plan` | the plan's figures: `vram_budget`, `ram_budget`, `headroom_bytes`, `stream_buffer_bytes`, `staging_bytes`, `lookahead`, `warnings` |
| `options` | what shapes the run: `overlap`, `ram_in_memory`, `ram_preload`, `vram_preload`, `pinned_buffers`, `blocks_in_flight`, `adaptive_lookahead` |
| `pool` | the `id` of the pinned pool its worker uses, or `null` |

Block numbers are per model. A block is named by the pair (`model`, `block`).

### `tracks`

A track is a thread or a CUDA stream.

| Field | Meaning |
|---|---|
| `id` | the number used as `tid` by events; unique |
| `name` | the name to show |
| `kind` | `cpu` or `gpu`: whose clock the events' times are on |
| `role` | `main`, `worker`, `loader`, `drop`, `preload`, `reader`, `compute_stream`, `copy_stream` or `early_copy_stream` |
| `model` | optional: the model whose prefetcher owns it (a worker, a copy stream, a loader) |

The CPU's time is when work is queued; the GPU's time is when it runs. A stall of the GPU shows on a `gpu` track and often
not on the main thread. The recorder times the `gpu` tracks with CUDA events.

### `pools`

| Field | Meaning |
|---|---|
| `id` | unique |
| `kind` | `stage` (made by one `stage()`), `shared` (a `Coordinator`'s) or `early_loader` (the two buffers of one early loader) |
| `buffers` | how many pinned buffers it has when the recording starts |
| `models` | the models that use it |

### `phases`

`{model, name, start_us, end_us}`. The names the viewer knows: `build`, `plan`, `load`, `pass N` (a pass starts when its
first block starts and ends when its last block ends), `release`. Phases of one model do not overlap; phases of different
models can.

The recorder writes `load` and `pass N`. For a model started by `stage()`, `load` is the time `stream_load` takes. For a model
that was preloaded (`preload()`, or a `Coordinator` step), `load` runs from the moment the model is entered until it is
ready: the time it spent in the background before its turn is not part of it, and that work has its own `preload` slice and
its loads on the early loader's track. A model whose blocks are all unwrapped has no `learn`, so a new pass is counted when a
block name repeats.

## Events

`traceEvents` holds three kinds. Metadata (`ph` `M`: `thread_name`, `process_name`) is for Perfetto and is ignored by the
viewer. Counters (`ph` `C`) are optional series: `vram`, `ram` and `pinned`, each with the bytes in use per model in
`args`. The recorder does not write counters; the viewer works the bytes out from the events. Everything else is a slice (`ph` `X`) or an instant (`ph` `i`) with a `cat` from the table below, a `tid` that is a
track, and `args`.

| `cat` | Kind | Where in the code | Required `args` (besides `model`) |
|---|---|---|---|
| `call` | slice | the main thread launching a block (`wrapped`) | `block`, `pass` |
| `run` | slice | a block's kernels on the compute stream | `block`, `pass` |
| `wait` | slice | a wait: `reason` is `gpu` (`wait_for_room`), `staged` (`take`), `loader` (`wait_ready`), `pinned buffer` (`acquire`) or `preload` (entering a preloaded model) | `reason`; `block` when it is for one |
| `load` | slice | a kept block loaded through ordinary memory (`stream_load`, the loader thread, the early loader) | `block`, `from` (`disk`), `via` (`ram`), `to` (`gpu`), `bytes`, `source` (`mmap`, `read` or `chunked`) |
| `read` | slice | the RAM tier read into memory (`to` `ram`), or a disk block read into a pinned buffer (`to` `pinned:N`) | `block`, `from` (`disk`), `to`, `bytes`; `pool` when `to` is a pinned buffer |
| `stage` | slice | a RAM block copied into a pinned buffer (`_stage_job`) | `block`, `from` (`ram`), `to` (`pinned:N`), `pool`, `bytes`, `source` (`memory` or `mapped`) |
| `copy` | slice | a pinned buffer copied to the GPU, on a copy stream | `block`, `from` (`pinned:N`), `to` (`staged`), `pool`, `bytes` |
| `alloc` | slice | a pinned buffer allocated or enlarged (`acquire`) | `pool`, `buffer`, `bytes` |
| `buffer` | instant | a pinned buffer taken or given back (`name` starts with `acquire` or `release`) | `pool`, `buffer`, `block` |
| `mapped` | instant | a RAM-tier block left mapped from the file at load | `block`, `bytes` |
| `move` | slice | `release_gpu` or `restore_gpu` moving a kept block | `block`, `from`, `to`, `bytes` |
| `free` | instant | a block freed after it ran in a final pass | `block`, `tier`, `bytes` |
| `drop` | slice | RAM-tier tensors dropped on the drop thread | `block`, `bytes` |
| `preload` | slice | the next model loading in the background | none |
| `replan` | instant | the plan changed when a preloaded model was entered | `demoted`, `promoted` (lists of block numbers) |
| `cancel` | instant | a preloaded model given up | none |
| `lookahead` | instant | the look-ahead changed between passes | `from`, `to` |
| `pool` | instant | a pool grew or was trimmed (no `model`) | `pool`, `buffers` |

A block that is not wrapped (already in the compute dtype on the GPU, with no scale) has no wrapper: its `call`, `run` and
`free` are recorded by a forward hook. When its weights are still arriving while the model runs (`overlap`), its first call
goes through a gate that installs the hook, and is recorded there. A model that calls each block once, such as a decoder,
has all its calls from that gate.

Places in `from`, `to` and `via`: `disk`, `ram`, `gpu` (the block's own slot, for a kept block), `staged` (on the GPU,
compact, waiting for its turn) and `pinned:N`, a buffer of the pool in `pool` (N counts from 1).

Not recorded: the individual tensors of a block (one event covers a block), the chunks of the early loader (one `load`
event covers a block), and the weights outside the blocks.

## What the checker rejects

* a file that is not JSON, has no `traceEvents` list or no `trashPanda` section, or has a version it does not know;
* a model, track, pool or phase with a missing or wrong field, or an `id` that appears twice;
* an event on a track that does not exist, of a model that does not exist, with a `cat` it does not know, of the wrong
  kind (a slice where an instant is expected), or with a required `args` field missing;
* a `block` that is not a block of the event's model, a `pool` that does not exist, a `pinned:N` beyond the pool's buffers,
  a place, reason, source or tier that is not in the lists above;
* an event that starts before 0 or ends after `duration_us`.

## Versions

Format 1 (`trashPanda.version` 1, or no version) has one model, `blocks` and `pinned_buffers` at the top, and slices with
`args.block`. The checker still accepts it and the viewer still plays it. The recorder writes format 2.

## Open points

* GPU times are put on the CPU's axis when the file is written, from a CUDA event recorded when the recording began; the file
  holds the result, not the offset. On Windows the driver can stamp an event when its batch is sent, long after it was
  recorded, so every event is followed by a query that sends the queue at once. The agreement was measured on one machine
  (RTX 3060, Windows): a block's `run` starts within about 25 microseconds of its `call`.
* The clock starts when the session opens, which is when the first `stage()` or `Coordinator` block is entered. Building and
  planning the first model are inside the recording and have no phase.
* `copy` ends in `staged` for a block that is streamed. A block that the early loader puts on the GPU is a `load`.
* Counters are optional, and the viewer works without them (it can add up the events' bytes instead).
