# The trace viewer: what it has to show

`trash-panda trace FILE` opens a recorded run in the browser: blocks moving between disk, RAM and the GPU, a timeline, and
the time each block took. This page lists what the code does that the viewer has to represent, where in the code each
thing happens, and how much of it the viewer shows today. It is the list the viewer is checked against.

Status: the viewer exists. It plays recordings (`TRASH_PANDA_TRACE=1`, see [the trace file](trace-format.md)) and the
made-up traces in `tests/ui/data/`.

## Principles

* The viewer draws what the code does, and nothing that the code does not have. Every name on the screen is a name from
  the code (a thread, a stream, a tier, a buffer).
* Where a block stands says its tier (disk, RAM, GPU). Its color says which model it belongs to. Red says that somebody
  waits.
* A simplification is named as one. What is not shown is listed on the page.
* The viewer works for any number of models, however they were started: one `stage()`, several `stage()` calls in a row,
  `preload()` by hand, or a `Coordinator`. What only a `Coordinator` has (a shared pinned pool, the early loader) is shown
  only when the trace has it.

## Vocabulary

| Word | Meaning in the code |
|---|---|
| model | what one `stage()` (or one `Step` of a `Coordinator`) loads, runs and releases |
| block | one repeated layer of a model, found by `find_blocks()`; the unit that moves |
| tier | where a block waits between uses: `gpu`, `ram` or `disk` (`Tier` in `planner.py`) |
| pass | one call through all the blocks; it starts when the first block starts (`_Prefetcher.learn`) |
| staged | copied to the GPU ahead of its turn, still in its compact form |
| compact form | the stored dtype (fp8 with a scale, or the file's own dtype); the block is expanded to the compute dtype only while it runs |

## Threads and streams

| Name | Made in | What it does |
|---|---|---|
| main thread | the caller | runs the blocks (`wrapped`): waits for the GPU, takes the staged block, launches the forward |
| `trash-panda` worker | `_Prefetcher.__init__` (`max_workers=1`) | the stage jobs, one after the other: RAM to pinned buffer, or file to pinned buffer, then the copy is started |
| `trash-panda-loader` | `_BlockLoader` (`overlap=True`) | loads the kept blocks one at a time while the model runs |
| `trash-panda-drop` | `_Prefetcher.drop` | lets go of RAM-tier tensors in a final pass |
| `trash-panda-preload` | `PendingStage` | loads the next model's RAM tier, then runs the early loader |
| reader threads | `stream_load` (`ram_read_threads`, default 1) | read the RAM tier into memory when `ram_in_memory` is set |
| compute stream | PyTorch | the blocks' kernels and the expansion of compact weights |
| copy stream | `_Prefetcher` (one per model) | the copies from pinned buffers to the GPU |
| early loader's copy stream | `_StagedCopy` | the copies of the next model's kept blocks |

## How a block moves

Nothing goes from the file to the GPU without passing through RAM.

| Path | Code | Route | Notes |
|---|---|---|---|
| Kept block, no `overlap` | `stream_load` `store()` | file, read through a memory map, then `.to(cuda)` | a pageable copy, not a pinned one; on the main thread, before the model runs |
| Kept block, `overlap=True` | `_BlockLoader`, `DeferredLoad.load` | file, plain seek-and-read into new memory (outside `with checkpoint`), then `.to(cuda)` | on the loader thread, while the model runs; the first call of the block waits (`wait_ready`) |
| Kept block left over at entry | `PendingStage.__enter__` | as the first row, inside `with checkpoint` | when a preloaded model is entered without `overlap` |
| Kept block of the next model | `_EarlyLoader`, `_StagedCopy` | file, memory map, then 64 MB chunks through two pinned buffers of its own, on its own stream | waits for free VRAM, or for memory a final pass has freed |
| RAM tier, at load | `stream_load` | mapped from the file by default (not read until first use), or read into ordinary memory (`ram_in_memory`) | with preload, only the blocks that fit the free RAM are read (`_early_ram_prefixes`); the rest stay mapped |
| Pass, RAM block | `_stage_job` | RAM to a pinned buffer (`copy_`), then an asynchronous copy to the GPU | for a mapped block the first `copy_` also reads the file |
| Pass, disk block | `_stage_disk_job` | file straight into a pinned buffer (`read_into`), then the asynchronous copy | |
| At each run | `bring`, `expand` | the compact weights are expanded to the compute dtype on the GPU | a block already in the compute dtype on the GPU, with no scale, is not wrapped (`_needs_wrapping`) and nothing happens |
| Between passes | `release_gpu`, `restore_gpu` | the kept blocks' weights go to CPU memory and back | `quiesce` first: staged blocks are dropped |
| Final pass | `wrapped`, `drop`, `announce` | each block is freed after it runs | RAM-tier tensors are dropped on the drop thread; a `Release` (name, tier, bytes, CUDA event) is queued for a subscriber |

## Two clocks

The CPU only queues the GPU's work. The main thread's time and the GPU's time are different things:

* The main thread waits in three places: `wait_for_room` (until the GPU has finished all but `blocks_in_flight - 1`
  earlier blocks), `take` (until the worker has *issued* the copy, `staged.result()`; not until it has finished) and
  `wait_ready` (until the loader has loaded the block). The worker waits for a pinned buffer in `_PinnedPool.acquire`.
* The wait for a copy to finish is on the GPU: `compute.wait_event(event)` makes the compute stream wait. The CPU does not
  see it.
* So a stall of the GPU shows on the GPU's tracks (compute and copy streams) and only sometimes on the main thread.

The recorder times the GPU's side with CUDA events: one before and one after each block's kernels (`run`), and one before
and one after each copy (`copy`). The events are resolved when the file is written, and put on the CPU's axis from one event
recorded when the recording began. On Windows the driver can hold an event back and stamp it when its batch is sent, so each
event is followed by a query, which sends the queue at once. On one machine (RTX 3060, Windows) a block's `run` then starts
within about 25 microseconds of its `call`.

## One model

The plan (`planner.plan`) fixes what the viewer starts from:

* The first `n` blocks (in model order) are on the GPU (`resident_blocks`, or as many as fit). The others are in RAM or on
  disk. The disk blocks are spread evenly among the others (`_spread_disk_blocks`), not put last.
* Weights outside the blocks (`other_bytes`) are kept expanded on the GPU. They are not blocks and do not move.
* The VRAM it counts on: `vram_budget` (less `headroom_bytes` for activations), the room for one expanded block
  (`stream_buffer_bytes`, `stream_buffer_factor` times the largest expanded block, 1.3 by default) and the staging (`lookahead` times the largest stored
  block). The RAM it counts on: `ram_budget`, less the reserve and the pinned buffers.
* Pinned buffers are allocated the first time they are used, and PyTorch rounds each up to the next power of two
  (`_pinned_size`). A bigger block replaces a smaller buffer.

The life of a stage: `build`, `plan`, `stream_load`, `install_offload`, the passes, `release`.

## Passes and sampling steps

* A sampling step can be one pass, or two (guidance). The passes are the same blocks in the same order.
* The order comes from the order file of an earlier run, or from the first pass (`_in_order`, `learn`). A block that runs
  out of order triggers a warning, and the order seen is saved when the handle is removed.
* In each pass every RAM or disk block is staged again. `stage_ahead` runs after a block has been taken and stages the
  next `lookahead` blocks that need a copy.
* The last block is linked to the first once the second pass has begun, so from then on the look-ahead reaches into the
  next pass.
* A staged block that is no longer wanted (`_drop`) is cancelled if its job has not started.
* With `adaptive_lookahead` (off by default) `DepthTuner` changes the look-ahead between passes, and the pinned pool grows
  to match. The staging the plan counted on is then partly unused (`spare_staging_bytes`).
* Disk blocks are read again in every pass. A mapped RAM block is read from the file when it is first copied; later
  passes read it from the file cache, as long as the pages are still in it.

## Release

* `handle.final_pass()`: from then on each block is freed after it runs, and the model cannot run again.
* `handle.release_gpu()` and `restore_gpu()`: the kept blocks leave the GPU and come back, between passes.
* `release(model, handle)`: the hooks are removed, the worker stops, the weights are dropped, and the GPU and pinned host
  caches are emptied.

## Several models

Without a `Coordinator`:

* Each `stage()` makes its own pinned pool and its own worker, and gives them up when it ends.
* `preload()` by hand loads the next model while the current one runs. It has its own pool unless `pool=` is given, and an
  early loader only if `after=` is given.

With a `Coordinator`:

* One pinned pool serves every model; it is trimmed back to `pinned_buffers` when a stage ends.
* While a model runs, the next step is prepared: its RAM tier is read early (`ram_preload`) and its kept blocks are loaded
  early (`vram_preload`) into free VRAM and into the memory the running model's final pass frees.
* On entry a preloaded model is replanned to the VRAM that is really free (`_replan`): blocks planned for the GPU can go
  to RAM, and RAM blocks can move to the GPU.
* A step can run again, a model nobody asked for is given up, and leaving `with co:` drops the pool.

What the viewer needs from every event: which model it belongs to. Block numbers are per model.

## Memory over time

The plan gives the budgets; the events give the bytes. The viewer can show VRAM, RAM and pinned memory in use against the
budgets, split by model, once every event carries its bytes.

## Edge cases

* Without CUDA there is no worker, no copy stream and no staging; blocks are copied in `bring`.
* If pinning fails, blocks go through ordinary memory (a warning).
* Blocks that never ran are listed by the profile (`blocks that never ran`).
* `skip_prefixes` leaves blocks out of the plan.
* A model with no repeating blocks has nothing to offload (a plan warning).
* A preloaded model that is given up (`cancel`) finishes its reads first, and its memory is freed.

## What the profile already counts

With `TRASH_PANDA_PROFILE=1` the prefetcher counts figures that a recorded trace has to reproduce. They are the check on
the recorder. So far the number of staged blocks has been compared with the profile on a tiny model; the waits and the reads
have not been compared on a real pipeline.

| Profile figure | In the trace |
|---|---|
| main thread waited for staged blocks | the waits in `take` |
| main thread waited for the GPU to catch up | the waits in `wait_for_room` |
| main thread waited for the loader (once) | the waits in `wait_ready` |
| worker, disk reads: GB and seconds | the file reads into pinned buffers |
| worker, RAM copies into the pinned buffers: GB and seconds | the RAM-to-pinned copies |
| worker waited for a free pinned buffer | the waits in `acquire` |
| per pass: time, and the wait for staged blocks | the pass events |
| loader thread: blocks and seconds | the loader's loads |
| final pass freed N blocks, by tier | the release events |

## What the viewer shows today

| Item | Today |
|---|---|
| Disk, RAM and GPU as stacked bands, every block labeled | drawn; a model with more than 20 blocks gets smaller tiles |
| Several models | drawn: every model in every band, a color and a tag for each, and a lane for each in the strip at the top |
| Pinned buffers inside the RAM band | drawn: each pool with its buffers, named by what owns it (one model's, the shared pool of a `Coordinator`, an early loader's own two) |
| Staged slots and the running slot | simplified: the slots stand for the look-ahead's tensors; nothing is pre-allocated |
| A kept block loaded through ordinary memory | drawn, as a tile that stands in RAM, in no slot of its own, while it loads; the early loader's blocks stand in its own buffers; a tile on its way is drawn above the others |
| Memory map against plain read | named in the tooltip of the load (`Read by`), not drawn |
| The early loader's chunks | not drawn: one load for each block |
| A mapped RAM tier, read at its first use | not drawn |
| Passes, a pass menu, the look-ahead into the next pass | drawn |
| An adaptive look-ahead | the setting is on the card of the model; the changes (`lookahead` and `pool` events) are recorded, not drawn |
| Waits for staged blocks, the loader, the GPU, a pinned buffer or a preload | drawn, with the reason |
| The two clocks, GPU tracks | drawn: the compute stream and the copy streams are tracks of their own; a gap in the compute stream is drawn as idle, worked out from the runs |
| The final pass | drawn: a block freed after it ran is drawn dashed, and the RAM tier is dropped on the drop thread's track |
| `release_gpu`, `restore_gpu` | recorded (`move`), not drawn in the memory view |
| A replan at entry | drawn: a block that moved between the GPU and RAM gets a tile in its new tier |
| The plan: budgets, tiers, the weights outside the blocks | shown on the card of the selected model, and as budget lines in the memory chart |
| Memory in use over time | shown, worked out from the events (VRAM, RAM and pinned memory, by model); activations and other programs are not counted |
| Compact kept blocks expanded on every run | not drawn as tiles; counted in the memory in use |
| Order warnings, cancelled staging, blocks that never ran | not drawn |
| The recordings of the trace folder | a list at the left, read-only: click one to open it, fold the list away with the button |

## The made-up traces

The `scenario_` files in `tests/ui/data/` are written by `tests/ui/simulation.py`, which schedules the work the way the code
does. The rules it follows, each with the code it names, are in its docstring, and `tests/ui/test_scenarios.py` checks every
scenario against them. The simulation does not run a model, so its times are round numbers, and most scenarios read the RAM
tier into memory (`ram_in_memory`), which is not the default; `mapped_ram` leaves it mapped. `recorded_tiny.json` is a
recording of a tiny model on the CPU, written by an early version of the recorder. The other files in that folder are older
made-up traces that the tests of the page still use.
