# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com), versions follow [Semantic Versioning](https://semver.org).

## [0.0.19] - 2026-11-02
### Added
- `stage(..., overlap=True)`: the resident blocks are not loaded before the first call. A worker thread loads them one at a time, in the order the blocks run (the order given by `order=` or the order file), and the model starts as soon as its first block is on the GPU. A block that starts before its weights have arrived waits for them; if the loading fails, the call that needs the missing block raises a `RuntimeError` that names it, and a warning is given. Removing the handle stops the thread after the block it is loading. The weights outside the blocks, the blocks in RAM and the blocks on disk are loaded as before. Off by default
- `stream_load(..., defer_prefixes=...)`: resident blocks whose tensors are left on the meta device. `model._trash_panda_state.deferred.load(prefix)` loads one of them. `install_offload()` uses it to start the worker thread, and puts the hooks of such a block in when its weights have arrived, because the hooks are built from the tensors
- `offload_blocks(..., names=, hint=)`: a name for each block, which `install_offload()` now passes instead of setting it afterwards, and `hint=False` for a block installed while the model runs, which has no place in the order given
- `TRASH_PANDA_RAM_IN_MEMORY=1` (`ram_in_memory` in the config) reads the tensors of the blocks that wait in RAM into memory of their own while loading, with one big read per tensor (`Checkpoint.read_tensor()`), instead of keeping them mapped from the checkpoint file. `TRASH_PANDA_RAM_READ_THREADS` (`ram_read_threads`, default 1) is the number of threads that do the reading; the tensors are read in the order of the file and placed in that order. A read that fails stops the load. Off by default
- The profile (`TRASH_PANDA_PROFILE=1`) shows more. Each pass line now ends with the GB and GB/s of RAM copies and disk reads done while it ran, so a cold first pass shows as a low GB/s. The six slowest blocks of the first pass are listed with their time split into waiting for the copy, waiting for the GPU and the rest. With `overlap=True` a `loader thread:` line gives the time the loading took and the time the main thread waited for it. A `stream_load took X s (...)` line gives the time per tier, and with `ram_in_memory` a second line gives the RAM-tier read. A `memory while releasing` block is written at five points of `release()`: the RAM the machine has free, the working set and the private bytes of the process, the CPU tensors of 1 MB or more still alive, and, when more than 0.1 GB of them stays alive, what holds the biggest. It scans the live objects, which costs a little time, and only runs with the profile on
- README: a section "Settings and environment variables" that lists every setting with its default, what was measured, the current recommendation, and how to read the profile. The roadmap no longer lists the items that are done

### Fixed
- `release()` did not free the original tensors of wrapped blocks while the caller still held the handle, because the prefetcher keeps the chain of entries for as long as the handle lives, and each entry held its block's original tensors. Undoing the hooks now empties the entry. A test with wrapped blocks and a handle that outlives the release failed before and passes now. With the memory-mapped RAM tier this was not visible in the free RAM, because those pages are file-backed and count as available; with private tensors it is. When `handle.final_pass()` was used, the blocks were already freed as they ran
- On Windows, memory read into `torch.empty` tensors stayed in the process after it was freed, because PyTorch's CPU allocator keeps the blocks. After a model read with `ram_in_memory` was released, the next `plan()` saw 2.0 to 2.6 GB of RAM budget instead of about 9 GB, until Windows trimmed the working set later. Reproduced without the pipeline: 3 GB of tensors made with `torch.empty` (on the main thread or on worker threads) and freed left the working set at 3.4 GB for at least 7 s, while tensors made on a `bytearray` or a numpy array gave the memory back at once. `Checkpoint.read_tensor()` allocates with a `bytearray` (about 1 s of zero-fill for 7.3 GB). On the test machine the next plan then saw 8.2 to 9.3 GB

### Changed
- `stage(overlap=True)` does not remember the peak GPU memory for the next plan, because the load falls inside the measured part of the run
- `stream_load()` puts the code that places one tensor into a function that the deferred loading reuses; the result for a load without deferral is the same

### Notes
- The numbers below are single runs on the test machine (Windows 11, RTX 3060 with 12 GB, 16 GB of RAM, a SATA SSD), for a text encoder, a 12B diffusion model (fp8 with scales; 12 blocks, 5.21 GB, on the GPU and 20 blocks, 7.29 GB, in RAM) and a VAE, making one 1024 x 1024 image in 7 steps. Identical runs differed by 2 to 3 s in total time
- Every cold read ran at 0.36 to 0.42 GB/s whichever way it was read: the text encoder's load (3.63 GB, 9 s), the diffusion model's resident blocks (5.21 GB, 12.7 s) and its RAM tier in the first pass (0.37 to 0.42 GB/s). About 16 GB are read cold in a run, roughly 40 s of about 95 s. The slow first step of the diffusion model (20 to 22 s against 6.3 s for the others) is the first read of the RAM tier from the drive: the main thread waited 12.5 to 15 s for blocks in it, and about 2.8 s of the step is one-time warm-up in its first three blocks. Neither the missing prefetch (see 0.0.18) nor a busy GPU was the cause
- Defaults: 94.9, 95.6, 96.5 and 98.3 s in four runs. With `ram_in_memory` the first step takes 9.0 s and the sampling 47 s instead of 61 s, but the RAM tier is read at load, 7.29 GB in 20 to 23 s (0.31 GB/s with one thread, 0.34 to 0.36 with two), and the total is 104 to 108 s. The read is slower than the memory-mapped copy and no longer overlaps with the compute of the first step. Recommendation: leave it off, unless steady steps from the first one are worth about 10 s of total time and about 7 GB of private memory
- `stage(overlap=True)`: 97.4, 97.8, 98.8 and 96.5 s against 94.9 to 98.3 s without it, and 109.3 s together with `ram_in_memory` against 105.0 s. It cannot help much on this pipeline. A resident block takes about 1.1 s to load and 0.2 s to run, so only about 3 s of compute overlaps with the load of the 12 resident blocks, and the text encoder computes for 0.2 s against an 11.7 s load. The loader thread also reads with plain seek-and-read, at 0.25 to 0.28 GB/s against 0.41 GB/s for the memory-mapped reads of `stream_load()` for the same blocks. It is groundwork for starting one model's loads behind another model's compute, not a speed-up for this pipeline. Recommendation: leave it off
- A read-ahead of the RAM tier was built and removed before this release. Helper threads read the file ranges of the next blocks in large chunks, ahead of the copy, hoping the pages would be in the file cache for the memory-mapped copy. The first step got slower (34.7 s against 21 to 22 s; the copy still took 32 s for 7.3 GB), and the pages it read did not serve the memory-mapped copy
- The 0.0.18 entry and the README suspected that the missing prefetch, or a warm file cache, explained the slow first step. The measurements above replace those suspicions: it is the drive
- 254 tests pass on the CPU

## [0.0.18] - 2026-11-01
### Added
- The prefetcher is told the order of the blocks when the hooks are installed, so the first pass can be prefetched; before, it learned the order during the first pass and the first pass of any model, and every pass of a model that runs once, was not prefetched at all. By default the order is the one the plan lists the blocks in, which is the order the model defines them in. The order seen at run time still corrects the links from the second pass on
- `order=` on `stage()` and `install_offload()`: a list of block prefixes that run first, in that order; the other blocks follow in the plan's order. A name that is not in the plan raises a `ValueError`. With `order` given, the order file below is neither read nor written
- A warning, once per model, when the first blocks that run do not follow the order given. It names the blocks and says how to pass `order=`. For the Krea2 diffusion model it appears without `order=`, because the model runs its four text blocks before its 28 main blocks and defines them after
- With `order=None`, the order the blocks ran in is saved when the model is released, in `<checkpoint name>_order.json` next to the checkpoint, and read the next time. `order_file_destination=` on `stage()` and `install_offload()` is the folder to use instead. The file lists every block that ran, wrapped or not, so a block that never runs (the encoder blocks of the VAE in a decode) is absent. It is written only when there was no file or the order in it was wrong, a file that names a block the model does not have is ignored and replaced, and a folder that cannot be written gives one warning. One line is printed when the file is saved and one when it is used
- On the test machine, the Krea2 pipeline saved files for the text encoder (36 blocks), the diffusion model (32 blocks) and the VAE (6 of its 8 planned blocks), and the next run used the first two with no warning. The time did not change in a way that stands out from the variation between runs: 62 to 66 s for the 7 sampling steps, 8.98 to 9.46 s per step, with and without the order given. The first step of the diffusion model took 25.2 s before and 21.8 s with the order, against about 7 s for the others. The missing prefetch is therefore probably not the main reason the first step is slow; this is a conclusion from single runs, not a measurement of the cause

### Changed
- Resident blocks that are not wrapped get a forward pre-hook that notes the block's name when it starts, so that the order file lists them. The hooks are removed with the handle
- With `order=None`, a file named after the checkpoint is now created next to it by the first run of a model that has run blocks. Delete it to start over

### Notes
- The file's name is the checkpoint's file name plus `_order.json`. Two checkpoints with the same file name (diffusers models are usually all `diffusion_pytorch_model.safetensors`) share a name if `order_file_destination` is one folder, and then overwrite each other's file. The default location, next to the checkpoint, does not have this problem. How to name the file is not decided

## [0.0.17] - 2026-10-31
### Added
- `handle.final_pass()` on the handle that `install_offload()` and `offload_blocks()` return. The caller says which call is the last one (for example before the last denoising step, or, with classifier-free guidance, before the last of the two calls of that step); from then on every block is freed right after it has run, blocks that were on the GPU from the start included, and the model cannot be run again. The weights outside the blocks are not freed by it; `release()` does that. Calling the model after the final pass does not raise an error: the freed blocks are on the meta device and give meta tensors back
- On the test machine, with temporary logging, the memory PyTorch had allocated fell block by block during the final pass of each model of the Krea2 pipeline. Text encoder: 36 layers, from 4.36 to 1.00 GB, about 0.1 GB per layer. Diffusion model: 32 blocks, from about 6 to about 1 GB during the last call of the last step, with small rises in between from the blocks staged ahead. VAE: 8 planned blocks of 177 MB in total; the allocated memory rose during the decode, because the decode's own tensors grew by more than the blocks freed. The memory PyTorch had reserved did not change (4.89 GB for the encoder, 9.87 GB for the diffusion model), so Task Manager and `nvidia-smi` do not show the drop. Whether the next model can use the freed memory is not tested yet

### Changed
- **Breaking:** `stage()` now hands `(model, plan, handle)` to the `with` block instead of `(model, plan)`, so the block can call `handle.final_pass()`. Code that unpacks two values fails with an unpack error and needs a third name, for example `as (unet, p, _)`
- `ruff check` is set up for development (`ruff` in the dev dependencies, rules `E4`, `E7`, `E9` and `F`, line length 120). The code is not reformatted. It found two unused imports in `stage.py` and a lambda assigned to a name in `install_offload()`, both fixed

## [0.0.16] - 2026-10-30
### Added
- `plan()` and `stage()` take `key_map` and `skip_prefixes`, with the same meaning as in `stream_load()`. `plan()` sizes each block from the checkpoint tensors whose names map to the model's names through `key_map`, and leaves out the model entries under `skip_prefixes`: blocks under those prefixes are not planned, and their tensors are not counted in the weights outside the blocks. `stage()` passes both to `plan()` and to `stream_load()`. A checkpoint with other names than the model, or with parts the model does not need (the vision tower of a text encoder), can now go through `stage`
- On the test machine, the text encoder of the Krea2 pipeline (Qwen3-VL, vision tower skipped) ran through `stage`. The plan found all 36 language layers and placed them on the GPU in compact fp8 (3.63 GB), where the code written by hand kept them in RAM and streamed them. Sampling in the same run took 8.90 s per step, against 9.05 s before. The time to encode the prompt was not measured

### Changed
- `plan()` matches checkpoint tensors to blocks by the model's names (after `key_map`) instead of by the checkpoint's names. A checkpoint tensor that matches no entry of the model is no longer counted in the size of a block; it is counted with the weights outside the blocks. Nothing changes when the checkpoint and the model use the same names

## [0.0.15] - 2026-10-29
### Added
- `stage(build, path, *, key=None, **plan_kwargs)`, a context manager that runs one model of a pipeline. It calls `build()` for the model on the meta device, plans it (`plan_kwargs` go to `plan()`), loads it, installs the hooks and hands the model and its plan to the `with` block; when the block ends, also with an error, it calls `release()`. It records the peak GPU memory of the block under `key` (default: the checkpoint path) and passes it as `activation_vram` to the next plan with the same key in the same process, unless `activation_vram` is given. Nothing is recorded when the block raises, and the model cannot be used after the block. It does not apply adapters, and the checkpoint names must match the model's names (no `key_map` or `skip_prefixes` yet)
- On the test machine, the diffusion model and the VAE of a pipeline ran through `stage` with the same plan, the same sampling time (9.05 s per step, against 9.06 s) and the same memory returned on release as with the calls made by hand

### Fixed
- `measure_activation_vram()` no longer counts the blocks staged ahead by the look-ahead as activations. They are real GPU tensors and are in the peak, and the plan already reserves them separately (`Plan.staging_bytes`), so the measured value was too high by up to that amount and a plan made from it kept fewer blocks on the GPU than necessary. In a test on the test machine (RTX 3060, 12 GB) with a 12B-parameter model, the same run measured 0.59 GB with the look-ahead off and 1.43 GB with a look-ahead of 2 (0.87 GB staged); it now measures 0.56 GB with a look-ahead of 2. A run that makes a single pass through the blocks stages nothing, so for it the value can be too low
- `measure_activation_vram` can be imported from `trash_panda`, as the README example does; it was missing from the package's list of names

## [0.0.14] - 2026-10-28
### Changed
- `plan()` now counts each pinned buffer at the size PyTorch gives it. PyTorch rounds every pinned allocation up to the next power of two (measured on PyTorch 2.5.1: two buffers of 0.7 GB took 2.15 GB), so the RAM the planner reserved for the pinned buffers was too low. The RAM budget is now smaller by the difference, which can move a block from the RAM tier to disk when RAM is nearly full

## [0.0.13] - 2026-10-27
### Added
- `release(model, handle=None)`, which gives back the memory a model holds once it is no longer needed: it removes the offload hooks (when the handle from `install_offload()` or `offload_blocks()` is passed), replaces every weight and buffer with an empty tensor on the meta device, and empties the GPU cache and PyTorch's cache of pinned memory. The model cannot be run afterwards, and calling it twice is harmless. `handle.remove()` alone restores the blocks but frees nothing, because the weights stay alive. This makes it possible to run a pipeline in stages (for example text encoder, then UNet, then VAE) and give back each stage's RAM and VRAM before the next one is planned or loaded
- `examples/sdxl_benchmark.py` has a `trash-panda-release` mode and ends every single-mode run with a table of the RAM and VRAM that the whole system has free at several points, together with the memory held by the script's own process. When the process still holds more than 0.5 GB above its starting level, the script runs a probe in a fresh process (`--mode probe`) that shows what the CUDA libraries and pinned buffers hold; `--no-probe` skips it

### Changed
- The prefetcher no longer keeps a reference to the first block it saw once it is closed, so that block's weights can be freed

### Notes
- On the test machine (RTX 3060, 16 GB RAM) with the SDXL UNet streamed under a 3 GB limit, `release()` returned between 4.4 and 6.6 GB of RAM and about 9.5 GB of VRAM at once, while the rest of the pipeline was still loaded. Removing the handle with `handle.remove()` returned none of it
- PyTorch keeps freed pinned memory for reuse, and `torch.cuda.empty_cache()` does not release it. In the same test about 1.5 to 2 GB of RAM stayed in use after the release until the pinned cache was emptied, which `release()` now does when PyTorch provides the call

## [0.0.12] - 2026-10-26
### Added
- A warning (once per run) when pinned memory cannot be allocated. The library then copies blocks through ordinary memory, which is slower and no longer overlaps with compute; before, this happened silently
- `plan()` warns when the GPU part of the plan fills more than 90% of the free VRAM (`vram_fill_warn_fraction` in the config), because a GPU that runs out of memory silently spills into shared memory on Windows and every step becomes much slower. A plan that exceeds the VRAM budget still gets the existing, stronger warning, and only that one
- The prefetch profile now shows the time of each pass through the blocks and how much of it the main thread spent waiting for staged blocks (the first five and the last two passes when there are more than ten), and how long the worker waited for a free pinned buffer. The per-pass lines show whether the waiting is concentrated in the first pass
- `trash_panda.offload.config`, which holds the default values of the tuning settings in one place: the look-ahead depth, the number of pinned buffers, how far the GPU may lag behind the copies, the VRAM headroom and the working-area factor, the automatic RAM reserve, the margin around a measured activation size, the commit-limit warning, the thresholds for finding blocks, and the profiling switch. Each can be overridden with an environment variable named `TRASH_PANDA_<SETTING>` (for example `TRASH_PANDA_LOOKAHEAD=0`). An argument passed to a function still wins over the environment, and the environment over the shipped default
- `examples/sdxl_benchmark.py` takes a `--lookahead` option and prints the look-ahead it used, and it runs on machines without CUDA (the Apple GPU or the CPU; only the baseline and trash-panda modes, with no peak VRAM). The path without CUDA is untested
- `docs/pipeline-design.md`, which records the measurements and the proposals for coordinating memory and I/O across a pipeline (releasing finished stages, reading the RAM tier ahead of use, prefetching single-pass models)

### Changed
- `plan()`, `find_blocks()` and the prefetcher now take their defaults from the config when an argument is left out (`None`). The default values themselves are unchanged
## [0.0.11] - 2026-10-23
### Added
- A look-ahead queue for prefetching: the background thread now stages the next `lookahead` blocks that need a copy (default 2), not just the next one, and disk blocks are read ahead the same way. `install_offload(..., lookahead=)` and `plan(..., lookahead=)` set it; `0` turns prefetching off. A depth of 4 was no better than 2 in testing and uses more VRAM
- Back-pressure: the GPU is not allowed to fall more than two blocks behind the copies, so staged blocks cannot pile up in VRAM
- `plan()` reserves VRAM for the blocks staged ahead (`Plan.staging_bytes`, `Plan.lookahead`) and lists it in the summary, and spreads the disk blocks evenly through the model (`spread_disk=True`) so reads are not bunched together
- Opt-in prefetch profiling: `install_offload(..., profile=True)` or `TRASH_PANDA_PROFILE=1` prints, at exit, how long the worker spent on disk reads and RAM copies, how long the main thread waited for staged blocks (with how many were ready at once and the five slowest waits by block position) and for the GPU
- `examples/disk_threads_test.py`, which measures how many MB/s a drive gives with 1, 2, 4 and 8 reader threads

### Changed
- Blocks that are already on the GPU are no longer sent through the background thread when they run, so they cannot queue behind copies that are still waiting. This did not change the step time in testing (9.39 s to 9.26 s, within the run-to-run variation)

### Notes
- On the test machine (RTX 3060, 16 GB RAM, SATA SSD) the profile shows that with 11 blocks on disk the drive's speed (about 400 MB/s) sets the floor of the step time, and more reader threads did not help

## [0.0.10] - 2026-10-22
### Changed
- Documentation only: the README and `BENCHMARK.md` were corrected and extended (the roadmap item on sudden exits, the description of how far ahead disk blocks are read, a result for 0.0.9). No change to the code

## [0.0.9] - 2026-10-22
### Added
- Disk blocks are now read ahead of time: while one block computes, the background thread reads the next block's tensors from the checkpoint straight into a pinned buffer and starts the copy to the GPU, instead of reading them when the block is needed. The look-ahead is one block, so on a slow drive most of the read time is still not hidden; a deeper look-ahead is planned
- `Checkpoint.read_into()`, which reads a tensor's bytes from the file straight into a buffer you give it, such as a slice of a pinned buffer, with no intermediate copy
- `DiskReader.specs()`, which lists the key, dtype, shape and size of a disk block's tensors from the checkpoint's header
- `examples/`, a folder that is not part of the package, with an SDXL benchmark as a script and as a notebook. It compares plain `diffusers`, `diffusers` model offloading, `accelerate` and trash-panda on the same prompt and seed, with each mode in its own fresh process, and reports the time, the peak VRAM and the peak RAM
- `BENCHMARK.md`, with the test machine, a first SDXL comparison, and a section on what the numbers do not show (single runs, memory that shifts between runs, unequal VRAM between tools). The results are indicative only and do not show that trash-panda is faster than the other tools
- A "Benchmarks" section in the README that points to them

## [0.0.8] - 2026-10-21
### Added
- The copy into the pinned buffers now runs on a background thread, so the main thread keeps launching compute while the next block is being copied. On the same 12B model with all blocks in RAM (5.2 GB of RAM in use at the start) steps took about 9.0 s
- `handle.remove()` now also stops the background thread and frees the pinned buffers

### Changed
- `plan()` counts the pinned buffers in the RAM budget (2 buffers, each as large as the largest block that is not on the GPU, about 0.9 GB for a 12B fp8 model) when a GPU is present, so it no longer over-commits RAM
- Errors in the background copy are printed instead of silently falling back to the plain copy

## [0.0.7] - 2026-10-20
### Added
- Blocks waiting in RAM are now copied to the GPU through a small pool of pinned (page-locked) buffers, so the transfer is truly asynchronous and overlaps with compute. On a 12B model with all blocks in RAM (RTX 3060 with 12 GB, 16 GB of RAM) steps took about 9.2 to 9.4 s instead of 10.3 to 10.9 s. If pinned memory cannot be allocated, the plain copy is used

### Notes
- The pinned buffers use about 1 GB of RAM that Windows cannot page out, and the planner does not count it in the RAM budget yet

## [0.0.6] - 2026-10-20
### Added
- `plan(activation_vram=...)`: tell the planner how much GPU memory your activations need, instead of the fixed `headroom_vram` guess. The planner leaves that much plus a small margin (5% of the card's memory, at least 0.5 GB) and fills the rest of the GPU with resident blocks
- `measure_activation_vram(fn, plan)`: runs a real call once and returns the peak GPU memory it needed beyond the loaded model, to use as `activation_vram`
- Planner tests for the new RAM reserve and for `activation_vram`

### Changed
- `plan()`: the default `min_free_ram` is now `"auto"` (8% of the machine's RAM, between 1 and 4 GB) instead of a fixed 3 GB. Passing a value, including `0`, works as before

## [0.0.5] - 2026-10-19
### Added
- Prefetching: while one block computes, the next block's compact tensors are copied to the GPU on a second CUDA stream. The execution order is learned during the first pass. Applies to blocks waiting in RAM; disk blocks are not prefetched yet

### Changed
- License changed from Apache-2.0 to AGPL-3.0-only. Versions 0.0.1 to 0.0.4 remain available under Apache-2.0

## [0.0.4] - 2026-10-18
### Fixed
- Windows error 1455 ("paging file too small") when reading disk-tier blocks while the model runs. Each read used to memory-map the whole checkpoint; it now seeks and reads only the bytes it needs

### Changed
- While loading (`with checkpoint:` / `stream_load`) the checkpoint is opened once with safetensors' memory map, so tensors waiting in RAM stay file-backed and Windows can drop and re-read them instead of paging them out. Disk-tier reads at run time use plain seek-and-read
- `Checkpoint` reads the safetensors header itself (byte offsets are kept in `TensorInfo`)

## [0.0.3] - 2026-10-18
### Added
- `plan()` and `Plan.manual()` to decide where each block lives (GPU, RAM or disk), automatically or by hand
- `stream_load()`: meta-device creation with lazy, tensor-by-tensor, sharded loading
- `install_offload()` and `offload_blocks()`: stream one block at a time through the compute device
- Support for fp32 / fp16 / bf16 as stored, and fp8 with or without scale tensors
- Optional quantize-on-load to fp8
- Load and plan reports
- CPU-only test suite and a test workflow that gates publishing

### Not yet supported
- Prefetching, int8 / 4-bit formats, adapters, multi-GPU

## [0.0.2] - 2026-10-18
### Added
- Packaging and trusted publishing to PyPI (no functionality yet)
