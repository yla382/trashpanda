# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com), versions follow [Semantic Versioning](https://semver.org).

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
