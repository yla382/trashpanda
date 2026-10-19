# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com), versions follow [Semantic Versioning](https://semver.org).

## [Unreleased]

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
