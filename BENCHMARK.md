# Benchmarks

Two sets of measurements, both from one machine. **Please read "What these numbers do not show" before using any
of them.** They are single runs, not averages, on a Windows desktop whose free memory changes from run to run.

## Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| RAM | 16 GB |
| Storage | SATA SSD (about 500 MB/s) |
| OS | Windows 11 Home |
| CPU | AMD Ryzen 7 3700X (8 cores) |
| PyTorch | 2.5.1 (CUDA 12.1 build) |
| Python / NVIDIA driver | Python 3.9.6 / driver 610.88 |

## 1. SDXL, compared with other tools

You can reproduce this one: the script is [examples/sdxl_benchmark.py](examples/sdxl_benchmark.py), and every mode
runs in its own fresh process.

Stable Diffusion XL base 1.0 in fp16, one image at 1024 x 1024, 20 steps, seed 1234. The UNet is the only part that
is offloaded. "UNet weights limit" is how much of the UNet's weights may sit on the GPU for `accelerate` and
trash-panda; "none" means no limit, so the whole UNet is on the GPU.

| Setup | UNet weights limit | RAM in use at start | Total (s) | Per step (s) | Encode + decode (s) | Peak VRAM (GB) | Peak RAM (GB) |
|---|---|---|---|---|---|---|---|
| diffusers, no offloading | none | 10.0 GB | 22.1 | 0.80 | 6.0 | 11.26 | 5.07 |
| diffusers model offloading | none | 9.8 GB | 20.5 | 0.84 | 3.7 | 5.59 | 8.71 |
| accelerate device map | 3 GB | 8.8 GB | 46.3 | 2.25 | 1.3 | 6.60 | 7.06 |
| accelerate device map | none | 9.1 GB | 22.6 | 0.80 | 6.6 | 11.26 | 5.21 |
| trash-panda | 3 GB | 10.0 GB | 30.8 | 1.45 | 1.7 | 7.84 | 8.54 |
| trash-panda | none | 8.4 GB | 18.9 | 0.72 | 4.6 | 11.25 | 5.20 |

The time is for the measured generation, after one warm-up generation. The VAE decodes in tiles in every mode,
because without it the decode overflowed the GPU in the "no offloading" mode and took minutes.

### What these numbers do not show

**They do not mean that trash-panda is better than the other tools.** This test is flawed, and the results are
indicative at most:

- **Each row is one run.** Only one setup was repeated, and that repeat agreed with the first run to within 0.01 s per
  step. Differences of 5 to 10 percent (for example 0.72 s against 0.80 s per step with no limit) are within what a
  single run can vary, and should not be read as one tool being faster.
- **The memory state was not controlled.** See the next section.
- **The limited runs are not an equal comparison.** The limit applies to the UNet's weights only, so the tools end up
  with different peak VRAM (6.60 GB for `accelerate`, 7.84 GB for trash-panda, for the same 3 GB limit). The extra VRAM
  that trash-panda used is not explained yet, and may be a weakness in the library.
- **The lowest peak VRAM is diffusers model offloading** (5.59 GB, at 0.84 s per step). It works because SDXL's UNet
  fits on the GPU on its own, and it cannot go lower than the size of the UNet. A model that does not fit on the GPU
  is the case where the other tools could matter, and this test does not cover it.
- **The fastest row is trash-panda with no limit** (0.72 s per step), but with no limit nothing is offloaded: the
  whole UNet is on the GPU, as in the first row, and the 10 percent gap is within what one run can vary.
- **One machine, one model, one image size.** Other GPUs, drives, operating systems and models can behave
  differently.

### Why the memory shifts

On Windows the memory that is in use at the start of a run is not stable. The "RAM in use at start" column ranges from
8.4 GB to 10.0 GB for the runs above, and it ranged from about 5 GB to over 10 GB over the days these were measured,
for reasons outside the benchmark:

- Windows keeps recently read files (such as the 7 GB of model weights) in memory, and gives that memory back only
  when something else needs it.
- Background programs (browsers, chat apps, game overlays, editors) grow and shrink while the machine is running.
- A Python process does not always give memory back to Windows when objects are deleted, so a long-running session
  can keep growing.

How much RAM is free at the start changes how many blocks fit in memory, whether Windows has to page, and how long
loading takes. The runs above were each started in a fresh process, and the starting value is printed for each, but
the machine was not rebooted between them. A fair benchmark would repeat each run several times from a similar
state, and report the median.

## 2. A 12B diffusion transformer (trash-panda only)

A different model, run only with trash-panda: one measurement on the same machine, to show the speed with the weights
split between the GPU and RAM.

| | |
|---|---|
| Model | a 12B diffusion transformer, fp8 with per-tensor scales |
| Text encoder / VAE | a 4B text encoder (fp8, streamed with trash-panda) and an image VAE |
| Image size and steps | 1024 x 1024, 7 steps |
| Seed | 2219653787 |
| What is timed | the sampling progress bar only (not model loading, prompt encoding or the VAE decode) |

Memory in use at the start of each run: about 5.2 to 5.7 GB of the 16 GB (a clean desktop, no browser or games open).
The encoder and VAE were loaded before the plan.

| Library version | Plan | Memory in use at start | Time (7 steps) | Per step |
|---|---|---|---|---|
| 0.0.7 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.7 GB | 1:04 | 9.2 s |
| 0.0.8 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.2 GB | 1:02 | 8.95 s |
| 0.0.9 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 6.1 GB | 1:04 | 9.26 s |
