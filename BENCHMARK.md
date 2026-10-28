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

Memory in use at the start of each run: about 5.2 to 6.1 GB of the 16 GB (a clean desktop, no browser or games open).
Up to 0.0.11 the encoder and VAE were loaded before the plan and stayed loaded during the sampling. In the 0.0.13 run
the stages run one after another: the encoder encodes the prompt and is released with `release()` before the UNet is
planned, and the VAE is built only after the sampling, once the UNet has been released.

| Library version | Plan | Memory in use at start | Time (7 steps) | Per step |
|---|---|---|---|---|
| 0.0.7 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.7 GB | 1:04 | 9.2 s |
| 0.0.8 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.2 GB | 1:02 | 8.95 s |
| 0.0.9 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 6.1 GB | 1:04 | 9.26 s |
| 0.0.11 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.2 GB | 1:04 | 9.20 s |
| 0.0.13 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 5.4 GB | 1:03 | 9.06 s |

With this much free memory the plan is the same as in the earlier rows, and the sampling time is unchanged within the
run-to-run variation (9.06 s against 9.20 s per step is a 1.5 percent difference from single runs). Releasing the
encoder before the plan does not make the sampling faster when the plan already fits in RAM. What it changes is the
memory: in the 0.0.13 run the encoder's release gave back 4.7 GB of RAM and 1.6 GB of VRAM before the UNet was planned,
and the release of the UNet gave back 10.3 GB of RAM and 10.4 GB of VRAM before the VAE was built. The RAM left free at
the end of the sampling was only 0.3 GB, because this run used `min_free_ram="0"` and no cap on the RAM tier.

### When memory is tight

The release matters more when less memory is free at the start, because the plan then has to leave blocks on disk, and
the drive (about 400 MB/s) sets the speed of each pass. Two runs started with about 8 GB in use:

| Flow | Plan | Time (7 steps) | Per step |
|---|---|---|---|
| Before: encoder and VAE stay loaded | 13 blocks on the GPU, 8 from RAM, 11 from disk | 1:35 | 13.7 s |
| After: encoder released before the plan | 12 blocks on the GPU, 18 from RAM, 2 from disk | 1:07 | 9.7 s |

**This is not a controlled comparison.** The two runs differ in more than the release: the plan arguments (a RAM cap of
5.3 GB in the first, none in the second, which also reserved no free RAM), the look-ahead (4 in the first, 2 in the
second) and the starting memory. It shows that the planner gives the UNet more RAM once the encoder has been released
(a RAM budget of 6.5 GB against 3.1 GB), not how much of the speed-up comes from the release.

### A wider image (2048 x 1024)

The same model and seed, with the image twice as wide, in the staged flow of 0.0.13. Memory in use at the start: 6.6 GB.
The plan was again 12 blocks on the GPU and 20 streamed from RAM, none on disk.

| Stage | Time |
|---|---|
| Sampling, 7 steps | 1:44 (14.9 s per step) |
| Building the VAE | 3.8 s |
| Decoding | 3.1 s |

The UNet was released before the VAE was built, so the decode had the whole card. The decode took 3.1 s, and the shared
GPU memory in Windows' Task Manager did not rise noticeably during the run (watched by eye, not logged). In earlier
runs at this size with the encoder, the UNet and the VAE all loaded at once, the shared GPU memory rose sharply and the
decode took minutes (also watched by eye, over several runs, not logged). The free VRAM after the decode read 0.00 GB.
That reading can include PyTorch's cached memory, and the peak memory the decode needed was not measured, so it shows
only that the decode filled the card, and a taller image may not fit.

This is one run, and the old flow was not run at this size, so there is no comparison here.
