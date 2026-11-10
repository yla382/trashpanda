# What we measured

Measurements on one machine, and the recommendation they lead to.

## What we measured

All numbers come from one machine (Windows 11, an RTX 3060 with 12 GB, 16 GB of RAM and a SATA SSD) and one pipeline:
a text encoder, a 12B diffusion model with fp8 weights and scale tensors, and a VAE, making one 1024 x 1024 image in 7
steps. The diffusion model's plan was 12 blocks (5.21 GB) on the GPU and 20 blocks (7.29 GB) in RAM, and the RAM in use
at the start was 5.5 to 6.5 GB. They are single runs. Identical runs differed by 2 to 3 s in total time, so a
difference smaller than that is not a result.

* **The drive sets the pace of every cold read.** The text encoder's blocks, the diffusion model's resident blocks and
  its RAM tier all ran at 0.36 to 0.42 GB/s the first time they were read, however they were read (the memory-mapped
  copy, one reader thread, two reader threads). A run reads about 16 GB cold, which is roughly 40 s of a total of about
  95 s. The first step of the diffusion model is the first read of its RAM tier: 20 to 22 s, against 6.3 s for every
  later step, and 12.5 to 15 s of that difference is the main thread waiting for blocks. About 2.8 s of the first step
  is one-time warm-up in its first three blocks.
* **Defaults:** 94.9, 95.6, 96.5 and 98.3 s in four runs.
* **`TRASH_PANDA_RAM_IN_MEMORY=1`:** the first step takes 9.0 s and the sampling 47 s instead of 61 s, because the read
  of the RAM tier has moved to the load: 7.29 GB in 20 to 23 s (0.31 GB/s with one thread, 0.34 to 0.36 with two). The
  total was 104 to 108 s, which is 8 to 12 s longer than with the defaults. The read is slower than the memory-mapped
  copy, and it no longer overlaps with the compute of the first step. Two threads read 16% faster than one; more were
  not tried, because both are close to what the drive delivers. The tensors are private memory, so they count against
  the Windows commit limit while the model is loaded.
* **`stage(overlap=True)`:** 96.5 to 98.8 s in four runs, against 94.9 to 98.3 s without it, and 109.3 s together with
  `TRASH_PANDA_RAM_IN_MEMORY=1` against 105.0 s. It cannot help much here. The compute of a model is small next to the
  load of its resident blocks: a resident block of the diffusion model takes about 1.1 s to load and 0.2 s to run, so
  only about 3 s of compute can overlap with the load, and the text encoder computes for 0.2 s against an 11.7 s load.
  The loader thread also reads with plain seek-and-read, which ran at 0.25 to 0.28 GB/s against 0.41 GB/s for the
  memory-mapped reads that `stream_load()` does for the same blocks.
* **`TRASH_PANDA_LOOKAHEAD`:** a depth of 4 was no better than 2 for disk blocks and uses more VRAM. With stand-in
  models (12 blocks of 0.07 to 0.54 GB, the file in the file cache, single runs) a deeper look-ahead shortened a pass by
  3% to 8%, only where the copies and the compute took about the same time (20 ms of compute per block), and only with
  as many pinned buffers as the look-ahead (4 with 4, not 4 with 2). Where the copies set the pace (no compute) or the
  compute did (100 ms per block) it made no difference beyond the noise of about 5% between runs.
  `TRASH_PANDA_ADAPTIVE_LOOKAHEAD=1` tries the depth by itself. In a check on the GPU its output was identical to the
  fixed look-ahead and it cost nothing where compute set the pace, but it found the gain in one of nine cases (disk
  blocks, 20 ms: 5.3% faster) and missed it in the others, so it is off by default. Where the compute sets the pace
  (100 ms per block) it lowers the look-ahead to 1, which was 0.4% to 2% slower in the grid and 0.8% in the check on the
  GPU, and frees one block of staging for the next model.
* **A read-ahead of the RAM tier** through ordinary file reads, ahead of the copy, made the first step slower (34.7 s
  against 21 to 22 s) and was removed: the pages it read did not serve the memory-mapped copy.
* **Memory after a release.** On Windows, PyTorch's own CPU allocator keeps blocks after they are freed (3 GB of
  tensors freed, working set unchanged), so memory read into `torch.empty` tensors still counted as taken when the next
  model was planned (a RAM budget of 2.0 to 2.6 GB, against about 9 GB). `TRASH_PANDA_RAM_IN_MEMORY` allocates through
  Python's allocator instead, which gives the memory back at once, and after the fix the next plan saw 8.2 to 9.3 GB.
* **Taking the resident blocks off the GPU and back** (`release_gpu()` / `restore_gpu()`) cost about 0.25 s per GB to
  release and 0.3 s per GB to restore, for blocks in the compute dtype and for fp8 blocks with scales alike. Stand-in
  blocks of two 4096 x 4096 linear layers, built on the GPU, median of 5 cycles: 1 GB took 0.22 s and 0.27 s, 2 GB took
  0.45 s and 0.58 s, 4 GB took 1.04 s and 1.27 s (fp8: 1.32 s), which is 3.9 to 4.6 GB/s out and 3.0 to 3.7 GB/s back.
  The VRAM they held was free afterwards, and the first pass after the restore took as long as any other (within the
  noise). A plain tensor moves to the GPU at 5.1 GB/s from ordinary memory and 12.9 GB/s from pinned memory, so pinned
  copies could make the restore up to 2.5 times faster, at the price of pinned RAM as large as the weights. Nothing
  above 4 GB was run, because the free RAM was between 2.6 and 9.7 GB during the measurement; with 2.6 GB free the copy
  to the CPU fell to 1.2 GB/s.
* **The Coordinator on the test pipeline** (one 1024 x 1024 image, 7 steps, no overlap, single runs; the start RAM and
  the state of the file cache differed from run to run, which moved the totals by several seconds). Three plain
  `stage()` calls took 98.4 s (start RAM 9.7 GB). A `Coordinator` with the defaults (both preload arguments on, every
  stage) took 107.3 s (9.5 GB), 111.7 s and 113.9 s (6.5 to 7.1 GB). At matched RAM the whole 8.9 s gap between the
  first two is in the diffusion model's phase (76.1 s against 84.9 s). Preparing it during the encoder read its RAM tier
  into memory, 7.29 GB in 21.6 s (0.33 GB/s), of which the 2.7 s encoder hid 3.6 s, so the model waited 18 to 20 s at
  entry; the first denoise step then took 9.7 s instead of 22.3 s, which was not enough to make up for it. The mapped
  read overlaps the cold read with the compute of the first step. Preparing the VAE while the diffusion model ran
  loaded 5 or 6 of its 6 blocks early and shortened the steps after the diffusion model by 2 to 3 s.
* **`ram_preload=False` for the encoder, `vram_preload` left on** took 99.5 s (start RAM 8.2 GB): 2 of the diffusion
  model's 12 GPU blocks (0.87 GB) loaded during the encoder, its load took 12.2 s instead of 15.7 s, and the encoder
  phase took about 3 s longer (the prompt was encoded in 3.9 s instead of 1.1 s). The window is the encoder's 3 to 4 s
  and the drive reads 0.4 GB/s, so it can hide about 1.5 GB at most. The cost in the encoder was not traced; one
  possibility is that loading into free VRAM makes the allocator ask the driver for new memory while the encoder runs.
  One run with the diffusion model's weights in the file cache (the encoder loaded in 2.7 s instead of 12 s) took
  81.3 s, but it is not comparable with the others.
* **Where the time goes** (the 107.3 s run, from the time between the stages): loading weights took 48.3 s (the
  encoder 12.0 s, the diffusion model 36.3 s, 45% of the run), the compute 52.6 s (49%), the rest about 6 s. A run reads
  about 16 GB through the drive. On a drive that reads 3 GB/s that part would take a few seconds, and preparing the
  next model while the encoder runs should pay off then.
* **`overlap=True` in a `Coordinator`** with the defaults took 105.1 s (start RAM 6.5 GB, no table), against 107.3 to
  113.9 s without it. The encoder's blocks load on a worker thread while the diffusion model's RAM tier is read, so the
  model waited 1.2 s at entry instead of 18 s, at the price of a longer first step. The difference from the runs
  without it is within the spread of those runs.

## Current recommendation

* Leave every setting at its default. The defaults were the fastest total in these measurements.
* Leave `TRASH_PANDA_RAM_IN_MEMORY` off unless you want steady steps from the first one, and have the RAM and the commit
  headroom for it (about 7 GB of private memory on the test pipeline). It does not make a single run shorter.
* Leave `overlap` off. It is groundwork for starting the next model's loads behind compute that is long enough to hide
  them (see the roadmap), not a speed-up for one pipeline.
* A slow first step is the first read of the RAM tier from the drive. The profile shows it as a first pass with a low
  GB/s in the RAM copies and a large "waited". Only a faster drive shortens it.
* Set `TRASH_PANDA_PROFILE=1` when you want to see where the time goes.
* In a `Coordinator`, pass `ram_preload=False` for a stage that is shorter than the read of the next model's RAM tier
  (the encoder before a diffusion model on a SATA drive), or `then=None` to prepare nothing. Preparing the VAE while the
  diffusion model runs is worth keeping on.
