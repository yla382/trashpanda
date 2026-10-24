# Pipeline coordination: design notes

Status: **ideas and measurements, not a design that is built.** Written after the 0.0.11 session so we can come back
to it. Everything marked *measured* comes from runs on one machine (see below). Everything marked *hypothesis* has
not been checked. Nothing here is promised or scheduled.

## 1. Why this document exists

trash-panda plans and runs **one model at a time**. Real pipelines have several large models in a row (text encoder,
diffusion model, VAE), and the memory and the drive are shared between them. Most of the problems we found in the
last session sit **between** the stages, not inside the copy path:

- GPU memory spilling into shared system memory (11.6 to 11.8 GB used) and a headroom default that was too thin.
- A shared-memory overflow after sampling at 2K, probably during the VAE decode.
- An encoder that is read at disk speed once per prompt, with nothing prefetched.
- Idle resident blocks and an idle encoder holding memory that the next stage could use.

The idea: plan the whole pipeline together, release what is done, and overlap loading one stage with running another.

How the prefetching and read-ahead work today is described in Appendix A.

## 2. The test machine and the model

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| RAM | 16 GB |
| Storage | SATA SSD, about 400 MB/s measured (500 MB/s on paper) |
| OS | Windows 11 Home |
| CPU / Python / PyTorch | Ryzen 7 3700X / 3.9.6 / 2.5.1 (CUDA 12.1) |

Krea2 pipeline (sizes are approximate, taken from the plan summaries):

| Model | Size | Notes |
|---|---|---|
| Diffusion transformer, 12B, fp8 with scales | about 13 GB | 32 blocks. Uncapped plan: 12 resident (5.2 GB), 20 streamed from RAM (7.3 GB), 0.6 GB outside the blocks |
| Text encoder, 4B, fp8 | about 4 GB | 36 blocks, 3.63 GB per pass, runs once per prompt |
| VAE | well under 1 GB (a guess) | decodes once |
| Total | about 17 GB | more than the 16 GB of RAM, more than the 12 GB of VRAM |

## 3. What we measured (0.0.11, single runs, so indicative only)

Run-to-run variation is about 0.5 s per step, because the free RAM at the start differs between runs (5.6 to 6.6 GB).
Do not read differences under that into anything.

**Look-ahead depth (RAM and disk blocks):**

| Plan | Look-ahead | Step time |
|---|---|---|
| Same plan, all streamed blocks in RAM | 0 / 2 | 10.50 s / 9.15 s (about 13% faster) |
| 13 resident, `max_ram="5.3GB"`, 11 blocks (4.1 GB) on disk | 0 | 18.97 s |
| the same | 2 | 15.83 s |
| the same | 4 | 16.47 s and 16.24 s |

**Profile of the capped plan (look-ahead 4), per step of 16.24 s:**

- Worker, disk reads: 4.10 GB in 10.99 s (374 MB/s).
- Worker, RAM copies: 2.80 GB in 1.48 s.
- Main thread waited for staged blocks: 10.32 s. Only 54 of 224 were ready (224 counted resident blocks too; see below).
- Main thread waited for the GPU: 0.17 s.
- The worker's own work adds up to about 12.5 s, so it was idle about 3.7 s per step. **Cause unknown.**

**Profile of the uncapped plan (look-ahead 2), 9.26 to 9.83 s per step:**

- RAM copies: 7.34 GB per step in 3.5 to 3.9 s (1.9 to 2.1 GB/s).
- Main thread waited 2.8 to 3.1 s per step. 119 of 140 streamed takes were ready (85%).
- The slowest waits were scattered through the pass (blocks 16, 17, 25, 27, 28), each late in only 1 of 7 passes, so each
  was a stall of roughly 1 to 3 s. A RAM copy takes about 0.11 s, so these are not normal copies.
- **Inference (not measured):** 21 of the 140 streamed takes were late, and the main thread waited 21.6 s in total, so
  about 1 s per late take. A cold read of one 0.37 GB block at 400 MB/s takes about 0.9 s. With 20 streamed blocks and
  each slow block late in exactly 1 of 7 passes, this fits the first pass reading the file-backed RAM tier from disk on
  first use. If so, nearly all the waiting is in step 1: a steady step would be about 6.2 s and the first about 28 s,
  and the 9.3 s average hides both. A per-step timer would settle it.

**Drive speed (`examples/disk_threads_test.py`, cold slices of a 9.6 GB file):** 429, 397, 395, 412 and 384 MB/s for
1, 2, 4, 8 and 1 threads. More reader threads do not help. The library's 374 MB/s is about 93% of the drive.

**Text encoder, one pass:** 3.63 GB copied in 8.6 to 9.6 s (0.4 GB/s), 0 of 36 blocks ready, main thread waited 8.7 to
9.7 s. This is about disk speed, though the blocks are meant to be in RAM.

**Resident blocks went through the worker queue** (a bug found by the profile: 224 takes per run instead of 140). Fixing
it was correct, but the step time did not change (9.39 s to 9.26 s, within the variation).

**Observed, not measured:** at 2K images, shared GPU memory sometimes overflows after sampling is done.

## 4. What the measurements say

1. With blocks on disk, the **drive's speed sets the floor**. Reading 4.1 GB at about 400 MB/s is 10 to 11 s, and no
   amount of prefetching or threading goes below it. Look-ahead 2 hides about a third of the read time.
2. Look-ahead 4 is no better than 2 and costs about 1.1 GB more VRAM. Depth 2 stays the default.
3. With every streamed block in RAM, the GPU does about 6 s of work per step and waits 2.8 to 3.1 s. The profile cannot
   say why.
4. The step time is not the whole story. Model loading, prompt encoding and the decode are not in the timed bar.

## 5. Ideas

### A. Release a stage when it is done (by hand first)

Today nothing releases a model for you. `handle.remove()` removes the hooks and frees the pinned buffers, and you have
to `del` the model and call `gc.collect()` and `torch.cuda.empty_cache()` yourself. The planner only sees the memory
that is free at the moment you call `plan()`, and does not know other models exist.

Test by hand before building anything: encode the prompt, release the encoder, then plan the diffusion model.
Compare the plan summary, step time and wait line against keeping the encoder, on a tight-RAM plan.

*Limit:* a Python process often keeps memory after `del`, so the release may not give everything back on Windows.
A reload costs a cold read of the encoder (about 9 s) if the prompts change.

### B. `release_gpu()` / `restore_gpu()` on the handle

Move the resident blocks off the GPU but keep the hooks, and move them back before the next generation. Loop:
sample, release, decode, restore. Costs not yet measured: the time to put 5.2 GB back for each image, and the RAM the
blocks need while they are off the GPU. The planner would have to count that RAM.

### C. Release during the last pass

On the last sampling step each block is used for the last time, so it can be freed as soon as it finishes. The VRAM is
then already free when the VAE decode starts (the VAE needs the final latent, so it cannot start earlier), which is
the fix for the 2K overflow without a pause between the stages. The library does not know which pass is last. The
sampler would have to say so (`handle.final_pass()` or a call from the step callback; `diffusers` has
`callback_on_step_end`). A second image needs the blocks restored (see B).

### D. Prefetch single-pass models from the first block

The prefetcher learns the block order during the first pass. A text encoder or VAE has one pass, so nothing is
prefetched. Give the prefetcher the block order up front. Then a single-pass model is read ahead like a disk-tier
model. Gain: total time becomes about `max(read, compute)` instead of `read + compute`. Reading is bound by the drive,
so the gain is small for a small encoder and larger for a big one.

### E. Overlap stages (preload the next model)

While the encoder runs, load as much of the diffusion model as there is spare VRAM and RAM for. When the encoder is
done, release it and start sampling with what has arrived while the rest keeps loading. The same for the VAE during
sampling. This needs:

- staged planning (A to C),
- a plan that **predicts** the memory that will be free after a stage,
- blocks that can change from streamed to resident while running (today a block's tier is fixed at load time),
- care about the drive, which is shared: preloading competes with an encoder that reads from disk.

### F. A cost model and a smarter planner

Choose depth, residency and the order of reads from measured disk speed, PCIe speed and compute time per block.
Planner items, in the order we think they pay off (they are also in the README roadmap):

1. Warn when a plan fills almost all of the GPU (the overflow is silent and slow).
2. Measure the activation size automatically and base the default headroom on it (1.5 GB is too thin).
3. Plan in stages (A to C).
4. Take memory held by other loaded models into account, or warn.
5. A KV-cache budget for language models.

### G. First-use stalls on the RAM tier

*Hypothesis:* the RAM tier is mapped from the checkpoint file, so Windows reads a block from disk the first time it is
touched. That would put the long stalls in the first pass and explain the encoder's 0.4 GB/s. If so, read those pages
into memory once when the hooks are installed. Test: show the wait for each pass separately.

### H. More pinned buffers

There are two pinned buffers. *Hypothesis:* the idle worker in the capped plan is waiting for a free buffer. Test: time
the wait for a buffer; then make the count an option and try 3 or 4 (about 0.4 GB of RAM each).

## 6. Rules that fell out of the discussion

- **Single pass versus loop.** An encoder or VAE runs each layer once, so a layer can be dropped right after it
  computes. The diffusion model runs the same blocks every step, so what is evicted is paid for again every step.
  Keep as many blocks resident as fit and stream only the rest. The value of a resident block grows with the number of
  steps (7 steps means 7 copies saved per image, 30 steps means 30). Language models behave like the loop.
- **Where the weights live decides what overlap can do.** From RAM, moving 7 GB over PCIe takes about a second, so
  there is little to hide. From disk, 13 GB takes about 33 s at 400 MB/s, and the encoder's few seconds of compute hide
  only a small part of it.
- **The drive is one shared resource.** Loading and streaming must share it. When the drive is the limit, reading
  further ahead only uses memory.
- **What if the GPU is faster than the loading?** It already waits: `take` blocks until the copy is done, and the
  compute stream waits for the copy's event, so a layer never runs on missing weights. The scheduling question is the
  order of reads (needed-first), which blocks stay resident, and how deep to read ahead (depth does not matter when
  the drive is the limit).

## 7. The Krea2 scenario

Cold run (nothing cached), at the drive's real 400 MB/s, rough numbers:

| Stage | Roughly |
|---|---|
| Read the encoder (4 GB) | 10 s |
| Encode the prompt | a few s |
| Read the transformer (13 GB) | 33 s |
| Sampling, 7 steps | about 65 s |
| VAE load and decode | a few s |
| **Total** | **about 110 to 115 s** |

- The reads (17 GB, about 43 s) are bound by the drive. Overlap cannot shorten them.
- Overlap can hide the compute beside them: the encoder's few seconds and the first sampling step (about 6 s).
  Estimate: 10 to 15 s, about 10% of a cold run, and almost nothing on a warm one.
- The larger effect is on the memory side: freeing the encoder lets the transformer have more resident blocks, which
  avoids a copy on every step, and avoids the decode overflow.
- A bigger encoder (several billion parameters, as in some recent pipelines; sizes from memory, check them) would
  raise the share that overlap can hide, and is the right scenario to test the idea on.

The user's proposal, in short: with 12 GB of VRAM, load the 4 GB encoder and run it; as each layer finishes, free it and
load transformer blocks into the space; once the encoder is done, sample with what has arrived while the rest keeps
loading; do the same for the VAE during sampling. It is sound, with two corrections: only single-pass models are
evicted layer by layer, and the encoder's compute is too short to hide a disk load.

## 8. Experiments, in order

Each step gives evidence before the next.

0. **Time every step** (the first step apart from the rest) and show the profile's wait for each pass. This is the
   cheapest test and may change the reading of every other number here (see the inference in section 3).
1. **Time the stages** of one Krea2 run (load encoder, encode, plan and load the transformer, sampling, VAE
   load and decode) with `time.perf_counter()` around each, and the profile on. This gives the first whole-pipeline
   number.
2. **Release by hand** between the stages on a tight-RAM plan (A), and after sampling before the decode at 2K (C by
   hand). Does the shared-memory overflow go away?
3. **Add a wait line per pass** to the profile (G). Are the long stalls all in pass 1?
4. **Time the worker's wait for a free pinned buffer** (H), then make the buffer count an option.
5. **Then decide** what to build: a release helper (B, C), the block order up front for single-pass models (D), or both.

A better benchmark method belongs here too: record the time of every step (first step apart from the rest), fix the
plan and change one thing per comparison, repeat each setting three times in alternating order and report the median,
and label cold and warm cache runs. It could be a small script in `examples/` that works for any model.

## 9. Open questions and risks

- How much RAM does a released model really give back on Windows?
- What does restoring 5.2 GB of resident blocks cost per image, and how much RAM does keeping them cost?
- Can a block change tier while running without breaking the hooks and the plan?
- Does preloading help when the encoder reads from disk, or do they just fight for the drive?
- More things moving at once may make the unexplained silent exits at high RAM more likely.
- Is the 2K overflow really the VAE decode with the resident blocks still on the GPU? Not confirmed.
- Other devices: the prefetching is CUDA-only. A Mac has unified memory, so splitting a model between GPU and RAM
  frees nothing there. Untested.

## 10. What is not claimed

- None of the ideas above is built, and none has a measured gain. The Krea2 scenario numbers are estimates.
- The numbers in section 3 are single runs on one machine. A gap under about 0.5 s per step is within the variation.
- Look-ahead 2 against 0 is a 13% to 17% gain in our runs. Against the earlier 0.0.8 on the all-RAM plan there was no
  change, because that plan was already near its limit.

## Appendix A. How prefetching and read-ahead work today

Prefetching (blocks in RAM) and read-ahead (blocks on disk) are one mechanism with two ways of filling the same
staging buffer. This describes version 0.0.11.

### A.1 The three tiers

- **GPU tier.** The first blocks of the model, as many as the VRAM budget allows. Their compact weights (fp8 plus
  scales) stay on the GPU permanently. They are still wrapped by the hooks, because every use has to expand fp8 to the
  compute dtype. "Resident" means compact on the GPU, not ready to use.
- **RAM tier.** Compact weights in CPU memory. They stay file-backed (mapped from the checkpoint), so Windows may drop
  them and read them again from the file instead of writing them to the page file.
- **Disk tier.** Not loaded at all. A reader reads the block's tensors from the checkpoint file each time.
- The planner chooses as few disk blocks as it takes for the remaining blocks to fit the RAM budget, and spaces them
  evenly among the streamed blocks, so each disk read has other blocks' compute next to it to hide behind.

### A.2 What happens when a wrapped block runs

1. The block is linked to the one that ran before it, which builds the order of execution.
2. The hook waits if two earlier blocks are still running on the GPU, so work cannot pile up.
3. The block's weights are taken from the staging queue, waiting for the background thread if they are not ready. If
   nothing was staged and the block needs a copy, it is staged on the spot. The compact weights are then expanded to
   the compute dtype and placed in the module's parameters.
4. The next blocks that need a copy (as many as the look-ahead) are queued for staging.
5. The block's real forward runs.
6. In every case, the compact originals are put back, which frees the expanded copy, and an event marks the end of the
   block's work on the compute stream.

### A.3 The prefetcher

- **One background thread** runs the staging jobs in the order they were queued.
  - A RAM job copies each of the block's tensors into a pinned buffer, then starts an asynchronous copy to the GPU on
    a second CUDA stream.
  - A disk job reads the file straight into the pinned buffer, then starts the same copy.
- The data always travels disk, then a pinned RAM buffer, then VRAM. The buffer is a short stop and is reused; a disk
  block does not become a RAM-tier block.
- Taking a staged block has two waits: for the thread to finish *issuing* the job (the memory copy or the file read),
  then for the compute stream to wait for the GPU copy's event.
- **The pinned pool** has 2 buffers, handed out in turn. A buffer is reused only after the GPU copy that read from it
  has finished. If pinned memory cannot be allocated, the job falls back to unpinned memory, which is not truly
  asynchronous. With prefetching off, or on a device without CUDA, a disk block is simply read when it is needed.
- Blocks that are already on the GPU never go through the thread.

### A.4 What the planner reserves for this

| Where | Reserve | Size |
|---|---|---|
| VRAM | working area for the block that is computing (the expanded copy) | 1.3 x the largest expanded block (1.13 GB in the Krea2 plan) |
| VRAM | staging area for blocks copied ahead | look-ahead x the largest compact block (868 MB at depth 2) |
| VRAM | headroom for activations and everything else | 1.5 GB by default, or a measured activation size plus a margin |
| RAM | pinned buffers | 2 x the largest block outside the GPU, counted in the RAM budget |
| RAM | system reserve | 8% of RAM, between 1 and 4 GB by default |

The VRAM staging scales with the look-ahead, but the number of pinned buffers is fixed at 2, whatever the depth.

### A.5 What follows from this

1. **The order of blocks matters.** Resident blocks come first, then the streamed ones. Queueing the next streamed blocks
   skips blocks that need no copy, so from the first block of a pass the first streamed blocks are already being
   staged, with all the resident compute as cover. The start of a pass is therefore not where the stalls are.
2. **Resident blocks pay the expansion on every step.** Expanding fp8 to the compute dtype happens on the compute
   stream and is not overlapped with anything.
3. **Later streamed blocks have little cover.** Each block after the first two is queued only about two blocks of
   compute (about 0.4 s) before it is needed. That is enough for a RAM copy (about 0.11 s) and not for a disk read
   (about 1 s).
4. **One thread serves everything in order,** so a slow disk read in front delays the RAM blocks behind it.
5. **The first use of a RAM-tier block may read the file.** Because the RAM tier is file-backed, the memory copy of a
   block that is not resident in memory becomes a disk read. This is the likely cause of the late blocks in the all-RAM
   profile (see the inference in section 3).
