# Pipeline coordination

**Status:** draft. Proposals and measurements; nothing in sections 7 and 8 is implemented.
**Applies to:** trash-panda 0.0.11.
**Conventions:** *Measured* means taken from instrumented runs on the machine in section 3. *Observed* means seen
during repeated runs but not timed. *Hypothesis* means not yet verified. All measurements are single runs unless stated.

## 1. Summary

trash-panda currently plans, loads and runs one model at a time. Image and video pipelines run several large models in
sequence (text encoder, diffusion model, VAE) that share the same GPU memory, system RAM and drive. Profiling the
0.0.11 prefetcher showed that the largest remaining costs are not in the per-block copy path. They are at the
boundaries between stages: cold reads on first use, memory held by finished models, and overflow of GPU memory during
the decode.

This document records the current behavior, the measurements, and a set of proposals to coordinate memory and I/O across
a pipeline: release finished stages, overlap loading with computation, and plan each stage from the memory that will
actually be free.

## 2. Problem statement

The following problems occur in a Krea2 pipeline (12B diffusion transformer, 4B text encoder, VAE) on a 12 GB GPU:

1. **The first sampling step is much slower than the rest.** Step 1 takes about 20 to 30 s while later steps are
   faster (observed). GPU utilization is low during most of step 1 (observed).
2. **A single-pass model is not prefetched.** The text encoder is copied at about disk speed (0.4 GB/s) with no block
   ready in advance (measured).
3. **Finished stages keep their memory.** Nothing releases a model when its stage ends. Resident blocks of the
   diffusion model stay on the GPU during the VAE decode (by design), and the encoder keeps its RAM and VRAM during
   sampling.
4. **GPU memory overflows into shared memory after sampling at 2K** (observed). The cause is not confirmed.
5. **The planner sees one model.** It reads the memory that is free at the moment `plan()` is called and has no
   knowledge of other models, so the result depends on the order in which models are loaded and released.

## 3. Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| RAM | 16 GB |
| Storage | SATA SSD, about 400 MB/s measured (500 MB/s nominal) |
| OS | Windows 11 Home |
| CPU / Python / PyTorch | Ryzen 7 3700X / 3.9.6 / 2.5.1 (CUDA 12.1) |

| Model | Size | Notes |
|---|---|---|
| Diffusion transformer, 12B, fp8 with scales | about 13 GB | 32 blocks. Auto plan: 12 resident (5.2 GB), 20 streamed from RAM (7.3 GB), 0.6 GB outside the blocks |
| Text encoder, 4B, fp8 | about 4 GB | 36 blocks, 3.63 GB per pass, one pass per prompt |
| VAE | under 1 GB (estimate) | one decode per image |
| Total | about 17 GB | exceeds both the 16 GB of RAM and the 12 GB of VRAM |

## 4. Measurements

Step-time variation between nominally identical runs is about 0.5 s, because the free RAM at the start differs (5.6 to
6.6 GB). Differences smaller than that are not significant.

### 4.1 Look-ahead depth

| Plan | Look-ahead | Step time |
|---|---|---|
| All streamed blocks in RAM | 0 / 2 | 10.50 s / 9.15 s (13% faster) |
| 13 resident, RAM capped at 5.3 GB, 11 blocks (4.1 GB) on disk | 0 | 18.97 s |
| same | 2 | 15.83 s |
| same | 4 | 16.47 s, 16.24 s |

### 4.2 Profile, capped plan (look-ahead 4), 16.24 s per step

| Quantity | Value |
|---|---|
| Worker, disk reads | 4.10 GB in 10.99 s (374 MB/s) |
| Worker, RAM copies | 2.80 GB in 1.48 s |
| Main thread waiting for staged blocks | 10.32 s per step |
| Main thread waiting for the GPU | 0.17 s per step |
| Worker busy time (reads plus copies) | about 12.5 s; idle about 3.7 s. Cause unknown |

### 4.3 Profile, auto plan (look-ahead 2), 9.26 to 9.83 s per step

| Quantity | Value |
|---|---|
| Worker, RAM copies | 7.34 GB per step in 3.5 to 3.9 s (1.9 to 2.1 GB/s) |
| Main thread waiting for staged blocks | 2.8 to 3.1 s per step; 119 of 140 streamed blocks ready |
| Slowest waits | blocks 16, 17, 25, 27, 28; each late in 1 of 7 passes; 1 to 3 s each |

A RAM copy of one block takes about 0.11 s, so the late blocks are not ordinary copies. 21 streamed blocks were late and
the main thread waited 21.6 s in total, which is about 1 s per late block. Reading one 0.37 GB block cold from the drive
takes about 0.9 s. With 20 streamed blocks, each slow block late in exactly one pass, and step 1 observed at 20 to
30 s, the data is consistent with the first pass reading the file-backed RAM tier from disk (section 7.7). If that holds,
a steady-state step is about 6.2 s and the first about 28 s. This is an inference; a per-step timer would confirm it.

### 4.4 Drive throughput

Cold slices of a 9.6 GB file read with 1, 2, 4, 8 and 1 threads: 429, 397, 395, 412 and 384 MB/s. Additional reader
threads do not increase throughput. The prefetcher reaches about 93% of this rate.

### 4.5 Text encoder

One pass: 3.63 GB copied in 8.6 to 9.6 s (0.4 GB/s), 0 of 36 blocks ready when requested, 8.7 to 9.7 s spent waiting.
This is the rate of the drive, although the blocks are assigned to the RAM tier.

### 4.6 Other results

- Blocks already on the GPU were being routed through the prefetch queue. Removing this reduced the number of staged
  takes from 224 to 140 per run. Step time was unchanged (9.39 s to 9.26 s, within variation).
- At 2K output, shared GPU memory sometimes overflows after sampling finishes (observed, not measured).

## 5. Analysis

1. With blocks on disk, drive throughput sets a floor. Reading 4.1 GB at about 400 MB/s takes 10 to 11 s per step, and
   prefetching or additional threads cannot reduce it. Look-ahead 2 hides about one third of the read time.
2. Look-ahead 4 is not faster than 2 and reserves about 1.1 GB more VRAM. The default stays at 2.
3. In the auto plan the GPU computes for about 6 s per step. The remaining 2.8 to 3.1 s of the average is waiting that
   is concentrated in a few late blocks, not spread across all of them.
4. The timed sampling bar excludes model loading, prompt encoding and decoding, so step time understates the cost of a
   cold run.

## 6. Design principles

**Single-pass versus looped models.** An encoder or VAE uses each layer once, so a layer can be released as soon as it
has computed. A diffusion model or a language model reuses every block on each step or token, so an evicted block is
fetched again each time. The value of keeping a block resident grows with the number of repetitions: with 7 steps a
resident block saves 7 copies per image, with 30 steps it saves 30.

**Location of the weights bounds what overlap can achieve.** From RAM, 7 GB crosses PCIe in about a second, so there is
little to hide. From disk, 13 GB takes about 33 s at 400 MB/s, while the compute that can overlap with it (encoder, first
sampling step) is a few seconds.

**The drive is a single shared resource.** Loading one model while another streams from disk divides the same
bandwidth. When the drive is the limit, reading further ahead only consumes memory.

**Compute never runs on missing weights.** When the GPU is ahead of the loading, the main thread blocks until the
staged copy has been issued and the compute stream waits for its completion event. The scheduling decisions are the
order of reads (needed-first), which blocks stay resident, and the read-ahead depth.

## 7. Proposals

### 7.1 Release a stage when it finishes

*Current behavior.* `handle.remove()` removes the hooks and frees the pinned buffers. Releasing memory also requires
deleting the model and calling `gc.collect()` and `torch.cuda.empty_cache()`. The planner reads only the memory that is
free when `plan()` runs.

*Proposal.* Run the stages in order (encode, release, plan and load the diffusion model, sample, release, decode) and
plan each stage from the memory that is free at that point. First validate by hand, then provide a helper that performs
the sequence and reports memory before and after.

*Risks.* A Python process often retains memory after `del`, so the release may not return all of it on Windows. If
prompts change between images, a released encoder must be read again cold (about 9 s).

*Validation.* On a RAM-constrained plan, compare plan summary, step time and wait time with and without releasing the
encoder before planning.

### 7.2 `release_gpu()` and `restore_gpu()`

*Proposal.* Move the resident blocks off the GPU while keeping the hooks installed, and move them back before the next
generation: sample, release, decode, restore.

*Unknowns.* The time to restore 5.2 GB per image and the RAM the blocks occupy while off the GPU, which the planner
would have to account for.

### 7.3 Release during the last pass

*Proposal.* On the final sampling step each block is used for the last time and can be freed as soon as it finishes, so
the VRAM is available when the decode starts (the decode needs the final latent and cannot start earlier). This removes
the pause between stages and is the expected fix for the 2K overflow.

*Interface.* The sampler owns the step count, so the library cannot detect the last pass. A call such as
`handle.final_pass()` from the step callback would signal it; `diffusers` provides `callback_on_step_end`. Generating a
second image requires the blocks to be restored (7.2).

### 7.4 Prefetch single-pass models from the first block

*Current behavior.* The prefetcher learns the block order during the first pass. A single-pass model never benefits.

*Proposal.* Supply the block order when the hooks are installed, so a single-pass model is read ahead like a disk-tier
model. Total time becomes approximately `max(read, compute)` instead of `read + compute`. Because reading is bound by
the drive, the gain is small for a small encoder and larger for an encoder of several billion parameters.

### 7.5 Overlap stages

*Proposal.* While the encoder runs, load the part of the diffusion model that fits in spare VRAM and RAM. When the
encoder finishes, release it and begin sampling with the blocks already present while loading continues. Overlap the
VAE with the final sampling steps in the same way.

*Requirements.* Staged planning (7.1 to 7.3); a plan that predicts the memory that will be free after each stage;
blocks that change from streamed to resident while running (a block's tier is currently fixed at load time); and
scheduling that accounts for the encoder and the preload sharing the drive.

*Bound.* The reads happen once and take about 43 s for 17 GB at 400 MB/s. Overlap can hide only the compute that runs
during that period: the encoder (a few seconds) and the first sampling step (about 6 s). From step 2 onward all blocks
are already in memory, so no further reads remain to overlap. The expected saving is roughly 10 to 15 s of a cold run of
about 110 s, and close to nothing on a warm run.

### 7.6 Cost model and planner changes

Choose read-ahead depth, residency and the order of reads from measured drive speed, PCIe speed and per-block compute
time. Planner work in priority order:

1. Warn when a plan fills almost all of the GPU (the overflow is silent and slow).
2. Measure the activation size automatically and derive the default headroom from it (the fixed 1.5 GB is too small at
   larger resolutions).
3. Staged planning (7.1 to 7.3).
4. Account for memory held by other loaded models, or warn.
5. KV-cache budget for language models.

### 7.7 First-use reads of the RAM tier

*Hypothesis.* The RAM tier is mapped from the checkpoint file, so a block's pages are read from disk the first time it
is used. This would place the long waits in the first pass, explain the observed slow first step, and explain the
encoder's 0.4 GB/s.

*Proposal.* Touch the RAM-tier pages once in a background thread when the hooks are installed. This moves the cost
before sampling and overlaps it with whatever else the program does; it does not remove it. The tier should remain
file-backed: copying it into ordinary memory would restore the "paging file too small" failure that file-backed mapping
avoids.

*Validation.* Record the time of every step and the prefetch wait per pass. If pass 1 holds nearly all of the wait, the
hypothesis is confirmed.

### 7.8 Number of pinned buffers

The pool has a fixed size of 2, while the VRAM staging area scales with the look-ahead. *Hypothesis:* the idle time of
the worker in the capped plan (section 4.2) is spent waiting for a free buffer. *Validation:* time the wait for a
buffer, then make the count configurable and test 3 and 4 (about 0.4 GB of RAM each).

### 7.9 Consume mode for single-pass models

*Proposal.* For a model that runs a single pass (a text encoder, a VAE), drop each layer's weights as soon as the layer
has computed, including its source copy and not only the working copy on the GPU. Streamed blocks already release
their working copy after each use; what is retained today is the RAM-tier weights, which a single-pass model never
reads again. In consume mode a layer is read from the checkpoint into a pinned buffer, copied to the GPU, used and
discarded. This is the disk-tier path with the read-ahead of 7.4, so the memory footprint is the pinned buffers and one
or two layers, independent of the model's size.

*Default.* The mode is an explicit option (for example `single_pass=True`), and the default is to keep the weights. The
library cannot know in advance whether a model will run again. If it dropped the weights by default and the model
looped, every pass would read the file again, a slowdown that nothing reports and that falls on the largest models.
Keeping by default costs only memory held longer, which is visible and recoverable.

*Trade-off.* A repeated run reads the model from the drive again (about 9 s cold for a 4B encoder) unless the operating
system's file cache still holds it. The mode suits restricted hardware and single images; with enough memory, keeping
the weights is faster.

*Depends on.* 7.4 (block order up front) and 7.1 (release).

### 7.10 Stage-major execution for batches

*Observation.* The text encoder runs once per distinct prompt, not once per image. Images generated from the same prompt
need the encoder once, and its embeddings can be reused (in `diffusers` through `prompt_embeds`).

*Proposal.* For a batch, run the pipeline stage by stage instead of image by image: encode all prompts with the encoder
loaded once, release it, sample every image with the transformer loaded once, release it, then decode every image with
the VAE loaded once. Only small objects are kept between the stages (the embeddings, in megabytes, and the latents, a
few megabytes per image). Each model is loaded once per batch, so the reload cost of consume mode (7.9) is paid once
and not once per image.

*Scope.* Applicable when the whole batch is known in advance. It does not apply to interactive use, where prompts
arrive one at a time and the stages cannot be reordered (7.11).

### 7.11 Read-ahead into idle drive time

*Observation.* While the transformer samples with its streamed blocks in RAM, the drive is idle. On the reference
pipeline this lasts about 65 s per image, enough to read the 4 GB encoder several times.

*Proposal.* In a cyclic, one-image-at-a-time workload (encode, sample, decode, encode again), start reading the weights
the next stage will need while the current stage runs. This requires a prefetcher shared across models, so that the
block chain continues from the last block of one model to the first block of the next, and a per-stage memory policy:
keep the model, drop it after use, or drop it and read it back ahead of time. Where there is no room to hold the
data, the read can target only the operating system's file cache, whose pages count as available memory and are
reclaimed on demand.

*Limits.* The saving is bounded by the time of the reads that are hidden (about 10 s per image for a 4B encoder). On
very restricted RAM the cached pages may be evicted before use. 7.10 removes most of this cost for batches.

### 7.12 Release during the last pass, with the next stage preloaded

*Proposal.* Extends 7.3. On the final sampling step the sampler signals the last pass (for example
`handle.final_pass()` from the step callback; the library cannot detect it, because the sampler owns the step count).
As each block finishes in that pass, its weights are released, and the VRAM freed this way is available for the next
stage when the pass ends. Weights of the next stage load into that space in the background.

*What it gains.* For the decoder the weights are small, so the saving in load time is minor. The main effect is the room
for the decoder's activations, which is the suspected cause of the overflow at 2K. For an interactive loop, the encoder
of the next image can be preloaded in the same way.

*Safety.* A block's weights must not be freed while the GPU is still using them. Releasing the reference after the
block's work has been enqueued on the compute stream is safe in PyTorch's allocator, and the staged tensors already use
`record_stream` for this purpose. A test must show that the output of a run with the final-pass release is identical to
a normal run.

*Cost.* A further image requires the released blocks to be restored (7.2), a cost that has not been measured, so the
signal is optional.

## 8. Scenario: cold Krea2 run

Estimated at the measured drive rate of 400 MB/s (not measured end to end):

| Stage | Duration |
|---|---|
| Read the encoder (4 GB) | about 10 s |
| Encode the prompt | a few seconds |
| Read the transformer (13 GB) | about 33 s |
| Sampling, 7 steps | about 65 s |
| VAE load and decode | a few seconds |
| **Total** | **about 110 to 115 s** |

- The 17 GB of reads (about 43 s) are bound by the drive.
- Overlap saves roughly 10 to 15 s (7.5).
- Freeing the encoder lets the transformer keep more blocks resident, which avoids a copy per block on every step, and
  removes the cause of the decode overflow. These memory effects are expected to matter more than overlap.
- A larger encoder (some current pipelines use text encoders of several billion parameters; sizes to be verified for
  the models of interest) increases the share that can be overlapped and is the appropriate case for validating 7.4
  and 7.5.

## 9. Validation plan

Each step provides evidence for the next.

1. **Per-step timing and per-pass wait.** Record the time of every step (the first separately) and the prefetch wait per
   pass. Confirms or rejects 7.7 and changes the interpretation of the other averages.
2. **Stage timeline.** Time each stage of a complete run: load encoder, encode, plan and load the transformer, sampling,
   VAE load and decode. Gives the first whole-pipeline number.
3. **Manual release.** Release the encoder before planning on a RAM-constrained plan (7.1), and release the transformer
   before the decode at 2K (7.3 by hand). Check whether the overflow disappears.
4. **Pinned buffer wait.** Time the wait for a buffer and test 3 and 4 buffers (7.8).
5. **Staged run with a release helper.** Implement the helper of 7.1, which reports free RAM and VRAM before and after,
   and run the pipeline stage by stage on a constrained plan and at 2K. Compare the plan summary, the pass times, the
   memory reports and the total time with the unstaged run. This measures how much memory a release returns and
   decides whether 7.9 to 7.12 are worth building.
6. **Decide the implementation** from the results: a release helper (7.2, 7.3), block order up front (7.4), consume
   mode (7.9), or a combination.

The benchmark method should be extended in parallel: record per-step times, fix the plan and change one variable per
comparison, repeat each setting three times in alternating order and report the median, and label cold and warm cache
runs. A script in `examples/` that works with any model would serve this purpose.

## 10. Open questions and risks

- How much memory does a released model return on Windows?
- What is the cost of restoring the resident blocks for each image, and of keeping them in RAM meanwhile?
- Can a block change tier while running without invalidating the hooks and the plan?
- Does preloading help when the encoder itself reads from disk, or do the two compete for the drive?
- More concurrent activity may increase the likelihood of the unexplained process exits seen at high RAM usage.
- Is the 2K overflow caused by the decode running while the resident blocks are still on the GPU? Not confirmed.
- In consume mode (7.9), how often does a repeated run find the model still in the operating system's file cache?
- Can the embeddings and latents of a batch (7.10) always be passed between stages without loading the models they
  came from, for the pipelines of interest?
- Can the file cache be filled ahead of time (7.11) without raising the memory pressure that evicts it again?
- The prefetching is CUDA-only. On a machine with unified memory (Apple silicon) splitting a model between GPU and
  RAM frees nothing. Untested.

## 11. Limits of the evidence

- None of the proposals is implemented, and none has a measured gain. The scenario figures are estimates.
- Section 4 consists of single runs on one machine; differences under about 0.5 s per step are within the variation.
- Look-ahead 2 against 0 is a 13% to 17% gain in the test runs. Compared with the earlier 0.0.8 on the all-RAM plan
  there was no change, because that plan was already close to its limit.

## Appendix A. Prefetching and read-ahead in 0.0.11

Prefetching (blocks in RAM) and read-ahead (blocks on disk) are one mechanism with two ways of filling the same staging
buffer.

### A.1 Tiers

- **GPU tier.** The first blocks of the model, as many as the VRAM budget allows. Their compact weights (fp8 and
  scales) stay on the GPU. They are still wrapped by the hooks because every use expands fp8 to the compute dtype.
  Resident means compact on the GPU, not ready to use.
- **RAM tier.** Compact weights in CPU memory, file-backed (mapped from the checkpoint), so the operating system may
  drop them and read them again from the file instead of writing them to the page file.
- **Disk tier.** Not loaded. A reader reads the block's tensors from the checkpoint on every use.
- The planner selects as few disk blocks as are needed for the remaining blocks to fit the RAM budget and spaces them
  evenly among the streamed blocks, so each read has other blocks' compute beside it.

### A.2 Execution of a wrapped block

1. The block is linked to the block that ran before it, which builds the execution order.
2. The hook waits if two earlier blocks are still running on the GPU, to prevent work from accumulating.
3. The block's weights are taken from the staging queue, waiting for the background thread if they are not ready. If
   nothing was staged and the block needs a copy, it is staged immediately. The compact weights are expanded to the
   compute dtype and placed in the module's parameters.
4. The next blocks that need a copy (as many as the look-ahead) are queued for staging.
5. The block's forward runs.
6. In all cases the compact originals are restored, which frees the expanded copy, and an event marks the end of the
   block's work on the compute stream.

### A.3 Prefetcher

- One background thread executes staging jobs in the order they were queued.
  - A RAM job copies each tensor of the block into a pinned buffer and starts an asynchronous copy to the GPU on a
    second CUDA stream.
  - A disk job reads the file directly into the pinned buffer and starts the same copy.
- Data always moves disk, then a pinned RAM buffer, then VRAM. The buffer is a reused waypoint; a disk block does not
  become a RAM-tier block.
- Taking a staged block involves two waits: for the thread to finish issuing the job (the memory copy or the file
  read), and for the compute stream to wait for the GPU copy's completion event.
- The pinned pool holds 2 buffers, used in turn. A buffer is reused only after the GPU copy that read from it has
  finished. If pinned memory cannot be allocated, the job falls back to unpinned memory, which is not fully
  asynchronous. With prefetching disabled, or without CUDA, a disk block is read when it is needed.
- Blocks already on the GPU do not pass through the thread.

### A.4 Memory reserved by the planner

| Where | Reserve | Size |
|---|---|---|
| VRAM | working area for the executing block (expanded copy) | 1.3 x the largest expanded block (1.13 GB in the Krea2 plan) |
| VRAM | staging area for blocks copied ahead | look-ahead x the largest compact block (868 MB at depth 2) |
| VRAM | headroom for activations and other use | 1.5 GB by default, or a measured activation size plus a margin |
| RAM | pinned buffers | 2 x the largest block outside the GPU, deducted from the RAM budget |
| RAM | system reserve | 8% of RAM, between 1 and 4 GB by default |

VRAM staging scales with the look-ahead; the number of pinned buffers is fixed at 2 for every depth.

### A.5 Consequences

1. **Block order.** Resident blocks come first and streamed blocks follow. Queueing skips blocks that need no copy, so
   from the first block of a pass the first streamed blocks are already staged, with all the resident compute as
   cover. The boundary between passes is therefore not where the waits occur.
2. **Expansion cost.** Resident blocks pay the fp8 expansion on every step. It runs on the compute stream and is not
   overlapped with anything.
3. **Limited cover for later blocks.** Each streamed block after the first two is queued about two blocks of compute
   (roughly 0.4 s) before it is needed. That covers a RAM copy (about 0.11 s) but not a disk read (about 1 s).
4. **Head-of-line blocking.** A single thread serves all jobs in order, so a slow disk read delays the RAM blocks
   queued behind it.
5. **First-use reads.** Because the RAM tier is file-backed, copying a block whose pages are not in memory becomes a
   disk read. This is the probable cause of the late blocks in section 4.3.
