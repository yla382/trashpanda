# Limitations and scope

What trash-panda does not do yet, and what is out of scope for now.

## What it does not do (yet)

* coordinating memory between several models that are loaded at the same time (for example a text encoder
  and a diffusion model). Each model is planned on its own from the memory that is free when you plan it, so
  plan them one after the other, or free the first model before loading the next. Models that run one after another
  are what `Coordinator` is for (see "Running a pipeline")
* full overlap of reading and computing for disk blocks: blocks waiting in RAM and on disk are prefetched through
  pinned buffers by a background thread, two blocks ahead by default, but the drive's speed sets a floor that no
  amount of prefetching removes (see "Known issues")
* Macs and other non-NVIDIA GPUs: the prefetching (a second CUDA stream, events and pinned buffers) is CUDA-only.
  On a Mac the library should fall back to the CPU, with plain copies and no prefetching, but this has not been tested
  on a Mac yet, and the Apple GPU (`mps`) has not been tried at all. A Mac's CPU and GPU also share one memory pool,
  so splitting a model between "GPU" and "RAM" does not free anything there
* int8 / 4-bit formats (fp8 and ordinary float dtypes only)
* LoRA/LoKr adapters, multi-GPU, model definitions

## Out of scope for now

The current direction is sequential model chaining: each model of a pipeline is loaded, run and released before the
next one is planned (for example text encoder, then diffusion model, then VAE). The following are not part of that
direction, or come after it, and are not planned for now:

* **ControlNet**: an extra network that steers a diffusion model with a guide image (edges, pose, depth). It runs
  alongside the diffusion model at every denoising step, so both models must be loaded at the same time
* **Speculative decoding**: a small draft model proposes several tokens and the large language model verifies them in
  one pass. Both models run throughout the generation, so both must be loaded at the same time
* **GGUF models**, possibly through `llama-cpp-python` as a separate backend
* **Computing the RAM tier on the CPU** instead of copying it to the GPU, as llama.cpp does, for token-by-token
  generation
* **Batching for language models**, so one weight copy serves many sequences
* **Multi-GPU support**
* **A KV-cache budget for language models**, from the context length you intend to use
* **4-bit and 8-bit storage** (for example NF4 with `bitsandbytes`, or our own implementation), to fit more of the
  model in RAM and on the GPU and halve the traffic
* **fp8 checkpoints with block-wise scales** (the `weight_scale_inv` layout used on the Hugging Face Hub), with one
  scale per 128 x 128 block of weights. It touches the scale lookup, the expansion and the staging of the scales
* **LoRA and other adapters** on offloaded blocks
* **A `from_pretrained`-style helper for `transformers` models** (find the weight files, build on the meta device with
  real buffers, plan, load and install the hooks in one call), ignoring buffers that are not in the checkpoint (rotary
  tables, position ids) instead of failing, and making `model.dtype`, `model.device` and `model.to(...)` behave on
  offloaded models so `generate()` and `pipeline()` work
* **A ComfyUI custom node** that loads and runs a model through trash-panda inside a ComfyUI workflow
* **A memory-mapped disk tier** as an experiment: map the checkpoint once and let the operating system keep recently
  used blocks cached, instead of an explicit split between RAM and disk. To be measured against explicit read-ahead,
  because a least-recently-used cache can thrash when the model is read in a cycle
* **Starting the next model's reads behind compute in a pipeline that runs the same stages for many inputs in a row**:
  the first model of the next input would be read while the last stage of the current one computes. It goes beyond
  `preload()`, which covers one run through the stages
* **Reading the order of the blocks from the source of `forward()`** with the `ast` module, to know the order of a
  model that has never run without calling it. It fails on code that is not plain (helper methods, dynamic
  dispatch, branches that depend on the inputs), and a wrong order is prefetched with confidence, which is worse
  than the definition order. `record_order()` calls the model on the meta device instead and sees the real order
* **Cheaper expansion**: reuse preallocated buffers and fuse the fp8 to bf16 conversion and scaling, instead of
  creating new tensors for every weight on every pass
* **A cost model** that chooses between GPU, RAM and disk from measured disk and PCIe speed
* **Logging**: an optional log through Python's `logging` module, and an optional structured file (for example
  `TRASH_PANDA_LOG=run.jsonl`) with the plan, memory and timings of a run. The visualization and a comparison of two
  runs could read it

These may be considered in the future. They are not part of the immediate plan.
