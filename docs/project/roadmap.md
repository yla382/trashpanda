# Roadmap

Model chaining is built: `stage()` loads, runs and releases one model, `preload()` loads the next model while one runs,
and a `Coordinator` runs a whole pipeline with one pinned pool (see [running a
pipeline](../guides/running-a-pipeline.md)). The items below are what is left around it, roughly in the order they are
planned. Nothing here is promised or scheduled.

## Speed

* The loader thread of `stage(overlap=True)` reads with plain seek-and-read, which ran at 0.25 to 0.28 GB/s against
  0.41 GB/s for the memory-mapped reads that `stream_load()` does for the same blocks (12 blocks, 5.21 GB: 18.5 to
  21.2 s against 12.7 s). Memory-mapped reads in the loader would remove most of that gap. The loader would have to open
  and keep its own mapping for the whole load, because mapping the whole file for every read charges the Windows commit
  limit (see the comment in `checkpoint.py`). What `overlap` can save on the test pipeline stays bounded by the compute
  that can overlap with the load, about 3 s, so this matters mostly for chaining. Two other parts of `overlap` are open:

    * The peak GPU memory is not remembered with `overlap=True`, because the load falls inside the measured part of the
      run. It has to be measured after the load, or with the weights subtracted.
    * The order of the weights outside the blocks (embeddings, input and output layers) in the order file, if they
      should be loaded in the order they are needed; today they are loaded before the first call.

* The number of pinned buffers as a tested option (`TRASH_PANDA_PINNED_BUFFERS`, 3 or 4 instead of 2), if the
  profile's "worker waited for a free pinned buffer" line shows the worker is held up by them.

## Compatibility

* `diffusers` pipelines: build the large components with `stream_load` and hand them to the pipeline class, so
  `from_pretrained` does not load everything into RAM first.

## Reliability and project

* Investigate sudden exits: rarely, and only when RAM is already high (about 12 GB in use on a 16 GB machine)
  before a large model is loaded, the process ends with no Python error, usually while the weights are being
  loaded. It is probably the operating system ending the process or a crash in native code, but this has not been
  confirmed. Next time, record the exit code and run with `PYTHONFAULTHANDLER=1`, and check whether it also happens
  from a clean memory state.
* Find out where a sudden exit goes wrong: when a run ends without a Python error, check the exit code right
  afterwards (`$LASTEXITCODE` in PowerShell) and run with `PYTHONFAULTHANDLER=1`. A large negative code such as
  `-1073741819` is a crash in native code, while a normal code points to the process being ended by the system. Also
  print the memory in use and the available commit (RAM plus page file) before loading, so a failed run shows what
  the machine had left.
* An option to keep chosen blocks or weights when `final_pass()` frees a model, if a model turns out to need it. By
  default every block is freed after it runs and the weights outside the blocks are left alone. A block that runs more
  than once in a pass (weight sharing as in ALBERT, or a caller that runs the model twice in one step) would be freed
  after its first run, and a weight tied between a block and an outside layer would be shared with a layer that still
  needs it. A list of name prefixes, like `skip_prefixes`, would leave those alone. It may not be needed.
* Tests on tiny `transformers` and `diffusers` models on the CPU, in CI.
* Tests on more models, GPUs and operating systems (Linux in particular), with benchmarks published here. A first
  test on a Mac is planned: the unit tests and `examples/sdxl_benchmark.py --size 512 --steps 10` in the baseline and
  trash-panda modes, to see whether it installs and runs on the CPU or the Apple GPU. Real Mac support, if there is
  demand, would be a separate piece of work (`mps`, unified memory, fp8 on the Apple GPU).
* A final project name, and a contributor agreement before outside code is accepted.
