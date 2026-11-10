# Known issues

* **Automatic plans change from run to run.** `plan()` uses the RAM that is free at the moment you call it, and
  free RAM varies. On a 16 GB machine the same script gave a RAM budget of 6.2 GB in one run and 8.0 GB in
  another, which moved several blocks between RAM and disk
* **Load other models before you plan.** If a second model (a text encoder, a VAE) is loaded after `plan()`, the
  memory the plan counted on is taken. In RAM, Windows pages the weights out and the first step is very slow. On the
  GPU, too many blocks stay resident and the overflow spills into shared memory, which can make every step several
  times slower. Create the other models first so the plan sees the memory that is really left
* **The default RAM reserve is a rule of thumb.** `min_free_ram="auto"` keeps 8% of the machine's RAM free (between
  1 and 4 GB, so about 1.3 GB on a 16 GB machine). If your other programs use a lot of memory, or very little, set
  `min_free_ram` yourself, for example `"500MB"` or `0`
* **The default VRAM headroom is a fixed guess.** `headroom_vram="1.5GB"` does not know how much memory your
  activations need (it grows with image size), and measuring it is a manual step for now (see "Fitting the GPU to
  your workload"). If too many blocks are resident, Windows silently spills GPU memory into shared system memory
  and every step becomes several times slower, with no error. Pass a measured `activation_vram`, lower
  `resident_blocks`, or set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control
  Panel so it fails loudly
* **Disk blocks are limited by the speed of the drive.** Blocks on disk are read ahead of time (two blocks ahead by
  default, `lookahead=`), which hid about a third of the read time in testing, but a read takes much longer than
  one block's compute on a slow drive. On the test machine's SATA SSD (about 400 MB/s measured, with 1, 2, 4 or 8
  reader threads alike) a plan with 11 blocks (4.1 GB) on disk took about 16 s per step, against about 9.3 s with
  every streamed block in RAM. A deeper look-ahead (4) was no better and uses more VRAM. These are single runs
* **The first step of a model with a RAM tier is slow.** The RAM tier is mapped from the checkpoint file, so each block
  is read from the drive the first time it is used. On the test machine that was 7.3 GB at about 0.4 GB/s: the first
  step of a diffusion model took 20 to 22 s against 6.3 s for the others, and 12.5 to 15 s of the difference was the
  main thread waiting for blocks. Every cold read on that machine ran at 0.36 to 0.42 GB/s, however it was read, so
  only a faster drive shortens it. `TRASH_PANDA_RAM_IN_MEMORY=1` moves the read to the load, before the first step, but
  the total time is not shorter (see "Settings and environment variables")
* **Checkpoints with the same file name share an order file name.** The order file is named after the checkpoint file
  plus `_order.json`. Diffusers models are usually all called `diffusion_pytorch_model.safetensors`, so with one shared
  `order_file_destination` they overwrite each other's file. With the default location, next to the checkpoint, this
  does not happen. A file that names a block the model does not have is ignored, so nothing breaks, but the file is
  rewritten on every run
* **A PyTorch warning with the profile on.** With `TRASH_PANDA_PROFILE=1`, a release may print
  `Deallocating Tensor that still has live PyObject references`. The memory trace of the
  profile keeps a weak reference to every large tensor it finds, and PyTorch warns when one of them is freed. It is
  harmless, nothing reads those tensors afterwards, and it appears from run to run only by chance. Without the profile
  it should not appear
* **Tested on one setup so far:** a 12B diffusion model (fp8 with scales) on Windows with an RTX 3060 (12 GB) and
  16 GB of RAM. Other models, GPUs and operating systems are not covered yet. The unit tests run on the CPU, and
  `examples/sdxl_benchmark.py` has a path for machines without CUDA (the Apple GPU or the CPU) that is untested
