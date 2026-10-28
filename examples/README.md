# Examples

Examples for trying trash-panda. They are not part of the package and are not installed with it: each one lists the
packages it needs.

| File | What it does |
|---|---|
| [sdxl_benchmark.py](sdxl_benchmark.py) | Compares ways of running SDXL on one GPU, each in its own fresh process: time per step, peak VRAM and peak RAM |

## SDXL benchmark

It generates one image with the same prompt, seed and size in each mode and reports the total time, the time per
step, the time for prompt encoding and decoding, the peak VRAM and the peak RAM of the process.

Modes:

| Mode | What it runs |
|---|---|
| `baseline` | plain `diffusers`, everything on the GPU |
| `model-offload` | `diffusers` model offloading (`enable_model_cpu_offload()`) |
| `accelerate` | an `accelerate` device map that limits how much of the UNet's weights may sit on the GPU |
| `trash-panda` | trash-panda with the same limit on the UNet's weights |
| `trash-panda-release` | like `trash-panda`, but the UNet is released with `trash_panda.release()` after the run; the memory table at the end shows how much RAM and VRAM comes back. Not part of `all` |
| `all` | every mode above except `trash-panda-release`, each in its own fresh process, followed by a table of the results |
| `clean` | shows the size of the downloaded model in the Hugging Face cache; add `--yes` to delete it |

### Install

Use an environment with a CUDA build of PyTorch (a plain `pip install torch` on Windows gives a CPU-only build):

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate huggingface_hub psutil dumpsterpanda
```

The first run downloads the fp16 weights of `stabilityai/stable-diffusion-xl-base-1.0` (about 7 GB) from the
Hugging Face Hub. This model is not gated, so no account or login is needed.

**On a Mac, or any machine without an NVIDIA GPU:** a plain `pip install torch` is enough. The script uses the Apple
GPU (`mps`) if there is one and the CPU otherwise. Only the `baseline` and `trash-panda` modes run there, the peak
VRAM shows as `n/a`, and trash-panda's prefetching is CUDA-only, so the numbers say whether it works, not how fast it
is. This path has not been tested yet. Use a small image so it finishes in reasonable time:

```
python sdxl_benchmark.py --mode baseline --size 512 --steps 10
python sdxl_benchmark.py --mode trash-panda --size 512 --steps 10
```

### Run

From this folder:

```
python sdxl_benchmark.py --mode baseline
python sdxl_benchmark.py --mode model-offload
python sdxl_benchmark.py --mode accelerate --unet-vram 3GB
python sdxl_benchmark.py --mode trash-panda --unet-vram 3GB
python sdxl_benchmark.py --mode trash-panda-release --unet-vram 3GB
python sdxl_benchmark.py --mode all --unet-vram 3GB
```

Each single-mode run ends with a table of the RAM (and VRAM) that the whole system has free at several points: at the
start, after the pipeline is loaded, after the run, after cleaning up and after the pipeline is deleted. A third pair of
columns shows the RAM held by the script's own process (its working set), which tells memory the script holds apart
from memory that other programs or the system hold. Run
`trash-panda` and `trash-panda-release` one after the other to compare `handle.remove()` with `release()`.

Options:

| Option | Meaning | Default |
|---|---|---|
| `--unet-vram` | limit on the UNet's weights on the GPU for `accelerate` and `trash-panda`; `none` for no limit | `3GB` |
| `--steps` | number of denoising steps | `20` |
| `--size` | width and height of the image | `1024` |
| `--seed` | random seed | `1234` |
| `--no-warmup` | skip the warm-up generation (the first run includes one-time CUDA setup, so it is not reported) | off |
| `--save-dir` | save the generated images in this folder | nothing is saved |

The last line of each run starts with `RESULT` and holds the numbers as JSON.

### Saving the images

By default no image is saved. To keep them, give a folder:

```
python sdxl_benchmark.py --mode trash-panda --unet-vram 3GB --save-dir images
python sdxl_benchmark.py --mode all --unet-vram 3GB --save-dir images
```

The folder is created if needed, and each mode writes one file named after it, for example
`images/sdxl_trash-panda.png`. Only the measured run is saved, not the warm-up. With the same seed and the same
weights, the modes should give nearly identical pictures, so comparing the saved files is a quick check that
nothing went wrong.

### Getting numbers you can compare

- Close other programs first, and note how much memory is in use at the start (each run prints it).
- Run every mode in its own process (`--mode all` does that). Memory left behind by one run changes the next.
- One run is one measurement. Repeat it before drawing conclusions from small differences.

### Clean up

The downloaded model stays in the Hugging Face cache (by default `~/.cache/huggingface/hub`). To see how much space
it uses, and to delete it:

```
python sdxl_benchmark.py --mode clean
python sdxl_benchmark.py --mode clean --yes
```

Other projects that use the same model share this cache and would download it again.
