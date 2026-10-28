"""SDXL benchmark: the same prompt and seed run with plain diffusers, diffusers offloading, accelerate and trash-panda.

Every mode should run in a fresh process, so that memory left behind by one run cannot change the next one.
`--mode all` does that for you: it starts this script once per mode and prints a table at the end.

    python sdxl_benchmark.py --mode baseline
    python sdxl_benchmark.py --mode trash-panda --unet-vram 3GB
    python sdxl_benchmark.py --mode all --unet-vram 3GB
    python sdxl_benchmark.py --mode trash-panda-release --unet-vram 3GB
    python sdxl_benchmark.py --mode clean            # shows the size of the downloaded model; add --yes to delete it

Every single-mode run ends with a table of the memory the whole system had free at several points. In the
"trash-panda-release" mode the UNet is released with trash_panda.release() after the run, so the table shows how much of
its RAM and VRAM comes back; "trash-panda" ends with handle.remove() instead, for comparison. The release mode is not part
of `--mode all`.

If the process still holds more than 0.5 GB above its starting level when the run ends (CUDA only), the script first
tries to empty PyTorch's cache of pinned memory, and then starts the probe: `--mode probe` runs in a fresh process and
shows how much RAM the CUDA context, cuDNN/cuBLAS and two pinned buffers each hold, and whether freeing them gives it
back. `--no-probe` skips it.

This script is not part of the trash-panda package. It needs its own packages:

    pip install diffusers transformers accelerate huggingface_hub psutil torch dumpsterpanda

The first run downloads the fp16 weights of stabilityai/stable-diffusion-xl-base-1.0 (about 7 GB).

Without an NVIDIA GPU (a Mac, for example) the script uses the Apple GPU (mps) if there is one, and the CPU otherwise.
Only the "baseline" and "trash-panda" modes run there, the peak VRAM is not measured, and trash-panda's prefetching is
CUDA-only, so those numbers say whether it works, not how fast it is. This path is untested. Use a smaller image,
for example:

    python sdxl_benchmark.py --mode baseline --size 512 --steps 10
    python sdxl_benchmark.py --mode trash-panda --size 512 --steps 10
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import psutil
import torch
from accelerate import dispatch_model, infer_auto_device_map
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download, scan_cache_dir
from trash_panda import detect_hardware, find_blocks, install_offload, plan, release, stream_load

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "a weathered fisherman in a yellow raincoat on a wooden dock at sunrise, holding a lantern, photorealistic"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
COMPUTE_DTYPE = "fp32" if DEVICE == "cpu" else "fp16"

MODES = ["baseline", "model-offload", "accelerate", "trash-panda", "trash-panda-release"]
AVAILABLE = MODES if DEVICE == "cuda" else ["baseline", "trash-panda", "trash-panda-release"]    # the others need a CUDA GPU
COMPARED = [m for m in AVAILABLE if m != "trash-panda-release"]       # what --mode all runs
PROBE_THRESHOLD_GB = 0.5      # a run that leaves its process holding more than this triggers the memory probe
LABELS = {
    "baseline": "diffusers, everything on the GPU",
    "model-offload": "diffusers model offloading",
    "accelerate": "accelerate device map",
    "trash-panda": "trash-panda",
    "trash-panda-release": "trash-panda, released after the run",
}


class PeakRam:
    """Samples this process's memory in the background and remembers the highest value (GB)."""

    def __init__(self):
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            self.peak = max(self.peak, proc.memory_info().rss / 1e9)
            time.sleep(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def free_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def note_memory(points, label):
    """Remember how much RAM (and VRAM) the whole system has free right now, and how much RAM this process holds."""
    hw = detect_hardware()
    process = psutil.Process(os.getpid()).memory_info().rss / 1e9
    points.append((label, hw.ram_available / 1e9, hw.vram_free / 1e9 if DEVICE == "cuda" else None, process))


def memory_table(points):
    """Aligned text columns; 'change' is against the row above, so each row shows what its step did."""
    width = max(len(label) for label, *_ in points)
    lines = [f"  {'':<{width}}  {'RAM free':>9}  {'change':>7}  {'VRAM free':>10}  {'change':>7}  {'process':>8}  {'change':>7}"]
    previous = None
    for label, ram, vram, process in points:
        ram_change = "" if previous is None else f"{ram - previous[1]:+.2f}"
        vram_free = "n/a" if vram is None else f"{vram:.2f}"
        vram_change = "" if previous is None or vram is None else f"{vram - previous[2]:+.2f}"
        process_change = "" if previous is None else f"{process - previous[3]:+.2f}"
        lines.append(f"  {label:<{width}}  {ram:>9.2f}  {ram_change:>7}  {vram_free:>10}  {vram_change:>7}"
                     f"  {process:>8.2f}  {process_change:>7}")
        previous = (label, ram, vram, process)
    first, last = points[0], points[-1]
    lines.append("\n  Compared with the start (the system's free memory and the process's own memory need not match):")
    lines.append(f"    system RAM free : {last[1] - first[1]:+.2f} GB")
    if first[2] is not None:
        lines.append(f"    system VRAM free: {last[2] - first[2]:+.2f} GB")
    lines.append(f"    this process    : {last[3] - first[3]:+.2f} GB")
    return "\n".join(lines)


def probe_process_memory():
    """In a fresh process: how much RAM does each thing a CUDA process keeps cost, and is it given back?"""
    points = []

    def note(label):
        gc.collect()
        time.sleep(0.5)                    # let the system numbers settle
        note_memory(points, label)

    note("start")
    torch.zeros(1, device="cuda")
    note("CUDA context")
    x = torch.randn(1, 64, 128, 128, device="cuda", dtype=torch.float16)
    conv = torch.nn.Conv2d(64, 64, 3, padding=1).half().cuda()
    conv(x)
    flat = x.view(64, -1)[:, :1024]
    torch.matmul(flat, flat.T)
    sync()
    note("one conv + matmul (cuDNN, cuBLAS)")
    del x, conv, flat

    buffers = [torch.empty(int(0.7e9), dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    for buffer in buffers:
        buffer.fill_(1)
    note("2 pinned buffers of 0.7 GB")
    del buffers, buffer                    # the loop variable would otherwise keep the last buffer alive
    note("pinned buffers deleted")
    free_cache()
    note("free_cache()")
    empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_host_cache is not None:
        empty_host_cache()
        note("pinned cache emptied")

    big = torch.empty(int(1.1e9), dtype=torch.uint8, pin_memory=True)     # just above 2**30 bytes (1.07 GB)
    big.fill_(1)
    note("1 pinned buffer of 1.1 GB")
    del big
    if empty_host_cache is not None:
        empty_host_cache()
        note("pinned cache emptied again")
    print(memory_table(points))


def generate(pipe, args, label, save=False):
    """One timed generation. Returns a dict of results."""
    pipe.vae.enable_tiling()          # decode the image in tiles, so the VAE needs far less memory
    step_times = []

    def on_step(_pipe, _step, _timestep, kwargs):
        sync()
        step_times.append(time.perf_counter())
        return kwargs

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator("cuda" if DEVICE == "cuda" else "cpu").manual_seed(args.seed)
    sync()
    start = time.perf_counter()
    step_times.append(start)
    with PeakRam() as ram:
        image = pipe(PROMPT, num_inference_steps=args.steps, height=args.size, width=args.size,
                     generator=generator, callback_on_step_end=on_step).images[0]
    sync()
    total = time.perf_counter() - start
    steps = [b - a for a, b in zip(step_times, step_times[1:])]
    if save and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        image.save(os.path.join(args.save_dir, f"sdxl_{label}.png"))
    return {
        "mode": label,
        "total_s": round(total, 1),
        "per_step_s": round(sum(steps) / len(steps), 2),
        "other_s": round(total - sum(steps), 1),      # prompt encoding and the VAE decode
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None,
        "peak_ram_gb": round(ram.peak, 2),
    }


def load_pipeline(**kwargs):
    return StableDiffusionXLPipeline.from_pretrained(
        REPO, torch_dtype=DTYPE, variant="fp16", use_safetensors=True, **kwargs
    )


def vram_limit(args):
    """The limit on the UNet's weights in bytes or as a string, or None for no limit."""
    if args.unet_vram.lower() in ("none", "auto"):
        return None
    return args.unet_vram


def run_baseline(args):
    pipe = load_pipeline().to(DEVICE)
    return pipe, None


def run_model_offload(args):
    pipe = load_pipeline()
    pipe.enable_model_cpu_offload()
    return pipe, None


def run_accelerate(args):
    pipe = load_pipeline()
    pipe.vae.to("cuda")
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    limit = vram_limit(args)
    gpu_room = limit if limit is not None else torch.cuda.mem_get_info()[0]
    device_map = infer_auto_device_map(
        pipe.unet,
        max_memory={0: gpu_room, "cpu": "100GB"},     # the weights are already in RAM; this only has to be big enough
        no_split_module_classes=getattr(pipe.unet, "_no_split_modules", None),
    )
    on_gpu = sum(1 for place in device_map.values() if place == 0)
    print(f"{on_gpu} of {len(device_map)} parts of the UNet are on the GPU, the rest on the CPU")
    pipe.unet = dispatch_model(pipe.unet, device_map=device_map, main_device="cuda")
    return pipe, None


def run_trash_panda(args):
    weights = hf_hub_download(REPO, "unet/diffusion_pytorch_model.fp16.safetensors")
    unet_config = UNet2DConditionModel.load_config(REPO, subfolder="unet")
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(unet_config)

    for group in find_blocks(unet):
        print(f"{len(group.prefixes):3d} blocks, for example {group.prefixes[0]}")

    limit = vram_limit(args) if DEVICE == "cuda" else None      # the limit is only used on a CUDA GPU
    extra = {} if args.lookahead is None else {"lookahead": args.lookahead}    # otherwise the library's default
    if limit is None:
        p = plan(unet, weights, compute_dtype=COMPUTE_DTYPE, **extra)
    else:
        p = plan(unet, weights, compute_dtype=COMPUTE_DTYPE, max_vram=limit, headroom_vram=0, **extra)
    print(p)
    print(f"look-ahead: {p.lookahead}")
    print(stream_load(unet, weights, plan=p, device=DEVICE))
    handle = install_offload(unet)
    unet.eval().requires_grad_(False)

    pipe = load_pipeline(unet=unet)
    pipe.vae.to(DEVICE)                    # not pipe.to(DEVICE): that would move the UNet's weights too
    pipe.text_encoder.to(DEVICE)
    pipe.text_encoder_2.to(DEVICE)
    return pipe, handle


RUNNERS = {
    "baseline": run_baseline,
    "model-offload": run_model_offload,
    "accelerate": run_accelerate,
    "trash-panda": run_trash_panda,
    "trash-panda-release": run_trash_panda,
}


def run_one(mode, args):
    """Run one mode in this process and print its result as a line that --mode all can read."""
    print(f"RAM in use at the start: {psutil.virtual_memory().used / 1e9:.1f} GB")
    memory = []
    note_memory(memory, "start")
    pipe, handle = RUNNERS[mode](args)
    note_memory(memory, "pipeline loaded")
    if not args.no_warmup:
        generate(pipe, args, "warmup")     # the first run includes one-time CUDA setup, so it is not reported
    result = generate(pipe, args, mode, save=True)
    note_memory(memory, "after the run")
    if handle is not None:
        if mode == "trash-panda-release":
            release(pipe.unet, handle)     # drops the UNet's weights, stops the copy thread, frees the GPU cache
            note_memory(memory, "UNet released")
        else:
            handle.remove()                # restores the blocks, stops the copy thread and frees the pinned buffers
            note_memory(memory, "handle removed")
        time.sleep(5)                      # Windows may take back freed memory a little late
        note_memory(memory, "5 s later")
    del pipe
    gc.collect()
    free_cache()
    note_memory(memory, "pipeline deleted")
    if DEVICE == "cuda" and memory[-1][3] - memory[0][3] > PROBE_THRESHOLD_GB:
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)    # PyTorch keeps freed pinned memory for reuse
        if empty_host_cache is not None:
            empty_host_cache()
            note_memory(memory, "pinned cache emptied")
    print("\nMemory in GB: what the whole system has free, and what this process holds ('change' is against the row above):\n"
          + memory_table(memory))
    held = memory[-1][3] - memory[0][3]
    if DEVICE == "cuda" and held > PROBE_THRESHOLD_GB and not args.no_probe:
        print(f"\nThis process still holds {held:.2f} GB more than at the start. "
              "What does a fresh process hold for the usual suspects?\n", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", "probe"])
    print("RESULT " + json.dumps(result))


def table(results, start_ram):
    lines = [
        "| Setup | Total (s) | Per step (s) | Encode + decode (s) | Peak VRAM (GB) | Peak RAM (GB) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {LABELS.get(r['mode'], r['mode'])} | {r['total_s']} | {r['per_step_s']} | {r['other_s']} | "
            f"{'n/a' if r['peak_vram_gb'] is None else r['peak_vram_gb']} | {r['peak_ram_gb']} |"
        )
    return "\n".join(lines)


def run_all(args):
    """Run every mode in its own fresh process and print a table of the results."""
    forwarded = ["--unet-vram", args.unet_vram, "--steps", str(args.steps), "--size", str(args.size),
                 "--seed", str(args.seed)]
    if args.lookahead is not None:
        forwarded += ["--lookahead", str(args.lookahead)]
    if args.no_warmup:
        forwarded.append("--no-warmup")
    forwarded.append("--no-probe")         # the memory tables are not shown here, so the probe would only slow it down
    if args.save_dir:
        forwarded += ["--save-dir", args.save_dir]
    results = []
    for mode in COMPARED:
        print(f"\n=== {mode} ===  (RAM in use before it starts: {psutil.virtual_memory().used / 1e9:.1f} GB)")
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", mode, *forwarded],
                              capture_output=True, text=True)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if done.returncode == 0 and lines:
            results.append(json.loads(lines[-1][len("RESULT "):]))
            print("done")
        else:
            print(f"failed (exit code {done.returncode}). Last lines of its output:")
            print("\n".join((done.stdout + done.stderr).strip().splitlines()[-15:]))
    print("\n" + table(results, None))


def clean(args):
    """Show the size of the downloaded model in the Hugging Face cache, and delete it with --yes."""
    cached = next((r for r in scan_cache_dir().repos if r.repo_id == REPO and r.repo_type == "model"), None)
    if cached is None:
        print("Nothing to delete: the model is not in the cache.")
        return
    print(f"{cached.repo_id}: {cached.size_on_disk_str} in {cached.repo_path}")
    if args.yes:
        shutil.rmtree(cached.repo_path)
        print("Deleted. Other projects that use this model will download it again.")
    else:
        print("Nothing was deleted. Add --yes to delete it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=[*MODES, "all", "clean", "probe"],
                        help="what to run; 'all' runs every mode in its own process; 'clean' manages the download; "
                             "'probe' measures what a fresh CUDA process holds in RAM (a run starts it by itself when needed)")
    parser.add_argument("--unet-vram", default="3GB",
                        help="limit on the UNet's weights on the GPU for accelerate and trash-panda; 'none' for no limit")
    parser.add_argument("--lookahead", type=int, default=None,
                        help="trash-panda only: how many blocks ahead to copy (0 turns prefetching off); default is the library's")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024, help="width and height of the image")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up generation")
    parser.add_argument("--no-probe", action="store_true",
                        help="do not run the memory probe at the end when the process still holds memory")
    parser.add_argument("--save-dir", default=None, help="save the generated images here (by default nothing is saved)")
    parser.add_argument("--yes", action="store_true", help="with --mode clean: really delete the downloaded model")
    args = parser.parse_args()
    if args.mode in MODES and args.mode not in AVAILABLE:
        parser.error(f"--mode {args.mode} needs a CUDA GPU; on this machine ({DEVICE}) use one of: {', '.join(AVAILABLE)}")

    if args.mode == "all":
        run_all(args)
    elif args.mode == "clean":
        clean(args)
    elif args.mode == "probe":
        probe_process_memory()
    else:
        run_one(args.mode, args)


if __name__ == "__main__":
    main()
