# Benchmarks

A single measurement of trash-panda on one machine and one model. It is one run, not an average, and free memory
on the machine changes from run to run, so small differences are within the noise.

## Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| RAM | 16 GB |
| Storage | SATA SSD (about 500 MB/s) |
| OS | Windows 11 Home |
| CPU | AMD Ryzen 7 3700X (8 cores) |
| PyTorch | 2.5.1 (CUDA 12.1 build) |
| Python / NVIDIA driver | TODO |
| Model | a 12B diffusion transformer, fp8 with per-tensor scales |
| Text encoder / VAE | a 4B text encoder (fp8, streamed with trash-panda) and an image VAE |
| Image size and steps | 1024 x 1024, 7 steps |
| What is timed | the sampling progress bar only (not model loading, prompt encoding or the VAE decode) |

## Baseline

Memory in use at the start of the run: about **5.7 GB** of the 16 GB (a clean desktop, no browser or games open).
The encoder and VAE were loaded before the plan.

## Result

| Library version | Plan | Time (7 steps) | Per step |
|---|---|---|---|
| 0.0.7 | 12 blocks on the GPU, 20 streamed from RAM, none on disk | 1:04 | 9.2 s |
