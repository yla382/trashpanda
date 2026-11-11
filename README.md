<p align="center">
  <img src="https://raw.githubusercontent.com/yla382/trashpanda/main/assets/raccoon_1.png" width="260" alt="A raccoon holding a circuit board in a trash can">
</p>

<h1 align="center">trash-panda</h1>

<p align="center">
  <strong>Run models bigger than your GPU: stream weights layer by layer across GPU, RAM and disk.</strong>
</p>

<p align="center">
  <a href="https://yla382.github.io/trashpanda/">Documentation</a> &middot;
  <a href="https://yla382.github.io/trashpanda/quickstart/">Quick start</a> &middot;
  <a href="https://yla382.github.io/trashpanda/guides/running-a-stage/">Guides</a> &middot;
  <a href="https://yla382.github.io/trashpanda/reference/">API Guide</a> &middot;
  <a href="https://yla382.github.io/trashpanda/design/">Design</a> &middot;
  <a href="https://yla382.github.io/trashpanda/project/roadmap/">Roadmap</a> &middot;
  <a href="CHANGELOG.md">Changelog</a> &middot;
  <a href="https://pypi.org/project/dumpsterpanda/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/dumpsterpanda/"><img src="https://img.shields.io/pypi/v/dumpsterpanda?color=607d8b" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-607d8b" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-607d8b" alt="AGPL-3.0 License">
  <a href="https://github.com/yla382/trashpanda/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/yla382/trashpanda/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://yla382.github.io/trashpanda/project/known-issues/"><img src="https://img.shields.io/badge/status-pre--alpha-f59e0b" alt="Pre-alpha"></a>
</p>

> **Status: early (Phase 1).** The API may change. The package name is temporary

**Documentation: [yla382.github.io/trashpanda](https://yla382.github.io/trashpanda/)**

## Why

Loading a big checkpoint the usual way needs two full copies in RAM, and a model that does not fit in VRAM
simply fails. trash-panda builds the model without allocating weights (meta device), reads the checkpoint
tensor by tensor, and keeps each block of the model wherever it fits best:

| Tier | Where the block waits | Cost per forward pass |
|---|---|---|
| `gpu`  | on the GPU, in compact form (for example fp8) | only the expansion to bf16 |
| `ram`  | in system RAM, in compact form | a transfer over PCIe |
| `disk` | in the checkpoint file | a file read |

Only one block at a time is expanded to the compute dtype, so peak VRAM is roughly one block plus your
activations, whatever the model size.

It works with any floating-point checkpoint: fp32, fp16 and bf16 are streamed as they are (no quantization
needed), and fp8 (with or without scale tensors) is kept compact so more fits.

## Install

Install PyTorch first (the build that matches your GPU, see https://pytorch.org), then:

```
pip install dumpsterpanda
```

The package is imported as `trash_panda`. For the command line (`trash-panda plan` and `trash-panda info`), install the
extra: `pip install "dumpsterpanda[cli]"`. To try the latest development version straight from GitHub instead:

```
pip install git+https://github.com/yla382/trashpanda.git
```

### Why is it called `dumpsterpanda` on PyPI? 🦝😢

The project is called **trash-panda**, but the name [`trashpanda`](https://pypi.org/project/trashpanda/#description)
was already taken on PyPI, and PyPI wouldn't let `trash-panda` sit that close to it either. So the raccoon moved
one dumpster over. The import is still `trash_panda`.

## Quick start

One model. `stage()` builds it, plans where each block lives, loads it, and gives its memory back when the block ends:

```python
import torch
from trash_panda import stage

def build():
    with torch.device("meta"):                  # the structure only: no weights are allocated
        return MyModel(config).eval()

with stage(build, "model.safetensors") as (model, p, handle):
    print(p)                                    # where each block lives
    out = model(x)
                                                # here the model's RAM and VRAM are free again
```

The planner decides from the checkpoint's sizes and the memory that is free. Steer it with arguments such as
`resident_blocks=12`, `max_vram="9GB"` or `min_free_ram="4GB"`.

Several models, one after another. A `Coordinator` shares one pinned pool among them and can prepare the next model
while the current one runs:

```python
from trash_panda import Coordinator, Step

co = Coordinator(
    Step("text", build_text, "text.safetensors"),
    Step("unet", build_unet, "unet.safetensors", resident_blocks=12),
    Step("vae", build_vae, "vae.safetensors", skip_prefixes=("encoder.",)),
)
with co:
    with co.stage("text", ram_preload=False) as (text, p, handle):    # its RAM tier is not read early
        cond = encode(text, ...)
    with co.stage("unet") as (unet, p, handle):
        latents = denoise(unet, cond, ...)
    with co.stage("vae") as (vae, p, handle):
        image = decode(vae, latents)
```

## Command line

Look at a checkpoint without loading it. `plan` shows where each block would live on this machine, and `info` shows what is
in the file:

```
trash-panda plan model.safetensors
trash-panda info model.safetensors
```

For a small example checkpoint (16 blocks, 424.5 MB), with the limits set low on purpose so that all three tiers are
used (`--max-vram 400MB --headroom-vram 50MB --max-ram 150MB --min-free-ram 0`), `plan` prints this (the hardware
lines and the warnings are left out):

```text
example.safetensors: 16 blocks, 424.5 MB
┌─────────────┬─────────┬────────┬──────────────┬─────────────┬──────────────┐
│ Group       │       N │ Blocks │          GPU │         RAM │         Disk │
├─────────────┼─────────┼────────┼──────────────┼─────────────┼──────────────┤
│ blocks.<N>. │ 0 to 15 │     16 │ 9 (226.5 MB) │ 3 (75.5 MB) │ 4 (100.7 MB) │
└─────────────┴─────────┴────────┴──────────────┴─────────────┴──────────────┘
┌──────┬──────────┬───────────────────────────────────┬──────────┐
│      │ Budget   │ What it holds                     │     Size │
├──────┼──────────┼───────────────────────────────────┼──────────┤
│ GPU  │ 350.0 MB │ 9 blocks kept on the GPU          │ 226.5 MB │
│      │          │ Input and output layers           │  21.8 MB │
│      │          │ Workspace for streaming blocks in │  83.1 MB │
│      │          │ Unused                            │  18.7 MB │
├──────┼──────────┼───────────────────────────────────┼──────────┤
│ RAM  │ 82.9 MB  │ 3 blocks                          │  75.5 MB │
│      │          │ Unused                            │   7.4 MB │
├──────┼──────────┼───────────────────────────────────┼──────────┤
│ Disk │ none     │ 4 blocks that did not fit in RAM  │ 100.7 MB │
└──────┴──────────┴───────────────────────────────────┴──────────┘
GPU budget: 400.0 MB allowed by --max-vram, less 50.0 MB kept for activations
RAM budget: 150.0 MB allowed by --max-ram, less 67.1 MB kept for the system and copy buffers
```

## Documentation

* [Quick start](https://yla382.github.io/trashpanda/quickstart/): from install to a first run
* Guides: [running a stage](https://yla382.github.io/trashpanda/guides/running-a-stage/),
  [running a pipeline](https://yla382.github.io/trashpanda/guides/running-a-pipeline/),
  [storage dtype](https://yla382.github.io/trashpanda/guides/storage-dtype/) and the
  [command line](https://yla382.github.io/trashpanda/guides/command-line/)
* [API reference](https://yla382.github.io/trashpanda/reference/): every public function and class, with its
  parameters and defaults, and the [settings](https://yla382.github.io/trashpanda/reference/settings/) with their
  `TRASH_PANDA_*` environment variables
* [Design](https://yla382.github.io/trashpanda/design/): how it works, in detail, and what we measured
* [Known issues](https://yla382.github.io/trashpanda/project/known-issues/),
  [limitations](https://yla382.github.io/trashpanda/project/limitations/) and the
  [roadmap](https://yla382.github.io/trashpanda/project/roadmap/)
* [CHANGELOG.md](CHANGELOG.md)

## Benchmarks

Measurements on one machine, and a script to reproduce them with SDXL, are in [BENCHMARK.md](BENCHMARK.md) and
[examples/](examples/). They are single runs on a Windows desktop whose free memory changes from run to run, so they
are indicative only, and they do not show that trash-panda is faster than the other tools.

## Development

```
pip install pytest psutil
pytest
```

Tests run on the CPU with a tiny fake model. The documentation is built with MkDocs (`pip install -r docs/requirements.txt`,
then `mkdocs serve`).

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
