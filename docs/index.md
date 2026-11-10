---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![A raccoon holding a circuit board in a trash can](assets/hero.png)

<div markdown>

# trash-panda

<p class="tagline">Run models bigger than your GPU: stream weights layer by layer across GPU, RAM and disk.</p>

[Quick start](quickstart.md){ .md-button .md-button--primary }
[API reference](reference/index.md){ .md-button }
[GitHub](https://github.com/yla382/trashpanda){ .md-button }

</div>

</div>

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quick start](quickstart.md)**

    From install to a first run, in a few minutes.

- :material-layers-triple: **[Running a stage](guides/running-a-stage.md)**

    Load, run and release one model.

- :material-pipe: **[Running a pipeline](guides/running-a-pipeline.md)**

    Several models one after another, with the next one loading while the current one runs.

- :material-book-open-variant: **[API reference](reference/index.md)**

    Every public function and class, with its parameters and defaults.

- :material-wrench-cog: **[Design](design/index.md)**

    How it works in detail, and the measurements behind the defaults.

</div>

## Why

Loading a big checkpoint the usual way needs two full copies in RAM, and a model that does not fit in VRAM simply
fails. trash-panda builds the model without allocating weights (meta device), reads the checkpoint tensor by tensor,
and keeps each block of the model wherever it fits best:

| Tier | Where the block waits | Cost per forward pass |
|---|---|---|
| `gpu`  | on the GPU, in compact form (for example fp8) | only the expansion to bf16 |
| `ram`  | in system RAM, in compact form | a transfer over PCIe |
| `disk` | in the checkpoint file | a file read |

Only one block at a time is expanded to the compute dtype, so peak VRAM is roughly one block plus your activations,
whatever the model size.

## Install

Install PyTorch first (the build that matches your GPU, see [pytorch.org](https://pytorch.org)), then:

```
pip install dumpsterpanda
```

The package is imported as `trash_panda`.
