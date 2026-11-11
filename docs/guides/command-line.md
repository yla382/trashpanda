# Command line

Two commands look at a checkpoint without loading it. `trash-panda info` says what is in the file, and
`trash-panda plan` says where the blocks of the model would live on this machine. Both read only the headers of the
safetensors file; no weights are loaded.

## Install

The command line is an optional extra. Install it into the environment that has your PyTorch:

```
pip install "dumpsterpanda[cli]"
```

The extra adds [Typer](https://typer.tiangolo.com) and [Rich](https://rich.readthedocs.io). Without it, the command
stops and says which package to install; `import trash_panda` does not need them.

!!! note "Command not found"
    `pip` puts the `trash-panda` command in a folder of the environment it installs into (`Scripts` or `bin`). Inside an
    activated virtual environment or conda environment that folder is on your PATH already. With a `--user` or system
    install it may not be: add that folder to your PATH, which `pip` names in its warning, or activate the environment
    you installed into.

## `trash-panda info`

```
trash-panda info model.safetensors
```

Below is a small example checkpoint with 16 blocks, plus a few tensors outside them.

```text
example.safetensors   424.5 MB   90 tensors   1 file
Data types
┌──────────┬─────────┬──────────┬───────┐
│ Dtype    │ Tensors │     Size │ Share │
├──────────┼─────────┼──────────┼───────┤
│ bfloat16 │      74 │ 424.4 MB │  100% │
│ float32  │      16 │  65.5 KB │   <1% │
└──────────┴─────────┴──────────┴───────┘
Block groups
┌─────────────┬─────────┬────────┬───────────┬──────────┬───────────────────┐
│ Group       │       N │ Blocks │ Per block │    Total │ Tensors per block │
├─────────────┼─────────┼────────┼───────────┼──────────┼───────────────────┤
│ blocks.<N>. │ 0 to 15 │     16 │   25.2 MB │ 402.7 MB │                 5 │
└─────────────┴─────────┴────────┴───────────┴──────────┴───────────────────┘
Outside the blocks: 21.8 MB
┌──────────────────────────────┬─────────┬──────────┐
│ Name                         │ Tensors │     Size │
├──────────────────────────────┼─────────┼──────────┤
│ embed.weight                 │       1 │   8.4 MB │
│ head.weight                  │       1 │   8.4 MB │
│ refiner.layers.0.proj.weight │       1 │   2.1 MB │
│ refiner.layers.1.proj.weight │       1 │   2.1 MB │
│ text_proj.*                  │       6 │ 786.4 KB │
└──────────────────────────────┴─────────┴──────────┘
Inside one block (blocks.0.)
┌─────────────────┬──────────┬─────────────┬────────┐
│ Tensor          │ Dtype    │ Shape       │   Size │
├─────────────────┼──────────┼─────────────┼────────┤
│ mlp.down.weight │ bfloat16 │ 1024 x 4096 │ 8.4 MB │
│ mlp.up.weight   │ bfloat16 │ 4096 x 1024 │ 8.4 MB │
│ attn.qkv.weight │ bfloat16 │ 3072 x 1024 │ 6.3 MB │
│ attn.out.weight │ bfloat16 │ 1024 x 1024 │ 2.1 MB │
│ norm.weight     │ float32  │ 1024        │ 4.1 KB │
└─────────────────┴──────────┴─────────────┴────────┘
```

* **Data types** counts the tensors and bytes of each dtype, so you can see at once whether a file is bf16, fp8 with
  separate scale tensors, or a mix.
* **Block groups** are the repeated blocks, found from the names. The pattern `blocks.<N>.` means the blocks are called
  `blocks.0.`, `blocks.1.` and so on, and the column `N` gives the numbers. These are the exact names to use for `--skip`
  and for `skip_prefixes`.
* **Outside the blocks** are the tensors that belong to no block, such as the embeddings and the output layer. Four or
  more tensors under one name become one row (`text_proj.*`); fewer are listed by their full names.
* **Inside one block** lists the tensors of the first block of the first group, biggest first, so you can see what a
  block is made of.

`--keys` also lists every tensor of the file with its dtype, shape and size.

The sizes are as stored in the file. The planner counts the weights outside the blocks expanded to the compute dtype,
so for an fp8 file the number in `info` and the number in `plan` can differ.

## `trash-panda plan`

```
trash-panda plan model.safetensors
```

The plan is made for the memory that is free right now, so it changes with what else is running. To plan for another
machine or to try a setting, pass the limits yourself. This one pretends the GPU has 400 MB and RAM 150 MB:

```
trash-panda plan example.safetensors --max-vram 400MB --headroom-vram 50MB --max-ram 150MB --min-free-ram 0
```

```text
GPU: NVIDIA GeForce RTX 3060, VRAM 11.80 GB free of 12.88 GB
RAM: 4.25 GB available of 17.13 GB
Memory the system will still let programs reserve (RAM plus page file or swap): 7.97 GB
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

WARNING: 4 blocks will be read from disk on every pass; this is slow unless the drive is fast
(NVMe).
```

* The first lines are the hardware the plan was made for. The line about memory the system will still let programs
  reserve (RAM plus page file or swap) appears only on systems where it can be read.
* The first table is what the model is made of, and how many blocks of each group wait in each tier.
* The second table is the memory. Each tier has a budget, and the rows below it are what fills it: the blocks, the weights
  outside the blocks, and the workspace for streaming blocks in. `Unused` is what is left, and a red `Over the budget`
  row means the plan needs more than the budget. The two lines under the table say where each budget comes from.
* A line that starts with `WARNING:` says what to look at, as in `stage()`.

`--blocks` also prints one row for every block with its tier. All options are on the
[command line reference](../reference/cli.md); they are the arguments of
[`plan()`](../reference/planning.md#trash_panda.offload.planner.plan).

## Good to know

* The plan is an estimate. The blocks are found from the names and sizes in the checkpoint, not from the model's code, so
  `stage()`, which plans from the real model, can differ. That happens when the blocks of a group do not all hold the same
  tensors, or when the checkpoint needs a `key_map`.
* A checkpoint with no group of numbered blocks has nothing to offload, and `plan` says so.
