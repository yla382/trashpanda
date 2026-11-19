# Command line

Two commands look at a checkpoint without loading it. `trash-panda info` says what is in the file, and
`trash-panda plan` says where the blocks of the model would live on this machine. Both read only the headers of the
safetensors file; no weights are loaded. `trash-panda trace` shows a run that was recorded, in the browser, and
`trash-panda config` prints the settings and what they are now. `trash-panda hardware` prints what this machine has, for
a bug report.

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

## `trash-panda trace`

`plan` says what would happen. `trace` shows what did happen: a run that was recorded, replayed in the browser.

### Recording a run

Set `TRASH_PANDA_TRACE` to `1` before the run:

```powershell
$env:TRASH_PANDA_TRACE = "1"
```

In bash, use `export TRASH_PANDA_TRACE=1`. The run is recorded from the start of a `stage()` block, or of a
`with Coordinator` block, until that block ends. The file is written then, also when the block raises, and its path is
printed:

```text
trash-panda: trace written to C:\Users\you\.cache\trash_panda\traces\20261114-143310-869.json
```

* The file goes to the trace folder: `TRASH_PANDA_TRACE_DIR`, or by default the `traces` folder of the cache folder
  (`TRASH_PANDA_HOME`, or by default `~/.cache/trash_panda`). The folder is made if it is missing.
* The name is the time the recording began, to the millisecond, so every recording is a new file and the names sort by
  time.
* A file holds one recording. A `Coordinator` records all its models in one file. A `stage()` that runs while another
  recording is open adds its model to that recording. Two `stage()` calls one after the other each write a file.
* Recording adds CUDA timing events and a timestamp to each block, copy and wait. The cost has not been measured, so take
  timings from a run without it.
* Without `TRASH_PANDA_TRACE` nothing is recorded, and the recording hooks do nothing.

### Viewing it

```
trash-panda trace            # the newest recording
trash-panda trace FILE       # one file
```

A small server runs on this machine only (`127.0.0.1`) and the browser opens; press Ctrl+C to stop it. `--port` picks the
port and `--no-open` only prints the address. The page shows, from the top:

* **Models in this run**: a lane for each model on one time axis, with its phases (`load`, `pass 1`, ...) and the work it
  did in the background before its turn. Click a lane to select the model.
* **Disk, RAM and GPU**: the blocks of every model, each model in its own color and tag, moving between the tiers while
  the run plays or when you drag the slider. RAM holds the pinned buffers of each pool: a model's own, the shared pool of
  a `Coordinator`, and the two buffers of an early loader. A tile that is being read through ordinary memory stands in RAM,
  in no slot of its own.
* **Timeline**: a track for each thread and CUDA stream (the main thread, the worker, the loader threads, the compute
  stream, the copy streams). The wheel zooms, a drag pans and a click moves the playhead. A gap in the GPU's work is drawn as
  idle; it is worked out from the runs, not recorded.
* **Memory in use**: VRAM, RAM and pinned memory over time, stacked by model, against the budgets of the plans. It is
  worked out from the events; activations and what other programs use are not counted.
* **The time of each block**, and the panels at the sides: what happens at the playhead, the totals of the selected model
  (loading, running, waiting) and of the whole run, and the plan of the selected model.

At the left is the list of the recordings in the trace folder, newest first. Click one to open it; the button at the top
folds the list away. The list only reads: nothing on the page changes a file. What the page does not show is listed at the
bottom of the page, and in [the viewer notes](../design/trace-viewer.md).

### Listing and removing recordings

```
trash-panda trace --list
```

prints a table of the trace folder: the file, when it was written, its size, its duration and its models. A file that is not
a trace shows `invalid`.

```
trash-panda trace --clean --keep 10
trash-panda trace --clean --older-than 30d
```

removes recordings. `--keep N` keeps the newest N and removes the others. `--older-than` takes a number and `m`, `h`, `d` or
`w` (`90m`, `12h`, `30d`, `2w`). With both, a file is removed only if it is beyond the newest N and older than the age;
with neither, every file is removed. The command lists the files and asks before it deletes them, unless you pass `--yes`.
Recording never deletes anything. All the options are on the [command line reference](../reference/cli.md).

## `trash-panda config`

```
trash-panda config
```

prints a table with every environment variable that trash-panda reads and the value it has now: the two folders
(`TRASH_PANDA_HOME` and `TRASH_PANDA_TRACE_DIR`, shown as the paths they resolve to) and then each tuning setting. A value
that nobody set is followed by `(default)`; a value that comes from the environment is shown without it, in bold.

```
| TRASH_PANDA_LOOKAHEAD       | 0           |
| TRASH_PANDA_PINNED_BUFFERS  | 2 (default) |
```

Two things are called out under the table. A variable that starts with `TRASH_PANDA_` but is not one of these (usually a typo)
is listed as ignored. A value that cannot be read, such as `TRASH_PANDA_LOOKAHEAD=many`, is marked `(invalid)` and the command
exits with `1`, because the library would raise an error when it ran. What each setting does is on the
[settings reference](../reference/settings.md).

## `trash-panda hardware`

```
trash-panda hardware
```

prints a table of what this machine has: the versions of trash-panda, Python and PyTorch, the operating system, the device
a model would run on, and the memory. Paste it into a bug report. Each row has a value, and the rows that need a word of
explanation have a note.

```
| Item               | Value                     | Note                          |
|--------------------+---------------------------+-------------------------------|
| trash-panda        | v0.2.3                    |                               |
| Device             | cuda                      | Where a model runs by default.|
| Unified memory     | no                        | ...                           |
| RAM                | 2.66 GiB available of 15.95 GiB |                         |
| GPU                | NVIDIA GeForce RTX 3060         |                         |
| VRAM               | 10.99 GiB free of 12.00 GiB     |                         |
```

* **Device** is `cuda`, `mps` (the Apple GPU) or `cpu`.
* **Unified memory** says whether the GPU and the CPU share one memory pool, as on an Apple silicon Mac. There, splitting a
  model between "GPU" and "RAM" frees nothing. The planner does not use this yet; trash-panda treats an `mps` machine as
  having no VRAM.
* **RAM** is what is free now, out of what is installed. **Commit headroom** (Windows only) is how much more memory the
  system lets programs reserve, which is RAM plus the page file; it is the number to look at when a process ends without an
  error.
* With CUDA, the table also shows the **GPU**, its **VRAM** (free now, out of the total), the CUDA version PyTorch was built
  for, the GPU's **compute capability** (its architecture version), whether it supports **bf16**, and whether it computes
  in **fp8** itself (compute capability 8.9 or higher). On an `mps` machine one row says the GPU memory is shared with RAM; on a
  machine with no GPU the row says none was detected.

Sizes are in GiB (1024 x 1024 x 1024 bytes), the unit that memory is sold and shown in, so a "16 GB" machine reads about
16 GiB and a "12 GB" card reads 12 GiB. `plan` counts in decimal gigabytes, so the same card is 12.88 GB there. The
figures are read when the command runs, so free memory changes from one run to the next.

## Good to know

* The plan is an estimate. The blocks are found from the names and sizes in the checkpoint, not from the model's code, so
  `stage()`, which plans from the real model, can differ. That happens when the blocks of a group do not all hold the same
  tensors, or when the checkpoint needs a `key_map`.
* A checkpoint with no group of numbered blocks has nothing to offload, and `plan` says so.
