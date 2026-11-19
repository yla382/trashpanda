# Command line

The `trash-panda` command is installed with the extra `dumpsterpanda[cli]`. How to read its output is on the
[command line guide](../guides/command-line.md). Run `trash-panda --help` or `trash-panda <command> --help` for the
same list in the terminal.

## `trash-panda plan`

```
trash-panda plan CHECKPOINT [OPTIONS]
```

Print where each block of a model would live on this machine, without loading it. It is an estimate: the blocks are found
from the names and sizes in the checkpoint, not from the model's code.

`CHECKPOINT` is a `.safetensors` file, a folder of shards, or the index json of a sharded checkpoint.

| Option | Default | Meaning |
|---|---|---|
| `--max-vram TEXT` | the free VRAM | VRAM to plan for, such as `"9GB"` |
| `--max-ram TEXT` | the available RAM | RAM to plan for, such as `"10GB"` |
| `--min-free-ram TEXT` | `auto` | RAM kept free for the system: `auto` or a size such as `"2GB"` |
| `--resident-blocks INTEGER` | as many as fit | force this many blocks onto the GPU |
| `--headroom-vram TEXT` | `1.5GB` | VRAM kept free for activations |
| `--compute-dtype TEXT` | `bf16` | dtype a block is expanded to when it runs |
| `--storage-dtype TEXT` | `auto` | dtype blocks wait in: `auto`, `fp8`, ... |
| `--no-disk` | off | keep blocks in RAM instead of reading them from disk |
| `--skip TEXT` | none | leave out checkpoint entries with this prefix; can be given more than once |
| `--blocks` | off | also list every block with its tier |

Each option is the argument of [`plan()`](planning.md#trash_panda.offload.planner.plan) with the same meaning:
`--max-vram` is `max_vram`, `--no-disk` is `allow_disk=False`, and so on.

## `trash-panda info`

```
trash-panda info CHECKPOINT [OPTIONS]
```

Show what is inside a checkpoint: its size, data types, groups of repeated blocks, the tensors outside the blocks, and the
tensors of one block. Sizes are as stored in the file.

| Option | Default | Meaning |
|---|---|---|
| `--keys` | off | also list every tensor with its dtype, shape and size |

## `trash-panda trace`

```
trash-panda trace [FILE] [OPTIONS]
```

Replay a recorded run in the browser, list the recordings, or remove old ones. Without `FILE` it opens the newest file of
the trace folder (`TRASH_PANDA_TRACE_DIR`, see [settings](settings.md#trace-files)). A recording is made by running with
`TRASH_PANDA_TRACE=1`. The command does not import PyTorch.

| Option | Default | Meaning |
|---|---|---|
| `FILE` | the newest recording | a trace file written by trash-panda |
| `--port INTEGER` | `0` | the port to serve on; `0` picks a free one |
| `--no-open` | off | do not open the browser, only print the address |
| `--list` | off | list the trace files of the trace folder and stop |
| `--clean` | off | remove trace files from the trace folder; it asks first |
| `--keep INTEGER` | none | with `--clean`: keep the newest N files and remove the others |
| `--older-than TEXT` | none | with `--clean`: remove files older than this: `90m`, `12h`, `30d` or `2w` |
| `--yes` | off | with `--clean`: do not ask first |

`--list` and `--clean` work on the trace folder and take no file. `--keep`, `--older-than` and `--yes` belong to `--clean`.
With both limits a file is removed only if it is beyond the newest N and older than the age; with neither limit every file is
removed. How to read the page is on the [command line guide](../guides/command-line.md#trash-panda-trace).

## `trash-panda config`

```
trash-panda config
```

No arguments or options. Prints a table of the variables trash-panda reads, with their current values. A value nobody set is
marked `(default)`. See the [command line guide](../guides/command-line.md#trash-panda-config) and the
[settings](settings.md).

## `trash-panda hardware`

```
trash-panda hardware
```

No arguments or options. Prints a table with the columns Item, Value and Note: the versions of trash-panda, Python and
PyTorch, the operating system, the device (`cuda`, `mps` or `cpu`), whether the memory is unified, the RAM, the commit
headroom on Windows, and with CUDA the GPU, its VRAM, the CUDA version, the compute capability, and bf16 and fp8 support.
See the [command line guide](../guides/command-line.md#trash-panda-hardware).

## Exit codes

`0` when the command ran. `1` when `trash-panda trace` has nothing to open (no recording in the trace folder, or a file
that is not a trace), when a cleanup was declined, or when `trash-panda config` finds a value that cannot be read. `2` when the arguments are wrong, for example a checkpoint that does
not exist.
