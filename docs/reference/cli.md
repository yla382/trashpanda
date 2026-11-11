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

## Exit codes

`0` when the command ran. `2` when the arguments are wrong, for example a checkpoint that does not exist.
