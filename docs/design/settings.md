# Settings in detail

The full descriptions of the tuning settings. The short list, for lookup, is in the [settings reference](../reference/settings.md).

Every tuning setting is kept, with its default, in `trash_panda.offload.config`. Each can be changed with an
environment variable named `TRASH_PANDA_<SETTING>` in capitals, for example `TRASH_PANDA_LOOKAHEAD=0`. The variable is
read when a function runs, not when the library is imported. An argument passed to a function wins over the
environment, and the environment wins over the default. Switches take `1` / `0`, `true` / `false`, `yes` / `no` or
`on` / `off`.

In PowerShell a variable lasts for that window only:

```
$env:TRASH_PANDA_PROFILE = "1"
Remove-Item Env:TRASH_PANDA_PROFILE
```

In bash, use `export TRASH_PANDA_PROFILE=1` and `unset TRASH_PANDA_PROFILE`.

| Variable | Default | What it does |
|---|---|---|
| `TRASH_PANDA_PROFILE` | `0` | Print where loading and prefetching spent their time (see "Reading the profile") |
| `TRASH_PANDA_LOOKAHEAD` | `2` | Blocks ahead whose weights are copied while one computes. `0` turns prefetching off |
| `TRASH_PANDA_PINNED_BUFFERS` | `2` | Page-locked host buffers that blocks are copied through |
| `TRASH_PANDA_BLOCKS_IN_FLIGHT` | `2` | The GPU may run at most this many blocks behind the copies |
| `TRASH_PANDA_ADAPTIVE_LOOKAHEAD` | `0` | After the first passes, try a deeper look-ahead (up to `TRASH_PANDA_LOOKAHEAD_MAX`) where the main thread waits, and a look-ahead of 1 where it does not, and keep the change only if the passes do not get slower (see "What we measured"). The staging that is no longer used is lent to the next model (see "Loading the next model while one runs"). Off by default |
| `TRASH_PANDA_LOOKAHEAD_MAX` | `4` | The deepest look-ahead that the setting above tries. The pinned buffers grow to match, and a depth is only taken if the driver reports free VRAM for the extra staged blocks |
| `TRASH_PANDA_RAM_IN_MEMORY` | `0` | Read the tensors of the blocks that wait in RAM into memory of their own while loading, instead of keeping them mapped from the checkpoint file |
| `TRASH_PANDA_RAM_READ_THREADS` | `1` | Threads that do that reading. Only used with `TRASH_PANDA_RAM_IN_MEMORY` |
| `TRASH_PANDA_MIN_FREE_RAM` | `auto` | RAM kept free for the system: `auto` (the three settings below) or a size such as `2GB` |
| `TRASH_PANDA_RAM_RESERVE_FRACTION` | `0.08` | With `auto`, the share of the machine's RAM kept free for the system. The result stays between the minimum and the maximum below |
| `TRASH_PANDA_RAM_RESERVE_MIN_GB` | `1.0` | With `auto`, the least RAM kept free, in GiB, however small the share above comes out |
| `TRASH_PANDA_RAM_RESERVE_MAX_GB` | `4.0` | With `auto`, the most RAM kept free, in GiB, however large the share above comes out |
| `TRASH_PANDA_RAM_RESERVE_UNKNOWN_GB` | `2.0` | The reserve when the machine's RAM cannot be read |
| `TRASH_PANDA_COMMIT_WARN_FRACTION` | `0.9` | Warn when the blocks in RAM need more than this share of the Windows commit limit |
| `TRASH_PANDA_HEADROOM_VRAM` | `1.5GB` | VRAM kept free for activations when none was measured |
| `TRASH_PANDA_STREAM_BUFFER_FACTOR` | `1.3` | Working area for a streamed block, as a multiple of its expanded size |
| `TRASH_PANDA_ACTIVATION_MARGIN_FRACTION` | `0.05` | Safety margin added to a measured activation size, as a share of the VRAM. It is never smaller than the minimum below |
| `TRASH_PANDA_ACTIVATION_MARGIN_MIN_GB` | `0.5` | The smallest safety margin added to a measured activation size, in GiB |
| `TRASH_PANDA_VRAM_FILL_WARN_FRACTION` | `0.9` | Warn when the planned GPU use is above this share of the free VRAM |
| `TRASH_PANDA_FIND_BLOCKS_MIN_FRACTION` | `0.01` | A group of layers must hold at least this share of the model's bytes to count as the blocks |
| `TRASH_PANDA_FIND_BLOCKS_MIN_BLOCKS` | `2` | The least number of blocks a group of layers must have to count as the blocks |

Two options are arguments of `stage()`, not environment variables: `overlap=True` (see "Running a stage") and `order=`
with `order_file_destination=`.
