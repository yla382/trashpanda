# Reading the profile

What the output of `TRASH_PANDA_PROFILE=1` means.

## Reading the profile

With `TRASH_PANDA_PROFILE=1`, each stage prints a report when its hooks are removed:

* `stream_load took X s (...)`: the time of the load, by tier. With `overlap=True` the blocks that are loaded while the
  model runs are not counted.
* `read N GB of streamed-block tensors into memory in X s (Y GB/s, T threads)`: only with `TRASH_PANDA_RAM_IN_MEMORY`.
* `per pass`: for each pass through the blocks, its time, the time the main thread waited for staged blocks, and the GB
  and GB/s of RAM copies and disk reads done meanwhile. A first pass much slower than the others, with a low GB/s, is
  cold reads.
* `first pass, main thread time by block`: the six slowest blocks of the first pass, with their time split into waiting
  for the copy (`staged`), waiting for the GPU (`gpu`) and the rest (launching work, one-time warm-up).
* `loader thread`: with `overlap=True`, how long the resident blocks took to load and how long the main thread waited
  for them.
* `final pass freed N blocks over X s (...)`: after `handle.final_pass()`, the blocks and GB freed in each tier and the
  time from the first block freed to the last. With a subscriber (`handle.subscribe()`) it also gives the most
  `Release` objects the queue held at once and how many were unread when the profile was printed.
* `blocks that never ran`: named blocks that were loaded but that no call used, with their size. If the model never uses
  them, leave them out with `skip_prefixes`. It is printed once some block has run.
* `memory while releasing`: at five points of `release()`, the RAM the machine has free, the working set and the private
  bytes of the process, and the CPU tensors of 1 MB or more that are still alive. If more than 0.1 GB stays alive it
  also names what holds the biggest. Read the working set, not the private bytes: the private bytes include GPU memory
  that Windows charges to the process. The profile scans the live objects at these five points, which costs a little
  time.
