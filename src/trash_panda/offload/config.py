"""Default values of the tuning settings, in one place.

`DEFAULTS` holds the values the library ships with. `current()` returns them with any environment overrides applied
(`TRASH_PANDA_<NAME>`, for example `TRASH_PANDA_LOOKAHEAD=0`). It is read when a function runs, not when the library
is imported, so a test can change a default with `monkeypatch.setenv` or by replacing `config.DEFAULTS`.

Precedence: an argument passed to a function, then the environment, then `DEFAULTS`.

Only settings that are worth tuning live here. Format facts and invariants (the scale suffix, the dtype tables, the
buffer alignment, the single worker thread) stay next to the code that depends on them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

ENV_PREFIX = "TRASH_PANDA_"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Defaults:
    # prefetching
    lookahead: int = 2                          # blocks ahead whose weights are copied while one computes (0 = off)
    pinned_buffers: int = 2                     # page-locked host buffers that blocks are copied through
    blocks_in_flight: int = 2                   # the GPU may run at most this many blocks behind the copies

    # VRAM
    headroom_vram: str = "1.5GB"                # VRAM kept free for activations when none was measured
    stream_buffer_factor: float = 1.3           # working area for a streamed block, as a multiple of its expanded size
    activation_margin_fraction: float = 0.05    # margin added to a measured activation size: this share of VRAM ...
    activation_margin_min_gb: float = 0.5       # ... but at least this many GiB

    # RAM
    min_free_ram: str = "auto"                  # RAM kept free for the system: "auto" (below) or a size such as "2GB"
    ram_reserve_fraction: float = 0.08          # "auto" keeps this share of the machine's RAM free ...
    ram_reserve_min_gb: float = 1.0             # ... but at least this many GiB
    ram_reserve_max_gb: float = 4.0             # ... and at most this many
    ram_reserve_unknown_gb: float = 2.0         # reserve when the machine's RAM cannot be read
    commit_warn_fraction: float = 0.9           # warn when streamed blocks need more than this share of the commit limit

    # finding the blocks of a model
    find_blocks_min_fraction: float = 0.01      # a group must hold at least this share of the model's bytes
    find_blocks_min_blocks: int = 2             # ... and have at least this many blocks

    # diagnostics
    profile: bool = False                       # print where the prefetching spent its time


DEFAULTS = Defaults()


def _parse(name: str, default, text: str):
    """The environment value `text` as the type of `default`."""
    var = ENV_PREFIX + name.upper()
    try:
        if isinstance(default, bool):               # bool before int: bool is a subclass of int
            lowered = text.strip().lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError("expected 1/0, true/false, yes/no or on/off")
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
        return text.strip()
    except ValueError as err:
        raise ValueError(f"{var}={text!r} is not valid: {err}") from None


def current() -> Defaults:
    """`DEFAULTS` with the `TRASH_PANDA_<NAME>` environment overrides applied."""
    base = DEFAULTS
    changes = {}
    for f in fields(base):
        raw = os.environ.get(ENV_PREFIX + f.name.upper())
        if raw:
            changes[f.name] = _parse(f.name, getattr(base, f.name), raw)
    return replace(base, **changes) if changes else base
