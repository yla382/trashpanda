"""Running the models of a pipeline one after another: one pinned pool for all of them, and each model loading in the
background while the one before it runs."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Optional, Tuple

from . import config
from .hooks import _PinnedPool, _empty_host_cache
from .stage import PendingStage, preload
from .stage import stage as open_stage

_NEXT = object()                                          # `then` was not given: the step after this one
_RESERVED = ("pool", "running", "after")                  # arguments that a Coordinator sets itself


class Step:
    """One model of a pipeline: a name, and the arguments of stage() for it.

    `Step("dit", build, path, overlap=True, skip_prefixes=("encoder.",))`. `build`, `path`, `key`, `key_map`,
    `skip_prefixes`, `order`, `order_file_destination`, `overlap`, `max_vram`, `max_ram` and the other arguments of plan()
    are those of stage(). `pool`, `running` and `after` are set by the Coordinator and cannot be given."""

    def __init__(self, name: str, build, path, **options) -> None:
        taken = [k for k in _RESERVED if k in options]
        if taken:
            raise TypeError(f"Step() does not take {', '.join(taken)}: the Coordinator sets it")
        self.name    = name
        self.build   = build
        self.path    = path
        self.options = options


class Coordinator:
    """Runs the models of a pipeline in order and moves memory from one to the next.

        co = Coordinator(
            Step("text", build_text, text_path),
            Step("dit",  build_dit,  dit_path, overlap=True),
            Step("vae",  build_vae,  vae_path, skip_prefixes=("encoder.",)),
        )
        with co:
            with co.stage("text") as (text, p, handle):
                emb = encode(text, prompt)
            with co.stage("dit") as (dit, p, handle):
                latents = sample(dit, emb)
            with co.stage("vae") as (vae, p, handle):
                image = decode(vae, latents)

    `co.stage(name)` gives `(model, plan, handle)` like stage() does, and the model is released when the block ends.
    - One pinned pool lives from `with co:` to the end of that block and is used by every model, instead of each model
      pinning its own buffers again. Its buffers grow to the largest block of any model and do not shrink. Each plan
      counts the buffers the pool has already, so only what they still have to grow by is kept back from the RAM tier.
      A model that enlarged the pool with more buffers (the look-ahead tuner does) has them dropped when its stage ends,
      so the pool goes back to the configured number of buffers.
    - While a model runs, the next step is prepared: planned for the memory the running model will give back (see
      preload()). `then` names that step: a step's name, None when there is none, and by default the step after the one
      that runs (nothing follows the last). Two flags of `co.stage()` say what is prepared, both True by default:
      `ram_preload` reads its RAM tier into memory in the background (False leaves it mapped from the file; reading it
      is slow on a SATA drive, and only pays off if the running stage is long enough to hide it), and `vram_preload`
      loads its GPU blocks early, into VRAM that is free and, if the running model's `handle.final_pass()` is used,
      into the memory the final pass frees. With both False the next step loads like stage() does.
    - A step can run again (a loop over images): `then="text"` in the last step's block would preload it.
    - Asking for a step other than the preloaded one gives the preloaded model up (its reads finish first) and loads the
      asked one as stage() does.
    - Leaving `with co:` gives up a preloaded model that was never asked for, drops the pool and gives its pinned memory
      back.

    A stage cannot be started inside another stage of the same Coordinator: a model is released before the next one is
    entered. `co.stage()` outside `with co:` raises a RuntimeError."""

    def __init__(self, *steps: Step) -> None:
        if not steps:
            raise ValueError("a Coordinator needs at least one Step")
        names = [s.name for s in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"the names of the steps must be different, got {names}")
        self.steps = list(steps)
        self.pool: Optional[_PinnedPool] = None
        self._pending: Optional[Tuple[str, PendingStage]] = None        # (name of the step, the model loading for it)
        self._busy = False

    def __enter__(self) -> "Coordinator":
        if self.pool is not None:
            raise RuntimeError("this Coordinator is already in a `with` block")
        self.pool = _PinnedPool()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._give_up_pending()
        finally:
            self.pool = None
            gc.collect()
            _empty_host_cache()
        return False

    def _step(self, name: str) -> Step:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"no step named {name!r}; the steps are {[s.name for s in self.steps]}")

    def _following(self, step: Step, then) -> Optional[Step]:
        """The step to preload while `step` runs."""
        if then is None:
            return None
        if then is not _NEXT:
            return self._step(then)
        i = self.steps.index(step)
        return self.steps[i + 1] if i + 1 < len(self.steps) else None

    def _give_up_pending(self) -> None:
        held, self._pending = self._pending, None
        if held is not None:
            held[1].cancel()

    def _take_pending(self, name: str) -> Optional[PendingStage]:
        """The preloaded model for step `name`, or None (a preloaded model for another step is given up)."""
        held, self._pending = self._pending, None
        if held is None:
            return None
        if held[0] == name:
            return held[1]
        held[1].cancel()
        return None

    @contextmanager
    def stage(self, name: str, *, then=_NEXT, ram_preload: bool = True, vram_preload: bool = True):
        """Run step `name`: the block gets `(model, plan, handle)`, and the model is released when it ends.

        `then` names the step to prepare while this one runs: by default the next step in the order of the Coordinator;
        None when there is none. What is prepared is set by the two flags, which are about that next step:
        `ram_preload` reads its RAM tier into memory in the background (False: it stays mapped from the file, and is
        read when its blocks first run); `vram_preload` loads its GPU blocks early, into VRAM that is free and into the
        memory this stage frees in its final pass (False: they load when the step starts). With both False nothing is
        prepared and the next step loads like stage() does."""
        if self.pool is None:
            raise RuntimeError("use the Coordinator in a `with` block: `with co: ... co.stage(name)`")
        if self._busy:
            raise RuntimeError("a stage of this Coordinator is running; leave its block before starting another")
        step = self._step(name)
        following = self._following(step, then) if (ram_preload or vram_preload) else None
        pending = self._take_pending(name)
        self._busy = True
        try:
            entered = pending if pending is not None else open_stage(step.build, step.path, pool=self.pool, **step.options)
            with entered as (model, p, handle):
                if following is not None:
                    self._pending = (following.name, preload(
                        following.build, following.path, running=p, pool=self.pool, ram_preload=ram_preload,
                        after=handle.subscribe() if vram_preload else None, **following.options))
                yield model, p, handle
        finally:
            self._busy = False
            self._trim_pool()

    def _trim_pool(self) -> None:
        """Give back the buffers that a model enlarged the pool with (the look-ahead tuner adds some): the next model
        cycles through all the buffers there are, so extra ones would be pinned for nothing. The pool keeps the
        configured number of buffers, at the size they grew to."""
        if self.pool is not None and self.pool.trim(config.current().pinned_buffers):
            _empty_host_cache()
