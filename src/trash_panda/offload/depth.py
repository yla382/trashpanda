"""Choosing how many blocks ahead the prefetcher stages its copies (the look-ahead)."""

from __future__ import annotations

import statistics
from typing import List, Tuple


class DepthTuner:
    """Hill climbing on the look-ahead, from the passes the prefetcher has timed. It is fed one finished pass at a time
    (`update(took, waited)`: the seconds the pass took and the seconds the main thread spent waiting for staged blocks)
    and answers with the look-ahead to use from the next pass on.

    The wait alone does not say whether a deeper look-ahead would help: a pass that is limited by the copies waits a lot
    and gains nothing from more depth, and so does a pass that is limited by the compute (measured on stand-in models:
    the gain, up to about 8%, was only where copies and compute were about equal). So the depth is tried, and kept only
    if the passes get faster:

      - the first `warm` passes are left out (cold reads), and so is the pass after each change of depth
      - the passes are taken `window` at a time, and the median of a window is what is compared
      - if the main thread waited less than `min_wait` of the pass time, there is nothing to gain from more depth. The
        staging it does not need is VRAM that the next model can use, so the shallowest look-ahead (1) is tried: it is
        kept if the window is not slower than the first one by more than `max_slowdown`, if not the depth goes back.
        Either way it stops
      - otherwise the depth goes up by one; if the window is faster by at least `min_gain` it is kept and the climb
        goes on up to `max_depth`, if not the depth goes back and it stops

    Once it has stopped, the depth does not change again."""

    def __init__(self, depth: int, max_depth: int, *, warm: int = 2, window: int = 2,
                 min_gain: float = 0.03, min_wait: float = 0.10, max_slowdown: float = 0.02) -> None:
        self.depth     = depth
        self.start     = depth
        self._lowering = False                        # True while the shallowest look-ahead is being tried
        self.max_slowdown = max_slowdown
        self.max_depth = max(max_depth, depth)
        self.window    = max(window, 1)
        self.min_gain  = min_gain
        self.min_wait  = min_wait
        self.base      = None                         # the median pass time at the depth that is kept
        self.done      = False
        self._skip     = warm
        self._seen: List[Tuple[float, float]] = []

    def stop(self) -> None:
        """Keep the depth as it is from now on."""
        self.done = True

    def update(self, took: float, waited: float) -> int:
        """Take one finished pass; returns the look-ahead for the next one."""
        if self.done:
            return self.depth
        if self._skip > 0:
            self._skip -= 1
            return self.depth
        self._seen.append((took, waited))
        if len(self._seen) < self.window:
            return self.depth
        t = statistics.median(p[0] for p in self._seen)
        w = statistics.median(p[1] for p in self._seen)
        self._seen = []
        waiting = w >= self.min_wait * t
        if self.base is None:                         # the first window is the depth we started with
            self.base = t
            if not waiting and self.depth > 1:        # nothing to gain from more depth: does less cost anything?
                self._lowering = True
                self.depth = 1
                self._skip = 1
                return self.depth
        elif self._lowering:                          # the shallowest look-ahead: keep it if it is not slower
            if t > self.base * (1 + self.max_slowdown):
                self.depth = self.start
            self.done = True
            return self.depth
        elif t <= self.base * (1 - self.min_gain):    # the deeper look-ahead paid off: keep it
            self.base = t
        else:                                         # it did not: go back and stay
            self.depth -= 1
            self.done = True
            return self.depth
        if not waiting or self.depth >= self.max_depth:
            self.done = True
            return self.depth
        self.depth += 1
        self._skip = 1
        return self.depth
