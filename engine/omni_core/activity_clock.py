"""Per-owner active time; real audit dates and timer tools remain wall-clock.

Decay/idle cadence markers and cold working-page recency use these coordinates.
An unloaded interval contributes nothing, including across process restart.
No neuron or provenance timestamp is bulk shifted to simulate suspension.
"""
from __future__ import annotations

import math
import time


class ActivityClock:
    FORMAT = "omni-native-activity-clock"

    def __init__(self, elapsed=0.0, *, paused=False, clock=time.monotonic):
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or type(paused) is not bool:
            raise ValueError("native activity clock coordinates are invalid")
        self._elapsed, self._paused, self._clock = float(elapsed), paused, clock
        self._anchor = self._tick()

    def _tick(self):
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("native activity clock needs a finite monotonic reading")
        return float(value)

    @property
    def paused(self):
        return self._paused

    def now(self):
        return self._elapsed if self._paused else self._elapsed + max(0.0, self._tick() - self._anchor)

    def pause(self):
        if not self._paused:
            self._elapsed = self.now()
            self._paused = True
        return self.snapshot()

    def resume(self):
        if self._paused:
            self._anchor = self._tick()
            self._paused = False

    def snapshot(self):
        return {"format": self.FORMAT, "version": 1, "elapsed": self.now(), "paused": self._paused}

    @classmethod
    def from_state(cls, state, *, legacy_markers=(), clock=time.monotonic):
        if state is None:
            # Ordinary old learned checkpoints can start a new active-time
            # coordinate without charging an unknown offline wall interval.
            # An exact suspended execution separately requires its sealed clock.
            values = [float(value) for value in legacy_markers if type(value) in (int, float) and math.isfinite(value) and value >= 0]
            return cls(max(values, default=0.0), paused=True, clock=clock)
        if (type(state) is not dict or set(state) != {"format", "version", "elapsed", "paused"}
                or state["format"] != cls.FORMAT or type(state["version"]) is not int or state["version"] != 1):
            raise ValueError("native activity clock format is invalid")
        return cls(state["elapsed"], paused=state["paused"], clock=clock)
