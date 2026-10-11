"""Auto-click while the crosshair is sitting on a target.

Kept as a state machine driven by the detect loop rather than a routine that
presses, sleeps and releases: that loop is the thing feeding the aim estimator,
and blocking it for the length of a click would stall detection and leave the
estimator reading a gap in the data as real target motion.

Nothing here talks to the mouse. update() reports what should happen and the
caller performs it, so the decision is testable without a Razer device.
"""
from __future__ import annotations

NONE, PRESS, RELEASE = 0, 1, 2


def covers(
    cx: float, cy: float, x: float, y: float, w: float, h: float, scale: float
) -> bool:
    """Whether the crosshair sits inside a box shrunk to `scale` of its size.

    Shrinking matters because a detection box reaches past the player it
    bounds, so the full box counts as a hit while the crosshair is still on
    empty space beside them.
    """
    return abs(cx - x) <= abs(w) * scale * 0.5 and abs(cy - y) <= abs(h) * scale * 0.5


class Trigger:
    def __init__(self) -> None:
        self.delay = 0.04
        self.cooldown = 0.14
        self.hold = 0.012
        self.scale = 0.55
        self.clicks = 0
        self.pressed = False
        self._enter: float | None = None
        self._down_at = 0.0
        # Far enough back that the cooldown gates the gap between clicks and
        # never the first one.
        self._up_at = float("-inf")

    def configure(
        self,
        *,
        delay_ms: float | None = None,
        cooldown_ms: float | None = None,
        hold_ms: float | None = None,
        box_scale: float | None = None,
    ) -> None:
        if delay_ms is not None:
            self.delay = max(0.0, float(delay_ms)) / 1000.0
        if cooldown_ms is not None:
            self.cooldown = max(0.0, float(cooldown_ms)) / 1000.0
        if hold_ms is not None:
            self.hold = max(0.001, float(hold_ms)) / 1000.0
        if box_scale is not None:
            self.scale = max(0.05, min(1.0, float(box_scale)))

    def update(self, on_target: bool, now: float) -> int:
        """Advance the state machine and return NONE, PRESS or RELEASE."""
        # Release on time whatever the target is doing. Holding the button
        # because the target is still under the crosshair would turn a click
        # into an unbroken burst, and losing the target mid-press must not
        # leave the button down.
        if self.pressed:
            if now - self._down_at >= self.hold:
                self.pressed = False
                self._up_at = now
                return RELEASE
            return NONE

        if not on_target:
            self._enter = None
            return NONE

        if self._enter is None:
            self._enter = now
            return NONE
        if now - self._enter < self.delay:
            return NONE
        if now - self._up_at < self.cooldown:
            return NONE

        self.pressed = True
        self._down_at = now
        self.clicks += 1
        return PRESS

    def disarm(self) -> bool:
        """Drop all state; returns True when a release is still owed.

        Called when the bind drops, the mode changes or the pipeline stops, any
        of which can happen mid-press and would otherwise strand the button.
        """
        owed = self.pressed
        self.pressed = False
        self._enter = None
        self._up_at = float("-inf")
        return owed
