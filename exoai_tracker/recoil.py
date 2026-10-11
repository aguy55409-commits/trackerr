"""Recoil compensation - pulls down while shooting."""
from __future__ import annotations

import time


class RecoilController:
    """
    Simple recoil compensation that pulls the mouse down at a steady rate.
    
    Supports:
    - Configurable vertical and horizontal pull
    - Ramp-up: gradually increases strength over time
    - Fractional accumulator for sub-pixel precision
    """
    
    def __init__(
        self,
        strength: float = 1.5,
        horizontal: float = 0.0,
        interval_ms: float = 10.0,
        ramp_ms: float = 100.0,
    ) -> None:
        self.strength = strength      # Pixels per tick (vertical)
        self.horizontal = horizontal  # Pixels per tick (horizontal, + = right)
        self.interval_ms = interval_ms
        self.ramp_ms = ramp_ms
        
        self._active = False
        self._start_time = 0.0
        self._last_tick = 0.0
        self._accum_x = 0.0
        self._accum_y = 0.0
    
    def configure(
        self,
        strength: float | None = None,
        horizontal: float | None = None,
        interval_ms: float | None = None,
        ramp_ms: float | None = None,
    ) -> None:
        if strength is not None:
            self.strength = strength
        if horizontal is not None:
            self.horizontal = horizontal
        if interval_ms is not None:
            self.interval_ms = max(1.0, interval_ms)
        if ramp_ms is not None:
            self.ramp_ms = max(0.0, ramp_ms)
    
    def start(self) -> None:
        """Called when the recoil bind is pressed."""
        if not self._active:
            self._active = True
            self._start_time = time.perf_counter()
            self._last_tick = self._start_time
            self._accum_x = 0.0
            self._accum_y = 0.0
    
    def stop(self) -> None:
        """Called when the recoil bind is released."""
        self._active = False
        self._accum_x = 0.0
        self._accum_y = 0.0
    
    def step(self) -> tuple[int, int]:
        """
        Returns (dx, dy) to move the mouse this tick.
        Call this at your move loop rate.
        Returns (0, 0) if not active or not time for a tick yet.
        """
        if not self._active or self.strength == 0.0:
            return 0, 0
        
        now = time.perf_counter()
        elapsed_since_tick = (now - self._last_tick) * 1000.0
        
        if elapsed_since_tick < self.interval_ms:
            return 0, 0
        
        self._last_tick = now
        
        # Calculate ramp multiplier (0 to 1)
        if self.ramp_ms > 0:
            elapsed_since_start = (now - self._start_time) * 1000.0
            ramp = min(1.0, elapsed_since_start / self.ramp_ms)
        else:
            ramp = 1.0
        
        # Apply strength with ramp
        pull_y = self.strength * ramp
        pull_x = self.horizontal * ramp
        
        # Accumulate fractional pixels
        self._accum_y += pull_y
        self._accum_x += pull_x
        
        # Extract whole pixels
        dy = int(self._accum_y)
        dx = int(self._accum_x)
        
        # Keep the fractional remainder
        self._accum_y -= dy
        self._accum_x -= dx
        
        return dx, dy
    
    @property
    def active(self) -> bool:
        return self._active
