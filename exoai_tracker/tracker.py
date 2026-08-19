from __future__ import annotations

import math
from typing import Literal

from exoai_tracker.binds import is_bind_held
from exoai_tracker.mouse import MouseController

MouseMethod = Literal["razer", "win32", "sendinput", "arduino", "custom"]


class TargetTracker:
    """Smooth error-space aim with adaptive gain."""

    def __init__(self, mouse_method: MouseMethod = "razer") -> None:
        self._mouse_method: MouseMethod = mouse_method
        self._bind_key = "right"
        self._track_speed = 3.5
        self._lead = 1.0
        self._smooth = 0.25
        self._fx = 0.0
        self._fy = 0.0
        self._ex = 0.0
        self._ey = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._has = False
        self._stamp = 0.0
        self._last_ix = 0
        self._last_iy = 0
        self._conf = 0.0
        self.mouse: MouseController | None = None

    def configure(
        self,
        track_speed: float,
        loop_interval: float,
        screen_aspect: float | None = None,
        lead: float = 1.0,
        smooth: float = 0.25,
    ) -> None:
        self._track_speed = max(0.01, min(5.0, track_speed))
        self._lead = max(0.0, min(1.5, lead))
        self._smooth = max(0.05, min(0.6, smooth))

    def set_bind_key(self, bind_key: str) -> None:
        self._bind_key = bind_key

    def set_mouse_method(self, mouse_method: MouseMethod) -> None:
        if self._mouse_method != mouse_method:
            self._mouse_method = mouse_method
            self.mouse = None

    def reset_aim(self) -> None:
        self._fx = self._fy = 0.0
        self._ex = self._ey = 0.0
        self._vx = self._vy = 0.0
        self._has = False
        self._stamp = 0.0
        self._last_ix = self._last_iy = 0
        self._conf = 0.0

    def last_move(self) -> tuple[int, int]:
        return self._last_ix, self._last_iy

    def mouse_failures(self) -> int:
        return 0

    def ensure_mouse(self) -> MouseController:
        if self.mouse is None:
            self.mouse = MouseController(self._mouse_method)
        return self.mouse

    def bind_held(self) -> bool:
        return is_bind_held(self._bind_key)

    def has_lock(self) -> bool:
        return self._has

    def confidence(self) -> float:
        return self._conf

    def remaining(self) -> tuple[float, float]:
        return self._ex, self._ey

    def observe(self, target_x: float, target_y: float, center_x: int, center_y: int, now: float, conf: float) -> None:
        ex = target_x - center_x
        ey = target_y - center_y
        if self._has and self._stamp > 0.0:
            dt = max(0.002, min(0.05, now - self._stamp))
            meas_vx = (ex - self._ex) / dt
            meas_vy = (ey - self._ey) / dt
            a = 1.0 - self._smooth
            self._vx = self._vx * self._smooth + meas_vx * a
            self._vy = self._vy * self._smooth + meas_vy * a
            speed = math.hypot(self._vx, self._vy)
            max_vel = 3000.0
            if speed > max_vel:
                scale = max_vel / speed
                self._vx *= scale
                self._vy *= scale
        else:
            self._vx = 0.0
            self._vy = 0.0
        self._ex = ex
        self._ey = ey
        self._stamp = now
        self._has = True
        self._conf = conf

    def tick(self, dt: float, infer_s: float) -> tuple[int, int]:
        if not self._has:
            self._last_ix = self._last_iy = 0
            return 0, 0

        dt = max(0.0003, min(0.016, dt))

        pred_time = min(infer_s * self._lead * 0.85, 0.06)
        ax = self._ex + self._vx * pred_time
        ay = self._ey + self._vy * pred_time
        dist = math.hypot(ax, ay)

        if dist < 0.5:
            self._fx = self._fy = 0.0
            self._last_ix = self._last_iy = 0
            return 0, 0

        s = self._track_speed
        # Map 0.01..5.0 into a usable pull strength (old formula starved low speeds)
        strength = 0.18 + (s / 5.0) * 0.72
        strength *= min(1.0, dt * 180.0)
        strength = max(0.20, min(0.95, strength))

        if dist > 80.0:
            strength = min(0.98, strength * 1.25)
        elif dist < 12.0:
            strength *= 0.55 + 0.45 * (dist / 12.0)

        mx = ax * strength
        my = ay * strength

        max_step = (6000.0 + s * 3500.0) * dt
        mag = math.hypot(mx, my)
        if mag > max_step:
            scale = max_step / mag
            mx *= scale
            my *= scale

        mx += self._fx
        my += self._fy
        ix = int(round(mx))
        iy = int(round(my))
        self._fx = mx - ix
        self._fy = my - iy

        if ix == 0 and iy == 0 and dist > 2.0:
            if abs(ax) >= abs(ay):
                ix = 1 if ax > 0 else -1
            else:
                iy = 1 if ay > 0 else -1
            self._fx = self._fy = 0.0

        if ix == 0 and iy == 0:
            self._last_ix = self._last_iy = 0
            return 0, 0

        self._ex -= ix
        self._ey -= iy
        self.ensure_mouse().move_relative(ix, iy)
        self._last_ix = ix
        self._last_iy = iy
        return ix, iy
