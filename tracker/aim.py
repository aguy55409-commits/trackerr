from __future__ import annotations

import math

from tracker.engine import Det
from tracker.mouse import Mouse


class Aimer:
    """Closest target + hard pull toward it while bind is held."""

    def __init__(self) -> None:
        self.speed = 3.0
        self.fov = 200
        self.offset_y = 0.55
        self._fx = 0.0
        self._fy = 0.0
        self.last_dx = 0
        self.last_dy = 0
        self.mouse = Mouse()

    def pick(self, dets: list[Det], cx: int, cy: int) -> Det | None:
        if not dets:
            return None
        # Soft FOV prefer, but still lock nearest enemy if none inside FOV
        fov_sq = float(max(40, self.fov) ** 2)
        best_in: Det | None = None
        best_in_s = 1e18
        best_any: Det | None = None
        best_any_s = 1e18
        for d in dets:
            name = d.name.lower()
            if any(x in name for x in ("ally", "teammate", "friend", "squad")):
                continue
            ay = d.y - d.h * (self.offset_y - 0.5)
            dist_sq = (d.x - cx) ** 2 + (ay - cy) ** 2
            score = dist_sq - d.conf * 2500.0
            if "head" in name or "knock" in name:
                score -= 3500.0
            cand = Det(x=d.x, y=ay, w=d.w, h=d.h, conf=d.conf, cls=d.cls, name=d.name)
            if score < best_any_s:
                best_any_s = score
                best_any = cand
            if dist_sq <= fov_sq and score < best_in_s:
                best_in_s = score
                best_in = cand
        return best_in or best_any

    def step(self, target: Det | None, cx: int, cy: int, dt: float) -> None:
        self.last_dx = self.last_dy = 0
        if target is None:
            self._fx = self._fy = 0.0
            return

        dt = max(0.002, min(0.05, dt))
        ex = target.x - cx
        ey = target.y - cy
        dist = math.hypot(ex, ey)
        if dist < 0.75:
            return

        # Aggressive pull — speed 0.1..5 maps to ~25%..95%
        s = max(0.1, min(5.0, self.speed))
        pull = 0.30 + 0.65 * (s / 5.0)
        mx = ex * pull
        my = ey * pull

        cap = (8000.0 + s * 4000.0) * dt
        mag = math.hypot(mx, my)
        if mag > cap and mag > 0:
            mx *= cap / mag
            my *= cap / mag

        mx += self._fx
        my += self._fy
        ix = int(round(mx))
        iy = int(round(my))
        self._fx = mx - ix
        self._fy = my - iy

        if ix == 0 and iy == 0:
            if abs(ex) >= abs(ey):
                ix = 1 if ex > 0 else -1
            else:
                iy = 1 if ey > 0 else -1
            self._fx = self._fy = 0.0

        self.mouse.move(ix, iy)
        self.last_dx, self.last_dy = ix, iy
