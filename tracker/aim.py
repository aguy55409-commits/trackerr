from __future__ import annotations

import math

from tracker.engine import Det
from tracker.mouse import Mouse


class Aimer:
    """Closest-in-FOV lock + simple proportional mouse pull."""

    def __init__(self) -> None:
        self.speed = 3.0
        self.fov = 120
        self.offset_y = 0.55  # 0=feet … 1=head
        self._fx = 0.0
        self._fy = 0.0
        self.last_dx = 0
        self.last_dy = 0
        self.mouse = Mouse("razer")

    def pick(self, dets: list[Det], cx: int, cy: int) -> Det | None:
        if not dets:
            return None
        fov_sq = float(self.fov * self.fov)
        best: Det | None = None
        best_score = 1e18
        for d in dets:
            name = d.name.lower()
            if "ally" in name or "teammate" in name or "friend" in name:
                continue
            # Aim higher on box for heads / bodies
            ay = d.y - d.h * (self.offset_y - 0.5)
            dist_sq = (d.x - cx) ** 2 + (ay - cy) ** 2
            if dist_sq > fov_sq:
                continue
            score = dist_sq - d.conf * 2000.0
            if "head" in name:
                score -= 4000.0
            if score < best_score:
                best_score = score
                # store aim point in x/y
                best = Det(x=d.x, y=ay, w=d.w, h=d.h, conf=d.conf, cls=d.cls, name=d.name)
        return best

    def step(self, target: Det | None, cx: int, cy: int, dt: float) -> None:
        self.last_dx = self.last_dy = 0
        if target is None:
            self._fx = self._fy = 0.0
            return

        dt = max(0.001, min(0.02, dt))
        ex = target.x - cx
        ey = target.y - cy
        dist = math.hypot(ex, ey)
        if dist < 1.0:
            return

        pull = 0.25 + 0.7 * (self.speed / 5.0)
        pull = max(0.2, min(0.95, pull))
        mx = ex * pull
        my = ey * pull
        cap = (5000.0 + self.speed * 3000.0) * dt
        mag = math.hypot(mx, my)
        if mag > cap:
            mx *= cap / mag
            my *= cap / mag

        mx += self._fx
        my += self._fy
        ix = int(round(mx))
        iy = int(round(my))
        self._fx = mx - ix
        self._fy = my - iy

        if ix == 0 and iy == 0 and dist > 2.0:
            ix = 1 if abs(ex) >= abs(ey) and ex > 0 else (-1 if abs(ex) >= abs(ey) else 0)
            iy = 0 if ix != 0 else (1 if ey > 0 else -1)
            self._fx = self._fy = 0.0

        if ix or iy:
            self.mouse.move(ix, iy)
            self.last_dx, self.last_dy = ix, iy
