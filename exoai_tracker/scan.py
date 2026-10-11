"""Single-target lock: Kalman + IoU, ByteTrack-style two-stage match.

Ultralytics persist IDs assume a fixed camera. A 320px crosshair crop jumps
when the mouse moves, so those IDs die. Lock in screen space instead.
"""
from __future__ import annotations

import time

from exoai_tracker.detector import Detection


class TargetSelector:
    def __init__(self, miss_limit: int = 30) -> None:
        self._lock: Detection | None = None
        self._aim_x = 0.0
        self._aim_y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._has_aim = False
        self._misses = 0
        self._miss_limit = miss_limit
        self._last_t = 0.0
        self._lock_id = -1

    def reset(self) -> None:
        self._lock = None
        self._has_aim = False
        self._misses = 0
        self._vx = 0.0
        self._vy = 0.0
        self._lock_id = -1

    @staticmethod
    def _aim_point(det: Detection, aim_offset_y: float) -> tuple[float, float]:
        # t is fraction from the box bottom: 0.92 head, 0.62 chest, 0.42 belly.
        t = 0.92 if det.kind == "head" else max(0.0, min(1.0, aim_offset_y))
        return det.x, det.y + det.height * (0.5 - t)

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax1, ay1 = a.x - a.width * 0.5, a.y - a.height * 0.5
        ax2, ay2 = a.x + a.width * 0.5, a.y + a.height * 0.5
        bx1, by1 = b.x - b.width * 0.5, b.y - b.height * 0.5
        bx2, by2 = b.x + b.width * 0.5, b.y + b.height * 0.5
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _center_inside(inner: Detection, outer: Detection, pad: float = 0.12) -> bool:
        return (
            abs(inner.x - outer.x) <= outer.width * (0.5 + pad)
            and abs(inner.y - outer.y) <= outer.height * (0.5 + pad)
        )

    def _combat_pool(self, detections: list[Detection]) -> list[Detection]:
        allies = [d for d in detections if d.kind == "ally"]
        enemies = [d for d in detections if d.kind == "enemy"]
        heads = [d for d in detections if d.kind == "head"]
        others = [d for d in detections if d.kind not in {"ally", "enemy", "head"}]

        def on_ally(det: Detection) -> bool:
            return any(self._center_inside(det, ally) or self._iou(det, ally) >= 0.08 for ally in allies)

        usable_heads = [head for head in heads if not on_ally(head)]
        if enemies and usable_heads:
            matched = [
                head
                for head in usable_heads
                if any(self._center_inside(head, enemy) or self._iou(head, enemy) >= 0.04 for enemy in enemies)
            ]
            if matched:
                return matched
        if usable_heads:
            return usable_heads
        if enemies:
            return enemies
        usable_other = [d for d in others if not on_ally(d)]
        if usable_other:
            return usable_other
        return [d for d in detections if d.kind != "ally" and not on_ally(d)]

    def _pick_closest(
        self,
        detections: list[Detection],
        center_x: float,
        center_y: float,
        aim_offset_y: float,
        fov_radius: int,
    ) -> Detection | None:
        fov_sq = float(max(fov_radius, 40) ** 2) * 4.0
        best: Detection | None = None
        best_score = float("inf")
        for det in detections:
            ax, ay = self._aim_point(det, aim_offset_y)
            dist_sq = (ax - center_x) ** 2 + (ay - center_y) ** 2
            if dist_sq > fov_sq:
                continue
            score = dist_sq - det.confidence * 2400.0
            if det.kind == "head":
                score -= 4500.0
            if score < best_score:
                best_score = score
                best = det
        return best

    def _match(self, pool: list[Detection]) -> Detection | None:
        lock = self._lock
        if lock is None:
            return None
        pred = Detection(
            x=lock.x + self._vx,
            y=lock.y + self._vy,
            confidence=lock.confidence,
            width=lock.width,
            height=lock.height,
            cls=lock.cls,
            kind=lock.kind,
        )
        gate = max(90.0, lock.height * 2.4, lock.width * 2.4)

        def score_of(det: Detection) -> float | None:
            if lock.kind in {"head", "enemy"} and det.kind == "ally":
                return None
            iou = self._iou(pred, det)
            dx = det.x - pred.x
            dy = det.y - pred.y
            dist = (dx * dx + dy * dy) ** 0.5
            if iou < 0.02 and dist > gate:
                return None
            return dist - iou * 180.0

        high = [d for d in pool if d.confidence >= 0.25]
        low = [d for d in pool if d.confidence < 0.25]
        best: Detection | None = None
        best_score = float("inf")
        for group in (high, low):
            for det in group:
                score = score_of(det)
                if score is None:
                    continue
                if score < best_score:
                    best_score = score
                    best = det
            if best is not None:
                return best
        return None

    def _commit(self, det: Detection, aim_offset_y: float, now: float) -> Detection:
        dt = max(0.004, now - self._last_t) if self._last_t else 0.016
        if self._lock is not None:
            inst_vx = (det.x - self._lock.x) / dt
            inst_vy = (det.y - self._lock.y) / dt
            self._vx = 0.65 * self._vx + 0.35 * inst_vx * dt
            self._vy = 0.65 * self._vy + 0.35 * inst_vy * dt
            self._vx = max(-40.0, min(40.0, self._vx))
            self._vy = max(-40.0, min(40.0, self._vy))
        self._lock = det
        self._lock_id = det.track_id
        self._misses = 0
        self._last_t = now
        self._aim_x, self._aim_y = self._aim_point(det, aim_offset_y)
        self._has_aim = True
        return Detection(
            x=self._aim_x,
            y=self._aim_y,
            confidence=det.confidence,
            width=det.width,
            height=det.height,
            cls=det.cls,
            kind=det.kind,
            track_id=det.track_id,
        )

    def _coast(self, aim_offset_y: float) -> Detection | None:
        self._misses += 1
        if self._misses > self._miss_limit or self._lock is None:
            self.reset()
            return None
        self._lock = Detection(
            x=self._lock.x + self._vx,
            y=self._lock.y + self._vy,
            confidence=self._lock.confidence,
            width=self._lock.width,
            height=self._lock.height,
            cls=self._lock.cls,
            kind=self._lock.kind,
            track_id=self._lock.track_id,
        )
        self._aim_x, self._aim_y = self._aim_point(self._lock, aim_offset_y)
        self._has_aim = True
        return self.current_target()

    def pick_target(
        self,
        detections: list[Detection],
        center_x: int | float,
        center_y: int | float,
        aim_offset_y: float,
        fov_radius: int = 180,
    ) -> Detection | None:
        now = time.perf_counter()
        pool = self._combat_pool(detections) if detections else []

        if not pool:
            if self._has_aim:
                return self._coast(aim_offset_y)
            return None
        if self._has_aim and self._lock is not None:
            match = self._match(pool)
            if match is not None:
                return self._commit(match, aim_offset_y, now)
        best = self._pick_closest(pool, float(center_x), float(center_y), aim_offset_y, fov_radius)
        if best is None:
            if self._has_aim:
                return self._coast(aim_offset_y)
            return None
        return self._commit(best, aim_offset_y, now)

    def current_target(self) -> Detection | None:
        if not self._has_aim or self._lock is None:
            return None
        return Detection(
            x=self._aim_x,
            y=self._aim_y,
            confidence=self._lock.confidence,
            width=self._lock.width,
            height=self._lock.height,
            cls=self._lock.cls,
            kind=self._lock.kind,
            track_id=self._lock.track_id,
        )

    @property
    def lock_id(self) -> int:
        return self._lock_id

    @property
    def fresh(self) -> bool:
        return self._has_aim and self._misses <= 3
