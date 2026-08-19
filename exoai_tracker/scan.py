from __future__ import annotations

import math

from exoai_tracker.detector import Detection


class TargetSelector:
    """Closest in-FOV lock with persistence. Aim math lives on the tracker."""

    def __init__(self) -> None:
        self._lock: Detection | None = None
        self._aim_x = 0.0
        self._aim_y = 0.0
        self._has_aim = False
        self._misses = 0

    def reset(self) -> None:
        self._lock = None
        self._has_aim = False
        self._misses = 0

    @staticmethod
    def _aim_point(x: float, y: float, height: float, aim_offset_y: float) -> tuple[float, float]:
        t = max(0.0, min(1.0, aim_offset_y))
        return x, y + height * (0.5 - t)

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
        return [d for d in detections if d.kind != "ally" and not on_ally(d)]

    def _pick_detection(
        self,
        detections: list[Detection],
        center_x: int,
        center_y: int,
        aim_offset_y: float,
        fov_radius: int,
    ) -> Detection | None:
        fov_sq = float(fov_radius * fov_radius) * 1.18
        best: Detection | None = None
        best_score = float("inf")
        lock = self._lock
        pool = self._combat_pool(detections)

        for det in pool:
            offset = 0.5 if det.kind == "head" else aim_offset_y
            ax, ay = self._aim_point(det.x, det.y, det.height, offset)
            dist_sq = (ax - center_x) ** 2 + (ay - center_y) ** 2
            if dist_sq > fov_sq:
                continue
            score = dist_sq - det.confidence * 2400.0
            if det.kind == "head":
                score -= 4500.0
            if lock is not None:
                iou = self._iou(lock, det)
                if iou >= 0.06:
                    score -= 7000.0 * iou
                cdist = math.hypot(lock.x - det.x, lock.y - det.y)
                if cdist < max(28.0, lock.height * 1.8):
                    score -= 3200.0
            if score < best_score:
                best_score = score
                best = det
        return best

    def pick_target(
        self,
        detections: list[Detection],
        center_x: int,
        center_y: int,
        aim_offset_y: float,
        fov_radius: int = 180,
    ) -> Detection | None:
        if not detections:
            self.reset()
            return None

        best = self._pick_detection(detections, center_x, center_y, aim_offset_y, fov_radius)
        if best is None:
            self.reset()
            return None

        self._lock = best
        self._misses = 0
        offset = 0.5 if best.kind == "head" else aim_offset_y
        ax, ay = self._aim_point(best.x, best.y, best.height, offset)
        self._aim_x = ax
        self._aim_y = ay
        self._has_aim = True
        return Detection(
            x=ax,
            y=ay,
            confidence=best.confidence,
            width=best.width,
            height=best.height,
            cls=best.cls,
            kind=best.kind,
        )

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
        )

    @property
    def fresh(self) -> bool:
        return self._has_aim and self._misses == 0
