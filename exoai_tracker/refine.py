from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _create_cv_tracker() -> Any | None:
    for factory in (
        lambda: cv2.legacy.TrackerMOSSE_create(),
        lambda: cv2.TrackerMIL_create(),
    ):
        try:
            return factory()
        except Exception:
            continue
    return None


class PatchLock:
    """
    Non-YOLO lock refinement between detections.
    Uses OpenCV correlation tracker when available, else template matching.
    """

    def __init__(self) -> None:
        self._tracker = None
        self._use_tracker = False
        self._template: np.ndarray | None = None
        self._screen_cx = 0.0
        self._screen_cy = 0.0
        self._box_w = 0.0
        self._box_h = 0.0
        self._region: dict[str, int] | None = None
        self._active = False
        self._failures = 0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._tracker = None
        self._use_tracker = False
        self._template = None
        self._region = None
        self._active = False
        self._failures = 0

    def nudge(self, dx: float, dy: float) -> None:
        if self._active:
            self._screen_cx += dx
            self._screen_cy += dy

    @staticmethod
    def _screen_to_frame(sx: float, sy: float, region: dict[str, int]) -> tuple[int, int]:
        return int(sx - region["left"]), int(sy - region["top"])

    @staticmethod
    def _frame_to_screen(fx: float, fy: float, region: dict[str, int]) -> tuple[float, float]:
        return fx + region["left"], fy + region["top"]

    def _clamp_roi(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int] | None:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def init(
        self,
        frame_rgb: np.ndarray,
        region: dict[str, int],
        screen_cx: float,
        screen_cy: float,
        box_w: float,
        box_h: float,
    ) -> bool:
        self.reset()
        if frame_rgb is None or frame_rgb.size == 0:
            return False

        fh, fw = frame_rgb.shape[:2]
        fx, fy = self._screen_to_frame(screen_cx, screen_cy, region)
        bw = max(12, int(box_w))
        bh = max(12, int(box_h))
        roi = self._clamp_roi(fx - bw // 2, fy - bh // 2, bw, bh, fw, fh)
        if roi is None:
            return False

        x, y, w, h = roi
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        patch = gray[y : y + h, x : x + w]
        if patch.size == 0:
            return False

        tracker = _create_cv_tracker()
        if tracker is not None:
            try:
                ok = tracker.init(frame_rgb, (x, y, w, h))
                if ok:
                    self._tracker = tracker
                    self._use_tracker = True
            except Exception:
                self._tracker = None
                self._use_tracker = False

        self._template = patch.astype(np.float32)
        self._screen_cx = screen_cx
        self._screen_cy = screen_cy
        self._box_w = float(box_w)
        self._box_h = float(box_h)
        self._region = dict(region)
        self._active = True
        self._failures = 0
        return True

    def _search_template(self, gray: np.ndarray) -> tuple[float, float] | None:
        if self._template is None or self._region is None:
            return None
        fh, fw = gray.shape[:2]
        fx, fy = self._screen_to_frame(self._screen_cx, self._screen_cy, self._region)
        th, tw = self._template.shape[:2]
        pad_x = max(24, int(tw * 1.6))
        pad_y = max(24, int(th * 1.6))
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(fw, fx + pad_x + tw)
        y2 = min(fh, fy + pad_y + th)
        if x2 - x1 <= tw or y2 - y1 <= th:
            return None

        search = gray[y1:y2, x1:x2]
        if search.size == 0:
            return None
        result = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, peak_loc = cv2.minMaxLoc(result)
        if peak < 0.38:
            return None
        cx = x1 + peak_loc[0] + tw * 0.5
        cy = y1 + peak_loc[1] + th * 0.5
        return self._frame_to_screen(cx, cy, self._region)

    def update(
        self,
        frame_rgb: np.ndarray,
        region: dict[str, int],
    ) -> tuple[float, float] | None:
        if not self._active or self._region is None:
            return None

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        found: tuple[float, float] | None = None

        if self._use_tracker and self._tracker is not None:
            try:
                ok, box = self._tracker.update(frame_rgb)
                if ok:
                    x, y, w, h = box
                    found = self._frame_to_screen(x + w * 0.5, y + h * 0.5, region)
            except Exception:
                self._use_tracker = False

        if found is None:
            found = self._search_template(gray)

        if found is None:
            self._failures += 1
            if self._failures > 8:
                self.reset()
            return None

        self._failures = 0
        self._screen_cx, self._screen_cy = found
        self._region = dict(region)
        return found
