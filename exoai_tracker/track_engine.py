"""Single detect/lock/move loop: persist ID lock, independent timers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from exoai_tracker.detector import Detection
from exoai_tracker.mouse_backends import MouseBackend
from exoai_tracker.scan import TargetSelector


@dataclass
class TrackConfig:
    speed: float = 0.5
    fov: int = 300
    aim_height: float = 0.35
    max_move: int = 32767
    overshoot_ratio: float = 0.96
    deadzone_px: float = 1.0
    detect_interval_ms: float = 0.0
    move_interval_ms: float = 0.0
    idle_sleep_ms: float = 0.0
    miss_limit: int = 36
    release_debounce_ms: float = 250.0
    tracker: str = ""


@dataclass
class TrackStats:
    moves: int = 0
    detections: int = 0
    last_dx: int = 0
    last_dy: int = 0


def compute_move(
    err_x: float,
    err_y: float,
    *,
    speed: float,
    max_move: int,
    overshoot_ratio: float,
    deadzone_px: float,
    carry_x: float = 0.0,
    carry_y: float = 0.0,
) -> tuple[int, int, float, float]:
    """Return integer dx/dy plus updated sub-pixel carry."""
    if abs(err_x) < deadzone_px and abs(err_y) < deadzone_px:
        return 0, 0, carry_x, carry_y

    raw_x = err_x * speed * overshoot_ratio + carry_x
    raw_y = err_y * speed * overshoot_ratio + carry_y
    if max_move > 0:
        if abs(raw_x) > max_move:
            raw_x = max_move if raw_x > 0 else -max_move
        if abs(raw_y) > max_move:
            raw_y = max_move if raw_y > 0 else -max_move

    dx = int(round(raw_x))
    dy = int(round(raw_y))
    return dx, dy, raw_x - dx, raw_y - dy


class TrackEngine:
    """Detect on one timer, hold a track_id, move on another."""

    def __init__(
        self,
        detector: Any,
        mouse: MouseBackend | None,
        config: TrackConfig,
        is_held: Callable[[], bool],
        get_fov: Callable[[], int] | None = None,
    ) -> None:
        self.detector = detector
        self.mouse = mouse
        self.config = config
        self.is_held = is_held
        self.get_fov = get_fov
        self.scanner = TargetSelector(miss_limit=config.miss_limit)
        self.stats = TrackStats()
        self.detections: list[Detection] = []
        self.debug_frame = None
        self._cx = float(detector.center_x)
        self._cy = float(detector.center_y)
        self._err_x = 0.0
        self._err_y = 0.0
        self._has_target = False
        self._conf = 0.0
        self._carry_x = 0.0
        self._carry_y = 0.0
        self._last_detect = 0.0
        self._last_move = 0.0
        self._held = False
        self._release_since = 0.0
        self.infer_count = 0

    def _detect_interval(self) -> float:
        return max(0.0, self.config.detect_interval_ms / 1000.0)

    def _move_interval(self) -> float:
        return max(0.0, self.config.move_interval_ms / 1000.0)

    def _idle_interval(self) -> float:
        return max(0.0, self.config.idle_sleep_ms / 1000.0)

    def _fov(self) -> int:
        if self.get_fov is not None:
            return int(self.get_fov())
        return int(self.config.fov)

    def _normalize(self, raw) -> list[Detection]:
        if not raw:
            return []
        first = raw[0]
        if isinstance(first, Detection):
            return raw
        return [
            Detection(
                x=t.x,
                y=t.y,
                confidence=t.confidence,
                width=t.width,
                height=t.height,
                cls=0,
                kind="enemy",
            )
            for t in raw
        ]

    def _release(self) -> None:
        # Clear the lock only. Do not rebuild Ultralytics trackers — that
        # hitch is what made mouse-flicks take ~3s to recover.
        self.scanner.reset()
        self._has_target = False
        self._conf = 0.0
        self._carry_x = 0.0
        self._carry_y = 0.0
        self.stats.last_dx = 0
        self.stats.last_dy = 0

    def has_lock(self) -> bool:
        return self._has_target

    def confidence(self) -> float:
        return self._conf

    def remaining(self) -> tuple[float, float]:
        return self._err_x, self._err_y

    def last_move(self) -> tuple[int, int]:
        return self.stats.last_dx, self.stats.last_dy

    def lock_id(self) -> int:
        return self.scanner.lock_id

    def _closest_fallback(self, detections: list[Detection], aim_height: float) -> Detection | None:
        usable = [d for d in detections if d.kind != "ally"]
        if not usable:
            return None
        best = min(
            usable,
            key=lambda d: (d.x - self._cx) ** 2 + (d.y - self._cy) ** 2,
        )
        t = 0.92 if best.kind == "head" else aim_height
        ax = best.x
        ay = best.y + best.height * (0.5 - t)
        return Detection(
            x=ax,
            y=ay,
            confidence=best.confidence,
            width=best.width,
            height=best.height,
            cls=best.cls,
            kind=best.kind,
            track_id=best.track_id,
        )

    def tick(self) -> bool:
        """Legacy single-thread tick (detect + move). Prefer detect_step/move_step."""
        if not self.detect_step():
            return True
        self.move_step()
        return True

    def detect_step(self) -> float:
        """Run one detection pass. Returns seconds spent."""
        now = time.perf_counter()
        if not self._update_hold_state(now):
            return 0.0

        self._cx = float(self.detector.center_x)
        self._cy = float(self.detector.center_y)
        t0 = time.perf_counter()
        fov = self._fov()
        try:
            raw, frame, _ = self.detector.detect(fov, wait_for_new=False)
        except Exception:
            raw, frame = [], getattr(self, "debug_frame", None)
        self.detections = self._normalize(raw)
        self.debug_frame = frame
        self._last_detect = now
        self.infer_count += 1
        target = self.scanner.pick_target(
            self.detections,
            self._cx,
            self._cy,
            self.config.aim_height,
            fov_radius=fov,
        )
        if target is None:
            target = self._closest_fallback(self.detections, self.config.aim_height)
        if target is None:
            self._has_target = False
        else:
            self._has_target = True
            self._err_x = target.x - self._cx
            self._err_y = target.y - self._cy
            self._conf = target.confidence
            self.stats.detections += 1
        return time.perf_counter() - t0

    def move_step(self) -> None:
        """Fire one mouse move from current error (no detect)."""
        if not self.is_held() or not self._has_target:
            return
        move_every = self._move_interval()
        now = time.perf_counter()
        if move_every > 0.0 and now - self._last_move < move_every:
            return

        dx, dy, self._carry_x, self._carry_y = compute_move(
            self._err_x,
            self._err_y,
            speed=self.config.speed,
            max_move=self.config.max_move,
            overshoot_ratio=self.config.overshoot_ratio,
            deadzone_px=self.config.deadzone_px,
            carry_x=self._carry_x,
            carry_y=self._carry_y,
        )
        self._last_move = now
        self.stats.last_dx = dx
        self.stats.last_dy = dy
        if self.mouse is not None and (dx != 0 or dy != 0):
            self.mouse.move(dx, dy)
            self._err_x -= dx
            self._err_y -= dy
            self.stats.moves += 1

    def _update_hold_state(self, now: float) -> bool:
        held = self.is_held()
        if not held:
            if self._held:
                if self._release_since <= 0.0:
                    self._release_since = now
                debounce = max(0.0, self.config.release_debounce_ms) / 1000.0
                if now - self._release_since < debounce:
                    held = True
            if not held:
                if self._held:
                    self._release()
                    pause = getattr(self.detector, "set_tracking_active", None)
                    if callable(pause):
                        pause(False)
                self._held = False
                self._release_since = 0.0
                idle = self._idle_interval()
                if idle > 0.0:
                    time.sleep(idle)
                return False
        else:
            self._release_since = 0.0
        if not self._held:
            resume = getattr(self.detector, "set_tracking_active", None)
            if callable(resume):
                resume(True)
        self._held = True
        return True

    def run(self, *, status_every: int = 50) -> None:
        try:
            while self.tick():
                if self.stats.moves and self.stats.moves % status_every == 0:
                    print(
                        f"Moves: {self.stats.moves} | "
                        f"Detects: {self.stats.detections} | "
                        f"Last: ({self.stats.last_dx}, {self.stats.last_dy}) | "
                        f"id={self.scanner.lock_id} | "
                        f"{self.mouse.label() if self.mouse is not None else 'none'}"
                    )
        except KeyboardInterrupt:
            print(f"\nStopped. Moves: {self.stats.moves}, Detects: {self.stats.detections}")
