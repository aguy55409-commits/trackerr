"""Ultra-low-latency aim: detect thread + zero-sleep move thread."""
from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field

from exoai_tracker.track_engine import TrackEngine


def enable_fast_timer() -> None:
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def set_thread_priority(level: int = 2) -> None:
    """0=normal, 1=above, 2=highest."""
    try:
        k32 = ctypes.windll.kernel32
        handle = k32.GetCurrentThread()
        if level >= 2:
            k32.SetThreadPriority(handle, 2)
        elif level == 1:
            k32.SetThreadPriority(handle, 1)
    except Exception:
        pass


@dataclass
class FastStats:
    detect_hz: float = 0.0
    infer_ms: float = 0.0
    move_hz: float = 0.0
    loop_us: float = 0.0
    detections: list = field(default_factory=list)
    debug_frame: object | None = None


class FastLoop:
    """Detect on one thread; move spins with no sleep while bind is held."""

    def __init__(self, engine: TrackEngine) -> None:
        self.engine = engine
        self.stats = FastStats()
        self._stop = threading.Event()
        self._detect_thread: threading.Thread | None = None
        self._move_thread: threading.Thread | None = None
        self._detect_count = 0
        self._move_count = 0
        self._detect_t0 = time.perf_counter()
        self._move_t0 = time.perf_counter()
        self._last_loop_us = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._detect_thread = threading.Thread(
            target=self._detect_loop, name="fast-detect", daemon=True
        )
        self._move_thread = threading.Thread(
            target=self._move_loop, name="fast-move", daemon=True
        )
        self._detect_thread.start()
        self._move_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._detect_thread, self._move_thread):
            if t and t.is_alive():
                t.join(timeout=1.5)
        self._detect_thread = None
        self._move_thread = None

    def _detect_loop(self) -> None:
        set_thread_priority(1)
        engine = self.engine
        interval = max(0.0, engine.config.detect_interval_ms / 1000.0)
        while not self._stop.is_set():
            if not engine.is_held():
                time.sleep(0.001)
                continue
            now = time.perf_counter()
            if interval > 0.0 and now - engine._last_detect < interval:
                continue
            ms = engine.detect_step() * 1000.0
            self._detect_count += 1
            elapsed = now - self._detect_t0
            if elapsed >= 0.25:
                self.stats.detect_hz = self._detect_count / elapsed
                self.stats.infer_ms = ms
                self.stats.detections = list(engine.detections)
                self.stats.debug_frame = engine.debug_frame
                self._detect_count = 0
                self._detect_t0 = now

    def _move_loop(self) -> None:
        set_thread_priority(2)
        engine = self.engine
        while not self._stop.is_set():
            if not engine.is_held():
                time.sleep(0.0005)
                continue
            t0 = time.perf_counter()
            engine.move_step()
            self._move_count += 1
            self._last_loop_us = (time.perf_counter() - t0) * 1e6
            elapsed = time.perf_counter() - self._move_t0
            if elapsed >= 0.25:
                self.stats.move_hz = self._move_count / elapsed
                self.stats.loop_us = self._last_loop_us
                self._move_count = 0
                self._move_t0 = time.perf_counter()
