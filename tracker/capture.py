from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray
    left: int
    top: int
    seq: int


class ScreenCapture:
    """Center-crop screen capture via MSS (GDI)."""

    def __init__(self, size: int = 320) -> None:
        self.size = max(160, int(size))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        self.screen_w = 0
        self.screen_h = 0
        self.center_x = 0
        self.center_y = 0
        self._region = {"left": 0, "top": 0, "width": self.size, "height": self.size}

    def start(self) -> None:
        with mss.mss() as sct:
            mon = sct.monitors[1]
        self.screen_w = mon["width"]
        self.screen_h = mon["height"]
        left = mon["left"]
        top = mon["top"]
        self.center_x = left + self.screen_w // 2
        self.center_y = top + self.screen_h // 2
        half = self.size // 2
        self._region = {
            "left": self.center_x - half,
            "top": self.center_y - half,
            "width": self.size,
            "height": self.size,
        }
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _loop(self) -> None:
        with mss.mss() as sct:
            while not self._stop.is_set():
                try:
                    shot = sct.grab(self._region)
                    bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
                    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
                    self._seq += 1
                    with self._lock:
                        self._frame = Frame(rgb=rgb, left=self._region["left"], top=self._region["top"], seq=self._seq)
                except Exception:
                    time.sleep(0.005)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame
