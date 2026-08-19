from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Literal

MouseMethod = Literal["razer", "win32"]
user32 = ctypes.windll.user32

MOUSEEVENTF_MOVE = 0x0001

_DEBUG_LOG = Path.home() / "tracker_mouse_debug.txt"


def _log(msg: str) -> None:
    try:
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


class MouseController:
    """Mouse input via Razer Synapse (rzctl), Win32 fallback if Razer fails."""

    def __init__(self, method: str = "razer") -> None:
        self._requested = method if method in ("razer", "win32") else "razer"
        self.method: str = self._requested
        self._razer = None
        self.fell_back = False
        self._fallback_reason = ""
        self._moves = 0
        self._fail_streak = 0
        self._last_dx = 0
        self._last_dy = 0

        if self.method == "razer":
            self._init_razer()

        _log(f"init method={self.method} fell_back={self.fell_back} reason={self._fallback_reason!r}")

    def _init_razer(self) -> bool:
        try:
            from exoai_tracker.rzctl.rzctl import RZCONTROL

            razer = RZCONTROL()
            if not razer.init():
                raise RuntimeError("Razer Synapse driver not found / RZCONTROL missing")
            self._razer = razer
            self.method = "razer"
            self.fell_back = False
            self._fallback_reason = ""
            return True
        except Exception as exc:
            self._razer = None
            self.method = "win32"
            self.fell_back = True
            self._fallback_reason = str(exc)
            return False

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    def move_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return

        self._last_dx = int(dx)
        self._last_dy = int(dy)
        used = None

        if self.method == "razer" and self._razer is not None:
            try:
                ok = self._razer.mouse_move(int(dx), int(dy), True)
                if ok:
                    used = "razer"
                    self._fail_streak = 0
                    self._moves += 1
                    if self._moves <= 5 or self._moves % 60 == 0:
                        _log(f"razer ok dx={dx} dy={dy} n={self._moves}")
                    return
                self._fail_streak += 1
                if self._fail_streak >= 8:
                    # Razer handle went stale — reopen once, then fall through
                    if not self._init_razer():
                        _log(f"razer dead → win32 ({self._fallback_reason})")
            except Exception as exc:
                self._fail_streak += 1
                _log(f"razer exc: {exc}")

        user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
        used = "win32"
        self._moves += 1
        if self._moves <= 5 or self._moves % 60 == 0:
            _log(f"{used} dx={dx} dy={dy} n={self._moves}")

    @property
    def label(self) -> str:
        if self.fell_back:
            return "Win32 (Razer unavailable)"
        return "Razer Synapse" if self.method == "razer" else "Win32"
