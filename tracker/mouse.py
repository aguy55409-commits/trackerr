from __future__ import annotations

import ctypes
from typing import Literal

from exoai_tracker.rzctl.rzctl import RZCONTROL

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
Method = Literal["razer", "win32"]


class Mouse:
    def __init__(self, method: Method = "razer") -> None:
        self.method: Method = method
        self.label = "Win32"
        self._rz: RZCONTROL | None = None
        if method == "razer":
            try:
                rz = RZCONTROL()
                if rz.init():
                    self._rz = rz
                    self.label = "Razer"
                else:
                    self.method = "win32"
            except Exception:
                self.method = "win32"

    def move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if self.method == "razer" and self._rz is not None:
            try:
                if self._rz.mouse_move(int(dx), int(dy), True):
                    return
            except Exception:
                pass
        user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
