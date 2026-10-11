from __future__ import annotations

import ctypes
from tracker.rzctl import RazerMouse

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001


class Mouse:
    def __init__(self) -> None:
        self._rz: RazerMouse | None = None
        self.label = "Win32"
        try:
            rz = RazerMouse()
            if rz.init():
                self._rz = rz
                self.label = "Razer"
        except Exception:
            pass

    def move(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if self._rz is not None:
            try:
                if self._rz.move(int(dx), int(dy)):
                    return
            except Exception:
                pass
        user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def test_nudge(self) -> str:
        import time

        class P(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        def cur():
            p = P()
            user32.GetCursorPos(ctypes.byref(p))
            return p.x, p.y

        b = cur()
        self.move(80, 0)
        time.sleep(0.06)
        a = cur()
        self.move(-(a[0] - b[0] or 80), 0)
        return f"{self.label}: +80 -> {a[0]-b[0]}px"
