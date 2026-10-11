"""SendInput mouse backend — no driver needed, works everywhere."""
from __future__ import annotations

import ctypes
import math
import time
from ctypes import Structure, Union, c_long, c_short, c_ulong, c_ushort

# --- Win32 structs for SendInput ---

PUL = ctypes.POINTER(c_ulong)


class _KeyBdInput(Structure):
    _fields_ = [
        ("wVk", c_ushort),
        ("wScan", c_ushort),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", PUL),
    ]


class _HardwareInput(Structure):
    _fields_ = [
        ("uMsg", c_ulong),
        ("wParamL", c_short),
        ("wParamH", c_ushort),
    ]


class _MouseInput(Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_ulong),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", PUL),
    ]


class _InputUnion(Union):
    _fields_ = [
        ("ki", _KeyBdInput),
        ("mi", _MouseInput),
        ("hi", _HardwareInput),
    ]


class _Input(Structure):
    _fields_ = [
        ("type", c_ulong),
        ("ii", _InputUnion),
    ]


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

_extra = c_ulong(0)
_extra_ptr = ctypes.pointer(_extra)


def _send_mouse(dx: int, dy: int, flags: int = MOUSEEVENTF_MOVE) -> bool:
    """Send a single mouse input event."""
    ii = _InputUnion()
    ii.mi = _MouseInput(dx, dy, 0, flags, 0, _extra_ptr)
    inp = _Input(INPUT_MOUSE, ii)
    return ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp)) == 1


class SendInputMouse:
    """SendInput-based mouse movement.
    
    No driver required. Less stealthy than rzctl but works on any Windows PC.
    Supports optional interpolation (breaking moves into small steps).
    """

    name = "sendinput"

    def __init__(self, interpolate: bool = False, step_delay: float = 0.0005) -> None:
        self._interpolate = interpolate
        self._step_delay = step_delay
        self._ok = 0
        self._pkt = 0
        self._last = (0, 0)
        self._down = False

    def init(self) -> bool:
        """SendInput is always available on Windows."""
        return True

    def ready(self) -> bool:
        return True

    def move(self, dx: int, dy: int) -> bool:
        """Move the mouse by (dx, dy) pixels.
        
        If interpolation is enabled, breaks the move into 1-pixel steps.
        """
        if dx == 0 and dy == 0:
            return True

        self._pkt += 1
        self._last = (dx, dy)

        if self._interpolate:
            return self._move_interpolated(dx, dy)
        
        if _send_mouse(dx, dy):
            self._ok += 1
            return True
        return False

    def _move_interpolated(self, dx: int, dy: int) -> bool:
        """Break the move into small steps along the line."""
        length = int(math.hypot(dx, dy))
        if length == 0:
            return True

        # Unit vector
        ux = dx / length
        uy = dy / length

        sent_x = sent_y = 0
        for i in range(1, length + 1):
            # Target position at step i
            target_x = int(round(ux * i))
            target_y = int(round(uy * i))
            
            # Delta from what we've already sent
            step_dx = target_x - sent_x
            step_dy = target_y - sent_y
            
            if step_dx != 0 or step_dy != 0:
                if not _send_mouse(step_dx, step_dy):
                    return False
                sent_x = target_x
                sent_y = target_y
                
                if self._step_delay > 0:
                    time.sleep(self._step_delay)

        self._ok += 1
        return True

    def press(self) -> bool:
        """Left mouse button down."""
        if _send_mouse(0, 0, MOUSEEVENTF_LEFTDOWN):
            self._down = True
            return True
        return False

    def release(self) -> bool:
        """Left mouse button up."""
        if _send_mouse(0, 0, MOUSEEVENTF_LEFTUP):
            self._down = False
            return True
        return False

    @property
    def down(self) -> bool:
        return self._down

    def close(self) -> None:
        pass

    def label(self) -> str:
        mode = "interp" if self._interpolate else "direct"
        return f"SendInput · {mode} · {self._ok}/{self._pkt}"

    def packet_count(self) -> int:
        return self._pkt

    def ok_count(self) -> int:
        return self._ok

    def last_move(self) -> tuple[int, int]:
        return self._last
