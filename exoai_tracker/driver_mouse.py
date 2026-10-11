"""Mouse movement via mouse_event - working in Apex."""
from __future__ import annotations

import ctypes

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001

_METHOD = "mouse_event"
_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0


def init() -> bool:
    """No init needed for mouse_event."""
    print("Using mouse_event for mouse movement")
    return True


def move_mouse(dx: int, dy: int) -> bool:
    """Move mouse via mouse_event."""
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
    _OK_COUNT += 1
    return True


def packet_count() -> int:
    return _PKT_COUNT


def ok_count() -> int:
    return _OK_COUNT


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def label() -> str:
    return f"mouse_event · {_OK_COUNT}/{_PKT_COUNT}"


def ready() -> bool:
    return True


class DriverMouseBackend:
    """Same mouse_event path that works in Apex."""

    name = "mouse_event"

    def init(self) -> bool:
        return init()

    def move(self, dx: int, dy: int) -> bool:
        return move_mouse(dx, dy)

    def close(self) -> None:
        return None

    def label(self) -> str:
        return label()

    def ready(self) -> bool:
        return True

    def packet_count(self) -> int:
        return packet_count()

    def ok_count(self) -> int:
        return ok_count()

    def last_move(self) -> tuple[int, int]:
        return last_move()
