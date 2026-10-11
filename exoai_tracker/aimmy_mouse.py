"""Aimmy-compatible mouse movement — mirrors InputLogic/MouseManager.cs + RZMouse.cs."""
from __future__ import annotations

import ctypes
import os
import random
import time
from ctypes import wintypes
from pathlib import Path

from exoai_tracker.paths import app_dir

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0

_PREV_X = 0.0
_PREV_Y = 0.0
_EMA_ENABLED = False
_SMOOTHING = 0.5


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


class _RzDll:
    def __init__(self) -> None:
        self._dll: ctypes.CDLL | None = None
        self._ready = False
        self._load()

    def _dll_paths(self) -> list[Path]:
        root = app_dir()
        return [
            root / "rzctl.dll",
            root / "exoai_tracker" / "rzctl" / "rzctl.dll",
            Path(__file__).resolve().parent.parent / "rzctl" / "rzctl.dll",
        ]

    def _load(self) -> bool:
        for path in self._dll_paths():
            if not path.is_file():
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(path.parent))
                dll = ctypes.CDLL(str(path))
                dll.init.restype = ctypes.c_bool
                dll.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
                dll.mouse_move.restype = None
                self._dll = dll
                self._ready = bool(dll.init())
                return self._ready
            except Exception:
                continue
        self._dll = None
        self._ready = False
        return False

    def move(self, dx: int, dy: int) -> bool:
        if self._dll is None and not self._load():
            return False
        try:
            self._dll.mouse_move(int(dx), int(dy), True)
            return True
        except Exception:
            self._ready = False
            return self._load() and self.move(dx, dy)

    @property
    def ready(self) -> bool:
        return self._ready


_RZ = _RzDll()
_PKT_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0


def _ema(prev: float, cur: float, factor: float) -> float:
    return (cur * factor) + (prev * (1.0 - factor))


def _lerp(t: float, end_x: int, end_y: int) -> tuple[int, int]:
    return int(end_x * t), int(end_y * t)


def _sendinput_move(dx: int, dy: int) -> None:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _mouse_event_move(dx: int, dy: int) -> None:
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def _deliver(dx: int, dy: int, method: str) -> None:
    global _PKT_COUNT
    if dx == 0 and dy == 0:
        return

    if method == "razer":
        if _RZ.move(dx, dy):
            _PKT_COUNT += 1
            time.sleep(0.001)
            return
        method = "mouse_event"

    if method == "sendinput":
        _sendinput_move(dx, dy)
        _PKT_COUNT += 1
        return

    # Aimmy default fallback
    _mouse_event_move(dx, dy)
    _PKT_COUNT += 1


def track_speed_to_sensitivity(track_speed: float) -> float:
    """Map UI speed 0.5–5.0 → Aimmy MouseSensitivity (higher = slower)."""
    speed = max(0.5, min(5.0, track_speed))
    return max(0.05, min(0.92, 1.0 - (speed / 5.0) * 0.88))


def move_crosshair(
    detected_x: int,
    detected_y: int,
    screen_width: float,
    screen_height: float,
    track_speed: float,
    movement_path: str = "Linear",
    jitter: int = 0,
    mouse_method: str = "razer",
    center_x: float | None = None,
    center_y: float | None = None,
) -> tuple[int, int]:
    """
    Exact port of Aimmy MouseManager.MoveCrosshair(detectedX, detectedY).
    detectedX/Y = absolute screen coordinates of aim point.
    """
    global _PREV_X, _PREV_Y, _LAST_DX, _LAST_DY

    if center_x is not None and center_y is not None:
        target_x = int(detected_x) - int(round(center_x))
        target_y = int(detected_y) - int(round(center_y))
    else:
        half_w = int(screen_width) // 2
        half_h = int(screen_height) // 2
        target_x = int(detected_x) - half_w
        target_y = int(detected_y) - half_h
    aspect = screen_width / screen_height if screen_height else 1.0

    sensitivity = track_speed_to_sensitivity(track_speed)
    t = 1.0 - sensitivity

    jx = random.randint(-jitter, jitter) if jitter > 0 else 0
    jy = random.randint(-jitter, jitter) if jitter > 0 else 0

    if movement_path == "Exponential":
        exp_t = 1.0 - max(0.0, sensitivity - 0.2)
        nx = int(target_x * (exp_t ** 3.0))
        ny = int(target_y * (exp_t ** 3.0))
    else:
        nx, ny = _lerp(t, target_x, target_y)

    if _EMA_ENABLED:
        nx = int(_ema(_PREV_X, nx, _SMOOTHING))
        ny = int(_ema(_PREV_Y, ny, _SMOOTHING))

    nx = max(-150, min(150, nx))
    ny = max(-150, min(150, ny))
    ny = int(ny / aspect)
    nx += jx
    ny += jy

    _deliver(nx, ny, mouse_method)
    _PREV_X = float(nx)
    _PREV_Y = float(ny)
    _LAST_DX = nx
    _LAST_DY = ny
    return nx, ny


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def packet_count() -> int:
    return _PKT_COUNT


def razer_ready() -> bool:
    return _RZ.ready


def label() -> str:
    if _RZ.ready:
        return f"Razer rzctl.dll · {_PKT_COUNT} pkts"
    return "Razer + mouse_event fallback"
