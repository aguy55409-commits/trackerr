from __future__ import annotations

import ctypes
from pathlib import Path

_DLL_PATH = Path(__file__).resolve().parent / "rzctl.dll"


class RzctlDll:
    """Aimmy-style rzctl.dll — init() + mouse_move(x, y, from_start_point)."""

    def __init__(self) -> None:
        self._dll: ctypes.CDLL | None = None
        self._ready = False
        self._load()

    def _load(self) -> bool:
        if not _DLL_PATH.is_file():
            return False
        try:
            dll = ctypes.CDLL(str(_DLL_PATH))
            dll.init.restype = ctypes.c_bool
            dll.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            dll.mouse_move.restype = None
            self._dll = dll
            self._ready = bool(dll.init())
            return self._ready
        except Exception:
            self._dll = None
            self._ready = False
            return False

    def init(self) -> bool:
        if self._dll is None:
            return self._load()
        self._ready = bool(self._dll.init())
        return self._ready

    def mouse_move(self, dx: int, dy: int, from_start_point: bool = True) -> bool:
        if self._dll is None and not self._load():
            return False
        try:
            self._dll.mouse_move(int(dx), int(dy), bool(from_start_point))
            return True
        except Exception:
            self._ready = False
            return self.init() and self.mouse_move(dx, dy, from_start_point)

    @property
    def ready(self) -> bool:
        return self._ready
