"""
Razer Synapse mouse via RZCONTROL driver.

Packet layout matches BlankyWacky/razerctl (modern Synapse):
https://github.com/BlankyWacky/razerctl

Requires Razer Synapse running so RZCONTROL exists.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from exoai_tracker.rzctl.rzctl_nt import (
    FILE_SHARE_READ,
    FILE_SHARE_WRITE,
    OPEN_EXISTING,
    find_sym_link,
)

kernel32 = ctypes.windll.kernel32

GENERIC_WRITE = 0x40000000
IOCTL_MOUSE = 0x88883020
INVALID = 0xFFFFFFFFFFFFFFFF

kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
kernel32.CreateFileW.restype = wintypes.HANDLE

kernel32.DeviceIoControl.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]
kernel32.DeviceIoControl.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


class MouseInputData(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("absolute_coord", ctypes.c_uint32),  # 0 relative, 65536 absolute
        ("button_flags", ctypes.c_uint16),
        ("movement", ctypes.c_int16),  # MUST be 1 for relative moves
        ("unk1", ctypes.c_uint32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("unk2", ctypes.c_uint32),
    ]


class RzControl(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("unk1", ctypes.c_uint32),
        ("type", ctypes.c_uint32),  # 1=keyboard 2=mouse
        ("data", MouseInputData),
    ]


_SIZE = ctypes.sizeof(RzControl)
assert _SIZE == 32, f"RzControl size {_SIZE} != 32"


def _valid(handle) -> bool:
    if handle is None:
        return False
    v = int(handle) & INVALID
    return 0 < v < INVALID


class RazerMouse:
    def __init__(self) -> None:
        self._handle = None

    def init(self) -> bool:
        self.close()
        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if not found or not name:
            return False

        sym = "\\\\?\\" + name
        handle = kernel32.CreateFileW(
            sym,
            GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if not _valid(handle):
            handle = kernel32.CreateFileW(
                sym,
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_EXISTING,
                0,
                None,
            )
        if not _valid(handle):
            return False
        self._handle = handle
        return True

    def close(self) -> None:
        if _valid(self._handle):
            kernel32.CloseHandle(self._handle)
        self._handle = None

    def _ioctl(self, packet: RzControl) -> bool:
        if not _valid(self._handle):
            return False
        junk = wintypes.DWORD(0)
        ok = bool(
            kernel32.DeviceIoControl(
                self._handle,
                IOCTL_MOUSE,
                ctypes.byref(packet),
                ctypes.sizeof(packet),
                None,
                0,
                ctypes.byref(junk),
                None,
            )
        )
        if not ok:
            self.init()
        return ok

    def move(self, dx: int, dy: int) -> bool:
        pkt = RzControl()
        pkt.unk1 = 0
        pkt.type = 2
        pkt.data.absolute_coord = 0
        pkt.data.button_flags = 0
        pkt.data.movement = 1
        pkt.data.unk1 = 0
        pkt.data.x = int(dx)
        pkt.data.y = int(dy)
        pkt.data.unk2 = 0
        return self._ioctl(pkt)


class RZCONTROL(RazerMouse):
    """Alias with old method name."""

    def mouse_move(self, x: int, y: int, from_start_point: bool = True) -> bool:
        return self.move(int(x), int(y))
