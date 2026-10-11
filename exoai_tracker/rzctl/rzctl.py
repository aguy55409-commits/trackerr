"""RZCONTROL mouse — API matches old rzctl, packet matches modern Synapse."""
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
MAX_VAL = 65536
INVALID = 0xFFFFFFFFFFFFFFFF

kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.DeviceIoControl.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
]
kernel32.DeviceIoControl.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class MouseInputData(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("absolute_coord", ctypes.c_uint32),
        ("button_flags", ctypes.c_uint16),
        ("movement", ctypes.c_int16),
        ("unk1", ctypes.c_uint32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("unk2", ctypes.c_uint32),
    ]


class RzControl(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("unk1", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("data", MouseInputData),
    ]


assert ctypes.sizeof(RzControl) == 32


def _valid(h) -> bool:
    if h is None:
        return False
    v = int(h) & INVALID
    return 0 < v < INVALID


class RZCONTROL:
    """One instance = one handle. Do NOT share a class-level handle."""

    def __init__(self) -> None:
        self.hDevice = None

    def init(self) -> bool:
        self.close()
        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if not found or not name:
            return False
        sym = "\\\\?\\" + name
        h = kernel32.CreateFileW(
            sym, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None
        )
        if not _valid(h):
            h = kernel32.CreateFileW(
                sym, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None
            )
        if not _valid(h):
            return False
        self.hDevice = h
        return True

    def close(self) -> None:
        if _valid(self.hDevice):
            try:
                kernel32.CloseHandle(self.hDevice)
            except Exception:
                pass
        self.hDevice = None

    def mouse_move(self, x: int, y: int, from_start_point: bool = True) -> bool:
        if not _valid(self.hDevice):
            if not self.init():
                return False
        pkt = RzControl()
        pkt.unk1 = 0
        pkt.type = 2
        pkt.data.absolute_coord = 0 if from_start_point else MAX_VAL
        pkt.data.button_flags = 0
        pkt.data.movement = 1
        pkt.data.unk1 = 0
        pkt.data.x = int(x)
        pkt.data.y = int(y)
        pkt.data.unk2 = 0
        junk = wintypes.DWORD(0)
        ok = bool(
            kernel32.DeviceIoControl(
                self.hDevice,
                IOCTL_MOUSE,
                ctypes.byref(pkt),
                ctypes.sizeof(pkt),
                None,
                0,
                ctypes.byref(junk),
                None,
            )
        )
        if not ok:
            if self.init():
                ok = bool(
                    kernel32.DeviceIoControl(
                        self.hDevice,
                        IOCTL_MOUSE,
                        ctypes.byref(pkt),
                        ctypes.sizeof(pkt),
                        None,
                        0,
                        ctypes.byref(junk),
                        None,
                    )
                )
        return ok

    def mouse_click(self, click_mask: int) -> None:
        if not _valid(self.hDevice) and not self.init():
            return
        pkt = RzControl()
        pkt.unk1 = 0
        pkt.type = 2
        pkt.data.absolute_coord = 0
        pkt.data.button_flags = int(click_mask) & 0xFFFF
        pkt.data.movement = 0
        pkt.data.x = 0
        pkt.data.y = 0
        junk = wintypes.DWORD(0)
        kernel32.DeviceIoControl(
            self.hDevice,
            IOCTL_MOUSE,
            ctypes.byref(pkt),
            ctypes.sizeof(pkt),
            None,
            0,
            ctypes.byref(junk),
            None,
        )
