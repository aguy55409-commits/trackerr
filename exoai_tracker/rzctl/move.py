"""Move the physical mouse — HID first, then RZCONTROL Nt/Win IOCTL, then dll.

Order per tick (what worked semi-choppy in Apex before):
  1) WriteFile / NtWriteFile on Razer HID (VID 1532)
  2) NtDeviceIoControlFile on RZCONTROL
  3) DeviceIoControl on RZCONTROL
  4) rzctl.dll
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

from exoai_tracker.rzctl.rzctl_nt import (
    FILE_SHARE_READ,
    FILE_SHARE_WRITE,
    OPEN_EXISTING,
    find_sym_link,
)

kernel32 = ctypes.windll.kernel32
ntdll = ctypes.windll.ntdll
user32 = ctypes.windll.user32

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
IOCTL_MOUSE = 0x88883020
# RZCONTROL button flags.
LEFT_DOWN = 1
LEFT_UP = 2
INVALID = 0xFFFFFFFFFFFFFFFF
RIM_TYPEMOUSE = 0
RIDI_DEVICENAME = 0x20000007

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
kernel32.WriteFile.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
]
kernel32.WriteFile.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

ntdll.NtDeviceIoControlFile.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG,
    ctypes.c_void_p, wintypes.ULONG,
]
ntdll.NtDeviceIoControlFile.restype = ctypes.c_long
ntdll.NtWriteFile.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p, ctypes.c_void_p,
]
ntdll.NtWriteFile.restype = ctypes.c_long


class IO_STATUS_BLOCK(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("Status", ctypes.c_long), ("Pointer", ctypes.c_void_p)]

    _anonymous_ = ("u",)
    _fields_ = [("u", _U), ("Information", ctypes.c_void_p)]


class RAWINPUTDEVICELIST(ctypes.Structure):
    _fields_ = [("hDevice", ctypes.c_void_p), ("dwType", wintypes.DWORD)]


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


PKT_SIZE = ctypes.sizeof(RzControl)


def _valid(h) -> bool:
    if h is None:
        return False
    v = int(h) & INVALID
    return 0 < v < INVALID


def _s8(v: int) -> int:
    return max(-127, min(127, int(v))) & 0xFF


def _dll_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    return [
        here / "rzctl.dll",
        root / "rzctl.dll",
        root / "sunone_aimbot" / "logic" / "rzctl.dll",
    ]


def _raw_mouse_paths() -> list[str]:
    num = ctypes.c_uint(0)
    user32.GetRawInputDeviceList(None, ctypes.byref(num), ctypes.sizeof(RAWINPUTDEVICELIST))
    if num.value == 0:
        return []
    devices = (RAWINPUTDEVICELIST * num.value)()
    user32.GetRawInputDeviceList(devices, ctypes.byref(num), ctypes.sizeof(RAWINPUTDEVICELIST))
    paths: list[str] = []
    for dev in devices:
        if int(dev.dwType) != RIM_TYPEMOUSE:
            continue
        nlen = ctypes.c_uint(0)
        user32.GetRawInputDeviceInfoW(dev.hDevice, RIDI_DEVICENAME, None, ctypes.byref(nlen))
        if nlen.value <= 0:
            continue
        buf = ctypes.create_unicode_buffer(nlen.value)
        user32.GetRawInputDeviceInfoW(dev.hDevice, RIDI_DEVICENAME, buf, ctypes.byref(nlen))
        if buf.value:
            paths.append(buf.value)
    razer = [p for p in paths if "1532" in p.upper()]
    other = [p for p in paths if "1532" not in p.upper()]
    return razer + other


class RazerMover:
    def __init__(self) -> None:
        self.handle = None
        self._hid = None
        self._hid_bytes = 4
        self._dll = None
        self._method = "none"
        self._pkt = 0
        self._ok = 0
        self._last = (0, 0)
        self._down = False

    def init(self) -> bool:
        hid_ok = self._open_hid()
        ioctl_ok = self._open_rz()
        dll_ok = self._load_dll()
        if hid_ok:
            self._method = "hid"
        elif ioctl_ok:
            self._method = "rz_nt"
        elif dll_ok:
            self._method = "rz_dll"
        return hid_ok or ioctl_ok or dll_ok

    def _open_file(self, path: str):
        for access in (GENERIC_READ | GENERIC_WRITE, GENERIC_WRITE, 0):
            h = kernel32.CreateFileW(
                path, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                None, OPEN_EXISTING, 0, None,
            )
            if _valid(h):
                return h
        return None

    def _open_hid(self) -> bool:
        if _valid(self._hid):
            return True
        for path in _raw_mouse_paths():
            h = self._open_file(path)
            if h is None:
                continue
            self._hid = h
            for size in (4, 3, 5, 8):
                if self._hid_write(0, 0, size):
                    self._hid_bytes = size
                    return True
            kernel32.CloseHandle(h)
            self._hid = None
        return False

    def _open_rz(self) -> bool:
        if _valid(self.handle):
            return True
        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if not found or not name:
            return False
        h = self._open_file("\\\\?\\" + name)
        if h is None:
            return False
        self.handle = h
        return True

    def _load_dll(self) -> bool:
        if self._dll is not None:
            return True
        import ctypes as ct

        for path in _dll_paths():
            if not path.is_file():
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(path.parent.resolve()))
                dll = ct.WinDLL(str(path.resolve()))
                dll.init.restype = ct.c_bool
                dll.mouse_move.argtypes = [ct.c_int, ct.c_int, ct.c_bool]
                dll.mouse_move.restype = None
                if hasattr(dll, "mouse_click"):
                    dll.mouse_click.argtypes = [ct.c_int]
                    dll.mouse_click.restype = None
                if dll.init():
                    self._dll = dll
                    return True
            except Exception:
                continue
        return False

    def _hid_write(
        self, dx: int, dy: int, size: int | None = None, buttons: int = 0
    ) -> bool:
        if not _valid(self._hid):
            return False
        n = int(size or self._hid_bytes)
        x = _s8(dx)
        y = _s8(dy)
        # Button bitmask, bit 0 being left. The longer reports lead with a
        # report id, so the mask sits one byte further in.
        b = int(buttons) & 0xFF
        if n == 3:
            raw = bytes((b, x, y))
        elif n == 5:
            raw = bytes((0, b, x, y, 0))
        elif n == 8:
            raw = bytes((0, b, x, y, 0, 0, 0, 0))
        else:
            raw = bytes((b, x, y, 0))
        buf = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        written = wintypes.DWORD(0)
        if kernel32.WriteFile(self._hid, ctypes.byref(buf), len(raw), ctypes.byref(written), None):
            if written.value > 0:
                return True
        io = IO_STATUS_BLOCK()
        status = ntdll.NtWriteFile(
            self._hid, None, None, None, ctypes.byref(io),
            ctypes.byref(buf), len(raw), None, None,
        )
        return status == 0

    def _packet(self, dx: int, dy: int) -> RzControl:
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
        return pkt

    def _button_packet(self, flags: int) -> RzControl:
        """A press or release: same packet as a move, but carrying no motion."""
        pkt = RzControl()
        pkt.unk1 = 0
        pkt.type = 2
        pkt.data.absolute_coord = 0
        pkt.data.button_flags = int(flags) & 0xFFFF
        pkt.data.movement = 0
        pkt.data.unk1 = 0
        pkt.data.x = 0
        pkt.data.y = 0
        pkt.data.unk2 = 0
        return pkt

    def _dll_click(self, flags: int) -> bool:
        if self._dll is None and not self._load_dll():
            return False
        fn = getattr(self._dll, "mouse_click", None)
        if fn is None:
            return False
        try:
            fn(int(flags))
            return True
        except Exception:
            return False

    def _buttons(self, flags: int, hid_mask: int) -> bool:
        """Send one button transition down the first path that takes it.

        move() deliberately fires every path at once, since a duplicated
        movement just lands as a slightly longer one. A duplicated button does
        not: three paths accepting a press is three shots, so this stops at the
        first success while keeping the same order of preference.
        """
        if not _valid(self._hid):
            self._open_hid()
        if not _valid(self.handle):
            self._open_rz()
        if self._hid_write(0, 0, buttons=hid_mask):
            return True
        pkt = self._button_packet(flags)
        if self._nt_ioctl(pkt):
            return True
        if self._win_ioctl(pkt):
            return True
        return self._dll_click(flags)

    def press(self) -> bool:
        ok = self._buttons(LEFT_DOWN, 1)
        self._down = ok
        return ok

    def release(self) -> bool:
        ok = self._buttons(LEFT_UP, 0)
        if ok:
            self._down = False
        return ok

    @property
    def down(self) -> bool:
        return self._down

    def _nt_ioctl(self, pkt: RzControl) -> bool:
        if not _valid(self.handle):
            return False
        io = IO_STATUS_BLOCK()
        status = ntdll.NtDeviceIoControlFile(
            self.handle, None, None, None, ctypes.byref(io),
            IOCTL_MOUSE, ctypes.byref(pkt), PKT_SIZE, None, 0,
        )
        return status == 0

    def _win_ioctl(self, pkt: RzControl) -> bool:
        if not _valid(self.handle):
            return False
        junk = wintypes.DWORD(0)
        return bool(
            kernel32.DeviceIoControl(
                self.handle, IOCTL_MOUSE, ctypes.byref(pkt), PKT_SIZE,
                None, 0, ctypes.byref(junk), None,
            )
        )

    def _dll_move(self, dx: int, dy: int) -> bool:
        if self._dll is None and not self._load_dll():
            return False
        try:
            self._dll.mouse_move(int(dx), int(dy), True)
            return True
        except Exception:
            self._dll = None
            return False

    def move(self, dx: int, dy: int) -> bool:
        dx = int(dx)
        dy = int(dy)
        if dx == 0 and dy == 0:
            return True
        self._pkt += 1
        self._last = (dx, dy)
        if not _valid(self._hid):
            self._open_hid()
        if not _valid(self.handle):
            self._open_rz()

        sent = False
        ox, oy = dx, dy
        while ox != 0 or oy != 0:
            sx = max(-127, min(127, ox))
            sy = max(-127, min(127, oy))
            ox -= sx
            oy -= sy
            # HID first — lowest level, what Apex sometimes still reads
            if self._hid_write(sx, sy):
                sent = True
                self._method = "hid"
            pkt = self._packet(sx, sy)
            if self._nt_ioctl(pkt):
                sent = True
                if self._method == "none":
                    self._method = "rz_nt"
            elif self._win_ioctl(pkt):
                sent = True
                if self._method == "none":
                    self._method = "rz_ioctl"
            if self._dll_move(sx, sy):
                sent = True
                if self._method == "none":
                    self._method = "rz_dll"
        if sent:
            self._ok += 1
        return sent

    @property
    def method(self) -> str:
        return self._method

    @property
    def pkt(self) -> int:
        return self._pkt

    @property
    def ok(self) -> int:
        return self._ok

    @property
    def last(self) -> tuple[int, int]:
        return self._last

    def ready(self) -> bool:
        return _valid(self._hid) or _valid(self.handle) or self._dll is not None
