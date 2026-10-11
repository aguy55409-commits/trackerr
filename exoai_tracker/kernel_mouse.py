"""Kernel-level mouse input for raw input games like Apex."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

ntdll = ctypes.windll.ntdll
kernel32 = ctypes.windll.kernel32

# Constants
GENERIC_WRITE = 0x40000000
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
INVALID_HANDLE = 0xFFFFFFFFFFFFFFFF

_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0
_HANDLE = None
_METHOD = "none"


class MOUSE_INPUT_DATA(ctypes.Structure):
    """Kernel mouse input structure (ntddmou.h)."""
    _pack_ = 1
    _fields_ = [
        ("UnitId", ctypes.c_ushort),
        ("Flags", ctypes.c_ushort),  # MOUSE_MOVE_RELATIVE = 0
        ("ButtonFlags", ctypes.c_ushort),
        ("ButtonData", ctypes.c_ushort),
        ("RawButtons", ctypes.c_ulong),
        ("LastX", ctypes.c_long),
        ("LastY", ctypes.c_long),
        ("ExtraInformation", ctypes.c_ulong),
    ]


def _find_mouse_device() -> str | None:
    """Find mouse device path."""
    # Try common mouse device paths
    paths = [
        r"\\.\PointerClass0",
        r"\\.\GLOBALROOT\Device\PointerClass0", 
        r"\\?\HID#VID_*&PID_*&MI_*#*#{378de44c-56ef-11d1-bc8c-00a0c91405dd}",
    ]
    
    for p in paths:
        try:
            h = kernel32.CreateFileW(
                p, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                None, OPEN_EXISTING, 0, None
            )
            if h and int(h) > 0 and int(h) != INVALID_HANDLE:
                kernel32.CloseHandle(h)
                return p
        except:
            continue
    
    return None


def _init_mouclass() -> bool:
    """Open handle to mouse class device."""
    global _HANDLE, _METHOD
    
    # Try RZCONTROL first (direct IOCTL)
    try:
        from exoai_tracker.rzctl.rzctl_nt import find_sym_link
        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if found and name:
            sym = "\\\\?\\" + name
            h = kernel32.CreateFileW(
                sym, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                None, OPEN_EXISTING, 0, None
            )
            if h and int(h) > 0 and int(h) != INVALID_HANDLE:
                _HANDLE = h
                _METHOD = "rzctl"
                return True
    except:
        pass
    
    return False


def _rz_move(dx: int, dy: int) -> bool:
    """Move via RZCONTROL IOCTL."""
    global _HANDLE
    
    if _HANDLE is None:
        return False
    
    # RZCONTROL packet structure
    class RzPacket(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("report_id", ctypes.c_uint32),
            ("packet_type", ctypes.c_uint32),
            ("unk0", ctypes.c_uint16),
            ("unk1", ctypes.c_uint16),
            ("button_flags", ctypes.c_uint16),
            ("unk2", ctypes.c_uint16),
            ("x", ctypes.c_int32),
            ("y", ctypes.c_int32),
        ]
    
    pkt = RzPacket()
    pkt.report_id = 0
    pkt.packet_type = 2  # Mouse move
    pkt.unk0 = 0
    pkt.unk1 = 1  # Relative movement
    pkt.button_flags = 0
    pkt.unk2 = 0
    pkt.x = int(dx)
    pkt.y = int(dy)
    
    IOCTL_MOUSE = 0x88883020
    junk = wintypes.DWORD(0)
    
    ok = kernel32.DeviceIoControl(
        _HANDLE, IOCTL_MOUSE, ctypes.byref(pkt), ctypes.sizeof(pkt),
        None, 0, ctypes.byref(junk), None
    )
    
    return bool(ok)


def _dll_move(dx: int, dy: int) -> bool:
    """Move via rzctl.dll."""
    try:
        # Try to load DLL if not already
        paths = [
            Path(__file__).resolve().parent / "rzctl" / "rzctl.dll",
            Path(__file__).resolve().parent.parent / "rzctl.dll",
        ]
        
        for p in paths:
            if not p.is_file():
                continue
            try:
                dll = ctypes.CDLL(str(p))
                dll.init.restype = ctypes.c_bool
                dll.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
                if dll.init():
                    dll.mouse_move(int(dx), int(dy), True)
                    return True
            except:
                continue
    except:
        pass
    return False


def _fallback_move(dx: int, dy: int) -> bool:
    """Fallback: mouse_event."""
    try:
        ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
        return True
    except:
        return False


def move_mouse(dx: int, dy: int) -> bool:
    """Move mouse using best available method."""
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY, _METHOD
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    # Try RZCONTROL IOCTL
    if _HANDLE is not None:
        if _rz_move(dx, dy):
            _OK_COUNT += 1
            return True
    
    # Try DLL
    if _dll_move(dx, dy):
        _OK_COUNT += 1
        _METHOD = "dll"
        return True
    
    # Fallback
    if _fallback_move(dx, dy):
        _OK_COUNT += 1
        _METHOD = "mouse_event"
        return True
    
    return False


def init() -> bool:
    """Initialize mouse input."""
    return _init_mouclass()


def packet_count() -> int:
    return _PKT_COUNT


def ok_count() -> int:
    return _OK_COUNT


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def label() -> str:
    return f"{_METHOD} · {_OK_COUNT}/{_PKT_COUNT}"


def ready() -> bool:
    return _HANDLE is not None or _METHOD != "none"
