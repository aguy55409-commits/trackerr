"""Mouse movement for games with raw input - tries multiple methods."""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0

# IOCTL constants for RZCONTROL
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
IOCTL_MOUSE = 0x88883020
INVALID_HANDLE = 0xFFFFFFFFFFFFFFFF


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


class MouseInputData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("unk0", ctypes.c_uint16),
        ("unk1", ctypes.c_uint16),
        ("button_flags", ctypes.c_uint16),
        ("unk2", ctypes.c_uint16),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class RzControlPacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("report_id", ctypes.c_uint32),
        ("packet_type", ctypes.c_uint32),
        ("data", MouseInputData),
    ]


_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0
_RZ_HANDLE = None
_DLL = None
_DLL_READY = False


def _find_rzcontrol() -> str | None:
    """Find RZCONTROL device symlink."""
    from exoai_tracker.rzctl.rzctl_nt import find_sym_link
    found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
    if found and name:
        return "\\\\?\\" + name
    return None


def _init_ioctl() -> bool:
    """Open handle to RZCONTROL device."""
    global _RZ_HANDLE
    
    if _RZ_HANDLE is not None:
        try:
            kernel32.CloseHandle(_RZ_HANDLE)
        except:
            pass
    _RZ_HANDLE = None
    
    sym = _find_rzcontrol()
    if not sym:
        return False
    
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    
    h = kernel32.CreateFileW(sym, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
    if h is None or (int(h) & INVALID_HANDLE) == INVALID_HANDLE or int(h) <= 0:
        h = kernel32.CreateFileW(sym, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
    
    if h is None or (int(h) & INVALID_HANDLE) == INVALID_HANDLE or int(h) <= 0:
        return False
    
    _RZ_HANDLE = h
    return True


def _ioctl_move(dx: int, dy: int) -> bool:
    """Send mouse move via DeviceIoControl."""
    global _RZ_HANDLE
    
    if _RZ_HANDLE is None:
        if not _init_ioctl():
            return False
    
    pkt = RzControlPacket()
    pkt.report_id = 0
    pkt.packet_type = 2
    pkt.data.unk0 = 0
    pkt.data.unk1 = 1  # Movement flag
    pkt.data.button_flags = 0
    pkt.data.unk2 = 0
    pkt.data.x = int(dx)
    pkt.data.y = int(dy)
    
    junk = wintypes.DWORD(0)
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    
    ok = kernel32.DeviceIoControl(
        _RZ_HANDLE, IOCTL_MOUSE, ctypes.byref(pkt), ctypes.sizeof(pkt),
        None, 0, ctypes.byref(junk), None
    )
    
    if not ok:
        _init_ioctl()
        return False
    return True


def _init_dll() -> bool:
    """Load rzctl.dll."""
    global _DLL, _DLL_READY
    
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
            dll.mouse_move.restype = None
            if dll.init():
                _DLL = dll
                _DLL_READY = True
                return True
        except:
            continue
    return False


def _dll_move(dx: int, dy: int) -> bool:
    """Move via rzctl.dll."""
    global _DLL, _DLL_READY
    
    if not _DLL_READY:
        if not _init_dll():
            return False
    
    try:
        _DLL.mouse_move(int(dx), int(dy), True)
        return True
    except:
        _DLL_READY = False
        return False


def _sendinput_move(dx: int, dy: int) -> None:
    """SendInput mouse move."""
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _mouse_event_move(dx: int, dy: int) -> None:
    """Legacy mouse_event."""
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def move_mouse(dx: int, dy: int) -> bool:
    """
    Move mouse using all available methods.
    Fires IOCTL + DLL + mouse_event for maximum compatibility.
    """
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    success = False
    
    # Method 1: Direct IOCTL to RZCONTROL
    if _ioctl_move(dx, dy):
        success = True
        time.sleep(0.0005)
    
    # Method 2: rzctl.dll
    if _dll_move(dx, dy):
        success = True
        time.sleep(0.0005)
    
    # Method 3: mouse_event fallback (may not work in Apex but worth trying)
    _mouse_event_move(dx, dy)
    
    if success:
        _OK_COUNT += 1
    
    return success


def init() -> bool:
    """Initialize all mouse backends."""
    ioctl_ok = _init_ioctl()
    dll_ok = _init_dll()
    return ioctl_ok or dll_ok


def packet_count() -> int:
    return _PKT_COUNT


def ok_count() -> int:
    return _OK_COUNT


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def label() -> str:
    ioctl = "IOCTL" if _RZ_HANDLE is not None else ""
    dll = "DLL" if _DLL_READY else ""
    parts = [p for p in [ioctl, dll] if p]
    if parts:
        return f"Razer ({'+'.join(parts)}) · {_OK_COUNT}/{_PKT_COUNT}"
    return f"mouse_event · {_PKT_COUNT}"


def ready() -> bool:
    return _RZ_HANDLE is not None or _DLL_READY
