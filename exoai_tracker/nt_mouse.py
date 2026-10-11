"""Direct NT syscall mouse input - bypasses user-mode hooks."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

ntdll = ctypes.windll.ntdll
kernel32 = ctypes.windll.kernel32

# NT constants
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
INVALID_HANDLE = 0xFFFFFFFFFFFFFFFF
IOCTL_MOUSE = 0x88883020

_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0
_HANDLE = None
_METHOD = "none"


class IO_STATUS_BLOCK(ctypes.Structure):
    """NT IO_STATUS_BLOCK."""
    _fields_ = [
        ("Status", ctypes.c_long),
        ("Information", ctypes.c_void_p),
    ]


class RzPacket(ctypes.Structure):
    """RZCONTROL mouse packet."""
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


def _find_rzcontrol() -> str | None:
    """Find RZCONTROL device."""
    try:
        from exoai_tracker.rzctl.rzctl_nt import find_sym_link
        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if found and name:
            return "\\\\?\\" + name
    except:
        pass
    return None


def _open_device(path: str):
    """Open device handle."""
    h = kernel32.CreateFileW(
        path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        None, OPEN_EXISTING, 0, None
    )
    if h and int(h) > 0 and int(h) != INVALID_HANDLE:
        return h
    return None


def _nt_ioctl(handle, ioctl_code: int, in_buf, in_size: int) -> bool:
    """Direct NtDeviceIoControlFile syscall - bypasses DeviceIoControl hooks."""
    io_status = IO_STATUS_BLOCK()
    
    # NtDeviceIoControlFile(FileHandle, Event, ApcRoutine, ApcContext,
    #                       IoStatusBlock, IoControlCode, InputBuffer, InputBufferLength,
    #                       OutputBuffer, OutputBufferLength)
    status = ntdll.NtDeviceIoControlFile(
        handle,           # FileHandle
        None,             # Event
        None,             # ApcRoutine
        None,             # ApcContext
        ctypes.byref(io_status),  # IoStatusBlock
        ioctl_code,       # IoControlCode
        ctypes.byref(in_buf) if in_buf else None,  # InputBuffer
        in_size,          # InputBufferLength
        None,             # OutputBuffer
        0                 # OutputBufferLength
    )
    
    # STATUS_SUCCESS = 0
    return status == 0


def _rz_move_nt(dx: int, dy: int) -> bool:
    """Move via direct NtDeviceIoControlFile to RZCONTROL."""
    global _HANDLE
    
    if _HANDLE is None:
        return False
    
    pkt = RzPacket()
    pkt.report_id = 0
    pkt.packet_type = 2
    pkt.unk0 = 0
    pkt.unk1 = 1
    pkt.button_flags = 0
    pkt.unk2 = 0
    pkt.x = int(dx)
    pkt.y = int(dy)
    
    return _nt_ioctl(_HANDLE, IOCTL_MOUSE, pkt, ctypes.sizeof(pkt))


def _rz_move_dll(dx: int, dy: int) -> bool:
    """Move via rzctl.dll."""
    from pathlib import Path
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
    return False


def _mouse_event_move(dx: int, dy: int) -> bool:
    """Fallback mouse_event."""
    try:
        ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
        return True
    except:
        return False


def move_mouse(dx: int, dy: int) -> bool:
    """Move mouse - tries NT syscall first, then DLL, then fallback."""
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY, _METHOD
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    # Try direct NT syscall to RZCONTROL
    if _HANDLE is not None:
        if _rz_move_nt(dx, dy):
            _OK_COUNT += 1
            _METHOD = "nt_ioctl"
            return True
    
    # Try DLL
    if _rz_move_dll(dx, dy):
        _OK_COUNT += 1
        _METHOD = "dll"
        return True
    
    # Fallback
    _mouse_event_move(dx, dy)
    _METHOD = "mouse_event"
    return True


def init() -> bool:
    """Initialize - open RZCONTROL handle."""
    global _HANDLE, _METHOD
    
    path = _find_rzcontrol()
    if path:
        _HANDLE = _open_device(path)
        if _HANDLE:
            _METHOD = "nt_ioctl"
            return True
    
    # Check if DLL works
    if _rz_move_dll(0, 0):
        _METHOD = "dll"
        return True
    
    _METHOD = "mouse_event"
    return True


def packet_count() -> int:
    return _PKT_COUNT


def ok_count() -> int:
    return _OK_COUNT


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def label() -> str:
    return f"{_METHOD} · {_OK_COUNT}/{_PKT_COUNT}"


def ready() -> bool:
    return _METHOD != "none"
