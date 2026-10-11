"""Mouse movement via alternative hooks that may bypass EAC."""
from __future__ import annotations

import ctypes
import struct
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
ntdll = ctypes.windll.ntdll

MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0

# For direct syscall
SYSCALL_NTUSERSEND = None  # Will be resolved dynamically


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


_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0
_METHOD = "none"


def _mouse_event_move(dx: int, dy: int) -> bool:
    """Legacy mouse_event - often hooked but worth trying."""
    try:
        user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
        return True
    except:
        return False


def _sendinput_move(dx: int, dy: int) -> bool:
    """Standard SendInput - usually hooked by EAC."""
    try:
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.dwFlags = MOUSEEVENTF_MOVE
        inp.mi.time = 0
        inp.mi.dwExtraInfo = None
        result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return result == 1
    except:
        return False


def _setcursor_move(dx: int, dy: int) -> bool:
    """SetCursorPos - absolute positioning, converted from relative."""
    try:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        new_x = pt.x + dx
        new_y = pt.y + dy
        user32.SetCursorPos(new_x, new_y)
        return True
    except:
        return False


def _clipcursor_move(dx: int, dy: int) -> bool:
    """Move via ClipCursor manipulation - unusual path."""
    try:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        
        # Create a 1px clip rect at target position
        new_x = pt.x + dx
        new_y = pt.y + dy
        
        rect = wintypes.RECT()
        rect.left = new_x
        rect.top = new_y
        rect.right = new_x + 1
        rect.bottom = new_y + 1
        
        # Clip, then release
        user32.ClipCursor(ctypes.byref(rect))
        user32.ClipCursor(None)
        return True
    except:
        return False


def _inject_thread_input(dx: int, dy: int) -> bool:
    """Use keybd_event path with mouse flags - unusual combination."""
    try:
        # Get foreground window's thread
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        
        tid = user32.GetWindowThreadProcessId(hwnd, None)
        my_tid = kernel32.GetCurrentThreadId()
        
        # Attach to target thread
        user32.AttachThreadInput(my_tid, tid, True)
        
        # Move cursor
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetCursorPos(pt.x + dx, pt.y + dy)
        
        # Detach
        user32.AttachThreadInput(my_tid, tid, False)
        return True
    except:
        return False


def _post_message_move(dx: int, dy: int) -> bool:
    """Post WM_MOUSEMOVE to foreground window."""
    try:
        WM_MOUSEMOVE = 0x0200
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        new_x = pt.x + dx
        new_y = pt.y + dy
        
        # Convert to client coords
        user32.ScreenToClient(hwnd, ctypes.byref(pt))
        client_x = pt.x + dx
        client_y = pt.y + dy
        
        lparam = (client_y << 16) | (client_x & 0xFFFF)
        user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
        return True
    except:
        return False


def _send_message_move(dx: int, dy: int) -> bool:
    """SendMessage with WM_INPUT simulation."""
    try:
        WM_MOUSEMOVE = 0x0200
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        
        # Move cursor first
        user32.SetCursorPos(pt.x + dx, pt.y + dy)
        
        # Then send message
        user32.GetCursorPos(ctypes.byref(pt))
        user32.ScreenToClient(hwnd, ctypes.byref(pt))
        lparam = (pt.y << 16) | (pt.x & 0xFFFF)
        user32.SendMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
        return True
    except:
        return False


# Method priority - try less common methods first
_METHODS = [
    ("thread_attach", _inject_thread_input),
    ("clip_cursor", _clipcursor_move),
    ("post_message", _post_message_move),
    ("send_message", _send_message_move),
    ("set_cursor", _setcursor_move),
    ("sendinput", _sendinput_move),
    ("mouse_event", _mouse_event_move),
]


def move_mouse(dx: int, dy: int) -> bool:
    """Try all mouse methods until one works."""
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY, _METHOD
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    # Try all methods
    for name, func in _METHODS:
        try:
            if func(dx, dy):
                _OK_COUNT += 1
                _METHOD = name
                return True
        except:
            continue
    
    return False


def init() -> bool:
    """Test which methods work."""
    global _METHOD
    
    # Test each method with a 0,0 move (no-op)
    for name, func in _METHODS:
        try:
            func(0, 0)
            _METHOD = name
            return True
        except:
            continue
    return False


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
