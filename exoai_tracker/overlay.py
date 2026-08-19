from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Callable

WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_CLOSE = 0x0010
WM_ERASEBKGND = 0x0014
WM_NCHITTEST = 0x0084
WM_TIMER = 0x0113
HTTRANSPARENT = -1
WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
LWA_COLORKEY = 0x00000001
SW_SHOWNA = 8
PS_SOLID = 0
NULL_BRUSH = 5
TRANSPARENT = 1
OPAQUE = 2
TA_CENTER = 6
TA_TOP = 0


@dataclass
class OverlayDebugData:
    detections: list[tuple[float, float, float]] = field(default_factory=list)
    target: tuple[float, float] | None = None
    detect_hz: float = 0.0
    infer_ms: float = 0.0
    state: str = ""

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
LRESULT = ctypes.c_ssize_t
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DefWindowProcW.restype = LRESULT
user32.BeginPaint.restype = wintypes.HDC
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
gdi32.CreatePen.restype = wintypes.HPEN
gdi32.GetStockObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.restype = wintypes.HGDIOBJ
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", ctypes.c_byte * 32),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.SetLayeredWindowAttributes.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
    wintypes.BYTE,
    wintypes.DWORD,
]
user32.SetTimer.argtypes = [
    wintypes.HWND,
    ctypes.c_size_t,
    wintypes.UINT,
    wintypes.LPVOID,
]
user32.SetTimer.restype = ctypes.c_size_t
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetMessageW.argtypes = [
    ctypes.POINTER(MSG),
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
]
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.DefWindowProcW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.InvalidateRect.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.RECT),
    wintypes.BOOL,
]
user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.FillRect.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(wintypes.RECT),
    wintypes.HBRUSH,
]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.PostMessageW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
gdi32.CreateSolidBrush.argtypes = [wintypes.DWORD]
gdi32.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.GetStockObject.argtypes = [ctypes.c_int]
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.Ellipse.argtypes = [
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.DWORD]
gdi32.SetTextAlign.argtypes = [wintypes.HDC, wintypes.UINT]
gdi32.CreateFontW.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPCWSTR,
]
gdi32.CreateFontW.restype = wintypes.HFONT
gdi32.TextOutW.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.LPCWSTR, ctypes.c_int]
gdi32.MoveToEx.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.POINTER(wintypes.POINT)]
gdi32.LineTo.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]


def rgb(red: int, green: int, blue: int) -> int:
    return red | (green << 8) | (blue << 16)


class FovOverlay:
    """Native click-through FOV ring with debug visualization."""

    def __init__(
        self,
        get_radius: Callable[[], int],
        get_enabled: Callable[[], bool],
        get_debug: Callable[[], OverlayDebugData] | None = None,
    ) -> None:
        self.get_radius = get_radius
        self.get_enabled = get_enabled
        self.get_debug = get_debug
        self.hwnd: int | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._last_state: tuple[bool, int, str] | None = None
        self._wndproc_ref = WNDPROC(self._wndproc)
        self._class_name = f"CarbonFovOverlay_{id(self)}"
        self._font: int | None = None
        self._thread = threading.Thread(target=self._run, name="exoai-overlay", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=3):
            raise RuntimeError("FOV overlay did not initialize")
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            instance = kernel32.GetModuleHandleW(None)
            wc = WNDCLASSW()
            wc.lpfnWndProc = self._wndproc_ref
            wc.hInstance = instance
            wc.lpszClassName = self._class_name
            if not user32.RegisterClassW(ctypes.byref(wc)):
                raise ctypes.WinError()

            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            ex_style = (
                WS_EX_LAYERED
                | WS_EX_TRANSPARENT
                | WS_EX_TOPMOST
                | WS_EX_TOOLWINDOW
                | WS_EX_NOACTIVATE
            )
            hwnd = user32.CreateWindowExW(
                ex_style,
                self._class_name,
                "Carbon FOV",
                WS_POPUP,
                0,
                0,
                width,
                height,
                None,
                None,
                instance,
                None,
            )
            if not hwnd:
                raise ctypes.WinError()
            self.hwnd = hwnd
            user32.SetLayeredWindowAttributes(hwnd, rgb(255, 0, 255), 0, LWA_COLORKEY)
            user32.SetTimer(hwnd, 1, 16, None)
            user32.ShowWindow(hwnd, SW_SHOWNA)
            self._font = gdi32.CreateFontW(
                14, 0, 0, 0, 700, 0, 0, 0, 0, 0, 0, 0, 0, "Segoe UI"
            )
            self._ready.set()

            msg = MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def _wndproc(
        self,
        hwnd: wintypes.HWND,
        message: int,
        wparam: int,
        lparam: int,
    ) -> int:
        if message == WM_NCHITTEST:
            return HTTRANSPARENT
        if message == WM_ERASEBKGND:
            return 1
        if message == WM_TIMER:
            debug = self.get_debug() if self.get_debug else OverlayDebugData()
            state = (self.get_enabled(), self.get_radius(), debug.state)
            if state != self._last_state or debug.target is not None:
                self._last_state = state
                user32.InvalidateRect(hwnd, None, True)
            return 0
        if message == WM_PAINT:
            ps = PAINTSTRUCT()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            background = gdi32.CreateSolidBrush(rgb(255, 0, 255))
            user32.FillRect(hdc, ctypes.byref(rect), background)
            gdi32.DeleteObject(background)

            if self.get_enabled():
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                cx, cy = width // 2, height // 2
                radius = self.get_radius()

                debug = self.get_debug() if self.get_debug else OverlayDebugData()

                pen = gdi32.CreatePen(PS_SOLID, 2, rgb(0, 255, 136))
                old_pen = gdi32.SelectObject(hdc, pen)
                old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(NULL_BRUSH))
                gdi32.Ellipse(hdc, cx - radius, cy - radius, cx + radius, cy + radius)
                gdi32.SelectObject(hdc, old_brush)
                gdi32.SelectObject(hdc, old_pen)
                gdi32.DeleteObject(pen)

                for dx, dy, conf in debug.detections:
                    det_pen = gdi32.CreatePen(PS_SOLID, 2, rgb(255, 80, 80))
                    det_brush = gdi32.CreateSolidBrush(rgb(255, 80, 80))
                    old_pen = gdi32.SelectObject(hdc, det_pen)
                    old_brush = gdi32.SelectObject(hdc, det_brush)
                    px, py = int(cx + dx), int(cy + dy)
                    r = max(3, int(conf * 8))
                    gdi32.Ellipse(hdc, px - r, py - r, px + r, py + r)
                    gdi32.SelectObject(hdc, old_brush)
                    gdi32.SelectObject(hdc, old_pen)
                    gdi32.DeleteObject(det_brush)
                    gdi32.DeleteObject(det_pen)

                if debug.target is not None:
                    tx, ty = debug.target
                    px, py = int(cx + tx), int(cy + ty)
                    tgt_pen = gdi32.CreatePen(PS_SOLID, 2, rgb(0, 255, 255))
                    old_pen = gdi32.SelectObject(hdc, tgt_pen)
                    gdi32.MoveToEx(hdc, px - 12, py, None)
                    gdi32.LineTo(hdc, px + 12, py)
                    gdi32.MoveToEx(hdc, px, py - 12, None)
                    gdi32.LineTo(hdc, px, py + 12)
                    gdi32.SelectObject(hdc, old_pen)
                    gdi32.DeleteObject(tgt_pen)

                if self._font and (debug.detect_hz > 0 or debug.state):
                    old_font = gdi32.SelectObject(hdc, self._font)
                    gdi32.SetBkMode(hdc, TRANSPARENT)
                    gdi32.SetTextAlign(hdc, TA_CENTER | TA_TOP)
                    gdi32.SetTextColor(hdc, rgb(0, 255, 136))
                    text = f"{debug.state}  {debug.detect_hz:.0f}hz  {debug.infer_ms:.1f}ms"
                    gdi32.TextOutW(hdc, cx, cy + radius + 8, text, len(text))
                    gdi32.SelectObject(hdc, old_font)

            user32.EndPaint(hwnd, ctypes.byref(ps))
            return 0
        if message == WM_CLOSE:
            user32.DestroyWindow(hwnd)
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def destroy(self) -> None:
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1)
