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
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


@dataclass
class DetectionBox:
    x: float  # center x relative to screen center
    y: float  # center y relative to screen center
    w: float  # width
    h: float  # height
    conf: float
    kind: str = "enemy"  # enemy, ally, head, etc.


@dataclass
class OverlayDebugData:
    detections: list[tuple[float, float, float]] = field(default_factory=list)
    boxes: list[DetectionBox] = field(default_factory=list)  # Full bounding boxes
    target: tuple[float, float] | None = None
    detect_hz: float = 0.0
    infer_ms: float = 0.0
    state: str = ""
    israel_mode: bool = False
    show_movement_line: bool = False

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
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
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
gdi32.Arc.argtypes = [
    wintypes.HDC,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
gdi32.Arc.restype = wintypes.BOOL


def rgb(red: int, green: int, blue: int) -> int:
    return red | (green << 8) | (blue << 16)


import math
import time

STAR_RADIUS = 16

# Ice palette shared with the menu. Width-1 pens only: thick GDI pens are slow.
_RING = {
    "ring_out": rgb(150, 205, 245),
    "ring_mid": rgb(225, 242, 255),
    "ring_in": rgb(90, 155, 215),
}
_LINE = (
    rgb(70, 130, 195),
    rgb(110, 175, 225),
    rgb(170, 215, 250),
    rgb(235, 248, 255),
)


def _approach(cur: float, target: float, rate: float, dt: float) -> float:
    if dt <= 0.0:
        return cur
    return cur + (target - cur) * (1.0 - math.exp(-rate * dt))


def _star_of_david_points(cx: int, cy: int, radius: int, angle: float) -> list[tuple[int, int]]:
    """Generate points for two overlapping triangles (Star of David)."""
    points = []
    # First triangle (pointing up)
    for i in range(3):
        a = angle + i * (2 * math.pi / 3) - math.pi / 2
        x = cx + int(radius * math.cos(a))
        y = cy + int(radius * math.sin(a))
        points.append((x, y))
    # Second triangle (pointing down)
    for i in range(3):
        a = angle + i * (2 * math.pi / 3) + math.pi / 2
        x = cx + int(radius * math.cos(a))
        y = cy + int(radius * math.sin(a))
        points.append((x, y))
    return points


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
        self._last_state: tuple | None = None
        self._topmost_tick = 0
        self._prev_dirty: wintypes.RECT | None = None
        self._wndproc_ref = WNDPROC(self._wndproc)
        self._class_name = f"CarbonFovOverlay_{id(self)}"
        self._font: int | None = None
        self._pens: dict[str, int] = {}
        self._clear_brush: int | None = None
        self._vis_radius: float | None = None
        self._line_x = 0.0
        self._line_y = 0.0
        self._anim_t = time.perf_counter()
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
            # Force topmost immediately
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW
            )
            # 60 Hz is enough for an eased ring. 240 Hz was repainting for no visual gain.
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

    def _dirty_rect(self, hwnd, debug: OverlayDebugData) -> wintypes.RECT:
        """Smallest screen region covering everything we draw this frame."""
        client = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(client))
        cx = (client.right - client.left) // 2
        cy = (client.bottom - client.top) // 2

        if debug.israel_mode:
            pad = STAR_RADIUS + 4
            extra = 0
        else:
            shown = self._vis_radius if self._vis_radius is not None else float(self.get_radius())
            pad = int(shown) + 8
            extra = 24  # stats text sits below the ring
            for box in debug.boxes:
                pad = max(
                    pad,
                    int(abs(box.x) + box.w * 0.5 + 4),
                    int(abs(box.y) + box.h * 0.5 + 4),
                )
            if abs(self._line_x) + abs(self._line_y) > 1.5:
                pad = max(pad, int(abs(self._line_x) + 12), int(abs(self._line_y) + 12))

        rect = wintypes.RECT()
        rect.left = max(client.left, cx - pad)
        rect.top = max(client.top, cy - pad)
        rect.right = min(client.right, cx + pad)
        rect.bottom = min(client.bottom, cy + pad + extra)
        return rect

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
            # Reassert topmost so fullscreen games can't cover us, but not every
            # tick: SetWindowPos at 240Hz churns the window manager and stutters.
            self._topmost_tick += 1
            # Twice a second left us sitting behind the game for up to half a
            # second whenever it grabbed the z-order, which looks like the
            # overlay not working. Ten times a second is still cheap at 60 Hz.
            if self._topmost_tick >= 6:
                self._topmost_tick = 0
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW
                )

            debug = self.get_debug() if self.get_debug else OverlayDebugData()
            animating = self._step_anim(debug)
            dirty = self._dirty_rect(hwnd, debug)
            state = (
                self.get_enabled(),
                self.get_radius(),
                debug.state,
                debug.israel_mode,
                debug.show_movement_line,
            )
            moving = (
                debug.israel_mode
                or debug.target is not None
                or bool(debug.boxes)
                or animating
            )
            if state != self._last_state or moving:
                self._last_state = state
                # Invalidate only what we draw. Invalidating the whole screen
                # here is what made tracking stutter.
                if self._prev_dirty is not None:
                    user32.InvalidateRect(hwnd, ctypes.byref(self._prev_dirty), True)
                user32.InvalidateRect(hwnd, ctypes.byref(dirty), True)
            self._prev_dirty = dirty
            return 0
        if message == WM_PAINT:
            ps = PAINTSTRUCT()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            # Clear only the invalidated region; clearing the whole screen every
            # frame is far too expensive at 240Hz.
            if not self._clear_brush:
                self._clear_brush = gdi32.CreateSolidBrush(rgb(255, 0, 255))
            user32.FillRect(hdc, ctypes.byref(ps.rcPaint), self._clear_brush)

            debug = self.get_debug() if self.get_debug else OverlayDebugData()
            # Israel mode draws on its own, with or without the FOV overlay on.
            # The movement line can show on its own while it eases in or out.
            line_live = abs(self._line_x) + abs(self._line_y) > 2.0
            if self.get_enabled() or debug.israel_mode or line_live:
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                cx, cy = width // 2, height // 2
                radius = self.get_radius()

                # Israel mode - only show spinning Star of David
                if debug.israel_mode:
                    angle = (time.time() * 3.0) % (2 * math.pi)  # Smooth rotation
                    pts = _star_of_david_points(cx, cy, STAR_RADIUS, angle)
                    star_pen = gdi32.CreatePen(PS_SOLID, 2, rgb(0, 120, 255))  # Blue
                    old_pen = gdi32.SelectObject(hdc, star_pen)
                    # Draw first triangle
                    gdi32.MoveToEx(hdc, pts[0][0], pts[0][1], None)
                    gdi32.LineTo(hdc, pts[1][0], pts[1][1])
                    gdi32.LineTo(hdc, pts[2][0], pts[2][1])
                    gdi32.LineTo(hdc, pts[0][0], pts[0][1])
                    # Draw second triangle
                    gdi32.MoveToEx(hdc, pts[3][0], pts[3][1], None)
                    gdi32.LineTo(hdc, pts[4][0], pts[4][1])
                    gdi32.LineTo(hdc, pts[5][0], pts[5][1])
                    gdi32.LineTo(hdc, pts[3][0], pts[3][1])
                    gdi32.SelectObject(hdc, old_pen)
                    gdi32.DeleteObject(star_pen)
                else:
                    if self.get_enabled():
                        self._paint_glass_fov(hdc, cx, cy)
                    elif line_live:
                        self._paint_motion_line(hdc, cx, cy)

                    if self.get_enabled():
                        for box in debug.boxes:
                            bx = int(cx + box.x)
                            by = int(cy + box.y)
                            left = int(bx - box.w * 0.5)
                            top = int(by - box.h * 0.5)
                            right = int(bx + box.w * 0.5)
                            bottom = int(by + box.h * 0.5)
                            if box.kind == "ally":
                                box_pen = self._pen("ally", rgb(120, 190, 255))
                            else:
                                box_pen = self._pen("enemy", rgb(255, 150, 150))
                            old_box_pen = gdi32.SelectObject(hdc, box_pen)
                            old_box_brush = gdi32.SelectObject(
                                hdc, gdi32.GetStockObject(NULL_BRUSH)
                            )
                            gdi32.Rectangle(hdc, left, top, right, bottom)
                            gdi32.SelectObject(hdc, old_box_brush)
                            gdi32.SelectObject(hdc, old_box_pen)

                    if self.get_enabled() and self._font and (debug.detect_hz > 0 or debug.state):
                        old_font = gdi32.SelectObject(hdc, self._font)
                        gdi32.SetBkMode(hdc, TRANSPARENT)
                        gdi32.SetTextAlign(hdc, TA_CENTER | TA_TOP)
                        gdi32.SetTextColor(hdc, rgb(186, 220, 255))
                        text = f"{debug.state}  {debug.detect_hz:.0f}hz  {debug.infer_ms:.1f}ms"
                        shown = int(self._vis_radius or radius)
                        gdi32.TextOutW(hdc, cx, cy + shown + 8, text, len(text))
                        gdi32.SelectObject(hdc, old_font)

            user32.EndPaint(hwnd, ctypes.byref(ps))
            return 0
        if message == WM_CLOSE:
            user32.DestroyWindow(hwnd)
            return 0
        if message == WM_DESTROY:
            self._drop_pens()
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _pen(self, key: str, color: int) -> int:
        pen = self._pens.get(key)
        if not pen:
            pen = gdi32.CreatePen(PS_SOLID, 1, color)
            self._pens[key] = pen
        return pen

    def _drop_pens(self) -> None:
        for pen in self._pens.values():
            if pen:
                gdi32.DeleteObject(pen)
        self._pens.clear()
        if self._clear_brush:
            gdi32.DeleteObject(self._clear_brush)
            self._clear_brush = None

    def _step_anim(self, debug: OverlayDebugData) -> bool:
        now = time.perf_counter()
        dt = min(0.05, max(0.0, now - self._anim_t))
        self._anim_t = now
        radius = float(self.get_radius())
        if self._vis_radius is None:
            self._vis_radius = radius
        self._vis_radius = _approach(self._vis_radius, radius, 14.0, dt)
        if debug.show_movement_line and debug.target is not None and not debug.israel_mode:
            gx, gy = float(debug.target[0]), float(debug.target[1])
        else:
            gx = gy = 0.0
        self._line_x = _approach(self._line_x, gx, 11.0, dt)
        self._line_y = _approach(self._line_y, gy, 11.0, dt)
        return (
            abs(self._vis_radius - radius) > 0.35
            or abs(self._line_x) + abs(self._line_y) > 1.5
        )

    def _paint_glass_fov(self, hdc, cx: int, cy: int) -> None:
        radius = max(1, int(round(self._vis_radius or self.get_radius())))
        old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(NULL_BRUSH))
        rings = (
            (radius + 2, "ring_out"),
            (radius, "ring_mid"),
            (radius - 3, "ring_in"),
        )
        for r, key in rings:
            if r < 2:
                continue
            pen = self._pen(key, _RING[key])
            old = gdi32.SelectObject(hdc, pen)
            gdi32.Ellipse(hdc, cx - r, cy - r, cx + r, cy + r)
            gdi32.SelectObject(hdc, old)
        self._glass_arc(hdc, cx, cy, max(2, radius - 1), 0.6, 2.4, "shine", rgb(255, 255, 255))
        self._glass_arc(hdc, cx, cy, radius + 2, 0.9, 2.1, "shine_soft", rgb(190, 225, 255))
        gdi32.SelectObject(hdc, old_brush)
        self._paint_motion_line(hdc, cx, cy)

    def _glass_arc(self, hdc, cx: int, cy: int, radius: int, a0: float, a1: float, key: str, color: int) -> None:
        pen = self._pen(key, color)
        old = gdi32.SelectObject(hdc, pen)
        x0 = int(cx + radius * math.cos(a0))
        y0 = int(cy - radius * math.sin(a0))
        x1 = int(cx + radius * math.cos(a1))
        y1 = int(cy - radius * math.sin(a1))
        gdi32.Arc(hdc, cx - radius, cy - radius, cx + radius, cy + radius, x0, y0, x1, y1)
        gdi32.SelectObject(hdc, old)

    def _paint_motion_line(self, hdc, cx: int, cy: int) -> None:
        ex = self._line_x
        ey = self._line_y
        length = math.hypot(ex, ey)
        if length < 2.0:
            return
        steps = 8
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            color = _LINE[min(len(_LINE) - 1, i * len(_LINE) // steps)]
            pen = self._pen(f"line{i}", color)
            old = gdi32.SelectObject(hdc, pen)
            gdi32.MoveToEx(hdc, int(cx + ex * t0), int(cy + ey * t0), None)
            gdi32.LineTo(hdc, int(cx + ex * t1), int(cy + ey * t1))
            gdi32.SelectObject(hdc, old)

        phase = (time.perf_counter() * 0.85) % 1.0
        bx = int(cx + ex * phase)
        by = int(cy + ey * phase)
        bead = self._pen("bead", rgb(255, 255, 255))
        old = gdi32.SelectObject(hdc, bead)
        brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(NULL_BRUSH))
        gdi32.Ellipse(hdc, bx - 3, by - 3, bx + 3, by + 3)
        end_x = int(cx + ex)
        end_y = int(cy + ey)
        gdi32.Ellipse(hdc, end_x - 4, end_y - 4, end_x + 4, end_y + 4)
        gdi32.SelectObject(hdc, brush)
        gdi32.SelectObject(hdc, old)

    def destroy(self) -> None:
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1)
