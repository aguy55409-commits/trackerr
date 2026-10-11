"""High-speed capture — multiple backends for maximum Hz."""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np

# Optional fast backends
try:
    import dxcam
except Exception:
    dxcam = None

try:
    from windows_capture import WindowsCapture
    HAS_WGC = True
except Exception:
    WindowsCapture = None
    HAS_WGC = False

# Windows GDI for BitBlt
try:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    HAS_GDI = True
except Exception:
    HAS_GDI = False


@dataclass
class Frame:
    rgb: np.ndarray
    left: int
    top: int
    seq: int


# Use Windows high-res timer
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass


def display_refresh_hz(default: int = 144) -> int:
    """Refresh rate of the primary display.

    This is the real ceiling on capture: the desktop only produces a new image
    once per refresh, so anything captured faster is a duplicate of a frame
    already seen.
    """
    class _DEVMODE(ctypes.Structure):
        _fields_ = [
            ("dmDeviceName", ctypes.c_wchar * 32),
            ("dmSpecVersion", ctypes.c_ushort),
            ("dmDriverVersion", ctypes.c_ushort),
            ("dmSize", ctypes.c_ushort),
            ("dmDriverExtra", ctypes.c_ushort),
            ("dmFields", ctypes.c_ulong),
            ("_pad0", ctypes.c_short * 4),
            ("_pad1", ctypes.c_short * 10),
            ("dmFormName", ctypes.c_wchar * 32),
            ("dmLogPixels", ctypes.c_ushort),
            ("dmBitsPerPel", ctypes.c_ulong),
            ("dmPelsWidth", ctypes.c_ulong),
            ("dmPelsHeight", ctypes.c_ulong),
            ("dmDisplayFlags", ctypes.c_ulong),
            ("dmDisplayFrequency", ctypes.c_ulong),
            ("_tail", ctypes.c_ulong * 8),
        ]

    try:
        dm = _DEVMODE()
        dm.dmSize = ctypes.sizeof(_DEVMODE)
        if ctypes.windll.user32.EnumDisplaySettingsW(None, -1, ctypes.byref(dm)):
            hz = int(dm.dmDisplayFrequency)
            if 20 < hz < 1000:
                return hz
    except Exception:
        pass
    return default


class Capture:
    """Multi-backend screen capture with automatic fallback."""
    
    METHODS = ("dxcam", "wgc", "bitblt", "mss")
    
    def __init__(
        self,
        size: int = 320,
        method: str = "dxcam",
        color: str = "BGR",
        fps_cap: int = 0,
        offset_x: int = 0,
        offset_y: int = 0,
        monitor: int = 0,
        window_title: str = "",
    ) -> None:
        self.size = max(160, int(size))
        self.method = method if method in self.METHODS else "dxcam"
        self.color = color.upper() if color.upper() in ("RGB", "BGR") else "BGR"
        self.fps_cap = max(0, int(fps_cap))
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self.monitor = max(0, int(monitor))
        self.window_title = str(window_title or "")
        self.fell_back = False
        self.active_method = self.method
        self.screen_w = 0
        self.screen_h = 0
        self.center_x = 0
        self.center_y = 0
        self.publish_hz = 0.0
        self._region = {"left": 0, "top": 0, "width": self.size, "height": self.size}
        self._bounds: tuple[int, int, int, int] = (0, 0, self.size, self.size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        # Triple buffer for async capture
        self._bufs = [
            np.empty((self.size, self.size, 3), dtype=np.uint8),
            np.empty((self.size, self.size, 3), dtype=np.uint8),
            np.empty((self.size, self.size, 3), dtype=np.uint8),
        ]
        self._wi = 0
        self._pub_n = 0
        self._pub_t0 = time.perf_counter()

    def start(self) -> None:
        with mss.mss() as sct:
            # monitors[0] is the union of all screens, so real ones start at 1.
            idx = min(self.monitor + 1, len(sct.monitors) - 1)
            mon = sct.monitors[max(1, idx)]
        self.screen_w = int(mon["width"])
        self.screen_h = int(mon["height"])
        left = int(mon["left"])
        top = int(mon["top"])
        self.center_x = left + self.screen_w // 2 + self.offset_x
        self.center_y = top + self.screen_h // 2 + self.offset_y
        half = self.size // 2
        self._region = {
            "left": self.center_x - half,
            "top": self.center_y - half,
            "width": self.size,
            "height": self.size,
        }
        r = self._region
        self._bounds = (r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"])
        self._stop.clear()
        self._pub_n = 0
        self._pub_t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def _publish(self, rgb: np.ndarray) -> None:
        self._seq += 1
        with self._lock:
            self._frame = Frame(
                rgb=rgb,
                left=self._region["left"],
                top=self._region["top"],
                seq=self._seq,
            )
        self._pub_n += 1
        now = time.perf_counter()
        if now - self._pub_t0 >= 0.25:
            self.publish_hz = self._pub_n / (now - self._pub_t0)
            self._pub_n = 0
            self._pub_t0 = now

    def _next_buf(self) -> np.ndarray:
        buf = self._bufs[self._wi]
        self._wi = (self._wi + 1) % len(self._bufs)
        return buf

    def _loop(self) -> None:
        """Try methods in order of preference."""
        methods = [
            ("dxcam", dxcam is not None, self._dxcam_loop),
            ("wgc", HAS_WGC, self._wgc_loop),
            ("bitblt", HAS_GDI, self._bitblt_loop),
            ("mss", True, self._mss_loop),
        ]
        chosen = {name: (avail, fn) for name, avail, fn in methods}

        # Honour the requested method first, then fall through the chain.
        pick = chosen.get(self.method)
        if pick and pick[0]:
            try:
                self.active_method = self.method
                self.fell_back = False
                pick[1]()
                return
            except Exception:
                pass
        
        # Fallback chain
        for name, available, fn in methods:
            if available:
                try:
                    self.active_method = name
                    self.fell_back = name != self.method
                    fn()
                    return
                except Exception:
                    continue
        
        # Last resort
        self.active_method = "mss"
        self.fell_back = True
        self._mss_loop()

    def _dxcam_loop(self) -> None:
        """DXCam DXGI desktop duplication.

        Blocks on dxcam's new-frame event rather than polling. Polling with
        grab(region=...) re-crops the existing buffer and returns a frame every
        call, so the same pixels were published several times over and the
        detector spent inference on images it had already seen.
        """
        assert dxcam is not None
        cam = dxcam.create(output_idx=0, output_color=self.color, max_buffer_len=16)
        if cam is None:
            raise RuntimeError("dxcam create failed")

        bounds = self._bounds
        started = False
        # No point asking for more than the display can present; the surplus is
        # duplicates. A little headroom absorbs jitter.
        target = min(480, int(display_refresh_hz() * 1.5))
        if self.fps_cap:
            target = min(target, self.fps_cap)
        min_dt = 1.0 / self.fps_cap if self.fps_cap else 0.0
        last_pub = 0.0

        try:
            try:
                cam.start(region=bounds, target_fps=target, video_mode=True)
            except TypeError:
                try:
                    cam.start(region=bounds, video_mode=True)
                except TypeError:
                    cam.start(region=bounds)
            started = True

            while not self._stop.is_set():
                frame = None
                try:
                    # Waits for a genuinely new frame.
                    frame = cam.get_latest_frame()
                except Exception:
                    time.sleep(0.0005)
                    continue

                if frame is None:
                    continue

                if min_dt:
                    now = time.perf_counter()
                    if now - last_pub < min_dt:
                        continue
                    last_pub = now

                dest = self._next_buf()
                if dest.shape != frame.shape:
                    dest = np.empty(frame.shape, dtype=np.uint8)
                    self._bufs[self._wi - 1] = dest
                np.copyto(dest, frame)
                self._publish(dest)

        finally:
            if started:
                try:
                    cam.stop()
                except Exception:
                    pass
            try:
                cam.release()
            except Exception:
                pass

    def _wgc_loop(self) -> None:
        """Windows.Graphics.Capture.

        Callback driven rather than polled, so every delivered frame is new.
        Unlike desktop duplication it can target a single window, which is the
        case that matters when DDA is unavailable or returns black.
        """
        if not HAS_WGC:
            raise RuntimeError("windows-capture not installed")

        left, top, right, bottom = self._bounds
        width = right - left
        height = bottom - top
        err: list[BaseException] = []

        window = (self.window_title or "").strip()
        kwargs: dict = {"cursor_capture": False, "draw_border": False}
        if window:
            kwargs["window_name"] = window
        else:
            kwargs["monitor_index"] = self.monitor + 1
        if self.fps_cap:
            # WGC takes a minimum gap between frames, in 100ns ticks.
            kwargs["minimum_update_interval"] = int(10_000_000 / self.fps_cap)
        cap = WindowsCapture(**kwargs)

        @cap.event
        def on_frame_arrived(frame, capture_control):  # noqa: ANN001
            if self._stop.is_set():
                capture_control.stop()
                return
            try:
                buf = frame.frame_buffer
                if window:
                    # A window frame is in window coordinates, so the screen
                    # rect means nothing here; take the centre of the window.
                    fh, fw = buf.shape[0], buf.shape[1]
                    x0 = max(0, (fw - width) // 2 + self.offset_x)
                    y0 = max(0, (fh - height) // 2 + self.offset_y)
                    bgra = buf[y0:y0 + height, x0:x0 + width]
                    if bgra.shape[0] != height or bgra.shape[1] != width:
                        return  # window smaller than the capture size
                else:
                    # Slice a view of the full-monitor buffer rather than
                    # calling frame.crop(), which copies the whole surface.
                    bgra = buf[top:bottom, left:right]
                dest = self._next_buf()
                if dest.shape[0] != height or dest.shape[1] != width:
                    dest = np.empty((height, width, 3), dtype=np.uint8)
                    self._bufs[self._wi - 1] = dest
                code = (
                    cv2.COLOR_BGRA2BGR if self.color == "BGR" else cv2.COLOR_BGRA2RGB
                )
                cv2.cvtColor(bgra, code, dst=dest)
                self._publish(dest)
            except Exception as exc:  # a raise here would kill WGC's thread
                err.append(exc)

        @cap.event
        def on_closed():
            self._stop.set()

        control = cap.start_free_threaded()
        try:
            while not self._stop.is_set():
                if err:
                    raise err[0]
                time.sleep(0.005)
        finally:
            try:
                control.stop()
            except Exception:
                pass

    def _bitblt_loop(self) -> None:
        """BitBlt GDI capture — recreates resources periodically to prevent leaks."""
        assert HAS_GDI
        
        SRCCOPY = 0x00CC0020
        DIB_RGB_COLORS = 0
        BI_RGB = 0
        
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.wintypes.DWORD),
                ("biWidth", ctypes.wintypes.LONG),
                ("biHeight", ctypes.wintypes.LONG),
                ("biPlanes", ctypes.wintypes.WORD),
                ("biBitCount", ctypes.wintypes.WORD),
                ("biCompression", ctypes.wintypes.DWORD),
                ("biSizeImage", ctypes.wintypes.DWORD),
                ("biXPelsPerMeter", ctypes.wintypes.LONG),
                ("biYPelsPerMeter", ctypes.wintypes.LONG),
                ("biClrUsed", ctypes.wintypes.DWORD),
                ("biClrImportant", ctypes.wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", ctypes.wintypes.DWORD * 3),
            ]
        
        left, top, right, bottom = self._bounds
        width = right - left
        height = bottom - top
        buffer_size = width * height * 4
        
        # Recreate GDI resources every N frames to prevent resource exhaustion
        REFRESH_INTERVAL = 5000
        frame_count = 0
        
        hdc_screen = None
        hdc_mem = None
        hbm = None
        buffer = None
        bmi = None
        
        def create_resources():
            nonlocal hdc_screen, hdc_mem, hbm, buffer, bmi
            # Clean up old resources first
            cleanup_resources()

            hdc_screen = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = -height  # Top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB

            # A DIB section hands back a pointer to the bitmap's own pixels, so
            # BitBlt writes where we can already read. CreateCompatibleBitmap
            # needs a GetDIBits conversion copy of the whole region afterwards,
            # which cost more than a refresh interval and halved the rate.
            bits = ctypes.c_void_p()
            gdi32.CreateDIBSection.restype = ctypes.wintypes.HBITMAP
            hbm = gdi32.CreateDIBSection(
                hdc_screen, ctypes.byref(bmi), DIB_RGB_COLORS,
                ctypes.byref(bits), None, 0,
            )
            if not hbm or not bits:
                raise RuntimeError("CreateDIBSection failed")
            gdi32.SelectObject(hdc_mem, hbm)
            buffer = ctypes.cast(
                bits, ctypes.POINTER(ctypes.c_ubyte * buffer_size)
            ).contents
        
        def cleanup_resources():
            nonlocal hdc_screen, hdc_mem, hbm
            if hbm:
                gdi32.DeleteObject(hbm)
                hbm = None
            if hdc_mem:
                gdi32.DeleteDC(hdc_mem)
                hdc_mem = None
            if hdc_screen:
                user32.ReleaseDC(0, hdc_screen)
                hdc_screen = None
        
        try:
            create_resources()
            
            while not self._stop.is_set():
                # Refresh resources periodically
                frame_count += 1
                if frame_count >= REFRESH_INTERVAL:
                    frame_count = 0
                    create_resources()
                
                # BitBlt writes straight into the DIB section's pixels, so
                # there is no GetDIBits copy to follow it.
                gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, left, top, SRCCOPY)
                gdi32.GdiFlush()

                img = np.ctypeslib.as_array(buffer).reshape(height, width, 4)
                dest = self._next_buf()
                if dest.shape[0] != height or dest.shape[1] != width:
                    dest = np.empty((height, width, 3), dtype=np.uint8)
                    self._bufs[self._wi - 1] = dest
                
                if self.color == "BGR":
                    cv2.cvtColor(img, cv2.COLOR_BGRA2BGR, dst=dest)
                else:
                    cv2.cvtColor(img, cv2.COLOR_BGRA2RGB, dst=dest)
                
                self._publish(dest)
                
        finally:
            cleanup_resources()

    def _mss_loop(self) -> None:
        """Pure MSS — recreates context periodically to prevent slowdowns."""
        # Recreate MSS context every N frames to prevent memory/performance issues
        REFRESH_INTERVAL = 3000
        frame_count = 0
        sct = None
        
        def create_sct():
            nonlocal sct
            if sct:
                try:
                    sct.close()
                except Exception:
                    pass
            sct = mss.mss()
        
        try:
            create_sct()
            
            while not self._stop.is_set():
                # Refresh MSS context periodically
                frame_count += 1
                if frame_count >= REFRESH_INTERVAL:
                    frame_count = 0
                    create_sct()
                
                try:
                    shot = sct.grab(self._region)
                    dest = self._next_buf()
                    h, w = shot.height, shot.width
                    if dest.shape[0] != h or dest.shape[1] != w:
                        dest = np.empty((h, w, 3), dtype=np.uint8)
                        self._bufs[self._wi - 1] = dest
                    bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(h, w, 4)
                    if self.color == "BGR":
                        cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=dest)
                    else:
                        cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=dest)
                    self._publish(dest)
                except Exception:
                    time.sleep(0.001)
                    
        finally:
            if sct:
                try:
                    sct.close()
                except Exception:
                    pass
