"""DXGI-only low-latency screen capture."""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import time
from dataclasses import dataclass

import dxcam
import numpy as np


@dataclass(slots=True)
class Frame:
    rgb: np.ndarray
    left: int
    top: int
    seq: int
    captured_at: float


def _monitor_rect(index: int) -> tuple[int, int, int, int]:
    rects: list[tuple[int, int, int, int]] = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.wintypes.RECT),
        ctypes.c_ssize_t,
    )

    def callback(_monitor, _dc, rect, _data):
        r = rect.contents
        rects.append((r.left, r.top, r.right, r.bottom))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(
        None, None, callback_type(callback), 0
    )
    if not rects:
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
        return 0, 0, width, height
    return rects[min(max(0, int(index)), len(rects) - 1)]


def _display_refresh_hz(default: int = 144) -> int:
    dc = ctypes.windll.user32.GetDC(None)
    try:
        hz = int(ctypes.windll.gdi32.GetDeviceCaps(dc, 116))  # VREFRESH
        return hz if 30 <= hz <= 1000 else default
    finally:
        if dc:
            ctypes.windll.user32.ReleaseDC(None, dc)


class Capture:
    """Continuous DXGI capture with an owned latest-frame ring."""

    METHODS = ("dxcam",)

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
        del method, window_title
        self.size = max(160, int(size))
        self.color = color.upper() if color.upper() in ("RGB", "BGR") else "BGR"
        self.fps_cap = max(0, int(fps_cap))
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self.monitor = max(0, int(monitor))
        self.method = "dxcam"
        self.active_method = "dxcam"
        self.fell_back = False

        self.screen_w = 0
        self.screen_h = 0
        self.center_x = 0
        self.center_y = 0
        self.publish_hz = 0.0
        # The FPS the capture is actually paced at (set in start()). Exposed so
        # the UI can show the real ceiling and whether fps_cap is overriding it.
        self.active_target_fps = 0

        self._screen_left = 0
        self._screen_top = 0
        self._bounds = (0, 0, self.size, self.size)
        self._frame: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._thread: threading.Thread | None = None
        self._pub_n = 0
        self._pub_t0 = time.perf_counter()
        self._error: BaseException | None = None
        # The inference thread may retain a frame while DXGI publishes newer
        # ones. If the ring is too small the producer wraps around and
        # overwrites a buffer the detector has not read yet, silently dropping
        # a detect -- the classic cause of the Hz dipping below the refresh
        # rate under load. 24 owned slots hold ~166 ms at 144 Hz (60+ ms even
        # at 360 Hz), leaving ample headroom for inference bursts.
        self._ring = max(16, 24)
        self._buffers = [
            np.empty((self.size, self.size, 3), dtype=np.uint8) for _ in range(self._ring)
        ]
        self._write_index = 0

    def start(self) -> None:
        left, top, right, bottom = _monitor_rect(self.monitor)
        self._screen_left, self._screen_top = left, top
        self.screen_w, self.screen_h = right - left, bottom - top
        self.center_x = left + self.screen_w // 2 + self.offset_x
        self.center_y = top + self.screen_h // 2 + self.offset_y

        half = self.size // 2
        # DXCam regions are relative to the selected output.
        local_left = self.center_x - left - half
        local_top = self.center_y - top - half
        local_left = max(0, min(local_left, self.screen_w - self.size))
        local_top = max(0, min(local_top, self.screen_h - self.size))
        self._bounds = (
            local_left,
            local_top,
            local_left + self.size,
            local_top + self.size,
        )

        self._frame = None
        self._seq = 0
        self._error = None
        self._pub_n = 0
        self._pub_t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._dxgi_loop, name="dxgi-capture", daemon=True
        )
        self._thread.start()

        # Fail startup explicitly instead of silently switching capture APIs.
        deadline = time.perf_counter() + 2.0
        while self._frame is None and self._error is None:
            if time.perf_counter() >= deadline:
                raise RuntimeError("DXGI capture timed out")
            time.sleep(0.001)
        if self._error is not None:
            raise RuntimeError(f"DXGI capture failed: {self._error}")

    def stop(self) -> None:
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def latest(self) -> Frame | None:
        # Assignment is atomic under CPython; avoiding a lock keeps the capture
        # and inference threads from blocking each other.
        return self._frame

    def wait_for_frame(self, after_seq: int, timeout: float = 0.02) -> Frame | None:
        """Wait until DXGI publishes a frame newer than ``after_seq``."""
        with self._frame_ready:
            frame = self._frame
            if (
                not self._stop.is_set()
                and (frame is None or frame.seq == after_seq)
            ):
                self._frame_ready.wait(timeout)
            return self._frame

    def _publish(self, pixels: np.ndarray) -> None:
        captured_at = time.perf_counter()
        owned = self._buffers[self._write_index]
        self._write_index = (self._write_index + 1) % len(self._buffers)
        if owned.shape != pixels.shape:
            owned = np.empty_like(pixels)
            self._buffers[self._write_index - 1] = owned
        np.copyto(owned, pixels)
        self._seq += 1
        with self._frame_ready:
            self._frame = Frame(
                rgb=owned,
                left=self._screen_left + self._bounds[0],
                top=self._screen_top + self._bounds[1],
                seq=self._seq,
                captured_at=captured_at,
            )
            self._frame_ready.notify_all()
        self._pub_n += 1
        now = time.perf_counter()
        elapsed = now - self._pub_t0
        if elapsed >= 0.25:
            self.publish_hz = self._pub_n / elapsed
            self._pub_n = 0
            self._pub_t0 = now

    def _dxgi_loop(self) -> None:
        try:
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 1
            )
        except Exception:
            pass
        camera = None
        started = False
        try:
            camera = dxcam.create(
                output_idx=self.monitor,
                output_color=self.color,
                max_buffer_len=2,
            )
            if camera is None:
                raise RuntimeError("dxcam.create returned no camera")

            # Processing DXGI duplicates above the monitor refresh rate only
            # steals GPU time from inference. An explicit cap can still
            # override this for diagnostics. fps_cap==0 tracks the live display
            # refresh, so a 144/240/360 Hz monitor is used automatically. The
            # rate is read once here; if the user changes the monitor refresh
            # while running, a Start/Stop re-reads it.
            target_fps = self.fps_cap or _display_refresh_hz()
            target_fps = max(30, min(1000, target_fps))
            self.active_target_fps = int(target_fps)
            camera.start(
                region=self._bounds,
                target_fps=target_fps,
                video_mode=True,
            )
            started = True

            while not self._stop.is_set():
                frame = camera.get_latest_frame()
                if frame is not None:
                    self._publish(frame)
                else:
                    time.sleep(0.0005)
        except BaseException as exc:
            self._error = exc
        finally:
            if camera is not None:
                if started:
                    try:
                        camera.stop()
                    except Exception:
                        pass
                try:
                    camera.release()
                except Exception:
                    pass
