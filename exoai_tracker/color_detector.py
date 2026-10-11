from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass

import cv2
import dxcam
import numpy as np

from exoai_tracker.dxgi_capture import _display_refresh_hz, _monitor_rect


def _lower_thread_priority() -> None:
    try:
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), -1
        )
    except Exception:
        pass


@dataclass
class ColorTarget:
    x: float
    y: float
    confidence: float
    width: float
    height: float
    area: int


# Common game highlight colors (HSV ranges)
COLOR_PRESETS = {
    "purple": {"lower": (125, 80, 80), "upper": (155, 255, 255)},   # Fortnite purple
    "yellow": {"lower": (20, 100, 100), "upper": (35, 255, 255)},   # Yellow outlines
    "red": {"lower": (0, 120, 100), "upper": (10, 255, 255)},       # Red enemy
    "red2": {"lower": (170, 120, 100), "upper": (180, 255, 255)},   # Red wraparound
    "orange": {"lower": (10, 100, 100), "upper": (25, 255, 255)},   # Orange highlight
    "green": {"lower": (35, 80, 80), "upper": (85, 255, 255)},      # Green (teammate)
    "white": {"lower": (0, 0, 200), "upper": (180, 30, 255)},       # White/bright
    "cyan": {"lower": (85, 100, 100), "upper": (100, 255, 255)},    # Cyan outline
}


class ColorDetector:
    """Fast color-based target detection - no neural network needed."""
    
    def __init__(
        self,
        capture_size: int = 320,
        color_preset: str = "purple",
        min_area: int = 15,
        max_area: int = 8000,
        capture_method: str = "dxcam",
        monitor: int = 0,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        self._capture_method = "dxcam"
        self._capture_size = capture_size
        self._min_area = min_area
        self._max_area = max_area
        self._debug_flag = False
        self._fov_radius_sq = 0
        
        # Color range in HSV
        preset = COLOR_PRESETS.get(color_preset, COLOR_PRESETS["purple"])
        self._lower = np.array(preset["lower"], dtype=np.uint8)
        self._upper = np.array(preset["upper"], dtype=np.uint8)
        self._color_preset = color_preset
        
        # Screen info
        self._monitor_index = max(0, int(monitor))
        left, top, right, bottom = _monitor_rect(self._monitor_index)
        self.screen_left = left
        self.screen_top = top
        self.screen_w = right - left
        self.screen_h = bottom - top
        self.center_x = self.screen_left + self.screen_w // 2 + int(offset_x)
        self.center_y = self.screen_top + self.screen_h // 2 + int(offset_y)
        self._region = self._build_region()
        
        # Buffers
        h, w = self._region["height"], self._region["width"]
        self._bufs = [
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
        ]
        self._write_i = 0
        self._hsv_buf = np.empty((h, w, 3), dtype=np.uint8)
        self._mask_buf = np.empty((h, w), dtype=np.uint8)
        
        # Threading
        self._latest_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._capture_stop = threading.Event()
        self._capture_error: Exception | None = None
        self._capture_fell_back = False
        self._frame_ready = threading.Event()
        self._cached_detect: tuple[list, np.ndarray | None, dict | None] | None = None
        self.frame_id = 0
        self._capture_seq = 0
        self._last_infer_seq = -1
        
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="carbon-color-capture", daemon=True
        )
        self._capture_thread.start()
    
    def _build_region(self) -> dict[str, int]:
        half = self._capture_size // 2
        left = max(self.screen_left, self.center_x - half)
        top = max(self.screen_top, self.center_y - half)
        width = min(self._capture_size, self.screen_left + self.screen_w - left)
        height = min(self._capture_size, self.screen_top + self.screen_h - top)
        return {"left": left, "top": top, "width": width, "height": height}
    
    def _capture_loop(self) -> None:
        _lower_thread_priority()
        try:
            self._capture_dxgi_loop()
        except Exception as exc:
            self._capture_error = exc
    
    def _capture_dxgi_loop(self) -> None:
        camera = None
        try:
            camera = dxcam.create(
                output_idx=self._monitor_index,
                output_color="RGB",
                max_buffer_len=4,
            )
        except Exception:
            raise RuntimeError("DXGI capture blocked")
        if camera is None:
            raise RuntimeError("DXGI camera creation failed")
        
        region = self._region
        local_left = region["left"] - self.screen_left
        local_top = region["top"] - self.screen_top
        bounds = (
            local_left,
            local_top,
            local_left + region["width"],
            local_top + region["height"],
        )
        started = False
        try:
            camera.start(
                region=bounds,
                target_fps=_display_refresh_hz(),
                video_mode=True,
            )
            started = True
            while not self._capture_stop.is_set():
                frame = camera.get_latest_frame()
                if frame is None:
                    continue
                self._publish_frame(frame)
        finally:
            if camera:
                if started:
                    try:
                        camera.stop()
                    except Exception:
                        pass
                try:
                    camera.release()
                except Exception:
                    pass
    
    def _publish_ready(self, frame: np.ndarray) -> None:
        self._write_i ^= 1
        self._capture_seq += 1
        with self._latest_lock:
            self._latest_frame = frame
        self._frame_ready.set()
    
    def _publish_frame(self, source: np.ndarray) -> None:
        dest = self._bufs[self._write_i]
        if dest.shape != source.shape:
            dest = np.empty(source.shape, dtype=np.uint8)
            self._bufs[self._write_i] = dest
        np.copyto(dest, source)
        self._publish_ready(dest)
    
    def set_color(self, preset: str) -> None:
        """Change the target color."""
        if preset in COLOR_PRESETS:
            p = COLOR_PRESETS[preset]
            self._lower = np.array(p["lower"], dtype=np.uint8)
            self._upper = np.array(p["upper"], dtype=np.uint8)
            self._color_preset = preset
    
    def set_fov_radius(self, fov_radius: int) -> None:
        self._fov_radius_sq = fov_radius * fov_radius
    
    def set_debug(self, enabled: bool) -> None:
        self._debug_flag = enabled
    
    def debug_enabled(self) -> bool:
        return self._debug_flag
    
    def capture_fell_back(self) -> bool:
        return self._capture_fell_back

    @property
    def capture_seq(self) -> int:
        return self._capture_seq
    
    def close(self) -> None:
        self._capture_stop.set()
        self._frame_ready.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1)
    
    def detect(
        self,
        fov_radius: int,
        *,
        wait_for_new: bool = False,
    ) -> tuple[list[ColorTarget], np.ndarray | None, dict | None]:
        if fov_radius * fov_radius != self._fov_radius_sq:
            self.set_fov_radius(fov_radius)
        
        # Get frame
        with self._latest_lock:
            frame = self._latest_frame
            capture_seq = self._capture_seq
        
        if frame is None:
            if self._cached_detect:
                return self._cached_detect
            return [], None, None
        
        if capture_seq == self._last_infer_seq:
            if wait_for_new:
                return [], None, None
            if self._cached_detect:
                return self._cached_detect
        
        self._last_infer_seq = capture_seq
        self.frame_id += 1
        
        # Color detection (very fast)
        cv2.cvtColor(frame, cv2.COLOR_RGB2HSV, dst=self._hsv_buf)
        cv2.inRange(self._hsv_buf, self._lower, self._upper, dst=self._mask_buf)
        
        # Find contours
        contours, _ = cv2.findContours(
            self._mask_buf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        targets: list[ColorTarget] = []
        region_left = self._region["left"]
        region_top = self._region["top"]
        cx0 = self.center_x
        cy0 = self.center_y
        fov_sq = self._fov_radius_sq
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area or area > self._max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            # Center of bounding box in screen coords
            tx = region_left + x + w / 2
            ty = region_top + y + h / 2
            
            # Check if in FOV
            dist_sq = (tx - cx0) ** 2 + (ty - cy0) ** 2
            if fov_sq > 0 and dist_sq > fov_sq:
                continue
            
            # Confidence based on area and aspect ratio
            aspect = h / max(w, 1)
            conf = min(1.0, area / 500) * min(1.0, aspect / 1.5)
            
            targets.append(ColorTarget(
                x=tx,
                y=ty,
                confidence=conf,
                width=float(w),
                height=float(h),
                area=int(area),
            ))
        
        # Sort by distance to center
        targets.sort(key=lambda t: (t.x - cx0) ** 2 + (t.y - cy0) ** 2)
        targets = targets[:6]  # Keep closest 6
        
        # Debug frame
        debug_frame = None
        if self._debug_flag:
            debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Draw mask overlay
            mask_color = np.zeros_like(debug_frame)
            mask_color[self._mask_buf > 0] = (0, 255, 0)
            debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_color, 0.3, 0)
            # Draw targets
            for t in targets:
                lx = int(t.x - region_left)
                ly = int(t.y - region_top)
                cv2.circle(debug_frame, (lx, ly), 5, (0, 0, 255), -1)
        
        result = (targets, debug_frame, self._region)
        self._cached_detect = result
        return result
