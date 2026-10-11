from __future__ import annotations

import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import webview

# Set drag region selector for frameless window
webview.DRAG_REGION_SELECTOR = '.titlebar'

from exoai_tracker.binds import PRESET_BINDS, format_bind, normalize_bind, wait_for_bind
from exoai_tracker.config import AppConfig, save_config


def ui_index() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        return base / "exoai_tracker" / "ui" / "index.html"
    return Path(__file__).resolve().parent / "ui" / "index.html"


class SettingsWindow:
    """WebView2 (Edge Chromium) control panel."""

    def __init__(
        self,
        config: AppConfig,
        on_apply: Callable[[AppConfig], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        is_running: Callable[[], bool],
    ) -> None:
        self.config = config
        self.on_apply = on_apply
        self.on_start = on_start
        self.on_stop = on_stop
        self.is_running = is_running
        self._lock = threading.Lock()
        self._stats = {
            "state": "IDLE",
            "detect_hz": 0.0,
            "infer_ms": 0.0,
            "target_conf": None,
            "detail": "Ready — press Start Tracker",
            "mouse_label": "Razer Synapse",
            "running": False,
        }
        self._window: webview.Window | None = None

    def set_stats(
        self,
        *,
        state: str,
        detect_hz: float,
        infer_ms: float,
        target_conf: float | None,
        detail: str,
        mouse_label: str | None = None,
    ) -> None:
        with self._lock:
            patch = {
                "state": state,
                "detect_hz": detect_hz,
                "infer_ms": infer_ms,
                "target_conf": target_conf,
                "detail": detail,
                "running": self.is_running(),
            }
            if mouse_label is not None:
                patch["mouse_label"] = mouse_label
            self._stats.update(patch)

    def set_status(self, text: str) -> None:
        with self._lock:
            self._stats["detail"] = text

    def set_mouse_label(self, text: str) -> None:
        with self._lock:
            self._stats["mouse_label"] = text

    def _snapshot(self) -> dict:
        with self._lock:
            data = dict(self._stats)
        data["running"] = self.is_running()
        return data

    def bootstrap(self) -> dict:
        data = self._snapshot()
        data["config"] = asdict(self.config)
        data["binds"] = list(PRESET_BINDS)
        return data

    def poll(self) -> dict:
        return self._snapshot()

    def apply_config(self, payload: dict) -> dict:
        cfg = self.config
        cfg.active_model = "AL"
        cfg.detection_mode = str(payload.get("detection_mode", cfg.detection_mode))
        cfg.detection_range = str(payload.get("detection_range", cfg.detection_range))
        cfg.input_mode = str(payload.get("input_mode", cfg.input_mode))
        cfg.bind_key = normalize_bind(str(payload.get("bind_key", cfg.bind_key)))
        cfg.controller_bind = normalize_bind(str(payload.get("controller_bind", cfg.controller_bind)))
        cfg.capture_method = str(payload.get("capture_method", cfg.capture_method))
        cfg.color_preset = str(payload.get("color_preset", cfg.color_preset))
        cfg.track_speed = float(payload.get("track_speed", cfg.track_speed))
        cfg.aim_lead = float(payload.get("aim_lead", cfg.aim_lead))
        cfg.aim_smooth = float(payload.get("aim_smooth", cfg.aim_smooth))
        cfg.aim_offset_y = float(payload.get("aim_offset_y", cfg.aim_offset_y))
        cfg.fov_radius = int(payload.get("fov_radius", cfg.fov_radius))
        cfg.fov_radius_held = int(payload.get("fov_radius_held", cfg.fov_radius_held))
        cfg.dynamic_fov_enabled = bool(payload.get("dynamic_fov_enabled", cfg.dynamic_fov_enabled))
        cfg.confidence = float(payload.get("confidence", cfg.confidence))
        cfg.show_debug = bool(payload.get("show_debug", cfg.show_debug))
        cfg.show_overlay = bool(payload.get("show_overlay", cfg.show_overlay))
        cfg.israel_mode = bool(payload.get("israel_mode", cfg.israel_mode))
        cfg.assist_mode = str(payload.get("assist_mode", cfg.assist_mode))
        cfg.trigger_delay_ms = float(payload.get("trigger_delay_ms", cfg.trigger_delay_ms))
        cfg.trigger_cooldown_ms = float(
            payload.get("trigger_cooldown_ms", cfg.trigger_cooldown_ms)
        )
        cfg.trigger_box_scale = float(
            payload.get("trigger_box_scale", cfg.trigger_box_scale)
        )
        # PID stabilizer
        cfg.pid_kp = float(payload.get("pid_kp", cfg.pid_kp))
        cfg.pid_ki = float(payload.get("pid_ki", cfg.pid_ki))
        cfg.pid_kd = float(payload.get("pid_kd", cfg.pid_kd))
        cfg.pid_ff = float(payload.get("pid_ff", cfg.pid_ff))
        cfg.pid_smoothing = float(payload.get("pid_smoothing", cfg.pid_smoothing))
        cfg.clamp()
        save_config(cfg)
        self.on_apply(cfg)
        return {"ok": True, "bind_label": format_bind(cfg.bind_key)}

    def capture_bind(self) -> str:
        key = wait_for_bind()
        self.config.bind_key = normalize_bind(key)
        save_config(self.config)
        self.on_apply(self.config)
        return self.config.bind_key

    def test_mouse(self) -> dict:
        from exoai_tracker.mouse import Mouse
        import time

        m = Mouse()
        if not m.init():
            msg = "Razer offline — open Synapse, plug Razer mouse, Run as Admin"
            self.set_status(msg)
            return {"ok": False, "ready": False, "message": msg}
        time.sleep(0.15)
        # Big obvious desktop nudges
        moves = [(80, 0), (-80, 0), (0, 60), (0, -60), (50, 50), (-50, -50)]
        ok = 0
        for dx, dy in moves:
            if m.move(dx, dy):
                ok += 1
            time.sleep(0.18)
        msg = f"Razer test {ok}/{len(moves)} · {m.label} — watch desktop cursor"
        self.set_status(msg)
        return {"ok": ok == len(moves), "ready": m.ok, "message": msg}

    def toggle(self) -> dict:
        try:
            if self.is_running():
                self.on_stop()
                self.set_stats(state="IDLE", detect_hz=0.0, infer_ms=0.0, target_conf=None, detail="Stopped")
            else:
                self.on_start()
                self.set_stats(state="SCAN", detect_hz=0.0, infer_ms=0.0, target_conf=None, detail="Running")
        except Exception as exc:
            self.set_status(f"Error: {exc}")
            raise
        return self._snapshot()

    def minimize(self) -> None:
        if self._window:
            self._window.minimize()

    def close_window(self) -> None:
        if self._window:
            self._window.destroy()

    def drag_window(self, start_x: int, start_y: int) -> None:
        """Drag window - called continuously during drag."""
        import ctypes
        import threading
        
        if not self._window:
            return
            
        win_x = self._window.x
        win_y = self._window.y
        
        def do_drag():
            import time
            user32 = ctypes.windll.user32
            
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
            last_x, last_y = start_x, start_y
            
            while user32.GetAsyncKeyState(0x01) & 0x8000:  # Left mouse button
                pt = POINT()
                user32.GetCursorPos(ctypes.byref(pt))
                
                dx = pt.x - last_x
                dy = pt.y - last_y
                
                if dx != 0 or dy != 0:
                    new_x = self._window.x + dx
                    new_y = self._window.y + dy
                    self._window.move(new_x, new_y)
                    last_x = pt.x
                    last_y = pt.y
                
                time.sleep(0.008)
        
        threading.Thread(target=do_drag, daemon=True).start()

    def run(self) -> None:
        index = ui_index()
        if not index.exists():
            raise FileNotFoundError(f"WebView2 UI missing: {index}")
        self._window = webview.create_window(
            "Carbon",
            url=index.as_uri(),
            js_api=self,
            width=800,
            height=600,
            min_size=(600, 400),
            background_color="#15171a",
            text_select=False,
        )
        webview.start(gui="edgechromium")
