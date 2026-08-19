from __future__ import annotations

import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import webview

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
    ) -> None:
        with self._lock:
            self._stats.update(
                {
                    "state": state,
                    "detect_hz": detect_hz,
                    "infer_ms": infer_ms,
                    "target_conf": target_conf,
                    "detail": detail,
                    "running": self.is_running(),
                }
            )

    def set_status(self, text: str) -> None:
        with self._lock:
            self._stats["detail"] = text

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
        cfg.active_model = str(payload.get("active_model", cfg.active_model))
        cfg.bind_key = normalize_bind(str(payload.get("bind_key", cfg.bind_key)))
        cfg.mouse_method = str(payload.get("mouse_method", cfg.mouse_method))
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

    def run(self) -> None:
        index = ui_index()
        if not index.exists():
            raise FileNotFoundError(f"WebView2 UI missing: {index}")
        self._window = webview.create_window(
            "Carbon",
            url=index.as_uri(),
            js_api=self,
            width=430,
            height=800,
            min_size=(400, 680),
            background_color="#08090c",
            text_select=False,
        )
        webview.start(gui="edgechromium")
