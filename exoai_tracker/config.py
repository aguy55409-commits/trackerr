from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from exoai_tracker.binds import normalize_bind
from exoai_tracker.models import get_preset, normalize_model_id
from exoai_tracker.paths import app_dir, resolve_data_path

CONFIG_PATH = app_dir() / "config.json"


@dataclass
class AppConfig:
    active_model: str = "FN"
    model_fn_path: str = "models/fn/best.engine"
    model_al_path: str = "models/al/best.engine"
    model_val_path: str = "models/val/best.engine"
    model_r6_path: str = "models/r6/best.engine"
    model_path: str = "models/fn/best.engine"
    bind_key: str = "right"
    track_speed: float = 4.2
    aim_lead: float = 1.05
    aim_smooth: float = 0.22
    fov_radius: int = 180
    dynamic_fov_enabled: bool = True
    fov_radius_held: int = 280
    confidence: float = 0.35
    capture_size: int = 320
    show_overlay: bool = True
    show_debug: bool = False
    aim_offset_y: float = 0.35
    mouse_method: str = "razer"
    loop_interval_ms: float = 1000.0 / 500.0
    detect_interval_ms: float = 0.0
    iou_threshold: float = 0.30
    capture_method: str = "dxcam"
    color_preset: str = "purple"

    def clamp(self) -> None:
        if self.capture_method not in ("dxcam", "mss"):
            self.capture_method = "dxcam"
        self.track_speed = max(0.01, min(5.0, float(self.track_speed)))
        self.aim_lead = max(0.0, min(1.5, float(self.aim_lead)))
        self.aim_smooth = max(0.05, min(0.9, float(self.aim_smooth)))
        self.fov_radius = max(40, min(600, int(self.fov_radius)))
        self.fov_radius_held = max(40, min(600, int(self.fov_radius_held)))
        if self.fov_radius_held < self.fov_radius:
            self.fov_radius_held = self.fov_radius
        self.confidence = max(0.1, min(0.95, float(self.confidence)))
        self.capture_size = max(320, min(1280, int(self.capture_size)))
        self.aim_offset_y = max(0.0, min(1.0, float(self.aim_offset_y)))
        self.loop_interval_ms = max(1.0, min(50.0, float(self.loop_interval_ms)))
        self.detect_interval_ms = max(0.0, min(50.0, float(self.detect_interval_ms)))
        self.iou_threshold = max(0.15, min(0.8, float(self.iou_threshold)))
        if self.mouse_method not in ("razer", "win32", "sendinput", "arduino", "custom"):
            self.mouse_method = "razer"
        self.active_model = normalize_model_id(self.active_model)
        self.model_path = self.resolve_model_path()
        self.bind_key = normalize_bind(self.bind_key)

    def effective_fov(self, bind_held: bool) -> int:
        if self.dynamic_fov_enabled and bind_held:
            return self.fov_radius_held
        return self.fov_radius

    def resolve_model_path(self) -> str:
        paths = {
            "FN": self.model_fn_path,
            "AL": self.model_al_path,
            "VAL": self.model_val_path,
            "R6": self.model_r6_path,
        }
        return str(resolve_data_path(paths.get(self.active_model, self.model_fn_path)))


DEFAULT_CONFIG = AppConfig()


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return deepcopy(DEFAULT_CONFIG)

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = AppConfig(**{f.name: data.get(f.name, getattr(DEFAULT_CONFIG, f.name)) for f in fields(AppConfig)})
        cfg.clamp()
        return cfg
    except (json.JSONDecodeError, TypeError, ValueError):
        return deepcopy(DEFAULT_CONFIG)


def save_config(config: AppConfig) -> None:
    config.clamp()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
