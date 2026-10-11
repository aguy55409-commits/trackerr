from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from exoai_tracker.binds import normalize_bind
from exoai_tracker.models import get_preset, normalize_model_id
from exoai_tracker.paths import app_dir, resolve_data_path

CONFIG_PATH = app_dir() / "config.json"


@dataclass
class AppConfig:
    active_model: str = "AL"
    model_fn_path: str = "models/fn/best.pt"
    model_al_path: str = "models/al/best.pt"
    model_val_path: str = "models/val/best.pt"
    model_r6_path: str = "models/r6/best.pt"
    model_path: str = "models/al/best.pt"
    bind_key: str = "right"
    controller_bind: str = "gp_lt"  # Bind when using controller mode
    track_speed: float = 4.2
    aim_lead: float = 1.05
    aim_smooth: float = 0.22
    fov_radius: int = 180
    dynamic_fov_enabled: bool = True
    fov_radius_held: int = 280
    confidence: float = 0.30
    capture_size: int = 480  # Larger = better long range, slower
    # Network input size, independent of the captured region. The weights were
    # trained at 960, so feeding a small crop 1:1 leaves distant players below
    # the size the model can resolve. Upscaling costs little because the model
    # is bound by per-layer launch overhead rather than resolution.
    net_size: int = 640  # 640 = better long range detection, ~1.2ms
    show_overlay: bool = False
    show_debug: bool = False
    show_movement_line: bool = False
    aim_offset_y: float = 0.35
    mouse_method: str = "razer"
    loop_interval_ms: float = 0.0
    detect_interval_ms: float = 0.0
    iou_threshold: float = 0.30
    capture_method: str = "dxcam"
    color_preset: str = "purple"
    detection_mode: str = "model"  # "model" or "color"
    detection_range: str = "balanced"  # "close", "balanced", "far"
    input_mode: str = "mouse"  # "mouse" or "controller"
    israel_mode: bool = False
    # What the assist does once it has a target: move the aim, click when the
    # crosshair is already on one, or both.
    assist_mode: str = "aim"  # "aim", "trigger", "both", "recoil"
    trigger_delay_ms: float = 40.0
    trigger_cooldown_ms: float = 140.0
    trigger_hold_ms: float = 12.0
    trigger_box_scale: float = 0.55
    # PID stabilizer settings
    pid_kp: float = 0.78          # Proportional gain (snap to target)
    pid_ki: float = 0.8           # Integral gain (kill steady-state error)
    pid_kd: float = 0.002         # Derivative gain (damp overshoot)
    pid_ff: float = 0.96          # Feedforward gain (match target velocity)
    pid_smoothing: float = 0.65   # Measurement filter (0=raw, 1=max smooth)
    # Advanced solver constants. Empty means "use the built-in defaults"; only
    # knobs the user actually moved are stored, so defaults can still improve.
    aim_tuning: dict = field(default_factory=dict)
    
    # === NEW: Advanced aim customization ===
    # Anti-snap: limits how fast the aim can move (pixels/frame). 0 = unlimited
    max_aim_speed: float = 0.0
    # Acceleration curve: 0 = instant full speed, 1 = slow ramp up
    aim_acceleration: float = 0.0
    # Sticky aim: seconds to keep targeting after target lost
    sticky_aim_time: float = 0.15
    # Target lock: minimum seconds before switching to a closer target
    target_lock_time: float = 0.08
    # Humanization: add micro-jitter (pixels). 0 = off
    humanize_jitter: float = 0.0
    # Humanization: random delay range (ms). 0 = off
    humanize_delay_ms: float = 0.0
    # Flick reduction: smoothing applied to large movements. 0 = off, 1 = max
    flick_reduction: float = 0.0
    # Easing curve: "linear", "ease_in", "ease_out", "ease_in_out"
    easing_curve: str = "linear"
    
    # === NEW: Detection filtering ===
    # Minimum detection box size (pixels). Filters tiny/distant targets
    min_box_size: int = 15
    # Maximum detection box size (pixels). Filters too-close/own-player
    max_box_size: int = 400
    # Minimum aspect ratio (h/w). Filters wide boxes that aren't players
    min_aspect_ratio: float = 0.8
    # Maximum aspect ratio (h/w). Filters tall thin boxes
    max_aspect_ratio: float = 4.0
    # Target priority: "closest", "highest_conf", "largest", "center"
    target_priority: str = "closest"
    # Ignore detections in bottom corners (own player filter)
    filter_own_player: bool = True
    # Smoothing: frames to average detection position. 1 = no smoothing
    detection_smoothing: int = 1
    # Prediction: lead moving targets by this many frames. 0 = off
    prediction_frames: int = 0
    
    # === Recoil Control ===
    # Enable recoil compensation (pulls down while shooting)
    recoil_enabled: bool = False
    # Strength: how much to pull down per tick (pixels)
    recoil_strength: float = 1.5
    # Horizontal compensation (+ = right, - = left)
    recoil_horizontal: float = 0.0
    # Delay between pulls (ms) - lower = faster compensation
    recoil_interval_ms: float = 10.0
    # Ramp up: gradually increase pull over this many ms. 0 = instant full strength
    recoil_ramp_ms: float = 100.0
    # Separate bind for recoil (empty = use main bind)
    recoil_bind: str = ""
    
    # Capture tuning.
    capture_fps_cap: int = 0      # 0 = uncapped (bounded by the display anyway)
    capture_offset_x: int = 0     # Shift the capture region off screen centre
    capture_offset_y: int = 0
    capture_monitor: int = 0      # 0-based monitor index
    capture_window: str = ""      # WGC only: capture this window instead

    def clamp(self) -> None:
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
        # The capture region has to cover the FOV or the outer ring of it can
        # never contain a detection. Capped at 960 because that is the largest
        # input the model can run at and still keep up: capturing a wider
        # region than that only means throwing the extra pixels away on the
        # resize, which costs detection range rather than adding it.
        # The capture region has to cover the FOV or the outer ring of it can
        # never contain a detection.
        need = 2 * max(self.fov_radius, self.fov_radius_held) + 32
        if self.capture_size < need:
            self.capture_size = min(1280, (need + 31) // 32 * 32)
        self.capture_size = (self.capture_size + 31) // 32 * 32
        # Runtime engines are fixed-shape. Restrict the UI/config to tested
        # profiles instead of accepting sizes that cannot load an engine.
        self.net_size = 448 if int(self.net_size) < 544 else 640
        self.aim_offset_y = max(0.0, min(1.0, float(self.aim_offset_y)))
        if self.assist_mode not in ("aim", "trigger", "both", "recoil"):
            self.assist_mode = "aim"
        self.trigger_delay_ms = max(0.0, min(500.0, float(self.trigger_delay_ms)))
        self.trigger_cooldown_ms = max(20.0, min(1000.0, float(self.trigger_cooldown_ms)))
        self.trigger_hold_ms = max(4.0, min(80.0, float(self.trigger_hold_ms)))
        self.trigger_box_scale = max(0.1, min(1.0, float(self.trigger_box_scale)))
        # PID stabilizer
        self.pid_kp = max(0.05, min(1.5, float(self.pid_kp)))
        self.pid_ki = max(0.0, min(5.0, float(self.pid_ki)))
        self.pid_kd = max(0.0, min(0.05, float(self.pid_kd)))
        self.pid_ff = max(0.0, min(1.2, float(self.pid_ff)))
        self.pid_smoothing = max(0.1, min(0.95, float(self.pid_smoothing)))
        self.loop_interval_ms = max(0.0, float(self.loop_interval_ms))
        self.detect_interval_ms = max(0.0, float(self.detect_interval_ms))
        self.iou_threshold = max(0.15, min(0.8, float(self.iou_threshold)))
        if self.detection_range not in ("close", "balanced", "far"):
            self.detection_range = "balanced"
        if self.input_mode not in ("mouse", "controller"):
            self.input_mode = "mouse"
        # Auto-adjust capture/net size based on range mode
        if self.detection_range == "close":
            self.capture_size = 448
            self.net_size = 448
        elif self.detection_range == "far":
            self.capture_size = max(640, self.capture_size)
            self.net_size = 640
        else:
            self.capture_size = max(self.capture_size, self.net_size)
        valid_mouse = ("razer", "sendinput", "sendinput_interp")
        if self.mouse_method not in valid_mouse:
            self.mouse_method = "razer"
        self.active_model = "AL"
        self.model_path = self.resolve_model_path()
        self.bind_key = normalize_bind(self.bind_key)
        self.capture_fps_cap = max(0, min(1000, int(self.capture_fps_cap)))
        self.capture_offset_x = max(-2000, min(2000, int(self.capture_offset_x)))
        self.capture_offset_y = max(-2000, min(2000, int(self.capture_offset_y)))
        self.capture_monitor = max(0, min(7, int(self.capture_monitor)))
        # Drop unknown or unusable tuning keys rather than carrying them into
        # the solver, where they would be silently ignored anyway.
        from exoai_tracker.aim import MoveSolver

        clean: dict[str, float] = {}
        for name, raw in (self.aim_tuning or {}).items():
            bounds = MoveSolver.TUNABLES.get(name)
            if bounds is None:
                continue
            try:
                clean[name] = max(bounds[0], min(bounds[1], float(raw)))
            except (TypeError, ValueError):
                continue
        self.aim_tuning = clean
        
        # New aim customization
        self.max_aim_speed = max(0.0, min(500.0, float(self.max_aim_speed)))
        self.aim_acceleration = max(0.0, min(1.0, float(self.aim_acceleration)))
        self.sticky_aim_time = max(0.0, min(2.0, float(self.sticky_aim_time)))
        self.target_lock_time = max(0.0, min(1.0, float(self.target_lock_time)))
        self.humanize_jitter = max(0.0, min(5.0, float(self.humanize_jitter)))
        self.humanize_delay_ms = max(0.0, min(50.0, float(self.humanize_delay_ms)))
        self.flick_reduction = max(0.0, min(1.0, float(self.flick_reduction)))
        if self.easing_curve not in ("linear", "ease_in", "ease_out", "ease_in_out"):
            self.easing_curve = "linear"
        
        # New detection filtering
        self.min_box_size = max(5, min(200, int(self.min_box_size)))
        self.max_box_size = max(50, min(800, int(self.max_box_size)))
        self.min_aspect_ratio = max(0.3, min(2.0, float(self.min_aspect_ratio)))
        self.max_aspect_ratio = max(1.0, min(8.0, float(self.max_aspect_ratio)))
        if self.target_priority not in ("closest", "highest_conf", "largest", "center"):
            self.target_priority = "closest"
        self.detection_smoothing = max(1, min(10, int(self.detection_smoothing)))
        self.prediction_frames = max(0, min(5, int(self.prediction_frames)))
        
        # Recoil control
        self.recoil_strength = max(0.0, min(20.0, float(self.recoil_strength)))
        self.recoil_horizontal = max(-10.0, min(10.0, float(self.recoil_horizontal)))
        self.recoil_interval_ms = max(1.0, min(100.0, float(self.recoil_interval_ms)))
        self.recoil_ramp_ms = max(0.0, min(500.0, float(self.recoil_ramp_ms)))
        if self.recoil_bind:
            self.recoil_bind = normalize_bind(self.recoil_bind)

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


# --- Named profiles -------------------------------------------------------
# A profile is a whole config saved under a name, so per-game or per-weapon
# setups can be swapped without retuning.

PROFILE_DIR = app_dir() / "profiles"
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_")


def _profile_file(name: str) -> Path:
    slug = "".join(c for c in name if c in _SAFE).strip()
    if not slug:
        raise ValueError("profile name must contain a letter or number")
    return PROFILE_DIR / f"{slug}.json"


def list_profiles() -> list[str]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def save_profile(name: str, config: AppConfig) -> None:
    path = _profile_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_profile(name: str) -> AppConfig:
    data = json.loads(_profile_file(name).read_text(encoding="utf-8"))
    cfg = AppConfig(
        **{
            f.name: data.get(f.name, getattr(DEFAULT_CONFIG, f.name))
            for f in fields(AppConfig)
        }
    )
    cfg.clamp()
    return cfg


def delete_profile(name: str) -> None:
    _profile_file(name).unlink(missing_ok=True)
