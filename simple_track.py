"""Simple aim tracker with pluggable mouse backends and performance tuning."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from exoai_tracker.binds import is_bind_held
from exoai_tracker.models import get_preset, get_capture_size, resolve_preset_weights
from exoai_tracker.mouse_backends import BACKEND_NAMES, init_backend
from exoai_tracker.track_engine import TrackConfig, TrackEngine

CONFIG_FILE = Path(__file__).parent / "track_config.json"
DEFAULT_CONFIG = {
    "active_model": "AL",
    "speed": 0.5,
    "fov": 300,
    "aim_height": 0.35,
    "max_move": 14,
    "overshoot_ratio": 0.96,
    "deadzone_px": 2,
    "detect_interval_ms": 16,
    "move_interval_ms": 8,
    "idle_sleep_ms": 20,
    "mouse_backend": "auto",
    "bind_key": "right",
    "tracker": "models/al/tracktrack.yaml",
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def config_menu() -> dict | None:
    cfg = load_config()
    print("\n=== TRACKER CONFIG ===")
    preset = get_preset(str(cfg.get("active_model", "AL")))
    print(f"Model: {preset.label} · {preset.description}")
    print(f"1. Speed: {cfg['speed']}")
    print(f"2. FOV: {cfg['fov']}px")
    print(f"3. Aim height: {cfg['aim_height']} (0=belly, 0.35=chest, 0.5=head)")
    print(f"4. Max move: {cfg['max_move']}px")
    print(f"5. Mouse backend: {cfg['mouse_backend']}  ({', '.join(BACKEND_NAMES)})")
    print(f"6. Detect interval: {cfg['detect_interval_ms']}ms")
    print(f"7. Move interval: {cfg['move_interval_ms']}ms")
    print(f"8. START")
    print("Q. Quit")

    try:
        choice = input("\nChoice: ").strip().lower()
    except EOFError:
        return cfg

    if choice == "1":
        try:
            cfg["speed"] = max(0.05, min(1.0, float(input("Speed (0.05-1.0): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "2":
        try:
            cfg["fov"] = max(100, min(500, int(input("FOV (100-500): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "3":
        try:
            cfg["aim_height"] = max(0.0, min(0.5, float(input("Height (0-0.5): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "4":
        try:
            cfg["max_move"] = max(5, min(50, int(input("Max move (5-50): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "5":
        backend = input(f"Backend ({'/'.join(BACKEND_NAMES)}): ").strip().lower() or "auto"
        if backend in BACKEND_NAMES:
            cfg["mouse_backend"] = backend
            save_config(cfg)
    elif choice == "6":
        try:
            cfg["detect_interval_ms"] = max(8, min(50, int(input("Detect ms (8-50): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "7":
        try:
            cfg["move_interval_ms"] = max(4, min(30, int(input("Move ms (4-30): "))))
            save_config(cfg)
        except ValueError:
            pass
    elif choice == "8":
        return cfg
    elif choice == "q":
        return None

    return config_menu()


def run_tracker(cfg: dict) -> None:
    print("\n=== Initializing mouse backend ===")
    mouse = init_backend(str(cfg.get("mouse_backend", "auto")))
    print(f"Using: {mouse.label()}")

    if mouse.name == "macro_bridge":
        print("Macro bridge active — configure Synapse using macro_bridge_setup.txt")

    print("\nLoading YOLO26 model...")
    from exoai_tracker.detector import YoloDetector

    model_id = str(cfg.get("active_model", "AL")).upper()
    preset = get_preset(model_id)
    model_path = resolve_preset_weights(model_id, "")
    capture_size = get_capture_size(model_id)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO26 weights not found: {model_path}")

    print(f"Model: {preset.label} ({preset.description})")
    print(f"Weights: {model_path.name} @ {capture_size}px")

    tracker_cfg = str(cfg.get("tracker", "models/al/tracktrack.yaml"))
    detector = YoloDetector(
        str(model_path),
        0.35,
        capture_size,
        target_class_ids=preset.target_class_ids,
        tracker=tracker_cfg,
    )

    track_cfg = TrackConfig(
        speed=float(cfg["speed"]),
        fov=int(cfg["fov"]),
        aim_height=float(cfg["aim_height"]),
        max_move=int(cfg["max_move"]),
        overshoot_ratio=float(cfg.get("overshoot_ratio", 0.96)),
        deadzone_px=float(cfg.get("deadzone_px", 2)),
        detect_interval_ms=float(cfg.get("detect_interval_ms", 16)),
        move_interval_ms=float(cfg.get("move_interval_ms", 8)),
        idle_sleep_ms=float(cfg.get("idle_sleep_ms", 20)),
        tracker=tracker_cfg,
    )

    bind_key = str(cfg.get("bind_key", "right"))

    print("\n=== TRACKING ===")
    print(f"Hold {bind_key.upper()} (or RMB/Mouse4/Mouse5)")
    print(
        f"Speed={track_cfg.speed} FOV={track_cfg.fov} Height={track_cfg.aim_height} "
        f"Detect={track_cfg.detect_interval_ms}ms Move={track_cfg.move_interval_ms}ms "
        f"Tracker={tracker_cfg}"
    )
    print(f"Backend: {mouse.name}\n")

    engine = TrackEngine(
        detector=detector,
        mouse=mouse,
        config=track_cfg,
        is_held=lambda: is_bind_held(bind_key)
        or is_bind_held("x1")
        or is_bind_held("x2"),
    )

    try:
        engine.run()
    finally:
        mouse.close()
        detector.close()


if __name__ == "__main__":
    print("=== AIM TRACKER ===")
    config = load_config() if "--auto" in sys.argv else config_menu()
    if config:
        if "--auto" in sys.argv:
            print(f"Auto mode: {config}")
        run_tracker(config)
