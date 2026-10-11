"""Clean app entry — GUI + pipeline with background model preload."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from exoai_tracker.binds import format_bind
from exoai_tracker.color_pipeline import ColorPipeline
from exoai_tracker.config import AppConfig, load_config, save_config
from exoai_tracker.controller_output import shutdown_shared_output
from exoai_tracker.native_ui import NativeUI
from exoai_tracker.infer import Infer
from exoai_tracker.models import get_preset, resolve_preset_weights
from exoai_tracker.overlay import DetectionBox, FovOverlay, OverlayDebugData
from exoai_tracker.pipeline import Pipeline, enable_1ms_timer

log = logging.getLogger("neverwin")


class TrackerApp:
    def __init__(self) -> None:
        self.config = load_config()
        self.pipe: Pipeline | ColorPipeline | None = None
        self._running = False
        self._starting = False
        self._ui_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._infer: Infer | None = None
        self._infer_path: Path | None = None
        self._infer_size: int | None = None
        self._infer_lock = threading.Lock()
        self._preload_err: str | None = None
        self._preload_generation = 0
        self._preload_active = threading.Event()
        self._restart_lock = threading.Lock()
        self._overlay: FovOverlay | None = None
        self._overlay_err: str | None = None
        self._overlay_data = OverlayDebugData()
        self.gui = NativeUI(
            config=self.config,
            on_apply=self.apply_config,
            on_start=self.start,
            on_stop=self.stop,
            is_running=lambda: self._running,
        )
        self._structural_state = {
            field: getattr(self.config, field, None) for field in self._STRUCTURAL
        }
        # Start overlay
        self._start_overlay()
        # Warm torch+engine while the window opens so Start is near-instant
        threading.Thread(target=self._preload_model, name="preload", daemon=True).start()

    def _start_overlay(self) -> None:
        try:
            self._overlay = FovOverlay(
                get_radius=lambda: self.config.effective_fov(self._running),
                # Not gated on running. Ticking the box did nothing visible
                # while stopped, which reads as a broken toggle rather than as
                # the overlay waiting for Start.
                get_enabled=lambda: self.config.show_overlay,
                get_debug=self._overlay_debug,
            )
            self._overlay_err = None
        except Exception as exc:
            self._overlay = None
            # Was swallowed silently, so a failed overlay looked identical to
            # one that simply refused to draw.
            self._overlay_err = f"{type(exc).__name__}: {exc}"

    def _overlay_debug(self) -> OverlayDebugData:
        # Read live so the star shows even while the pipeline is stopped.
        self._overlay_data.israel_mode = self.config.israel_mode
        self._overlay_data.show_movement_line = self.config.show_movement_line
        return self._overlay_data

    # Settings the running pipeline cannot absorb: they decide which pipeline
    # exists, how big its buffers are, or which weights are loaded. Changing
    # one means rebuilding rather than poking the live object.
    _STRUCTURAL = (
        "detection_mode",
        "detection_range",
        "capture_size",
        "net_size",
        "capture_method",
        "active_model",
        "model_path",
        # Capture geometry and rate are fixed when the backend starts.
        "capture_fps_cap",
        "capture_offset_x",
        "capture_offset_y",
        "capture_monitor",
        "capture_window",
        # Mouse driver requires rebuilding InputHandler.
        "mouse_method",
    )

    def apply_config(self, config: AppConfig) -> None:
        previous = self._structural_state
        current = {field: getattr(config, field, None) for field in self._STRUCTURAL}
        structural_changed = any(previous.get(k) != v for k, v in current.items())
        model_changed = any(
            previous.get(k) != current.get(k)
            for k in ("active_model", "model_path", "net_size")
        )
        mode_changed = previous.get("detection_mode") != current.get("detection_mode")
        self._structural_state = current
        self.config = config
        save_config(config)
        if model_changed:
            with self._infer_lock:
                self._preload_generation += 1
                self._infer = None
                self._infer_path = None
                self._infer_size = None
        if self.pipe is None:
            if (model_changed or mode_changed) and config.detection_mode == "model":
                threading.Thread(
                    target=self._preload_model, name="preload-refresh", daemon=True
                ).start()
            return

        if structural_changed:
            # Rebuild off the UI thread so the window stays responsive.
            self.gui.set_status("Applying settings…")
            threading.Thread(target=self._restart, name="restart",
                             daemon=True).start()
            return

        common = {
            "assist_mode": config.assist_mode,
            "trigger_delay_ms": config.trigger_delay_ms,
            "trigger_cooldown_ms": config.trigger_cooldown_ms,
            "trigger_hold_ms": config.trigger_hold_ms,
            "trigger_box_scale": config.trigger_box_scale,
            "pid_kp": config.pid_kp,
            "pid_ki": config.pid_ki,
            "pid_kd": config.pid_kd,
            "pid_ff": config.pid_ff,
            "pid_smoothing": config.pid_smoothing,
            "input_mode": config.input_mode,
            "controller_bind": config.controller_bind,
            "aim_tuning": config.aim_tuning,
            "max_aim_speed": config.max_aim_speed,
            "aim_acceleration": config.aim_acceleration,
            "sticky_aim_time": config.sticky_aim_time,
            "target_lock_time": config.target_lock_time,
            "humanize_jitter": config.humanize_jitter,
            "flick_reduction": config.flick_reduction,
            "easing_curve": config.easing_curve,
            "min_box_size": config.min_box_size,
            "max_box_size": config.max_box_size,
            "min_aspect_ratio": config.min_aspect_ratio,
            "max_aspect_ratio": config.max_aspect_ratio,
            "target_priority": config.target_priority,
            "filter_own_player": config.filter_own_player,
            "detection_smoothing": config.detection_smoothing,
            "prediction_frames": config.prediction_frames,
            # Recoil control
            "recoil_enabled": config.recoil_enabled,
            "recoil_strength": config.recoil_strength,
            "recoil_horizontal": config.recoil_horizontal,
            "recoil_interval_ms": config.recoil_interval_ms,
            "recoil_ramp_ms": config.recoil_ramp_ms,
            "recoil_bind": config.recoil_bind,
        }
        if isinstance(self.pipe, ColorPipeline):
            self.pipe.apply(
                bind=config.bind_key,
                speed=config.track_speed,
                fov=config.effective_fov(True),
                color_preset=config.color_preset,
                offset_y=config.aim_offset_y,
                **{k: v for k, v in common.items() if k not in {
                    "max_aim_speed", "aim_acceleration", "sticky_aim_time",
                    "target_lock_time", "humanize_jitter", "flick_reduction",
                    "easing_curve", "min_box_size", "max_box_size",
                    "min_aspect_ratio", "max_aspect_ratio", "target_priority",
                    "filter_own_player", "detection_smoothing", "prediction_frames",
                }},
            )
        else:
            self.pipe.apply(
                bind=config.bind_key,
                speed=config.track_speed,
                fov=config.effective_fov(True),
                conf=config.confidence,
                iou=config.iou_threshold,
                offset_y=config.aim_offset_y,
                **common,
            )

    def _restart(self) -> None:
        if not self._restart_lock.acquire(blocking=False):
            return
        try:
            was_running = self._running
            self.stop()
            if was_running:
                self._starting = True
                self._start_worker()
        finally:
            self._restart_lock.release()

    def _model_path(self) -> Path:
        preset = get_preset(self.config.active_model)
        path = resolve_preset_weights(
            self.config.active_model,
            self.config.resolve_model_path(),
        )
        # Prefer FP16 engines (2x faster inference) over FP32.
        p16_combined = path.parent / f"best_combined_{self.config.net_size}_fp16.engine"
        p16_sized = path.parent / f"best_{self.config.net_size}_fp16.engine"
        combined_engine = path.parent / f"best_combined_{self.config.net_size}.engine"
        sized_engine = path.parent / f"best_{self.config.net_size}.engine"
        if p16_combined.exists():
            path = p16_combined
        elif p16_sized.exists():
            path = p16_sized
        elif combined_engine.exists():
            path = combined_engine
        elif sized_engine.exists():
            path = sized_engine
        if not path.exists():
            raise FileNotFoundError(f"{preset.label} weights missing: {path}")
        return path

    def _preload_model(self) -> None:
        self._preload_active.set()
        try:
            with self._infer_lock:
                self._preload_err = None
                generation = self._preload_generation
            self.gui.set_status("Preloading model…")
            path = self._model_path()
            cfg = self.config
            infer = Infer(
                path,
                conf=cfg.confidence,
                iou=cfg.iou_threshold,
                imgsz=cfg.net_size,
                warmup=8,
            )
            with self._infer_lock:
                if generation != self._preload_generation:
                    return
                self._infer = infer
                self._infer_path = path
                self._infer_size = cfg.net_size
                self._preload_err = None
            if not self._running:
                self.gui.set_status(
                    f"Ready · {infer.backend} · {path.name} cached — press Start"
                )
        except Exception as exc:
            log.exception("Model preload failed")
            self._preload_err = f"{type(exc).__name__}: {exc}"
            if not self._running:
                self.gui.set_status(f"Preload failed: {self._preload_err}")
        finally:
            self._preload_active.clear()

    def _get_infer(self, path: Path) -> Infer:
        cfg = self.config
        with self._infer_lock:
            if (
                self._infer is not None
                and self._infer_path is not None
                and self._infer_path.resolve() == path.resolve()
                and self._infer_size == cfg.net_size
            ):
                self._infer.update(conf=cfg.confidence, iou=cfg.iou_threshold)
                return self._infer
        self.gui.set_status(f"Loading {path.name}…")
        infer = Infer(
            path,
            conf=cfg.confidence,
            iou=cfg.iou_threshold,
            imgsz=cfg.net_size,
            warmup=8,
        )
        with self._infer_lock:
            self._infer = infer
            self._infer_path = path
            self._infer_size = cfg.net_size
        return infer

    def start(self) -> None:
        if self._running or self._starting:
            return
        self._starting = True
        threading.Thread(target=self._start_worker, name="start", daemon=True).start()

    def _start_worker(self) -> None:
        if self.config.detection_mode == "color":
            self._start_color()
            return
        try:
            model = self._model_path()
            # Wait briefly if preload still running the same file
            deadline = time.perf_counter() + 45.0
            while self._preload_active.is_set() and time.perf_counter() < deadline:
                with self._infer_lock:
                    ready = (
                        self._infer is not None
                        and self._infer_path is not None
                        and self._infer_path.resolve() == model.resolve()
                        and self._infer_size == self.config.net_size
                    )
                if ready or self._preload_err:
                    break
                self.gui.set_status(f"Loading {model.name}…")
                time.sleep(0.05)

            infer = self._get_infer(model)
            cfg = self.config
            self.pipe = Pipeline(
                model,
                bind=cfg.bind_key,
                speed=cfg.track_speed,
                fov=cfg.fov_radius,
                conf=cfg.confidence,
                iou=cfg.iou_threshold,
                capture_size=cfg.capture_size,
                net_size=cfg.net_size,
                capture_method=cfg.capture_method,
                offset_y=cfg.aim_offset_y,
                get_fov=lambda: self.config.effective_fov(True),
                infer=infer,
                aim_tuning=cfg.aim_tuning,
                capture_fps_cap=cfg.capture_fps_cap,
                capture_offset_x=cfg.capture_offset_x,
                capture_offset_y=cfg.capture_offset_y,
                capture_monitor=cfg.capture_monitor,
                capture_window=cfg.capture_window,
                assist_mode=cfg.assist_mode,
                trigger_delay_ms=cfg.trigger_delay_ms,
                trigger_cooldown_ms=cfg.trigger_cooldown_ms,
                trigger_hold_ms=cfg.trigger_hold_ms,
                trigger_box_scale=cfg.trigger_box_scale,
                pid_kp=cfg.pid_kp,
                pid_ki=cfg.pid_ki,
                pid_kd=cfg.pid_kd,
                pid_ff=cfg.pid_ff,
                pid_smoothing=cfg.pid_smoothing,
                input_mode=cfg.input_mode,
                controller_bind=cfg.controller_bind,
                mouse_method=cfg.mouse_method,
                # Advanced aim
                max_aim_speed=cfg.max_aim_speed,
                aim_acceleration=cfg.aim_acceleration,
                sticky_aim_time=cfg.sticky_aim_time,
                target_lock_time=cfg.target_lock_time,
                humanize_jitter=cfg.humanize_jitter,
                flick_reduction=cfg.flick_reduction,
                easing_curve=cfg.easing_curve,
                # Detection filtering
                min_box_size=cfg.min_box_size,
                max_box_size=cfg.max_box_size,
                min_aspect_ratio=cfg.min_aspect_ratio,
                max_aspect_ratio=cfg.max_aspect_ratio,
                target_priority=cfg.target_priority,
                filter_own_player=cfg.filter_own_player,
                detection_smoothing=cfg.detection_smoothing,
                prediction_frames=cfg.prediction_frames,
                # Recoil control
                recoil_enabled=cfg.recoil_enabled,
                recoil_strength=cfg.recoil_strength,
                recoil_horizontal=cfg.recoil_horizontal,
                recoil_interval_ms=cfg.recoil_interval_ms,
                recoil_ramp_ms=cfg.recoil_ramp_ms,
                recoil_bind=cfg.recoil_bind,
            )
            self.pipe.start()
            self._running = True
            self._stop.clear()
            self._ui_thread = threading.Thread(target=self._stats_loop, name="ui-stats", daemon=True)
            self._ui_thread.start()
            active_bind = cfg.controller_bind if cfg.input_mode == "controller" else cfg.bind_key
            self.gui.set_status(f"Running · {model.name} · hold [{format_bind(active_bind)}]")
            self.gui.set_mouse_label(self.pipe.mouse.short)
        except Exception as exc:
            log.exception("Model pipeline start failed")
            if self.pipe is not None:
                try:
                    self.pipe.stop()
                except Exception:
                    pass
            self.pipe = None
            self._running = False
            self.gui.set_status(f"Start failed: {type(exc).__name__}: {exc}")
        finally:
            self._starting = False

    def _start_color(self) -> None:
        """Colour mode. No model, so nothing to preload or wait for."""
        try:
            cfg = self.config
            self.pipe = ColorPipeline(
                bind=cfg.bind_key,
                speed=cfg.track_speed,
                fov=cfg.fov_radius,
                capture_size=cfg.capture_size,
                capture_method=cfg.capture_method,
                capture_monitor=cfg.capture_monitor,
                capture_offset_x=cfg.capture_offset_x,
                capture_offset_y=cfg.capture_offset_y,
                color_preset=cfg.color_preset,
                offset_y=cfg.aim_offset_y,
                get_fov=lambda: self.config.effective_fov(True),
                aim_tuning=cfg.aim_tuning,
                assist_mode=cfg.assist_mode,
                trigger_delay_ms=cfg.trigger_delay_ms,
                trigger_cooldown_ms=cfg.trigger_cooldown_ms,
                trigger_hold_ms=cfg.trigger_hold_ms,
                trigger_box_scale=cfg.trigger_box_scale,
                pid_kp=cfg.pid_kp,
                pid_ki=cfg.pid_ki,
                pid_kd=cfg.pid_kd,
                pid_ff=cfg.pid_ff,
                pid_smoothing=cfg.pid_smoothing,
                input_mode=cfg.input_mode,
                controller_bind=cfg.controller_bind,
                mouse_method=cfg.mouse_method,
                # Recoil control
                recoil_enabled=cfg.recoil_enabled,
                recoil_strength=cfg.recoil_strength,
                recoil_horizontal=cfg.recoil_horizontal,
                recoil_interval_ms=cfg.recoil_interval_ms,
                recoil_ramp_ms=cfg.recoil_ramp_ms,
                recoil_bind=cfg.recoil_bind,
            )
            self.pipe.start()
            self._running = True
            self._stop.clear()
            self._ui_thread = threading.Thread(
                target=self._stats_loop, name="ui-stats", daemon=True
            )
            self._ui_thread.start()
            active_bind = cfg.controller_bind if cfg.input_mode == "controller" else cfg.bind_key
            self.gui.set_status(
                f"Running · colour {cfg.color_preset} · hold "
                f"[{format_bind(active_bind)}]"
            )
            self.gui.set_mouse_label(self.pipe.mouse.short)
        except Exception as exc:
            log.exception("Color pipeline start failed")
            if self.pipe is not None:
                try:
                    self.pipe.stop()
                except Exception:
                    pass
            self.pipe = None
            self._running = False
            self.gui.set_status(f"Start failed: {type(exc).__name__}: {exc}")
        finally:
            self._starting = False

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        if self._ui_thread and self._ui_thread.is_alive():
            self._ui_thread.join(timeout=1.0)
        self._ui_thread = None
        if self.pipe is not None:
            # Keep Infer cached — only tear down capture/mouse/worker
            self.pipe.stop()
            self.pipe = None
        name = self._infer_path.name if self._infer_path else "model"
        self.gui.set_status(f"Stopped · {name} still cached")

    def _stats_loop(self) -> None:
        # Colour mode has a detector rather than a capture, and no aimer.
        cx = cy = 0
        src = getattr(self.pipe, "capture", None) or getattr(self.pipe, "detector", None)
        if src is not None:
            cx = getattr(src, "center_x", 0)
            cy = getattr(src, "center_y", 0)
        while not self._stop.is_set() and self._running and self.pipe is not None:
            snapshot = getattr(self.pipe, "snapshot_stats", None)
            s = snapshot() if snapshot is not None else self.pipe.stats
            # Colour mode reports detect_ms and has no confidence score.
            infer_ms = getattr(s, "infer_ms", None)
            if infer_ms is None:
                infer_ms = getattr(s, "detect_ms", 0.0)
            self.gui.set_stats(
                state=s.state,
                detect_hz=s.detect_hz,
                infer_ms=infer_ms,
                target_conf=getattr(s, "conf", None),
                cap_hz=getattr(s, "cap_hz", 0.0),
                move_hz=getattr(s, "move_hz", 0.0),
                frame_ms=getattr(s, "frame_ms", 0.0),
                detail=(
                    s.detail
                    if not self._overlay_err
                    else f"{s.detail} · overlay error: {self._overlay_err}"
                ),
                mouse_label=s.mouse_label,
            )
            target_pos = None
            aimer = getattr(self.pipe, "aimer", None)
            if aimer is not None and aimer.target is not None:
                t = aimer.target
                target_pos = (t.x - cx, t.y - cy)
            elif getattr(self.pipe, "_has_target", False):
                target_pos = (self.pipe._tx - cx, self.pipe._ty - cy)
            self._overlay_data.target = target_pos
            if self.config.show_debug:
                self._overlay_data.boxes = [
                    DetectionBox(
                        x=d.x - cx,
                        y=d.y - cy,
                        w=d.w,
                        h=d.h,
                        conf=d.conf,
                        kind=d.kind,
                    )
                    for d in getattr(s, "detections", [])
                ]
            else:
                self._overlay_data.boxes = []
            self._overlay_data.detect_hz = s.detect_hz
            self._overlay_data.infer_ms = infer_ms
            self._overlay_data.state = s.state
            time.sleep(0.05)

    def run(self) -> None:
        try:
            self.gui.run()
        finally:
            self.stop()
            if self._overlay is not None:
                self._overlay.destroy()
            shutdown_shared_output()
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


def main() -> None:
    enable_1ms_timer()
    TrackerApp().run()


if __name__ == "__main__":
    main()
