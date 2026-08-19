from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from exoai_tracker.binds import format_bind
from exoai_tracker.color_detector import ColorDetector, ColorTarget
from exoai_tracker.config import AppConfig, load_config, save_config
from exoai_tracker.detector import Detection, YoloDetector
from exoai_tracker.gui import SettingsWindow
from exoai_tracker.models import get_preset, is_color_mode, resolve_preset_weights
from exoai_tracker.overlay import FovOverlay, OverlayDebugData
from exoai_tracker.scan import TargetSelector
from exoai_tracker.tracker import TargetTracker

AIM_LOOP_S = 1.0 / 144.0  # 144 Hz aim loop - smooth but not too fast
IDLE_LOOP_S = 1.0 / 60.0  # 60 Hz when idle to save CPU


def _enable_1ms_timer() -> None:
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def _precise_wait(seconds: float) -> None:
    if seconds > 0.0001:  # Only sleep if meaningful delay needed
        time.sleep(seconds)


@dataclass
class InferenceSnapshot:
    detections: list[Detection] = field(default_factory=list)
    debug_frame: object | None = None
    inference_ms: float = 0.0
    detect_hz: float = 0.0
    updated_at: float = 0.0
    snap_id: int = 0


class TrackerApp:
    def __init__(self) -> None:
        self.config = load_config()
        self.detector: YoloDetector | ColorDetector | None = None
        self._is_color_mode = False
        self.tracker = TargetTracker(self.config.mouse_method)  # type: ignore[arg-type]
        self.scanner = TargetSelector()
        self.overlay: FovOverlay | None = None
        self._running = False
        self._control_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loaded_model_path = ""
        self._loop_interval = self.config.loop_interval_ms / 1000.0
        self._detect_interval = self.config.detect_interval_ms / 1000.0
        self._snapshot = InferenceSnapshot()
        self._snapshot_lock = threading.Lock()
        self._overlay_state = ""
        self._capture_fallback_notified = False
        self._last_infer_s = 0.008
        self._apply_tracker_settings()
        self.gui = SettingsWindow(
            config=self.config,
            on_apply=self.apply_config,
            on_start=self.start,
            on_stop=self.stop,
            is_running=lambda: self._running,
        )

    def _apply_tracker_settings(self) -> None:
        self._loop_interval = self.config.loop_interval_ms / 1000.0
        self._detect_interval = self.config.detect_interval_ms / 1000.0
        aspect = 16.0 / 9.0
        if self.detector is not None and self.detector.screen_h > 0:
            aspect = self.detector.screen_w / self.detector.screen_h
        self.tracker.configure(
            track_speed=self.config.track_speed,
            loop_interval=self._loop_interval,
            screen_aspect=aspect,
            lead=self.config.aim_lead,
            smooth=self.config.aim_smooth,
        )
        self.tracker.set_bind_key(self.config.bind_key)

    def apply_config(self, config: AppConfig) -> None:
        old_model = self.config.active_model
        old_capture = self.config.capture_method
        old_color = getattr(self.config, 'color_preset', 'purple')
        self.config = config
        self._apply_tracker_settings()
        self.tracker.set_mouse_method(config.mouse_method)  # type: ignore[arg-type]
        
        # Reload detector if model, capture, or color preset changed
        if self._running and (old_model != config.active_model or old_capture != config.capture_method or 
                              (self._is_color_mode and old_color != config.color_preset)):
            self._load_detector()
            return
        
        if self._is_color_mode:
            if self.detector is not None:
                self.detector.set_debug(config.show_debug)
                self.detector.set_fov_radius(config.fov_radius)
            return
        
        preset = get_preset(config.active_model)
        if self.detector is not None and hasattr(self.detector, 'update_settings'):
            self.detector.update_settings(
                config.confidence,
                config.capture_size,
                preset.target_class_ids,
                iou=config.iou_threshold,
            )
            self.detector.set_debug(config.show_debug)
            self.detector.set_fov_radius(config.fov_radius)

    def _resolve_weights(self) -> Path:
        preset = get_preset(self.config.active_model)
        model_path = resolve_preset_weights(
            self.config.active_model,
            self.config.resolve_model_path(),
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"{preset.label} YOLO weights not found: {model_path}\n"
                f"Train or export models in {model_path.parent}"
            )
        return model_path

    def _load_detector(self) -> None:
        preset = get_preset(self.config.active_model)
        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self._is_color_mode = is_color_mode(self.config.active_model)
        
        if self._is_color_mode:
            # Color detection mode - no model needed
            self.detector = ColorDetector(
                capture_size=self.config.capture_size,
                color_preset=self.config.color_preset,
                capture_method=self.config.capture_method,
            )
            self.detector.set_debug(self.config.show_debug)
            self.detector.set_fov_radius(self.config.fov_radius)
            self._loaded_model_path = "color"
            self._apply_tracker_settings()
            self.gui.set_status(f"Color mode · {self.config.color_preset} · ultra fast")
            return

        model_path = self._resolve_weights()
        try:
            self.detector = YoloDetector(
                model_path=str(model_path),
                confidence=self.config.confidence,
                capture_size=self.config.capture_size,
                target_class_ids=preset.target_class_ids,
                iou=self.config.iou_threshold,
                capture_method=self.config.capture_method,
            )
        except Exception:
            if model_path.suffix.lower() == ".engine":
                pt = model_path.with_name("best.pt")
                if pt.exists():
                    model_path = pt
                    self.detector = YoloDetector(
                        model_path=str(model_path),
                        confidence=self.config.confidence,
                        capture_size=self.config.capture_size,
                        target_class_ids=preset.target_class_ids,
                        iou=self.config.iou_threshold,
                        capture_method=self.config.capture_method,
                    )
                else:
                    raise
            else:
                raise
        self.detector.set_debug(self.config.show_debug)
        self.detector.set_fov_radius(self.config.fov_radius)
        self._loaded_model_path = str(model_path)
        self._apply_tracker_settings()
        self.gui.set_status(f"Loaded {preset.label} · {model_path.name}")

    def start(self) -> None:
        if self._running:
            return
        self._load_detector()
        self.scanner.reset()
        self.tracker.reset_aim()
        self.tracker.mouse = None  # force fresh Razer handle each start
        mouse = self.tracker.ensure_mouse()
        if mouse.fell_back:
            self.gui.set_status(f"Razer unavailable, using Win32 · {mouse.fallback_reason[:80]}")
        else:
            self.gui.set_status(f"Mouse: {mouse.label} · bind [{format_bind(self.config.bind_key)}]")
        if self.config.show_overlay and self.overlay is None:
            try:
                self.overlay = FovOverlay(
                    get_radius=lambda: self.config.effective_fov(self.tracker.bind_held()),
                    get_enabled=lambda: self.config.show_overlay,
                    get_debug=self._get_overlay_debug,
                )
            except Exception as exc:
                self.config.show_overlay = False
                self.gui.set_status(f"Overlay disabled: {exc}")
        self._stop_event.clear()
        self._running = True
        self._capture_fallback_notified = False
        self._infer_thread = threading.Thread(target=self._infer_loop, name="exoai-infer", daemon=True)
        self._control_thread = threading.Thread(target=self._control_loop, name="exoai-control", daemon=True)
        self._infer_thread.start()
        self._control_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        for thread in (self._infer_thread, self._control_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2)
        self._infer_thread = None
        self._control_thread = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        cv2.destroyAllWindows()
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

    def _read_snapshot(self) -> InferenceSnapshot:
        snap = self._snapshot
        return InferenceSnapshot(
            detections=list(snap.detections),
            debug_frame=snap.debug_frame,
            inference_ms=snap.inference_ms,
            detect_hz=snap.detect_hz,
            updated_at=snap.updated_at,
            snap_id=snap.snap_id,
        )

    def _get_overlay_debug(self) -> OverlayDebugData:
        if not self.tracker.bind_held() or self.detector is None:
            return OverlayDebugData(state=self._overlay_state)
        snap = self._snapshot
        dets = list(snap.detections)
        hz = snap.detect_hz
        ms = snap.inference_ms
        cx, cy = self.detector.center_x, self.detector.center_y
        det_points = [(d.x - cx, d.y - cy, d.confidence) for d in dets]
        tgt_point = None
        if self.tracker.has_lock():
            tgt_point = self.tracker.remaining()
        return OverlayDebugData(
            detections=det_points,
            target=tgt_point,
            detect_hz=hz,
            infer_ms=ms,
            state=self._overlay_state,
        )

    def _infer_loop(self) -> None:
        assert self.detector is not None
        detect_count = 0
        detect_timer = time.perf_counter()
        cx = self.detector.center_x
        cy = self.detector.center_y
        last_seq = -1
        is_color = self._is_color_mode
        while not self._stop_event.is_set():
            tick = time.perf_counter()
            held = self.tracker.bind_held()
            fov = self.config.effective_fov(held)
            try:
                raw, frame, _region = self.detector.detect(fov, wait_for_new=False)
            except Exception as exc:
                self.gui.set_status(f"Detection stopped: {type(exc).__name__}: {exc}")
                self._running = False
                self._stop_event.set()
                return
            cur_seq = self.detector._capture_seq
            if cur_seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = cur_seq
            now = time.perf_counter()
            
            # Convert color targets to Detection for scanner compatibility
            if is_color and raw:
                raw = [
                    Detection(x=t.x, y=t.y, confidence=t.confidence, 
                              width=t.width, height=t.height, cls=0, kind="enemy")
                    for t in raw
                ]
            
            target = None
            if held and raw:
                target = self.scanner.pick_target(raw, cx, cy, self.config.aim_offset_y, fov_radius=fov)
                if target is None:
                    self.tracker.reset_aim()
            elif not held:
                self.tracker.reset_aim()
                self.scanner.reset()
            end_tick = time.perf_counter()
            infer_ms = (end_tick - tick) * 1000.0
            self._last_infer_s = infer_ms / 1000.0
            detect_count += 1
            elapsed_since_reset = end_tick - detect_timer
            if elapsed_since_reset > 0.05:
                rolling_hz = detect_count / elapsed_since_reset
            else:
                rolling_hz = self._snapshot.detect_hz
            if target is not None and self.scanner.fresh:
                self.tracker.observe(target.x, target.y, cx, cy, now, target.confidence)
                # Aim steps only in control loop — avoids double-move canceling itself
            self._snapshot.detections = raw
            self._snapshot.debug_frame = frame
            self._snapshot.inference_ms = infer_ms
            self._snapshot.detect_hz = rolling_hz
            self._snapshot.updated_at = tick
            self._snapshot.snap_id += 1
            if elapsed_since_reset >= 1.0:
                detect_count = 0
                detect_timer = end_tick

    def _control_loop(self) -> None:
        assert self.detector is not None
        preset_label = get_preset(self.config.active_model).label
        bind_label = format_bind(self.config.bind_key)
        stats_timer = 0.0
        last_step = time.perf_counter()
        while not self._stop_event.is_set():
            tick_start = time.perf_counter()
            dt = tick_start - last_step
            last_step = tick_start
            held = self.tracker.bind_held()
            snapshot = self._read_snapshot()
            loop_interval = AIM_LOOP_S if held else IDLE_LOOP_S
            if not held:
                state = "SCAN"
                detail = f"Hold [{bind_label}] to aim"
                conf = None
            else:
                self.tracker.tick(dt, self._last_infer_s)
                if self.tracker.has_lock():
                    state = "TRACK"
                    conf = self.tracker.confidence()
                    mx, my = self.tracker.last_move()
                    mouse = self.tracker.mouse
                    mlabel = mouse.label if mouse else "?"
                    detail = f"{mlabel} · move {mx},{my} · hold [{bind_label}]"
                elif snapshot.detections:
                    state = "FOV"
                    conf = max(d.confidence for d in snapshot.detections)
                    detail = f"{preset_label} · target in FOV · hold [{bind_label}]"
                else:
                    state = "SCAN"
                    conf = None
                    detail = f"{preset_label} · scanning · hold [{bind_label}]"
            self._overlay_state = state
            if not self._capture_fallback_notified and self.detector.capture_fell_back():
                self._capture_fallback_notified = True
                self.gui.set_status("DXGI blocked (Vanguard?) - using MSS capture")
            if tick_start - stats_timer >= 0.2:
                self.gui.set_stats(
                    state=state,
                    detect_hz=snapshot.detect_hz,
                    infer_ms=snapshot.inference_ms,
                    target_conf=conf,
                    detail=detail,
                )
                stats_timer = tick_start
            if held and self.config.show_debug and snapshot.debug_frame is not None:
                frame = snapshot.debug_frame
                cx = frame.shape[1] // 2
                cy = frame.shape[0] // 2
                r = min(self.config.effective_fov(True), min(cx, cy))
                cv2.circle(frame, (cx, cy), r, (0, 255, 136), 2)
                if self.tracker.has_lock():
                    ex, ey = self.tracker.remaining()
                    cv2.circle(frame, (int(cx + ex), int(cy + ey)), 5, (0, 255, 255), -1)
                for det in snapshot.detections:
                    lx = int(det.x - (self.detector.center_x - cx))
                    ly = int(det.y - (self.detector.center_y - cy))
                    cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)
                cv2.imshow("Carbon Debug", frame)
                cv2.waitKey(1)
            _precise_wait(loop_interval - (time.perf_counter() - tick_start))

    def run(self) -> None:
        self.gui.run()


def main() -> None:
    _enable_1ms_timer()
    TrackerApp().run()


if __name__ == "__main__":
    main()
