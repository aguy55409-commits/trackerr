"""Dual-thread: steady-cadence detect + move (no bursty seq gating)."""
from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path

from exoai_tracker.aim import HOSTILE, Aimer, DetectionFilter, TargetSelector, pick_target
from exoai_tracker.binds import is_bind_held
from exoai_tracker.dxgi_capture import Capture
from exoai_tracker.infer import Det, Infer
from exoai_tracker.input_handler import InputHandler
from exoai_tracker.recoil import RecoilController
from exoai_tracker.trigger import PRESS, RELEASE, Trigger, covers

DET_INTERVAL = 0.001   # 1ms - as fast as inference allows
MOVE_INTERVAL = 0.001  # 1ms = ~1000 Hz moves


def enable_1ms_timer() -> None:
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def _prio(level: int) -> None:
    try:
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), level
        )
    except Exception:
        pass


@dataclass
class Stats:
    state: str = "IDLE"
    detect_hz: float = 0.0
    move_hz: float = 0.0
    infer_ms: float = 0.0
    frame_ms: float = 0.0
    loop_us: float = 0.0
    conf: float | None = None
    detail: str = "Ready"
    detections: list[Det] = field(default_factory=list)
    mouse_label: str = "Razer"
    # The ceiling the capture is paced at (monitor refresh or fps_cap). Lets
    # the UI show "det Hz / ceiling Hz" so a drop below the ceiling is obvious.
    cap_hz: float = 0.0


class Pipeline:
    def __init__(
        self,
        model_path: Path,
        *,
        bind: str = "right",
        speed: float = 3.5,
        fov: int = 120,
        conf: float = 0.35,
        iou: float = 0.40,
        capture_size: int = 320,
        net_size: int = 640,
        capture_method: str = "dxcam",
        offset_y: float = 0.55,
        get_fov=None,
        infer: Infer | None = None,
        assist_mode: str = "aim",
        trigger_delay_ms: float = 40.0,
        trigger_cooldown_ms: float = 140.0,
        trigger_hold_ms: float = 12.0,
        trigger_box_scale: float = 0.55,
        pid_kp: float = 0.78,
        pid_ki: float = 0.8,
        pid_kd: float = 0.002,
        pid_ff: float = 0.96,
        pid_smoothing: float = 0.65,
        input_mode: str = "mouse",
        controller_bind: str = "gp_lt",
        aim_tuning: dict | None = None,
        capture_fps_cap: int = 0,
        capture_offset_x: int = 0,
        capture_offset_y: int = 0,
        capture_monitor: int = 0,
        capture_window: str = "",
        mouse_method: str = "razer",
        # Advanced aim settings
        max_aim_speed: float = 0.0,
        aim_acceleration: float = 0.0,
        sticky_aim_time: float = 0.15,
        target_lock_time: float = 0.08,
        humanize_jitter: float = 0.0,
        flick_reduction: float = 0.0,
        easing_curve: str = "linear",
        # Detection filtering
        min_box_size: int = 15,
        max_box_size: int = 400,
        min_aspect_ratio: float = 0.8,
        max_aspect_ratio: float = 4.0,
        target_priority: str = "closest",
        filter_own_player: bool = True,
        detection_smoothing: int = 1,
        prediction_frames: int = 0,
        # Recoil control
        recoil_enabled: bool = False,
        recoil_strength: float = 1.5,
        recoil_horizontal: float = 0.0,
        recoil_interval_ms: float = 10.0,
        recoil_ramp_ms: float = 100.0,
        recoil_bind: str = "",
    ) -> None:
        self._mouse_bind = bind
        self._mouse_method = mouse_method
        self._capture_size = capture_size
        self._controller_bind = controller_bind
        self._input_mode = input_mode
        self.bind = controller_bind if input_mode == "controller" else bind
        self.fov = fov
        self.offset_y = offset_y
        self.get_fov = get_fov
        self.assist_mode = assist_mode
        self.trigger = Trigger()
        self.trigger.configure(
            delay_ms=trigger_delay_ms,
            cooldown_ms=trigger_cooldown_ms,
            hold_ms=trigger_hold_ms,
            box_scale=trigger_box_scale,
        )
        self.capture = Capture(
            capture_size,
            capture_method,
            color="BGR",
            fps_cap=capture_fps_cap,
            offset_x=capture_offset_x,
            offset_y=capture_offset_y,
            monitor=capture_monitor,
            window_title=capture_window,
        )
        self.infer = infer or Infer(model_path, conf=conf, iou=iou, imgsz=net_size)
        self.infer.update(conf=conf, iou=iou)
        self.aimer = Aimer()
        self.aimer.configure(speed)
        self.aimer.configure_pid(kp=pid_kp, ki=pid_ki, kd=pid_kd, ff=pid_ff, smoothing=pid_smoothing)
        self.aimer.configure_tuning(aim_tuning)
        self.aimer.set_velocity_output(input_mode == "controller")
        self.aimer.configure_advanced(
            max_speed=max_aim_speed,
            acceleration=aim_acceleration,
            jitter=humanize_jitter,
            flick_reduction=flick_reduction,
            easing=easing_curve,
        )
        
        # Detection filtering and target selection
        self.det_filter = DetectionFilter(
            min_size=min_box_size,
            max_size=max_box_size,
            min_aspect=min_aspect_ratio,
            max_aspect=max_aspect_ratio,
            filter_own=filter_own_player,
            smoothing=detection_smoothing,
            capture_size=capture_size,
        )
        self.target_selector = TargetSelector(
            priority=target_priority,
            lock_time=target_lock_time,
            sticky_time=sticky_aim_time,
        )
        self._prediction_frames = prediction_frames
        
        # Recoil control
        self.recoil = RecoilController(
            strength=recoil_strength,
            horizontal=recoil_horizontal,
            interval_ms=recoil_interval_ms,
            ramp_ms=recoil_ramp_ms,
        )
        self._recoil_enabled = recoil_enabled
        # Recoil bind: if empty, uses main bind; otherwise its own
        self._recoil_bind = recoil_bind if recoil_bind else None
        
        self.mouse = InputHandler(mode=input_mode, mouse_method=mouse_method)
        self.stats = Stats()
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self._aim_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._det_t: threading.Thread | None = None
        self._mov_t: threading.Thread | None = None
        # Opt-in aim tracing, written to aimlog.txt. Off unless EXOAI_AIMLOG is
        # set, so the 1ms move loop pays nothing in normal use.
        self._log_on = bool(os.environ.get("EXOAI_AIMLOG"))
        self._log: list[str] = []
        self._log_n = 0

    def _aimlog(self, tag: str, dx: int, dy: int, sent: bool) -> None:
        if not self._log_on:
            return
        self._log_n += 1
        # Press/release markers always land; per-step lines are sampled.
        if tag in ("move", "zero") and self._log_n % 25:
            return
        s = self.aimer.solver
        ctrl = getattr(self.mouse, "_controller", None)
        aim = ""
        if ctrl is not None:
            aim = f" stick=({getattr(ctrl, '_aim_x', 0):.0f},{getattr(ctrl, '_aim_y', 0):.0f})"
        sens = s.sens
        self._log.append(
            f"{time.perf_counter():10.3f} {tag:<7} dx={dx:>4} dy={dy:>4} sent={sent!s:<5} "
            f"trim={s.trim:.3f} vel={s.velocity_output!s:<5} "
            f"sens={'none' if not sens else f'{sens:.2f}'} "
            f"err=({self.aimer.err_x:7.1f},{self.aimer.err_y:7.1f}) "
            f"mode={self.mouse.mode} ok={self.mouse.ok!s:<5}{aim}"
        )
        if tag == "release" or len(self._log) > 3000:
            self._flush_log()

    def snapshot_stats(self) -> Stats:
        with self._stats_lock:
            return replace(self.stats, detections=list(self.stats.detections))

    def _flush_log(self) -> None:
        if not self._log:
            return
        try:
            with open("aimlog.txt", "a", encoding="utf-8") as fh:
                fh.write("\n".join(self._log) + "\n")
        except OSError:
            pass
        self._log.clear()

    def start(self) -> None:
        enable_1ms_timer()
        if not self.mouse.init():
            raise RuntimeError("No input device found (Razer mouse or controller)")
        self.capture.start()
        self.aimer.set_aspect(self.capture.screen_w, self.capture.screen_h)
        self.stats.mouse_label = self.mouse.label
        end2 = "e2e" if getattr(self.infer, "_end2end", False) else "nms"
        cap_method = getattr(self.capture, 'active_method', self.capture.method)
        self.stats.detail = (
            f"{self.infer.path.name} · {self.infer.backend} · {end2} · "
            f"{cap_method} · {self.mouse.label}"
        )
        self._stop.clear()
        self._det_t = threading.Thread(target=self._detect_loop, name="detect", daemon=True)
        self._mov_t = threading.Thread(target=self._move_loop, name="move", daemon=True)
        self._det_t.start()
        self._mov_t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._det_t, self._mov_t):
            if t and t.is_alive():
                t.join(timeout=1.5)
        self._det_t = self._mov_t = None
        # After the loops are down, so nothing can press again behind us.
        self._disarm()
        self.capture.stop()
        self.aimer.reset()
        self.stats.state = "IDLE"
        self.stats.detail = "Stopped"

    def apply(
        self,
        *,
        bind: str | None = None,
        speed: float | None = None,
        fov: int | None = None,
        conf: float | None = None,
        iou: float | None = None,
        offset_y: float | None = None,
        assist_mode: str | None = None,
        trigger_delay_ms: float | None = None,
        trigger_cooldown_ms: float | None = None,
        trigger_hold_ms: float | None = None,
        trigger_box_scale: float | None = None,
        pid_kp: float | None = None,
        pid_ki: float | None = None,
        pid_kd: float | None = None,
        pid_ff: float | None = None,
        pid_smoothing: float | None = None,
        input_mode: str | None = None,
        controller_bind: str | None = None,
        aim_tuning: dict | None = None,
        max_aim_speed: float | None = None,
        aim_acceleration: float | None = None,
        sticky_aim_time: float | None = None,
        target_lock_time: float | None = None,
        humanize_jitter: float | None = None,
        flick_reduction: float | None = None,
        easing_curve: str | None = None,
        min_box_size: int | None = None,
        max_box_size: int | None = None,
        min_aspect_ratio: float | None = None,
        max_aspect_ratio: float | None = None,
        target_priority: str | None = None,
        filter_own_player: bool | None = None,
        detection_smoothing: int | None = None,
        prediction_frames: int | None = None,
        # Recoil control
        recoil_enabled: bool | None = None,
        recoil_strength: float | None = None,
        recoil_horizontal: float | None = None,
        recoil_interval_ms: float | None = None,
        recoil_ramp_ms: float | None = None,
        recoil_bind: str | None = None,
    ) -> None:
        if bind is not None:
            self._mouse_bind = bind
        if controller_bind is not None:
            self._controller_bind = controller_bind
        if input_mode is not None:
            self._input_mode = input_mode
            self.mouse.set_mode(input_mode)
            with self._aim_lock:
                self.aimer.set_velocity_output(input_mode == "controller")
        # Use correct bind based on mode
        self.bind = self._controller_bind if self._input_mode == "controller" else self._mouse_bind
        if assist_mode is not None and assist_mode != self.assist_mode:
            self.assist_mode = assist_mode
            self._disarm()
        self.trigger.configure(
            delay_ms=trigger_delay_ms,
            cooldown_ms=trigger_cooldown_ms,
            hold_ms=trigger_hold_ms,
            box_scale=trigger_box_scale,
        )
        if speed is not None:
            with self._aim_lock:
                self.aimer.configure(speed)
        # PID tuning (always under lock since it touches internal state)
        with self._aim_lock:
            self.aimer.configure_pid(
                kp=pid_kp, ki=pid_ki, kd=pid_kd, ff=pid_ff, smoothing=pid_smoothing
            )
            if aim_tuning is not None:
                # Reapply defaults first so clearing a knob in the UI restores
                # it rather than leaving the last value stuck.
                self.aimer.configure_tuning(Aimer.solver_defaults())
                self.aimer.configure_tuning(aim_tuning)
            self.aimer.configure_advanced(
                max_speed=max_aim_speed,
                acceleration=aim_acceleration,
                jitter=humanize_jitter,
                flick_reduction=flick_reduction,
                easing=easing_curve,
            )
        with self._target_lock:
            self.det_filter.configure(
                min_size=min_box_size,
                max_size=max_box_size,
                min_aspect=min_aspect_ratio,
                max_aspect=max_aspect_ratio,
                filter_own=filter_own_player,
                smoothing=detection_smoothing,
            )
            self.target_selector.configure(
                priority=target_priority,
                lock_time=target_lock_time,
                sticky_time=sticky_aim_time,
            )
            if prediction_frames is not None:
                self._prediction_frames = max(0, int(prediction_frames))
        if fov is not None:
            self.fov = int(fov)
        if offset_y is not None:
            self.offset_y = float(offset_y)
        if conf is not None or iou is not None:
            self.infer.update(conf=conf, iou=iou)
        
        # Recoil control
        if recoil_enabled is not None:
            self._recoil_enabled = recoil_enabled
        if recoil_bind is not None:
            self._recoil_bind = recoil_bind if recoil_bind else None
        self.recoil.configure(
            strength=recoil_strength,
            horizontal=recoil_horizontal,
            interval_ms=recoil_interval_ms,
            ramp_ms=recoil_ramp_ms,
        )

    def _disarm(self) -> None:
        """Let go of the button if the trigger was holding it."""
        if self.trigger.disarm():
            self.mouse.release()

    def _service_trigger(self, dets: list[Det], now: float) -> None:
        has_trigger = self.assist_mode in ("trigger", "both")
        if not has_trigger:
            self._disarm()
            return
        cx, cy = self.capture.center_x, self.capture.center_y
        # Measured against the raw boxes. pick_target hands back a detection
        # whose centre has been moved to the aim point, so testing that would
        # ask whether the crosshair is on the chosen bone rather than on the
        # player.
        on = any(
            d.kind in HOSTILE
            and covers(cx, cy, d.x, d.y, d.w, d.h, self.trigger.scale)
            for d in dets
        )
        act = self.trigger.update(on, now)
        if act == PRESS:
            self.mouse.press()
        elif act == RELEASE:
            self.mouse.release()

    def _fov_now(self) -> float:
        if self.get_fov is not None:
            return float(self.get_fov())
        return float(self.fov)

    def _wait_until(self, deadline: float) -> float:
        while not self._stop.is_set():
            now = time.perf_counter()
            rem = deadline - now
            if rem <= 0:
                return now
            time.sleep(min(rem, 0.001))
        return time.perf_counter()

    def _detect_loop(self) -> None:
        _prio(2)  # Highest priority
        n = 0
        t0 = time.perf_counter()
        last_seq = -1
        while not self._stop.is_set():
            if not is_bind_held(self.bind):
                with self._aim_lock:
                    self.aimer.reset()
                self._disarm()
                self.stats.state = "IDLE"
                time.sleep(0.002)
                continue

            frame = self.capture.wait_for_frame(last_seq, timeout=0.02)
            if frame is None or frame.seq == last_seq:
                continue
            last_seq = frame.seq

            t_inf = time.perf_counter()
            dets = self.infer.predict(frame.rgb, frame.left, frame.top)
            ms = (time.perf_counter() - t_inf) * 1000.0
            now = time.perf_counter()
            frame_ms = (now - getattr(frame, "captured_at", now)) * 1000.0

            cx, cy = self.capture.center_x, self.capture.center_y
            with self._target_lock:
                filtered = self.det_filter.filter(dets, cx, cy)
                tgt = self.target_selector.select(
                    filtered, cx, cy,
                    fov=self._fov_now(),
                    offset_y=self.offset_y,
                    now=now,
                )
                if tgt is not None:
                    tgt = self.det_filter.smooth(tgt, now)
                    if self._prediction_frames > 0:
                        tgt = self.det_filter.predict(
                            tgt, self._prediction_frames, now
                        )

            with self._aim_lock:
                self.aimer.observe(tgt, dets, offset_y=self.offset_y)
                tracking = self.aimer.target is not None

            self._service_trigger(filtered, time.perf_counter())

            with self._stats_lock:
                self.stats.detections = dets
                self.stats.infer_ms = ms
                self.stats.frame_ms = frame_ms
                self.stats.conf = None if tgt is None else tgt.conf
                self.stats.state = "TRACK" if tracking else ("FOV" if dets else "SCAN")
                self.stats.cap_hz = float(
                    getattr(self.capture, "active_target_fps", 0)
                )

            n += 1
            now = time.perf_counter()
            if now - t0 >= 0.25:
                with self._stats_lock:
                    self.stats.detect_hz = n / (now - t0)
                n = 0
                t0 = now

    def _move_loop(self) -> None:
        _prio(1)
        next_t = time.perf_counter()
        n = 0
        t0 = time.perf_counter()
        was_held = False
        while not self._stop.is_set():
            if not is_bind_held(self.bind):
                if was_held:
                    # Recentre the stick and start the next press from a clean
                    # solver: gain trim, integrals and accumulators all carry
                    # otherwise, so a hold that ended with a beaten-down gain
                    # would cripple the following one.
                    self.mouse.reset_controller()
                    with self._aim_lock:
                        self.aimer.solver.reset()
                    self._aimlog("release", 0, 0, False)
                    was_held = False
                next_t = time.perf_counter() + MOVE_INTERVAL
                time.sleep(0.001)
                continue
            if not was_held:
                self._aimlog("PRESS", 0, 0, False)
            was_held = True

            if self.assist_mode == "trigger":
                time.sleep(0.001)
                continue

            now = self._wait_until(next_t)
            next_t += MOVE_INTERVAL
            if now - next_t > 0.03:
                next_t = now + MOVE_INTERVAL

            t_m = time.perf_counter()
            
            # Check which features are active
            has_aim = self.assist_mode in ("aim", "both")
            has_recoil = self.assist_mode == "recoil"
            
            # Aim assist
            dx, dy = 0, 0
            if has_aim:
                with self._aim_lock:
                    dx, dy = self.aimer.step(
                        self.capture.center_x,
                        self.capture.center_y,
                        MOVE_INTERVAL,
                    )
            
            # Recoil compensation
            rx, ry = 0, 0
            if has_recoil:
                recoil_bind = self._recoil_bind or self.bind
                if is_bind_held(recoil_bind):
                    if not self.recoil.active:
                        self.recoil.start()
                    rx, ry = self.recoil.step()
                else:
                    if self.recoil.active:
                        self.recoil.stop()
            
            # Combine aim + recoil
            final_dx = dx + rx
            final_dy = dy + ry
            
            if final_dx or final_dy:
                sent = self.mouse.move(final_dx, final_dy)
                with self._aim_lock:
                    self.aimer.note_move(dx, dy)  # Only log aim portion for sens estimation
                self._aimlog("move", final_dx, final_dy, sent)
            else:
                self._aimlog("zero", 0, 0, False)
            with self._stats_lock:
                self.stats.loop_us = (time.perf_counter() - t_m) * 1e6
            n += 1

            if now - t0 >= 0.25:
                move_hz = n / (now - t0)
                n = 0
                t0 = now
                dx, dy = self.aimer.last_dx, self.aimer.last_dy
                sens = self.aimer.solver.sens
                sens_txt = (
                    f"{sens:.1f}px/ct {self.aimer.solver.lat_ms:.0f}ms"
                    if sens else "learning"
                )
                with self._stats_lock:
                    self.stats.move_hz = move_hz
                    self.stats.detail = (
                        f"{self.stats.state} · {self.stats.detect_hz:.0f}detHz · "
                        f"{move_hz:.0f}movHz · cap {self.capture.publish_hz:.0f}Hz · "
                        f"{self.stats.infer_ms:.1f}ms · move {dx},{dy} · "
                        f"age {self.stats.frame_ms:.1f}ms · "
                        f"err {self.aimer.err_x:.0f},{self.aimer.err_y:.0f} · "
                        f"sens {sens_txt} · "
                        f"miss {self.aimer._misses} · {self.mouse.label}"
                    )
