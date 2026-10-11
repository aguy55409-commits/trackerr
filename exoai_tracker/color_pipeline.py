"""Ultra-fast color detection pipeline — <1ms detect, no neural network."""
from __future__ import annotations

import ctypes
import math
import threading
import time
from dataclasses import dataclass, replace

from exoai_tracker.aim import MoveSolver
from exoai_tracker.binds import is_bind_held
from exoai_tracker.color_detector import ColorDetector, ColorTarget
from exoai_tracker.input_handler import InputHandler
from exoai_tracker.recoil import RecoilController
from exoai_tracker.trigger import PRESS, RELEASE, Trigger, covers


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
class ColorStats:
    state: str = "IDLE"
    detect_hz: float = 0.0
    move_hz: float = 0.0
    detect_ms: float = 0.0
    loop_us: float = 0.0
    detail: str = "Ready"
    mouse_label: str = "Razer"


class ColorPipeline:
    """Fast color-based tracking with smooth aim."""

    def __init__(
        self,
        *,
        bind: str = "right",
        speed: float = 2.5,
        fov: int = 120,
        capture_size: int = 320,
        capture_method: str = "dxcam",
        capture_monitor: int = 0,
        capture_offset_x: int = 0,
        capture_offset_y: int = 0,
        color_preset: str = "red",
        offset_y: float = 0.45,
        get_fov=None,
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
        mouse_method: str = "razer",
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
        self._controller_bind = controller_bind
        self._input_mode = input_mode
        self.bind = controller_bind if input_mode == "controller" else bind
        self.fov = fov
        self.offset_y = offset_y
        self.get_fov = get_fov
        self.speed = speed
        self.assist_mode = assist_mode
        self.trigger = Trigger()
        self.trigger.configure(
            delay_ms=trigger_delay_ms,
            cooldown_ms=trigger_cooldown_ms,
            hold_ms=trigger_hold_ms,
            box_scale=trigger_box_scale,
        )
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_ff = pid_ff
        self._pid_smoothing = pid_smoothing
        self.detector = ColorDetector(
            capture_size=capture_size,
            color_preset=color_preset,
            capture_method=capture_method,
            monitor=capture_monitor,
            offset_x=capture_offset_x,
            offset_y=capture_offset_y,
        )
        self.mouse = InputHandler(mode=input_mode, mouse_method=mouse_method)
        self.stats = ColorStats()
        self._stats_lock = threading.Lock()
        self.solver = MoveSolver()
        self.solver.configure(speed)
        self.solver.configure_pid(
            kp=pid_kp, ki=pid_ki, kd=pid_kd, ff=pid_ff, smoothing=pid_smoothing
        )
        self.solver.configure_tuning(aim_tuning)
        self.solver.velocity_output = input_mode == "controller"
        
        # Recoil control
        self.recoil = RecoilController(
            strength=recoil_strength,
            horizontal=recoil_horizontal,
            interval_ms=recoil_interval_ms,
            ramp_ms=recoil_ramp_ms,
        )
        self._recoil_enabled = recoil_enabled
        self._recoil_bind = recoil_bind if recoil_bind else None
        
        self._stop = threading.Event()
        self._lock = threading.Lock()
        
        # Latest measured target position (screen coords)
        self._tx = 0.0
        self._ty = 0.0
        self._has_target = False
        self._fresh = False
        self._misses = 0
        self._last_move = 0.0
        self._locked_xy: tuple[float, float] | None = None
        self._locked_at = 0.0
        
        self._det_t: threading.Thread | None = None
        self._mov_t: threading.Thread | None = None

    def snapshot_stats(self) -> ColorStats:
        with self._stats_lock:
            return replace(self.stats)

    def start(self) -> None:
        enable_1ms_timer()
        if not self.mouse.init():
            raise RuntimeError("No input device found (Razer mouse or controller)")
        self.stats.mouse_label = self.mouse.label
        self.stats.detail = f"Color · DXGI · {self.mouse.label}"
        self._stop.clear()
        self._has_target = False
        self._det_t = threading.Thread(target=self._detect_loop, name="color-detect", daemon=True)
        self._mov_t = threading.Thread(target=self._move_loop, name="color-move", daemon=True)
        self._det_t.start()
        self._mov_t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._det_t, self._mov_t):
            if t and t.is_alive():
                t.join(timeout=1.5)
        self._det_t = self._mov_t = None
        self.detector.close()
        self._has_target = False
        self._disarm()
        self.stats.state = "IDLE"
        self.stats.detail = "Stopped"

    def apply(
        self,
        *,
        bind: str | None = None,
        speed: float | None = None,
        fov: int | None = None,
        color_preset: str | None = None,
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
            velocity = input_mode == "controller"
            if velocity != self.solver.velocity_output:
                # Stale integrals and a stale gain fit do not carry between the
                # stick and mouse paths.
                self.solver.velocity_output = velocity
                self.solver.reset()
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
            self.speed = max(0.5, min(5.0, float(speed)))
            self.solver.configure(self.speed)
        # PID tuning (always under lock)
        with self._lock:
            self.solver.configure_pid(
                kp=pid_kp, ki=pid_ki, kd=pid_kd, ff=pid_ff, smoothing=pid_smoothing
            )
            if aim_tuning is not None:
                # Defaults first, so clearing a knob restores it.
                self.solver.configure_tuning(MoveSolver.tuning_defaults())
                self.solver.configure_tuning(aim_tuning)
        if fov is not None:
            self.fov = int(fov)
        if color_preset is not None:
            self.detector.set_color(color_preset)
        if offset_y is not None:
            self.offset_y = float(offset_y)
        
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
        if self.trigger.disarm():
            self.mouse.release()

    def _service_trigger(self, targets: list[ColorTarget], now: float) -> None:
        has_trigger = self.assist_mode in ("trigger", "both")
        if not has_trigger:
            self._disarm()
            return
        cx, cy = self.detector.center_x, self.detector.center_y
        on = any(
            covers(cx, cy, t.x, t.y, t.width, t.height, self.trigger.scale)
            for t in targets
        )
        act = self.trigger.update(on, now)
        if act == PRESS:
            self.mouse.press()
        elif act == RELEASE:
            self.mouse.release()

    def _fov_now(self) -> int:
        if self.get_fov is not None:
            return int(self.get_fov())
        return self.fov

    def _detect_loop(self) -> None:
        _prio(2)  # Highest priority
        n = 0
        t0 = time.perf_counter()
        cx = self.detector.center_x
        cy = self.detector.center_y
        last_seq = -1

        while not self._stop.is_set():
            if not is_bind_held(self.bind):
                with self._lock:
                    self._has_target = False
                    self._misses = 0
                    self._locked_xy = None
                    self.solver.reset()
                self._disarm()
                self.stats.state = "IDLE"
                time.sleep(0.001)
                continue

            seq = self.detector.capture_seq
            if seq == last_seq:
                time.sleep(0.0002)
                continue
            last_seq = seq
            t_det = time.perf_counter()
            targets, _, _ = self.detector.detect(self._fov_now())
            ms = (time.perf_counter() - t_det) * 1000.0

            # Before the offset below, which edits the target in place: that
            # would shift the box away from the player and make the hit test
            # ask about the aim point instead.
            self._service_trigger(targets, time.perf_counter())

            # Keep the same nearby blob instead of hopping whenever two color
            # targets exchange closest-to-crosshair order.
            best: ColorTarget | None = None
            if targets:
                best = targets[0]
                if self._locked_xy and time.perf_counter() - self._locked_at < 0.2:
                    lx, ly = self._locked_xy
                    candidate = min(
                        targets, key=lambda t: (t.x - lx) ** 2 + (t.y - ly) ** 2
                    )
                    if (candidate.x - lx) ** 2 + (candidate.y - ly) ** 2 < (
                        max(candidate.width, candidate.height) * 2.0
                    ) ** 2:
                        best = candidate
                self._locked_xy = (best.x, best.y)
                self._locked_at = time.perf_counter()

            with self._lock:
                if best:
                    self._tx = best.x
                    self._ty = best.y - best.height * (0.5 - self.offset_y)
                    self._has_target = True
                    self._fresh = True
                    self._misses = 0
                else:
                    self._misses += 1
                    if self._misses > 5:
                        self._has_target = False
                        self._fresh = False

            with self._stats_lock:
                self.stats.detect_ms = ms
                self.stats.state = "TRACK" if self._has_target else "SCAN"

            n += 1
            now = time.perf_counter()
            if now - t0 >= 0.25:
                with self._stats_lock:
                    self.stats.detect_hz = n / (now - t0)
                n = 0
                t0 = now

    def _move_loop(self) -> None:
        _prio(2)
        n = 0
        t0 = time.perf_counter()
        cx = self.detector.center_x
        cy = self.detector.center_y

        was_held = False
        while not self._stop.is_set():
            if not is_bind_held(self.bind):
                if was_held:
                    # Recentre the stick and start the next press from a clean
                    # solver: gain trim, integrals and accumulators all carry
                    # otherwise, so a hold that ended with a beaten-down gain
                    # would cripple the following one.
                    self.mouse.reset_controller()
                    with self._lock:
                        self.solver.reset()
                    was_held = False
                time.sleep(0.001)
                continue
            was_held = True

            # Trigger-only leaves the aim alone. Solving anyway would log the
            # counts as sent and fit the sensitivity to movement that never
            # happened.
            if self.assist_mode == "trigger":
                time.sleep(0.001)
                continue

            t_m = time.perf_counter()

            # One move per detection: acting between detections pushes on an
            # error already corrected but not re-measured, which orbits.
            with self._lock:
                fresh = self._fresh and self._has_target
                tx, ty = self._tx, self._ty
                self._fresh = False

            if not fresh:
                time.sleep(0.0005)
                continue

            # Check which features are active
            has_aim = self.assist_mode in ("aim", "both")
            has_recoil = self.assist_mode == "recoil"
            
            # Aim assist
            ix, iy = 0, 0
            if has_aim:
                with self._lock:
                    ix, iy = self.solver.solve(tx - cx, ty - cy)
            
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
            final_x = ix + rx
            final_y = iy + ry
            
            if final_x or final_y:
                self.mouse.move(final_x, final_y)
                if has_aim:
                    with self._lock:
                        self.solver.note_move(ix, iy)

            with self._stats_lock:
                self.stats.loop_us = (time.perf_counter() - t_m) * 1e6
            n += 1

            now = time.perf_counter()
            if now - t0 >= 0.25:
                move_hz = n / (now - t0)
                with self._stats_lock:
                    self.stats.move_hz = move_hz
                    self.stats.detail = (
                        f"{self.stats.state} · {self.stats.detect_hz:.0f}Hz · "
                        f"gain {self.solver.trim:.2f} · "
                        f"sens {f'{s:.1f}px/ct' if (s := self.solver.sens) else 'learning'} · "
                        f"{self.mouse.label}"
                    )
                n = 0
                t0 = now
