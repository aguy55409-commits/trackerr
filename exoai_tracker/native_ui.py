"""Neverlose-style ImGui menu.

Simplified sidebar navigation with clean flat design.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict
from typing import Callable

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp

from exoai_tracker.aim import MoveSolver
from exoai_tracker.binds import PRESET_BINDS, format_bind, normalize_bind, wait_for_bind
from exoai_tracker.config import (
    AppConfig,
    delete_profile,
    list_profiles,
    load_profile,
    save_config,
    save_profile,
)

APPLY_DEBOUNCE = 0.35
NAV_W = 156
ACCENT = (0.62, 0.86, 1.0, 1.0)


def _bind_label(name: str) -> str:
    return format_bind(name).replace("\U0001f3ae", "").strip()


class NativeUI:
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
            "detail": "Ready",
            "mouse_label": "Razer",
            "cap_hz": 0.0,
            "move_hz": 0.0,
            "frame_ms": 0.0,
        }
        self._dirty_at: float | None = None
        self._tab = 0
        self._capturing = False
        self._capture_target = "bind_key"
        self._profiles = list_profiles()
        self._profile_name = ""
        self._toast = ""
        self._toast_at = 0.0
        self._fade = 1.0
        self._fade_tab = 0
        self._pill_y: float | None = None
        self._pill_target: float | None = None
        self._tab_rects: list[tuple[float, float, float, float]] = []

    def set_stats(self, *, state: str, detect_hz: float, infer_ms: float,
                  target_conf: float | None, detail: str, mouse_label: str | None = None,
                  cap_hz: float = 0.0, move_hz: float = 0.0, frame_ms: float = 0.0) -> None:
        with self._lock:
            self._stats.update(state=state, detect_hz=detect_hz, infer_ms=infer_ms,
                               target_conf=target_conf, detail=detail,
                               cap_hz=cap_hz, move_hz=move_hz, frame_ms=frame_ms)
            if mouse_label:
                self._stats["mouse_label"] = mouse_label

    def set_status(self, text: str) -> None:
        with self._lock:
            self._stats["detail"] = text

    def set_mouse_label(self, text: str) -> None:
        with self._lock:
            self._stats["mouse_label"] = text

    def _touch(self, changed: bool) -> None:
        if changed:
            self._dirty_at = time.perf_counter()

    def _flush(self) -> None:
        if self._dirty_at and time.perf_counter() - self._dirty_at >= APPLY_DEBOUNCE:
            self._dirty_at = None
            self.config.clamp()
            save_config(self.config)
            self.on_apply(self.config)

    def _notify(self, text: str) -> None:
        self._toast = text
        self._toast_at = time.perf_counter()

    def _exp(self, cur: float, target: float, speed: float, eps: float = 0.004) -> float:
        """Frame-rate independent ease. One lerp per value, so 30 fps still looks fluid."""
        dt = imgui.get_io().delta_time
        if dt <= 0.0 or dt > 0.1:
            dt = 1.0 / 30.0
        nxt = cur + (target - cur) * (1.0 - math.exp(-speed * dt))
        if abs(target - nxt) < eps:
            return target
        return nxt

    def _tick_motion(self) -> None:
        if self._tab != self._fade_tab:
            # Stay readable while the page eases in. Never fade the whole window out.
            self._fade = 0.72
            self._fade_tab = self._tab
        self._fade = self._exp(self._fade, 1.0, 16.0)
        if self._pill_y is not None and self._pill_target is not None:
            self._pill_y = self._exp(self._pill_y, self._pill_target, 18.0, eps=0.4)
        moving = self._fade < 0.999 or (
            self._pill_y is not None
            and self._pill_target is not None
            and abs(self._pill_y - self._pill_target) > 0.4
        )
        try:
            hello_imgui.get_runner_params().fps_idling.fps_idle = 30.0 if moving else 20.0
        except Exception:
            pass

    def _begin_capture(self, target: str) -> None:
        if self._capturing:
            return
        self._capturing = True
        self._capture_target = target

        def worker():
            try:
                key = normalize_bind(wait_for_bind())
                setattr(self.config, self._capture_target, key)
                self.config.clamp()
                save_config(self.config)
                self.on_apply(self.config)
            except Exception:
                pass
            finally:
                self._capturing = False

        threading.Thread(target=worker, daemon=True).start()

    # --- liquid-glass helpers -----------------------------------------
    def _glass_card(self, label: str, size, pad=(14, 14)) -> bool:
        """Rounded translucent child with a soft border + top highlight."""
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 16.0)
        imgui.push_style_color(imgui.Col_.child_bg, (0.14, 0.20, 0.32, 0.92))
        imgui.push_style_color(imgui.Col_.border, (0.72, 0.88, 1.00, 0.55))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 1.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, pad)
        ok = imgui.begin_child(label, size, imgui.ChildFlags_.none)
        if ok:
            dl = imgui.get_window_draw_list()
            x0, y0 = imgui.get_window_pos()
            x1 = x0 + imgui.get_window_width()
            try:
                dl.add_line(
                    (x0 + 14, y0 + 1),
                    (x1 - 14, y0 + 1),
                    imgui.get_color_u32((0.90, 0.97, 1.00, 0.75)),
                    1.0,
                )
            except Exception:
                pass
        imgui.pop_style_var(2)
        imgui.pop_style_color(2)
        imgui.pop_style_var(1)
        return ok

    def _section(self, title: str) -> None:
        """Page title with a short accent mark. No extra panel."""
        imgui.dummy((0, 2))
        pos = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        try:
            dl.add_rect_filled(
                (pos[0], pos[1] + 2),
                (pos[0] + 3, pos[1] + 16),
                imgui.get_color_u32(ACCENT),
                2.0,
            )
        except Exception:
            pass
        imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + 12)
        imgui.text(title)
        imgui.dummy((0, 8))

    def _draw(self) -> None:
        self._flush()
        self._tick_motion()
        cfg = self.config
        running = self.is_running()
        with self._lock:
            st = dict(self._stats)

        display = imgui.get_io().display_size
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(display)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_bring_to_front_on_focus
        )
        imgui.begin("##main", None, flags)

        # One surface. A hairline under the title is the only chrome.
        dl = imgui.get_window_draw_list()
        origin = imgui.get_window_pos()
        try:
            dl.add_line(
                (origin.x + 18, origin.y + 52),
                (origin.x + display.x - 18, origin.y + 52),
                imgui.get_color_u32((0.75, 0.90, 1.0, 0.28)),
                1.0,
            )
        except Exception:
            pass

        imgui.set_cursor_pos((18, 16))
        imgui.text_colored(ACCENT, "NEVERWIN")
        imgui.same_line(0, 14)
        state_col = (0.45, 0.95, 0.70, 1.0) if running else (0.62, 0.68, 0.78, 1.0)
        det_hz = float(st.get("detect_hz") or 0.0)
        cap_hz = float(st.get("cap_hz") or 0.0)
        hz_txt = f"{det_hz:.0f} Hz" if cap_hz <= 0 else f"{det_hz:.0f}/{cap_hz:.0f} Hz"
        imgui.text_colored(state_col, f"{st['state']}   {hz_txt}")

        imgui.set_cursor_pos((16, 64))
        body_h = max(40.0, display.y - 80.0)
        self._nav(running, body_h)
        imgui.same_line(0, 0)

        imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (22, 8))
        page_open = imgui.begin_child("##page", (0, body_h), 0)
        if page_open:
            tabs = [
                self._tab_aimbot,
                self._tab_visuals,
                self._tab_detection,
                self._tab_performance,
                self._tab_advanced,
                self._tab_misc,
            ]
            if 0 <= self._tab < len(tabs):
                imgui.push_style_var(imgui.StyleVar_.alpha, self._fade)
                slide = (1.0 - self._fade) * 6.0
                cursor = imgui.get_cursor_pos()
                imgui.set_cursor_pos((cursor.x, cursor.y + slide))
                tabs[self._tab](cfg, st)
                imgui.pop_style_var()
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()

        if self._toast and time.perf_counter() - self._toast_at < 2.5:
            imgui.set_cursor_pos((display.x - 280, display.y - 40))
            imgui.push_style_var(imgui.StyleVar_.frame_rounding, 12.0)
            imgui.push_style_color(imgui.Col_.button, (0.12, 0.18, 0.28, 0.95))
            imgui.push_style_color(imgui.Col_.text, ACCENT)
            imgui.button(self._toast, (250, 26))
            imgui.pop_style_color(2)
            imgui.pop_style_var()

        imgui.end()

    def _nav(self, running: bool, body_h: float) -> None:
        imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (6, 4))
        if not imgui.begin_child("##nav", (NAV_W, body_h), 0):
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            return

        tabs = ["Aimbot", "Visuals", "Detection", "Perf", "Advanced", "Misc"]
        dl = imgui.get_window_draw_list()
        if self._tab_rects and 0 <= self._tab < len(self._tab_rects):
            x0, y0, x1, y1 = self._tab_rects[self._tab]
            self._pill_target = y0
            if self._pill_y is None:
                self._pill_y = y0
            try:
                dl.add_rect_filled(
                    (x0, self._pill_y),
                    (x1, self._pill_y + (y1 - y0)),
                    imgui.get_color_u32((0.55, 0.82, 1.0, 0.18)),
                    8.0,
                )
            except Exception:
                pass

        rects: list[tuple[float, float, float, float]] = []
        for i, name in enumerate(tabs):
            selected = self._tab == i
            imgui.push_style_var(imgui.StyleVar_.frame_rounding, 8.0)
            if selected:
                imgui.push_style_color(imgui.Col_.button, (0, 0, 0, 0))
                imgui.push_style_color(imgui.Col_.button_hovered, (0.55, 0.82, 1.0, 0.12))
                imgui.push_style_color(imgui.Col_.text, (0.96, 0.98, 1.0, 1.0))
            else:
                imgui.push_style_color(imgui.Col_.button, (0, 0, 0, 0))
                imgui.push_style_color(imgui.Col_.button_hovered, (1, 1, 1, 0.06))
                imgui.push_style_color(imgui.Col_.text, (0.62, 0.70, 0.80, 1.0))
            if imgui.button(name, (NAV_W - 20, 30)):
                if i != self._tab:
                    self._tab = i
                    self._fade = 0.72
                    self._fade_tab = i
            r0 = imgui.get_item_rect_min()
            r1 = imgui.get_item_rect_max()
            rects.append((r0.x, r0.y, r1.x, r1.y))
            imgui.pop_style_color(3)
            imgui.pop_style_var()
        self._tab_rects = rects

        # Start sits at the bottom of the same column, still inside this window.
        spare = imgui.get_content_region_avail().y
        if spare > 44:
            imgui.dummy((0, spare - 40))
        if running:
            cols = [
                (imgui.Col_.button, (0.55, 0.22, 0.28, 1.0)),
                (imgui.Col_.button_hovered, (0.70, 0.28, 0.34, 1.0)),
                (imgui.Col_.text, (0.98, 0.96, 0.96, 1.0)),
            ]
            label = "Stop"
        else:
            cols = [
                (imgui.Col_.button, (0.16, 0.62, 0.48, 1.0)),
                (imgui.Col_.button_hovered, (0.20, 0.74, 0.56, 1.0)),
                (imgui.Col_.text, (0.04, 0.12, 0.10, 1.0)),
            ]
            label = "Start"
        for c_, v in cols:
            imgui.push_style_color(c_, v)
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 8.0)
        if imgui.button(label, (NAV_W - 20, 32)):
            if running:
                self.on_stop()
            else:
                self.on_start()
        imgui.pop_style_var()
        imgui.pop_style_color(3)

        edge = imgui.get_window_pos()
        height = imgui.get_window_height()
        width = imgui.get_window_width()
        try:
            dl.add_line(
                (edge.x + width - 1, edge.y + 4),
                (edge.x + width - 1, edge.y + height - 4),
                imgui.get_color_u32((0.75, 0.90, 1.0, 0.22)),
                1.0,
            )
        except Exception:
            pass

        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()

    def _tab_aimbot(self, cfg: AppConfig, st: dict) -> None:
        self._section("Aimbot")
        
        # Bind
        imgui.text("Activation")
        mode_bind = "controller_bind" if cfg.input_mode == "controller" else "bind_key"
        current = getattr(cfg, mode_bind)
        
        imgui.set_next_item_width(120)
        if imgui.begin_combo("##bind", _bind_label(current)):
            for b in PRESET_BINDS:
                if imgui.selectable(_bind_label(b), b == current)[0]:
                    setattr(cfg, mode_bind, b)
                    self._touch(True)
            imgui.end_combo()
        imgui.same_line()
        if imgui.button("..." if not self._capturing else "?"):
            self._begin_capture(mode_bind)
        
        imgui.spacing()
        
        # Input mode
        imgui.text("Output")
        imgui.set_next_item_width(180)
        modes = ["mouse", "controller"]
        labels = ["Mouse", "Controller"]
        idx = modes.index(cfg.input_mode) if cfg.input_mode in modes else 0
        ch, idx = imgui.combo("##mode", idx, labels)
        if ch:
            cfg.input_mode = modes[idx]
            self._touch(True)
        
        if cfg.input_mode == "mouse":
            imgui.set_next_item_width(180)
            methods = ["razer", "sendinput", "sendinput_interp"]
            mlabels = ["Razer", "SendInput", "SendInput (smooth)"]
            idx = methods.index(cfg.mouse_method) if cfg.mouse_method in methods else 0
            ch, idx = imgui.combo("##mouse", idx, mlabels)
            if ch:
                cfg.mouse_method = methods[idx]
                self._touch(True)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Aim settings
        imgui.text("Aim Settings")
        
        imgui.set_next_item_width(200)
        ch, cfg.track_speed = imgui.slider_float("Speed", cfg.track_speed, 0.5, 5.0, "%.1f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.aim_offset_y = imgui.slider_float("Aim Height", cfg.aim_offset_y, 0.0, 1.0, "%.2f")
        self._touch(ch)
        imgui.text_disabled("0=feet, 0.5=chest, 1=head")
        
        imgui.spacing()
        
        imgui.set_next_item_width(200)
        ch, cfg.fov_radius = imgui.slider_int("FOV", cfg.fov_radius, 40, 400)
        self._touch(ch)
        
        ch, cfg.dynamic_fov_enabled = imgui.checkbox("Dynamic FOV", cfg.dynamic_fov_enabled)
        self._touch(ch)
        
        if cfg.dynamic_fov_enabled:
            imgui.set_next_item_width(200)
            ch, cfg.fov_radius_held = imgui.slider_int("FOV (held)", cfg.fov_radius_held, 40, 600)
            self._touch(ch)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Assist mode
        imgui.text("Assist Mode")
        imgui.set_next_item_width(180)
        assists = ["aim", "trigger", "both", "recoil"]
        alabs = ["Aim", "Trigger", "Aim + Trigger", "Recoil"]
        idx = assists.index(cfg.assist_mode) if cfg.assist_mode in assists else 0
        ch, idx = imgui.combo("##assist", idx, alabs)
        if ch:
            cfg.assist_mode = assists[idx]
            self._touch(True)
        
        # Trigger settings
        if cfg.assist_mode in ("trigger", "both"):
            imgui.spacing()
            imgui.text_disabled("Trigger Settings")
            imgui.set_next_item_width(200)
            ch, cfg.trigger_delay_ms = imgui.slider_float("Delay (ms)", cfg.trigger_delay_ms, 0, 200, "%.0f")
            self._touch(ch)
            imgui.set_next_item_width(200)
            ch, cfg.trigger_cooldown_ms = imgui.slider_float("Cooldown (ms)", cfg.trigger_cooldown_ms, 20, 500, "%.0f")
            self._touch(ch)
        
        # Recoil settings
        if cfg.assist_mode == "recoil":
            imgui.spacing()
            imgui.text_disabled("Recoil Settings")
            
            imgui.set_next_item_width(200)
            ch, cfg.recoil_strength = imgui.slider_float("Pull Strength", cfg.recoil_strength, 0.1, 15.0, "%.1f")
            self._touch(ch)
            
            imgui.set_next_item_width(200)
            ch, cfg.recoil_horizontal = imgui.slider_float("Horizontal", cfg.recoil_horizontal, -10.0, 10.0, "%.1f")
            self._touch(ch)
            
            imgui.set_next_item_width(200)
            ch, cfg.recoil_interval_ms = imgui.slider_float("Interval (ms)", cfg.recoil_interval_ms, 1.0, 50.0, "%.0f")
            self._touch(ch)
            
            imgui.set_next_item_width(200)
            ch, cfg.recoil_ramp_ms = imgui.slider_float("Ramp Up (ms)", cfg.recoil_ramp_ms, 0.0, 300.0, "%.0f")
            self._touch(ch)

    def _tab_visuals(self, cfg: AppConfig, st: dict) -> None:
        self._section("Visuals")
        
        ch, cfg.show_overlay = imgui.checkbox("Enable Overlay", cfg.show_overlay)
        self._touch(ch)
        
        ch, cfg.show_debug = imgui.checkbox("Show Detection Boxes", cfg.show_debug)
        self._touch(ch)

        ch, cfg.show_movement_line = imgui.checkbox(
            "Show Line of Movement", cfg.show_movement_line
        )
        self._touch(ch)
        imgui.text_disabled("Glass path from the center to the person being followed")
        
        ch, cfg.israel_mode = imgui.checkbox("Israel Mode", cfg.israel_mode)
        self._touch(ch)
        imgui.text_disabled("Shows regardless of overlay toggle")

    def _tab_detection(self, cfg: AppConfig, st: dict) -> None:
        self._section("Detection")
        
        # Mode
        imgui.text("Mode")
        imgui.set_next_item_width(180)
        dmodes = ["model", "color"]
        dlabs = ["AI Model", "Color"]
        idx = dmodes.index(cfg.detection_mode) if cfg.detection_mode in dmodes else 0
        ch, idx = imgui.combo("##dmode", idx, dlabs)
        if ch:
            cfg.detection_mode = dmodes[idx]
            self._touch(True)
        
        imgui.spacing()
        
        if cfg.detection_mode == "color":
            imgui.set_next_item_width(180)
            colors = ["purple", "red", "yellow", "cyan", "green"]
            idx = colors.index(cfg.color_preset) if cfg.color_preset in colors else 0
            ch, idx = imgui.combo("Color", idx, [c.title() for c in colors])
            if ch:
                cfg.color_preset = colors[idx]
                self._touch(True)
        else:
            imgui.set_next_item_width(200)
            ch, cfg.confidence = imgui.slider_float("Confidence", cfg.confidence, 0.1, 0.9, "%.2f")
            self._touch(ch)
            
            imgui.set_next_item_width(200)
            ch, cfg.iou_threshold = imgui.slider_float("IoU", cfg.iou_threshold, 0.15, 0.8, "%.2f")
            self._touch(ch)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Capture
        imgui.text("Capture")
        cfg.capture_method = "dxcam"
        imgui.text_colored(ACCENT, "DXGI Desktop Duplication")
        imgui.text_disabled("Only capture backend")
        
        imgui.set_next_item_width(200)
        ranges = ["close", "balanced", "far"]
        range_labels = ["Close (448)", "Balanced (448)", "Long Range (640)"]
        idx = ranges.index(cfg.detection_range) if cfg.detection_range in ranges else 1
        ch, idx = imgui.combo("Detection Range", idx, range_labels)
        if ch:
            cfg.detection_range = ranges[idx]
            cfg.net_size = 640 if cfg.detection_range == "far" else 448
            cfg.capture_size = cfg.net_size
            self._touch(True)
        imgui.text_disabled(
            f"DXGI region {cfg.capture_size}px · model {cfg.net_size}px"
        )
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Filtering
        imgui.text("Filtering")
        
        imgui.set_next_item_width(200)
        ch, cfg.min_box_size = imgui.slider_int("Min Box", cfg.min_box_size, 5, 100)
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.max_box_size = imgui.slider_int("Max Box", cfg.max_box_size, 100, 800)
        self._touch(ch)
        
        ch, cfg.filter_own_player = imgui.checkbox("Filter Own Player", cfg.filter_own_player)
        self._touch(ch)

    def _tab_performance(self, cfg: AppConfig, st: dict) -> None:
        self._section("Performance")

        # Live detection rate vs the capture ceiling. A detection Hz that sits
        # at the ceiling means the detector is at full throttle (screen-bound);
        # a gap between the two is where inference or GPU contention is biting.
        cap = float(st.get("cap_hz") or 0.0)
        det = float(st.get("detect_hz") or 0.0)
        infer_ms = float(st.get("infer_ms") or 0.0)
        move_hz = float(st.get("move_hz") or 0.0)
        frame_ms = float(st.get("frame_ms") or 0.0)

        at_ceiling = cap > 0 and det >= cap - 2.0
        det_color = (0.30, 0.90, 0.55, 1.0) if at_ceiling else (0.98, 0.75, 0.30, 1.0)
        cap_txt = f"/ {cap:.0f} Hz cap" if cap > 0 else "no cap"
        imgui.text_colored(det_color, f"DETECTION  {det:.0f} Hz  {cap_txt}")

        infer_max = f"~{1000.0 / infer_ms:.0f} Hz max" if infer_ms > 0 else ""
        imgui.text(f"Inference  {infer_ms:.2f} ms  {infer_max}")
        imgui.text(f"Mouse move  {move_hz:.0f} Hz     frame age  {frame_ms:.1f} ms")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # The resolution knobs that trade detection range against speed.
        # Smaller net_size / capture_size = faster inference and a higher
        # ceiling, at the cost of long-range reach. These are structural:
        # changing them rebuilds the pipeline on the next Start.
        imgui.text_colored(ACCENT, "RESOLUTION")
        imgui.text_disabled("Smaller = faster Hz, less long-range reach")

        imgui.set_next_item_width(200)
        ch, cfg.capture_size = imgui.slider_int("Capture Size", cfg.capture_size, 160, 1024)
        self._touch(ch)

        imgui.set_next_item_width(200)
        ch, cfg.net_size = imgui.slider_int("Network Size", cfg.net_size, 320, 960)
        self._touch(ch)
        if cfg.net_size % 32:
            self._notify("Net size should be a multiple of 32")

        # A capture FPS cap for diagnostics: 0 = follow the monitor refresh.
        imgui.set_next_item_width(200)
        ch, cfg.capture_fps_cap = imgui.slider_int("FPS Cap (0=auto)", cfg.capture_fps_cap, 0, 1000)
        self._touch(ch)
        imgui.text_disabled("0 = track monitor refresh (recommended)")

        imgui.spacing()
        if cap > 0 and infer_ms > 0:
            infer_cap = 1000.0 / infer_ms
            binding = "capture" if cap <= infer_cap else "inference"
            imgui.text_colored(ACCENT, f"Binding limit: {binding} "
                                       f"(cap {cap:.0f} vs infer ~{infer_cap:.0f} Hz)")

    def _tab_advanced(self, cfg: AppConfig, st: dict) -> None:
        self._section("Advanced")
        
        # Target selection
        imgui.text("Target Selection")
        
        imgui.set_next_item_width(180)
        pris = ["closest", "highest_conf", "largest", "center"]
        plabs = ["Closest", "Highest Conf", "Largest", "Center"]
        idx = pris.index(cfg.target_priority) if cfg.target_priority in pris else 0
        ch, idx = imgui.combo("Priority", idx, plabs)
        if ch:
            cfg.target_priority = pris[idx]
            self._touch(True)
        
        imgui.set_next_item_width(200)
        ch, cfg.target_lock_time = imgui.slider_float("Lock Time", cfg.target_lock_time, 0.0, 0.5, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.sticky_aim_time = imgui.slider_float("Sticky Time", cfg.sticky_aim_time, 0.0, 1.0, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.prediction_frames = imgui.slider_int("Prediction", cfg.prediction_frames, 0, 5)
        self._touch(ch)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Smoothing
        imgui.text("Smoothing")
        
        imgui.set_next_item_width(200)
        ch, cfg.max_aim_speed = imgui.slider_float("Max Speed", cfg.max_aim_speed, 0, 300, "%.0f")
        self._touch(ch)
        imgui.text_disabled("0 = unlimited")
        
        imgui.set_next_item_width(200)
        ch, cfg.flick_reduction = imgui.slider_float("Flick Reduce", cfg.flick_reduction, 0.0, 1.0, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.aim_acceleration = imgui.slider_float("Acceleration", cfg.aim_acceleration, 0.0, 1.0, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(180)
        eases = ["linear", "ease_in", "ease_out", "ease_in_out"]
        elabs = ["Linear", "Ease In", "Ease Out", "Ease In/Out"]
        idx = eases.index(cfg.easing_curve) if cfg.easing_curve in eases else 0
        ch, idx = imgui.combo("Easing", idx, elabs)
        if ch:
            cfg.easing_curve = eases[idx]
            self._touch(True)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Humanization
        imgui.text("Humanization")
        
        imgui.set_next_item_width(200)
        ch, cfg.humanize_jitter = imgui.slider_float("Jitter", cfg.humanize_jitter, 0.0, 3.0, "%.1f")
        self._touch(ch)
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # PID
        imgui.text("PID Controller")
        
        imgui.set_next_item_width(200)
        ch, cfg.pid_kp = imgui.slider_float("Kp", cfg.pid_kp, 0.1, 1.5, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.pid_ki = imgui.slider_float("Ki", cfg.pid_ki, 0.0, 3.0, "%.2f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.pid_kd = imgui.slider_float("Kd", cfg.pid_kd, 0.0, 0.02, "%.4f")
        self._touch(ch)
        
        imgui.set_next_item_width(200)
        ch, cfg.pid_ff = imgui.slider_float("Feedforward", cfg.pid_ff, 0.0, 1.2, "%.2f")
        self._touch(ch)

    def _tab_misc(self, cfg: AppConfig, st: dict) -> None:
        self._section("Misc")
        
        # Profiles
        imgui.text("Profiles")
        
        imgui.set_next_item_width(150)
        ch, self._profile_name = imgui.input_text("##pname", self._profile_name)
        imgui.same_line()
        if imgui.button("Save") and self._profile_name.strip():
            try:
                save_profile(self._profile_name, self.config)
                self._profiles = list_profiles()
                self._notify(f"Saved: {self._profile_name}")
            except ValueError as e:
                self._notify(str(e))
        
        imgui.spacing()
        
        for name in list(self._profiles):
            imgui.text(name)
            imgui.same_line(180)
            imgui.push_id(name)
            if imgui.small_button("Load"):
                loaded = load_profile(name)
                for k, v in asdict(loaded).items():
                    setattr(self.config, k, v)
                self.config.clamp()
                save_config(self.config)
                self.on_apply(self.config)
                self._notify(f"Loaded: {name}")
            imgui.same_line()
            if imgui.small_button("X"):
                delete_profile(name)
                self._profiles = list_profiles()
            imgui.pop_id()
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Info
        imgui.text("Status")
        imgui.text_disabled(f"Output: {st['mouse_label']}")
        imgui.text_disabled(f"Detection: {st['detect_hz']:.1f} Hz")
        imgui.text_disabled(f"Inference: {st['infer_ms']:.2f} ms")
        conf = st['target_conf']
        imgui.text_disabled(f"Confidence: {'-' if conf is None else f'{conf:.0%}'}")

    def run(self) -> None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "Neverwin"
        params.app_window_params.window_geometry.size = (760, 520)
        params.app_window_params.restore_previous_geometry = False
        params.app_window_params.resizable = False
        # The menu is the window. No OS title bar and no second panel behind it.
        params.app_window_params.borderless = True
        params.app_window_params.borderless_movable = True
        params.app_window_params.borderless_resizable = False
        params.app_window_params.borderless_closable = True
        params.app_window_params.borderless_highlight_color = (0.70, 0.88, 1.0, 0.35)
        params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )
        params.imgui_window_params.show_menu_bar = False
        # Limit UI framerate to reduce GPU competition with CUDA inference
        params.fps_idling.fps_idle = 20.0        # Sitting still: almost no GPU time
        params.fps_idling.enable_idling = True
        try:
            # Motion is eased in time, so 30 fps still reads as smooth.
            params.fps_idling.fps_active = 30.0
        except AttributeError:
            pass
        params.callbacks.setup_imgui_style = _liquid_glass_style
        params.callbacks.show_gui = self._draw
        addons = immapp.AddOnsParams()
        immapp.run(params, addons)


def _liquid_glass_style() -> None:
    """Liquid-glass theme: deep backdrop, rounded widgets, neon-cyan accent,
    translucent glass surfaces with soft borders."""
    style = imgui.get_style()

    style.window_rounding = 0.0
    style.frame_rounding = 10.0
    style.grab_rounding = 10.0
    style.tab_rounding = 10.0
    style.scrollbar_rounding = 12.0
    style.child_rounding = 14.0
    style.popup_rounding = 12.0
    style.indent_spacing = 12.0
    style.item_spacing = (10, 7)
    style.item_inner_spacing = (8, 5)
    style.frame_padding = (10, 6)
    style.window_padding = (14, 14)
    style.scrollbar_size = 10.0
    style.grab_min_size = 10.0

    style.window_border_size = 0.0
    style.frame_border_size = 0.0
    style.child_border_size = 0.0
    style.popup_border_size = 1.0
    style.tab_border_size = 0.0

    bg        = imgui.ImVec4(0.07,  0.10,  0.16,  1.0)
    glass     = imgui.ImVec4(0.12,  0.18,  0.28,  0.94)
    glass_hi  = imgui.ImVec4(0.18,  0.26,  0.38,  0.96)
    border    = imgui.ImVec4(0.62,  0.84,  1.00,  0.45)
    accent    = imgui.ImVec4(0.62,  0.86,  1.00,  1.0)
    accent_hi = imgui.ImVec4(0.82,  0.95,  1.00,  1.0)
    accent_dk = imgui.ImVec4(0.18,  0.42,  0.62,  1.0)
    text      = imgui.ImVec4(0.92,  0.94,  0.98,  1.0)
    text_dim  = imgui.ImVec4(0.55,  0.60,  0.70,  1.0)

    for idx, col in (
        (imgui.Col_.text, text),
        (imgui.Col_.text_disabled, text_dim),
        (imgui.Col_.window_bg, bg),
        (imgui.Col_.child_bg, glass),
        (imgui.Col_.popup_bg, imgui.ImVec4(0.09, 0.11, 0.17, 0.98)),
        (imgui.Col_.border, border),
        (imgui.Col_.border_shadow, imgui.ImVec4(0, 0, 0, 0)),
        (imgui.Col_.frame_bg, imgui.ImVec4(0.10, 0.16, 0.26, 0.90)),
        (imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.16, 0.24, 0.36, 0.95)),
        (imgui.Col_.frame_bg_active, imgui.ImVec4(0.18, 0.30, 0.44, 1.0)),
        (imgui.Col_.title_bg, glass),
        (imgui.Col_.title_bg_active, glass_hi),
        (imgui.Col_.title_bg_collapsed, glass),
        (imgui.Col_.menu_bar_bg, glass),
        (imgui.Col_.scrollbar_bg, imgui.ImVec4(0.06, 0.07, 0.11, 0.6)),
        (imgui.Col_.scrollbar_grab, imgui.ImVec4(0.25, 0.32, 0.45, 0.6)),
        (imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.32, 0.44, 0.62, 0.85)),
        (imgui.Col_.scrollbar_grab_active, accent),
        (imgui.Col_.check_mark, accent),
        (imgui.Col_.slider_grab, accent),
        (imgui.Col_.slider_grab_active, accent_hi),
        (imgui.Col_.button, imgui.ImVec4(0.14, 0.20, 0.32, 0.90)),
        (imgui.Col_.button_hovered, imgui.ImVec4(0.20, 0.32, 0.48, 0.95)),
        (imgui.Col_.button_active, accent_dk),
        (imgui.Col_.header, imgui.ImVec4(0.12, 0.16, 0.24, 0.5)),
        (imgui.Col_.header_hovered, imgui.ImVec4(0.16, 0.24, 0.34, 0.8)),
        (imgui.Col_.header_active, accent_dk),
        (imgui.Col_.separator, border),
        (imgui.Col_.separator_hovered, accent),
        (imgui.Col_.separator_active, accent),
        (imgui.Col_.resize_grip, imgui.ImVec4(0, 0, 0, 0)),
        (imgui.Col_.resize_grip_hovered, accent),
        (imgui.Col_.resize_grip_active, accent),
        (imgui.Col_.tab, glass),
        (imgui.Col_.tab_hovered, accent),
        (imgui.Col_.tab_selected, accent_dk),
        (imgui.Col_.plot_lines, accent),
        (imgui.Col_.plot_histogram, accent),
        (imgui.Col_.table_header_bg, imgui.ImVec4(0.12, 0.16, 0.24, 0.5)),
        (imgui.Col_.table_border_strong, border),
        (imgui.Col_.table_border_light, border),
    ):
        style.set_color_(idx, col)
