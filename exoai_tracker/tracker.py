"""Bind + mouse facade. Aim lock lives on TrackEngine."""
from __future__ import annotations

from exoai_tracker.binds import is_bind_held
from exoai_tracker.mouse_backends import MouseBackend, init_backend


class TargetTracker:
    def __init__(self, backend_name: str = "razer") -> None:
        self._bind_key = "right"
        self._speed = 0.3
        self._max_move = 127
        self._overshoot = 0.96
        self._deadzone = 2.0
        self._backend_name = backend_name
        self._mouse: MouseBackend | None = None

    def configure(self, track_speed: float, **kwargs) -> None:
        self._speed = max(0.15, min(1.0, float(track_speed) / 5.0))
        self._max_move = 127
        if "overshoot_ratio" in kwargs:
            self._overshoot = float(kwargs["overshoot_ratio"])
        if "deadzone_px" in kwargs:
            self._deadzone = float(kwargs["deadzone_px"])
        if "mouse_backend" in kwargs:
            self._backend_name = str(kwargs["mouse_backend"])

    def set_bind_key(self, key: str) -> None:
        self._bind_key = key

    def bind_held(self) -> bool:
        return is_bind_held(self._bind_key)

    def reset_aim(self) -> None:
        return None

    def ensure_mouse(self) -> "TargetTracker":
        if self._mouse is None:
            self._mouse = init_backend(self._backend_name)
        return self

    @property
    def mouse(self) -> "_MouseStats":
        return _MouseStats(self._mouse)

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def max_move(self) -> int:
        return self._max_move

    @property
    def overshoot(self) -> float:
        return self._overshoot

    @property
    def deadzone(self) -> float:
        return self._deadzone


class _MouseStats:
    def __init__(self, backend: MouseBackend | None) -> None:
        self._backend = backend

    @property
    def ok_moves(self) -> int:
        return 0 if self._backend is None else self._backend.ok_count()

    @property
    def short_label(self) -> str:
        if self._backend is None:
            return "none"
        return self._backend.name

    @property
    def label(self) -> str:
        if self._backend is None:
            return "none"
        return self._backend.label()
