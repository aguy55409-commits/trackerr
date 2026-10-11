"""Unified input handler — manual switch between mouse and controller."""
from __future__ import annotations

import time

from exoai_tracker.mouse import Mouse
from exoai_tracker.controller_output import get_shared_output, is_controller_connected


class InputHandler:
    """Routes aim to mouse or controller based on user setting."""
    
    # Controllers idle-sleep, so a failed init is not permanent. Retry at this
    # interval instead of silently staying on the mouse for the whole session.
    _RETRY_S = 1.0

    def __init__(self, mode: str = "mouse", mouse_method: str = "razer") -> None:
        self._mouse = Mouse(method=mouse_method)
        self._mouse_method = mouse_method
        # Shared across pipelines on purpose; see get_shared_output().
        self._controller = get_shared_output()
        self._mode = mode  # "mouse" or "controller"
        self._mouse_ok = False
        self._controller_ok = False
        self._last_try = 0.0

    def _try_controller(self) -> bool:
        """Attempt controller init, rate-limited. True once it is live."""
        if self._controller_ok:
            return True
        now = time.monotonic()
        if now - self._last_try < self._RETRY_S:
            return False
        self._last_try = now
        self._controller_ok = self._controller.init()
        return self._controller_ok
    
    def init(self) -> bool:
        """Initialize input handlers.

        Controller mode falls back to the mouse if no pad is present, but the
        fallback is reported through `label` so it is not mistaken for working
        controller output.
        """
        # Always init mouse
        self._mouse_ok = self._mouse.init()

        if self._mode == "controller":
            self._try_controller()
            return self._controller_ok or self._mouse_ok

        return self._mouse_ok
    
    def set_mode(self, mode: str) -> bool:
        """Switch input mode. Returns True if the target mode is live."""
        if mode == "controller":
            self._mode = "controller"
            self._last_try = 0.0  # user asked for it now, so retry immediately
            return self._try_controller()
        self._mode = "mouse"
        return self._mouse_ok
    
    def move(self, dx: int, dy: int) -> bool:
        """Move aim using selected input method."""
        if dx == 0 and dy == 0:
            return True
        
        if self._mode == "controller":
            # Deliberately does not fall through to the mouse: silently steering
            # the mouse while the UI says "Controller" is what made a sleeping
            # pad look like broken tracking.
            if not self._try_controller():
                return False
            return self._controller.move(dx, dy)
        if self._mouse_ok:
            return self._mouse.move(dx, dy)
        return False
    
    def press(self) -> bool:
        """Press primary button."""
        if self._mode == "controller":
            # For controller, we don't handle button presses here
            # The game reads real controller input
            return True
        return self._mouse.press()
    
    def release(self) -> bool:
        """Release primary button."""
        if self._mode == "controller":
            return True
        return self._mouse.release()
    
    @property
    def down(self) -> bool:
        if self._mode == "controller":
            return False
        return self._mouse.down
    
    @property
    def ok(self) -> bool:
        if self._mode == "controller":
            return self._controller_ok
        return self._mouse_ok
    
    @property
    def label(self) -> str:
        if self._mode == "controller":
            if self._controller_ok:
                return f"🎮 {self._controller.label}"
            return "🎮 No controller — connect/wake it"
        if self._mouse_ok:
            return self._mouse.label
        return "No input device"
    
    @property
    def short(self) -> str:
        if self._mode == "controller" and self._controller_ok:
            return "controller"
        return self._mouse.short
    
    @property
    def method(self) -> str:
        if self._mode == "controller":
            return "controller"
        return self._mouse.method
    
    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def using_controller(self) -> bool:
        return self._mode == "controller" and self._controller_ok
    
    def reset_controller(self) -> None:
        """Reset controller stick to center."""
        if self._controller_ok:
            self._controller.reset()
    
    def close(self) -> None:
        """Release this handler.

        The virtual pad is deliberately left running: it is shared and must
        survive Stop/Start, or the game loses the device it bound to.
        """
        self._controller.reset()
