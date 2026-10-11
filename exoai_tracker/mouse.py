"""Mouse backends — Razer driver or SendInput fallback."""
from __future__ import annotations


class Mouse:
    """Unified mouse interface with selectable backend.
    
    Backends:
    - razer: rzctl driver (hardware-level, stealthier, needs Synapse + admin)
    - sendinput: Win32 SendInput (works everywhere, less stealthy)
    - sendinput_interp: SendInput with interpolated movement (smoothest, slowest)
    """

    def __init__(self, method: str = "razer") -> None:
        self._method = method
        self._requested_method = method  # Keep track of what was originally requested
        self._backend = None
        self.ok = False

    def init(self) -> bool:
        # Lazy imports to avoid circular dependency
        if self._method == "sendinput":
            from exoai_tracker.mouse_backends.sendinput import SendInputMouse
            self._backend = SendInputMouse(interpolate=False)
        elif self._method == "sendinput_interp":
            from exoai_tracker.mouse_backends.sendinput import SendInputMouse
            self._backend = SendInputMouse(interpolate=True, step_delay=0.0003)
        else:  # razer (default)
            from exoai_tracker.rzctl.move import RazerMover
            self._backend = RazerMover()
        
        self.ok = self._backend.init()
        
        # Fallback to SendInput if the selected method fails
        if not self.ok and self._method not in ("sendinput", "sendinput_interp"):
            from exoai_tracker.mouse_backends.sendinput import SendInputMouse
            self._backend = SendInputMouse(interpolate=False)
            self.ok = self._backend.init()
            if self.ok:
                self._method = "sendinput"  # Update method to reflect fallback
        
        return self.ok

    def move(self, dx: int, dy: int) -> bool:
        if dx == 0 and dy == 0:
            return True
        if self._backend is None:
            return False
        return self._backend.move(int(dx), int(dy))

    def press(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.press()

    def release(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.release()

    @property
    def down(self) -> bool:
        if self._backend is None:
            return False
        return getattr(self._backend, "down", False)

    @property
    def label(self) -> str:
        if not self.ok or self._backend is None:
            if self._method == "razer":
                return "Razer offline — Synapse + admin"
            return f"{self._method} offline"
        
        if self._method.startswith("sendinput"):
            return self._backend.label()
        # Razer backend
        return f"Razer · {self._backend.method} · {self._backend.ok}/{self._backend.pkt}"

    @property
    def short(self) -> str:
        if not self.ok:
            return "none"
        return self._method

    @property
    def packets(self) -> int:
        if self._backend is None:
            return 0
        return getattr(self._backend, "ok", 0)

    @property
    def method(self) -> str:
        return self._method
