"""Razer mouse backend — wraps Mouse (RZCONTROL + dll)."""
from __future__ import annotations

from exoai_tracker.mouse import Mouse


class RazerBackend:
    name = "synapse"

    def __init__(self) -> None:
        self._m = Mouse()

    def init(self) -> bool:
        return self._m.init()

    def move(self, dx: int, dy: int) -> bool:
        return self._m.move(dx, dy)

    def close(self) -> None:
        try:
            self._m._rz.close()
        except Exception:
            pass

    def label(self) -> str:
        return self._m.label

    def ready(self) -> bool:
        return self._m.ok

    def packet_count(self) -> int:
        return self._m._pkt

    def ok_count(self) -> int:
        return self._m._ok

    def last_move(self) -> tuple[int, int]:
        return self._m._last
