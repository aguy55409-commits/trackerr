"""Pluggable mouse backends — Razer Synapse driver (same path as MOUSE FUNCTION)."""
from __future__ import annotations

import time

from exoai_tracker.mouse_backends.base import MouseBackend
from exoai_tracker.mouse_backends.razer import RazerBackend

BACKEND_NAMES = ("razer", "synapse")

_FACTORIES = {
    "razer": RazerBackend,
    "synapse": RazerBackend,
}


def create_backend(name: str = "razer") -> MouseBackend:
    key = (name or "razer").strip().lower()
    if key == "auto":
        key = "razer"
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"Unknown mouse backend: {name!r}. Use: {BACKEND_NAMES}")
    return factory()


def init_backend(name: str = "razer") -> MouseBackend:
    backend = create_backend(name)
    if not backend.init():
        raise RuntimeError(
            "Razer Synapse driver not found. Install Synapse 3 and plug in a Razer mouse."
        )
    return backend


def run_mouse_test(backend: MouseBackend | None = None) -> dict:
    """Quick in-game check: right, left, down, up via Razer driver."""
    own = backend is None
    if own:
        backend = init_backend("razer")

    time.sleep(0.1)
    moves = [(40, 0), (-40, 0), (0, 25), (0, -25)]
    ok = 0
    for dx, dy in moves:
        if backend.move(dx, dy):
            ok += 1
        time.sleep(0.25)

    if own:
        backend.close()

    ready = backend.ready() if not own else ok > 0
    return {
        "ok": ok == len(moves),
        "ready": ready,
        "message": (
            f"Razer driver test: {ok}/{len(moves)} moves sent"
            + (f" ({backend.label()})" if not own else "")
            + ". Watch crosshair in Apex."
        ),
    }


__all__ = [
    "BACKEND_NAMES",
    "MouseBackend",
    "RazerBackend",
    "create_backend",
    "init_backend",
    "run_mouse_test",
]
