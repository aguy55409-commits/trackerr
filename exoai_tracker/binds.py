from __future__ import annotations

import ctypes
from typing import Final

import keyboard

user32 = ctypes.windll.user32

# Virtual-key codes for mouse buttons on Windows.
MOUSE_VK: Final[dict[str, int]] = {
    "left": 0x01,
    "lmb": 0x01,
    "right": 0x02,
    "rmb": 0x02,
    "middle": 0x04,
    "mmb": 0x04,
    "mouse4": 0x05,
    "x1": 0x05,
    "mouse5": 0x06,
    "x2": 0x06,
}

BIND_ALIASES: Final[dict[str, str]] = {
    "rmb": "right",
    "lmb": "left",
    "mmb": "middle",
    "mouse right": "right",
    "mouse left": "left",
    "mouse middle": "middle",
    "mouse x1": "x1",
    "mouse x2": "x2",
    "mouse4": "x1",
    "mouse5": "x2",
}

DISPLAY_NAMES: Final[dict[str, str]] = {
    "right": "RMB",
    "left": "LMB",
    "middle": "MMB",
    "x1": "Mouse4",
    "x2": "Mouse5",
}

PRESET_BINDS: Final[tuple[str, ...]] = (
    "right",
    "left",
    "middle",
    "x1",
    "x2",
    "x",
    "shift",
    "ctrl",
    "alt",
    "caps lock",
)


def normalize_bind(name: str) -> str:
    key = name.strip().lower()
    return BIND_ALIASES.get(key, key)


def format_bind(name: str) -> str:
    normalized = normalize_bind(name)
    return DISPLAY_NAMES.get(normalized, normalized.upper())


def is_mouse_bind(name: str) -> bool:
    return normalize_bind(name) in {"left", "right", "middle", "x1", "x2"}


def is_bind_held(bind_key: str) -> bool:
    normalized = normalize_bind(bind_key)
    vk = MOUSE_VK.get(normalized)
    if vk is not None:
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)
    try:
        return keyboard.is_pressed(normalized)
    except Exception:
        return False


def wait_for_bind() -> str:
    """Block until a keyboard key or mouse button is pressed."""
    while True:
        for canonical in ("left", "right", "middle", "x1", "x2"):
            vk = MOUSE_VK[canonical]
            if user32.GetAsyncKeyState(vk) & 0x8000:
                return canonical

        event = keyboard.read_event(suppress=False)
        if event.event_type == "down" and event.name:
            return normalize_bind(event.name)
