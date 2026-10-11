from __future__ import annotations

import time
from typing import Final

import ctypes

user32 = ctypes.windll.user32
user32.VkKeyScanW.argtypes = [ctypes.c_wchar]
user32.VkKeyScanW.restype = ctypes.c_short
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetKeyState.argtypes = [ctypes.c_int]
user32.GetKeyState.restype = ctypes.c_short

# XInput for controller support
try:
    xinput = ctypes.windll.xinput1_4
except OSError:
    try:
        xinput = ctypes.windll.xinput1_3
    except OSError:
        try:
            xinput = ctypes.windll.xinput9_1_0
        except OSError:
            xinput = None

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

# XInput button constants
XINPUT_GAMEPAD_DPAD_UP = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
XINPUT_GAMEPAD_START = 0x0010
XINPUT_GAMEPAD_BACK = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_Y = 0x8000

TRIGGER_THRESHOLD = 128  # 0-255, trigger pressed threshold

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

SPECIAL_VK: Final[dict[str, int]] = {
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "caps lock": 0x14,
    "space": 0x20,
    "tab": 0x09,
    "x": 0x58,
}

# Controller button mappings
CONTROLLER_BUTTONS: Final[dict[str, int]] = {
    "gp_a": XINPUT_GAMEPAD_A,
    "gp_b": XINPUT_GAMEPAD_B,
    "gp_x": XINPUT_GAMEPAD_X,
    "gp_y": XINPUT_GAMEPAD_Y,
    "gp_lb": XINPUT_GAMEPAD_LEFT_SHOULDER,
    "gp_rb": XINPUT_GAMEPAD_RIGHT_SHOULDER,
    "gp_ls": XINPUT_GAMEPAD_LEFT_THUMB,
    "gp_rs": XINPUT_GAMEPAD_RIGHT_THUMB,
    "gp_start": XINPUT_GAMEPAD_START,
    "gp_back": XINPUT_GAMEPAD_BACK,
    "gp_up": XINPUT_GAMEPAD_DPAD_UP,
    "gp_down": XINPUT_GAMEPAD_DPAD_DOWN,
    "gp_left": XINPUT_GAMEPAD_DPAD_LEFT,
    "gp_right": XINPUT_GAMEPAD_DPAD_RIGHT,
}

# Trigger binds (special handling)
CONTROLLER_TRIGGERS: Final[tuple[str, ...]] = ("gp_lt", "gp_rt")

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
    "control": "ctrl",
}

DISPLAY_NAMES: Final[dict[str, str]] = {
    "right": "RMB",
    "left": "LMB",
    "middle": "MMB",
    "x1": "Mouse4",
    "x2": "Mouse5",
    "gp_a": "🎮 A",
    "gp_b": "🎮 B",
    "gp_x": "🎮 X",
    "gp_y": "🎮 Y",
    "gp_lb": "🎮 LB",
    "gp_rb": "🎮 RB",
    "gp_lt": "🎮 LT",
    "gp_rt": "🎮 RT",
    "gp_ls": "🎮 LS",
    "gp_rs": "🎮 RS",
    "gp_start": "🎮 Start",
    "gp_back": "🎮 Back",
    "gp_up": "🎮 D-Up",
    "gp_down": "🎮 D-Down",
    "gp_left": "🎮 D-Left",
    "gp_right": "🎮 D-Right",
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
    "gp_lt",
    "gp_rt",
    "gp_lb",
    "gp_rb",
    "gp_a",
    "gp_b",
    "gp_x",
    "gp_y",
)

MOUSE_CANONICAL: Final[tuple[str, ...]] = ("left", "right", "middle", "x1", "x2")


def normalize_bind(name: str) -> str:
    key = name.strip().lower()
    return BIND_ALIASES.get(key, key)


def format_bind(name: str) -> str:
    normalized = normalize_bind(name)
    return DISPLAY_NAMES.get(normalized, normalized.upper())


def is_mouse_bind(name: str) -> bool:
    return normalize_bind(name) in {"left", "right", "middle", "x1", "x2"}


def _vk_for_bind(name: str) -> int | None:
    normalized = normalize_bind(name)
    if normalized in MOUSE_VK:
        return MOUSE_VK[normalized]
    if normalized in SPECIAL_VK:
        return SPECIAL_VK[normalized]
    if len(normalized) == 1:
        scan = user32.VkKeyScanW(normalized[0])
        if scan != -1:
            return int(scan) & 0xFF
    return None


def _vk_held(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000) or bool(user32.GetKeyState(vk) & 0x8000)


def _get_controller_state(controller_id: int = 0) -> XINPUT_STATE | None:
    """Get XInput controller state. Returns None if not connected."""
    if xinput is None:
        return None
    state = XINPUT_STATE()
    result = xinput.XInputGetState(controller_id, ctypes.byref(state))
    if result == 0:  # ERROR_SUCCESS
        return state
    return None


def _controller_button_held(bind: str) -> bool:
    """Check if a controller button is held."""
    state = _get_controller_state()
    if state is None:
        return False
    
    # Handle triggers
    if bind == "gp_lt":
        return state.Gamepad.bLeftTrigger >= TRIGGER_THRESHOLD
    if bind == "gp_rt":
        return state.Gamepad.bRightTrigger >= TRIGGER_THRESHOLD
    
    # Handle buttons
    if bind in CONTROLLER_BUTTONS:
        return bool(state.Gamepad.wButtons & CONTROLLER_BUTTONS[bind])
    
    return False


def is_controller_bind(name: str) -> bool:
    """Check if bind is a controller bind."""
    normalized = normalize_bind(name)
    return normalized.startswith("gp_")


# Map mouse binds to controller equivalents
MOUSE_TO_CONTROLLER: Final[dict[str, str]] = {
    "right": "gp_lt",   # RMB → Left Trigger (ADS)
    "left": "gp_rt",    # LMB → Right Trigger (Shoot)
    "middle": "gp_rs",  # MMB → Right Stick Click
    "x1": "gp_lb",      # Mouse4 → Left Bumper
    "x2": "gp_rb",      # Mouse5 → Right Bumper
}


def get_controller_equivalent(bind: str) -> str:
    """Get controller equivalent of a mouse bind."""
    normalized = normalize_bind(bind)
    if is_controller_bind(normalized):
        return normalized
    return MOUSE_TO_CONTROLLER.get(normalized, "gp_lt")


def is_bind_held(bind_key: str, force_controller: bool = False) -> bool:
    normalized = normalize_bind(bind_key)
    
    # If forcing controller mode, map mouse binds to controller
    if force_controller and not is_controller_bind(normalized):
        normalized = get_controller_equivalent(normalized)
    
    # Check controller binds
    if is_controller_bind(normalized):
        return _controller_button_held(normalized)
    
    # Check keyboard/mouse binds
    vk = _vk_for_bind(normalized)
    if vk is None:
        return False
    return _vk_held(vk)


def wait_for_bind() -> str:
    """Poll Win32 key state and controller. No global keyboard hook."""
    time.sleep(0.18)
    while True:
        # Check mouse buttons
        for canonical in MOUSE_CANONICAL:
            if _vk_held(MOUSE_VK[canonical]):
                return canonical
        
        # Check controller buttons
        state = _get_controller_state()
        if state is not None:
            # Triggers
            if state.Gamepad.bLeftTrigger >= TRIGGER_THRESHOLD:
                return "gp_lt"
            if state.Gamepad.bRightTrigger >= TRIGGER_THRESHOLD:
                return "gp_rt"
            # Buttons
            for name, mask in CONTROLLER_BUTTONS.items():
                if state.Gamepad.wButtons & mask:
                    return name
        
        # Check special keys
        for name, vk in SPECIAL_VK.items():
            if name in BIND_ALIASES:
                continue
            if _vk_held(vk):
                return normalize_bind(name)
        
        # Check letter keys
        for code in range(ord("a"), ord("z") + 1):
            ch = chr(code)
            vk = _vk_for_bind(ch)
            if vk is not None and _vk_held(vk):
                return ch
        
        time.sleep(0.01)
