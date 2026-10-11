"""Controller output: passthrough of the real pad + aim assist on the right stick.

The real controller is read over XInput and mirrored onto a virtual DS4. The
virtual pad is a DualShock rather than an Xbox pad on purpose: a virtual X360
device would claim an XInput slot and we could end up reading our own output
back as if it were the player.
"""
from __future__ import annotations

import ctypes
import threading
import time

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


def is_controller_connected(controller_id: int = 0) -> bool:
    if xinput is None:
        return False
    state = XINPUT_STATE()
    return xinput.XInputGetState(controller_id, ctypes.byref(state)) == 0


def get_real_controller_state(controller_id: int = 0) -> XINPUT_STATE | None:
    if xinput is None:
        return None
    state = XINPUT_STATE()
    if xinput.XInputGetState(controller_id, ctypes.byref(state)) == 0:
        return state
    return None


try:
    import vgamepad as vg
    HAS_VGAMEPAD = True
except ImportError:
    vg = None
    HAS_VGAMEPAD = False


class ControllerOutput:
    STICK_MAX = 32767
    STICK_MIN = -32768

    # Below this the real sticks are treated as centred, so pad drift does not
    # leak into the mirrored output.
    STICK_DEADZONE = 5000

    POLL_DT = 0.004

    def __init__(self) -> None:
        self._gamepad = None
        self._ok = False
        self._lock = threading.Lock()
        self._aim_x = 0.0
        self._aim_y = 0.0
        # Stick counts per count of requested correction. A stick is a velocity
        # rather than a displacement, so this maps "how far off are we" onto
        # "how hard to push", not onto a distance to travel.
        self._sensitivity = 1500
        # Applied per poll once commands stop arriving, so the stick recentres
        # over ~60ms instead of staying locked over.
        self._decay = 0.85
        self._last_cmd = 0.0
        self._thread = None
        self._stop = threading.Event()

    # A correction older than this means the aim loop is no longer tracking,
    # so the stick should fall back to centre rather than hold its last push.
    STALE_CMD = 0.03

    def init(self) -> bool:
        # Idempotent: this instance is shared, so later callers must attach to
        # the existing pad rather than spawn a second one.
        if self._ok and self._gamepad is not None:
            return True
        if not HAS_VGAMEPAD:
            print("[Controller] vgamepad not installed")
            return False
        if not is_controller_connected():
            print("[Controller] No real controller detected")
            return False

        try:
            self._gamepad = vg.VDS4Gamepad()
            self._ok = True
        except Exception as exc:
            print(f"[Controller] Failed to init: {exc}")
            print("[Controller] Is the ViGEmBus driver installed?")
            self._ok = False
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._passthrough_loop, name="pad-passthrough", daemon=True
        )
        self._thread.start()
        print("[Controller] Passthrough active (real pad -> virtual DS4)")
        return True

    @staticmethod
    def _to_float(value: float) -> float:
        return max(-1.0, min(1.0, value / 32767.0))

    def _passthrough_loop(self) -> None:
        while not self._stop.is_set():
            if not self._ok or self._gamepad is None:
                time.sleep(self.POLL_DT)
                continue

            state = get_real_controller_state()
            if state is None:
                time.sleep(self.POLL_DT)
                continue

            gp = state.Gamepad
            dz = self.STICK_DEADZONE
            lx = gp.sThumbLX if abs(gp.sThumbLX) > dz else 0
            ly = gp.sThumbLY if abs(gp.sThumbLY) > dz else 0
            rx = gp.sThumbRX if abs(gp.sThumbRX) > dz else 0
            ry = gp.sThumbRY if abs(gp.sThumbRY) > dz else 0

            with self._lock:
                if time.perf_counter() - self._last_cmd > self.STALE_CMD:
                    self._aim_x *= self._decay
                    self._aim_y *= self._decay
                    if abs(self._aim_x) < 50:
                        self._aim_x = 0.0
                    if abs(self._aim_y) < 50:
                        self._aim_y = 0.0
                aim_x, aim_y = self._aim_x, self._aim_y

            rx = max(self.STICK_MIN, min(self.STICK_MAX, int(rx + aim_x)))
            ry = max(self.STICK_MIN, min(self.STICK_MAX, int(ry + aim_y)))

            try:
                # DS4 axes are float and its Y runs opposite to XInput's.
                self._gamepad.left_joystick_float(self._to_float(lx), self._to_float(-ly))
                self._gamepad.right_joystick_float(self._to_float(rx), self._to_float(-ry))
                self._gamepad.left_trigger_float(gp.bLeftTrigger / 255.0)
                self._gamepad.right_trigger_float(gp.bRightTrigger / 255.0)
                self._apply_buttons(gp.wButtons)
                self._gamepad.update()
            except Exception:
                pass

            time.sleep(self.POLL_DT)

    def _apply_buttons(self, buttons: int) -> None:
        up = bool(buttons & 0x0001)
        down = bool(buttons & 0x0002)
        left = bool(buttons & 0x0004)
        right = bool(buttons & 0x0008)
        d = vg.DS4_DPAD_DIRECTIONS
        if up and right:
            dpad = d.DS4_BUTTON_DPAD_NORTHEAST
        elif up and left:
            dpad = d.DS4_BUTTON_DPAD_NORTHWEST
        elif down and right:
            dpad = d.DS4_BUTTON_DPAD_SOUTHEAST
        elif down and left:
            dpad = d.DS4_BUTTON_DPAD_SOUTHWEST
        elif up:
            dpad = d.DS4_BUTTON_DPAD_NORTH
        elif down:
            dpad = d.DS4_BUTTON_DPAD_SOUTH
        elif left:
            dpad = d.DS4_BUTTON_DPAD_WEST
        elif right:
            dpad = d.DS4_BUTTON_DPAD_EAST
        else:
            dpad = d.DS4_BUTTON_DPAD_NONE
        self._gamepad.directional_pad(dpad)

        b = vg.DS4_BUTTONS
        mapping = (
            (0x1000, b.DS4_BUTTON_CROSS),
            (0x2000, b.DS4_BUTTON_CIRCLE),
            (0x4000, b.DS4_BUTTON_SQUARE),
            (0x8000, b.DS4_BUTTON_TRIANGLE),
            (0x0100, b.DS4_BUTTON_SHOULDER_LEFT),
            (0x0200, b.DS4_BUTTON_SHOULDER_RIGHT),
            (0x0040, b.DS4_BUTTON_THUMB_LEFT),
            (0x0080, b.DS4_BUTTON_THUMB_RIGHT),
            (0x0010, b.DS4_BUTTON_OPTIONS),
            (0x0020, b.DS4_BUTTON_SHARE),
        )
        for mask, button in mapping:
            if buttons & mask:
                self._gamepad.press_button(button)
            else:
                self._gamepad.release_button(button)

    def move(self, dx: int, dy: int) -> bool:
        """Feed an aim correction; the passthrough loop mixes it into the stick.

        The correction replaces the previous one rather than adding to it.
        Accumulating would treat a stick like a mouse and saturate it to full
        lock within a few frames.
        """
        if not self._ok:
            return False
        with self._lock:
            # Negated on Y so a downward screen correction pushes the stick down.
            self._aim_x = max(-self.STICK_MAX, min(self.STICK_MAX, dx * self._sensitivity))
            self._aim_y = max(-self.STICK_MAX, min(self.STICK_MAX, -dy * self._sensitivity))
            self._last_cmd = time.perf_counter()
        return True

    def reset(self) -> None:
        with self._lock:
            self._aim_x = 0.0
            self._aim_y = 0.0
            self._last_cmd = 0.0

    def set_sensitivity(self, sens: float) -> None:
        self._sensitivity = max(100, min(8000, sens))

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def label(self) -> str:
        if not HAS_VGAMEPAD:
            return "vgamepad not installed"
        if not self._ok:
            return "Controller offline"
        return "DS4 Passthrough"

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._gamepad:
            try:
                self._gamepad.reset()
                self._gamepad.update()
            except Exception:
                pass
            self._gamepad = None
        self._ok = False


# The virtual pad must outlive any single tracking run. Games bind to a pad
# when they start and ignore one that is hot-plugged later, so tearing it down
# on Stop and building a new one on Start leaves the game with a device it will
# not read - and with the real pad cloaked, no working controller at all.
_SHARED: ControllerOutput | None = None


def get_shared_output() -> ControllerOutput:
    """The process-wide virtual pad, created on first use."""
    global _SHARED
    if _SHARED is None:
        _SHARED = ControllerOutput()
    return _SHARED


def shutdown_shared_output() -> None:
    """Tear the virtual pad down. For process exit only."""
    global _SHARED
    if _SHARED is not None:
        _SHARED.close()
        _SHARED = None
