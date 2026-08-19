from exoai_tracker.rzctl.rzctl_nt import (
    FILE_SHARE_READ,
    FILE_SHARE_WRITE,
    OPEN_EXISTING,
    Structure,
    byref,
    c_int32,
    c_ulong,
    find_sym_link,
    kernel32,
    pointer,
    sizeof,
)


def enum(**enums):
    return type("Enum", (), enums)


MOUSE_CLICK = enum(
    LEFT_DOWN=1,
    LEFT_UP=2,
    RIGHT_DOWN=4,
    RIGHT_UP=8,
    SCROLL_CLICK_DOWN=16,
    SCROLL_CLICK_UP=32,
    BACK_DOWN=64,
    BACK_UP=128,
    FOWARD_DOWN=256,
    FOWARD_UP=512,
    SCROLL_DOWN=4287104000,
    SCROLL_UP=7865344,
)

KEYBOARD_INPUT_TYPE = enum(KEYBOARD_DOWN=0, KEYBOARD_UP=1)


class RZCONTROL_IOCTL_STRUCT(Structure):
    _fields_ = [
        ("unk0", c_int32),
        ("unk1", c_int32),
        ("max_val_or_scan_code", c_int32),
        ("click_mask", c_int32),
        ("unk3", c_int32),
        ("x", c_int32),
        ("y", c_int32),
        ("unk4", c_int32),
    ]


IOCTL_MOUSE = 0x88883020
MAX_VAL = 65536
RZCONTROL_MOUSE = 2
RZCONTROL_KEYBOARD = 1


def _valid_handle(handle) -> bool:
    if handle is None:
        return False
    value = int(handle) & 0xFFFFFFFFFFFFFFFF
    return 0 < value < 0xFFFFFFFFFFFFFFFF


class RZCONTROL:
    hDevice = None

    def init(self) -> bool:
        if _valid_handle(RZCONTROL.hDevice):
            kernel32.CloseHandle(RZCONTROL.hDevice)
        RZCONTROL.hDevice = None

        found, name = find_sym_link("\\GLOBAL??", "RZCONTROL")
        if not found or not name:
            return False

        sym_link = "\\\\?\\" + name
        # Original rzctl opens with desiredAccess=0
        handle = kernel32.CreateFileW(
            sym_link,
            0,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            0,
            0,
        )
        if not _valid_handle(handle):
            RZCONTROL.hDevice = None
            return False

        RZCONTROL.hDevice = handle
        return True

    def impl_mouse_ioctl(self, ioctl: RZCONTROL_IOCTL_STRUCT) -> bool:
        if not ioctl or not _valid_handle(RZCONTROL.hDevice):
            return False
        p_ioctl = pointer(ioctl)
        junk = c_ulong()
        b_result = kernel32.DeviceIoControl(
            RZCONTROL.hDevice,
            IOCTL_MOUSE,
            p_ioctl,
            sizeof(RZCONTROL_IOCTL_STRUCT),
            0,
            0,
            byref(junk),
            0,
        )
        if not b_result:
            self.init()
            return False
        return True

    def mouse_move(self, x: int, y: int, from_start_point: bool) -> bool:
        max_val = 0
        if not from_start_point:
            max_val = MAX_VAL
            if x < 1:
                x = 1
            if x > max_val:
                x = max_val
            if y < 1:
                y = 1
            if y > max_val:
                y = max_val
        mm = RZCONTROL_IOCTL_STRUCT(0, RZCONTROL_MOUSE, max_val, 0, 0, int(x), int(y), 0)
        return self.impl_mouse_ioctl(mm)

    def mouse_click(self, click_mask: int) -> bool:
        mm = RZCONTROL_IOCTL_STRUCT(0, RZCONTROL_MOUSE, 0, click_mask, 0, 0, 0, 0)
        return self.impl_mouse_ioctl(mm)
