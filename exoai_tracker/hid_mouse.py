"""Raw HID mouse injection - writes directly to mouse HID device."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

# Windows APIs
setupapi = ctypes.windll.setupapi
hid = ctypes.windll.hid
kernel32 = ctypes.windll.kernel32

# GUIDs
GUID_DEVINTERFACE_MOUSE = (0x378DE44C, 0x56EF, 0x11D1, 
    (0xBC, 0x8C, 0x00, 0xA0, 0xC9, 0x14, 0x05, 0xDD))
GUID_DEVINTERFACE_HID = (0x4D1E55B2, 0xF16F, 0x11CF,
    (0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30))

# Constants
DIGCF_PRESENT = 0x02
DIGCF_DEVICEINTERFACE = 0x10
GENERIC_WRITE = 0x40000000
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x01
FILE_SHARE_WRITE = 0x02
OPEN_EXISTING = 3
INVALID_HANDLE = 0xFFFFFFFFFFFFFFFF

_PKT_COUNT = 0
_OK_COUNT = 0
_LAST_DX = 0
_LAST_DY = 0
_HANDLE = None
_METHOD = "none"
_DEVICE_PATH = None


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("InterfaceClassGuid", GUID),
        ("Flags", wintypes.DWORD),
        ("Reserved", ctypes.POINTER(ctypes.c_ulong)),
    ]


class SP_DEVICE_INTERFACE_DETAIL_DATA_W(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("DevicePath", ctypes.c_wchar * 260),
    ]


class HIDP_CAPS(ctypes.Structure):
    _fields_ = [
        ("Usage", ctypes.c_ushort),
        ("UsagePage", ctypes.c_ushort),
        ("InputReportByteLength", ctypes.c_ushort),
        ("OutputReportByteLength", ctypes.c_ushort),
        ("FeatureReportByteLength", ctypes.c_ushort),
        ("Reserved", ctypes.c_ushort * 17),
        ("NumberLinkCollectionNodes", ctypes.c_ushort),
        ("NumberInputButtonCaps", ctypes.c_ushort),
        ("NumberInputValueCaps", ctypes.c_ushort),
        ("NumberInputDataIndices", ctypes.c_ushort),
        ("NumberOutputButtonCaps", ctypes.c_ushort),
        ("NumberOutputValueCaps", ctypes.c_ushort),
        ("NumberOutputDataIndices", ctypes.c_ushort),
        ("NumberFeatureButtonCaps", ctypes.c_ushort),
        ("NumberFeatureValueCaps", ctypes.c_ushort),
        ("NumberFeatureDataIndices", ctypes.c_ushort),
    ]


def _make_guid(data) -> GUID:
    g = GUID()
    g.Data1 = data[0]
    g.Data2 = data[1]
    g.Data3 = data[2]
    for i, b in enumerate(data[3]):
        g.Data4[i] = b
    return g


def _find_hid_mice() -> list[str]:
    """Find all HID mouse device paths."""
    devices = []
    all_hid = []
    
    guid = _make_guid(GUID_DEVINTERFACE_HID)
    
    # Get device info set
    hdev_info = setupapi.SetupDiGetClassDevsW(
        ctypes.byref(guid), None, None, 
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )
    
    if not hdev_info or hdev_info == INVALID_HANDLE:
        # Try without GUID filter
        hdev_info = setupapi.SetupDiGetClassDevsW(
            None, "HID", None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
        )
    
    if not hdev_info or hdev_info == INVALID_HANDLE:
        return devices
    
    try:
        dev_interface = SP_DEVICE_INTERFACE_DATA()
        dev_interface.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
        
        index = 0
        while index < 100:  # Limit iterations
            # Enumerate interfaces
            if not setupapi.SetupDiEnumDeviceInterfaces(
                hdev_info, None, ctypes.byref(guid), index, ctypes.byref(dev_interface)
            ):
                break
            
            # Get required size
            required_size = wintypes.DWORD(0)
            setupapi.SetupDiGetDeviceInterfaceDetailW(
                hdev_info, ctypes.byref(dev_interface), None, 0, 
                ctypes.byref(required_size), None
            )
            
            if required_size.value > 0:
                # Allocate buffer for path
                buf_size = required_size.value + 256
                buf = (ctypes.c_byte * buf_size)()
                detail = ctypes.cast(buf, ctypes.POINTER(SP_DEVICE_INTERFACE_DETAIL_DATA_W)).contents
                detail.cbSize = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
                
                if setupapi.SetupDiGetDeviceInterfaceDetailW(
                    hdev_info, ctypes.byref(dev_interface), ctypes.byref(detail),
                    buf_size, None, None
                ):
                    path = detail.DevicePath
                    all_hid.append(path)
                    # Check if it's a mouse (usage page 1, usage 2)
                    if _is_mouse_device(path):
                        devices.append(path)
            
            index += 1
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(hdev_info)
    
    print(f"Total HID devices found: {len(all_hid)}")
    return devices


def _is_mouse_device(path: str) -> bool:
    """Check if HID device is a mouse."""
    try:
        h = kernel32.CreateFileW(
            path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None
        )
        if not h or h == INVALID_HANDLE or int(h) <= 0:
            return False
        
        try:
            # Get preparsed data
            ppd = ctypes.c_void_p()
            if not hid.HidD_GetPreparsedData(h, ctypes.byref(ppd)):
                return False
            
            try:
                caps = HIDP_CAPS()
                if hid.HidP_GetCaps(ppd, ctypes.byref(caps)) == 0:  # HIDP_STATUS_SUCCESS
                    # Mouse: UsagePage=1 (Generic Desktop), Usage=2 (Mouse)
                    if caps.UsagePage == 1 and caps.Usage == 2:
                        return True
            finally:
                hid.HidD_FreePreparsedData(ppd)
        finally:
            kernel32.CloseHandle(h)
    except:
        pass
    return False


def _open_mouse_write(path: str):
    """Open mouse device for writing."""
    h = kernel32.CreateFileW(
        path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        None, OPEN_EXISTING, 0, None
    )
    if h and h != INVALID_HANDLE and int(h) > 0:
        return h
    return None


def _write_hid_report(dx: int, dy: int) -> bool:
    """Write raw HID mouse report."""
    global _HANDLE
    
    if _HANDLE is None:
        return False
    
    # Standard mouse HID report: [buttons, x, y, wheel]
    # Report format depends on device, but typical boot mouse is 3-4 bytes
    report = (ctypes.c_ubyte * 4)()
    report[0] = 0  # Buttons (none pressed)
    report[1] = dx & 0xFF if dx >= 0 else (256 + dx) & 0xFF  # X movement (signed byte)
    report[2] = dy & 0xFF if dy >= 0 else (256 + dy) & 0xFF  # Y movement (signed byte)
    report[3] = 0  # Wheel
    
    written = wintypes.DWORD(0)
    ok = kernel32.WriteFile(
        _HANDLE, ctypes.byref(report), 4, ctypes.byref(written), None
    )
    
    return bool(ok) and written.value > 0


def _hid_set_feature(dx: int, dy: int) -> bool:
    """Try HidD_SetOutputReport."""
    global _HANDLE
    
    if _HANDLE is None:
        return False
    
    # Output report with mouse movement
    report = (ctypes.c_ubyte * 5)()
    report[0] = 0  # Report ID
    report[1] = 0  # Buttons
    report[2] = dx & 0xFF if dx >= 0 else (256 + dx) & 0xFF
    report[3] = dy & 0xFF if dy >= 0 else (256 + dy) & 0xFF
    report[4] = 0  # Wheel
    
    return bool(hid.HidD_SetOutputReport(_HANDLE, ctypes.byref(report), 5))


def _mouse_event_move(dx: int, dy: int) -> bool:
    """Fallback mouse_event."""
    try:
        ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
        return True
    except:
        return False


def _sendinput_move(dx: int, dy: int) -> bool:
    """SendInput fallback."""
    try:
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
            ]
        
        class INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        
        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]
        
        inp = INPUT()
        inp.type = 0  # INPUT_MOUSE
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.dwFlags = 0x0001  # MOUSEEVENTF_MOVE
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return True
    except:
        return False


def move_mouse(dx: int, dy: int) -> bool:
    """Move mouse via HID injection."""
    global _PKT_COUNT, _OK_COUNT, _LAST_DX, _LAST_DY, _METHOD
    
    if dx == 0 and dy == 0:
        return True
    
    _PKT_COUNT += 1
    _LAST_DX = dx
    _LAST_DY = dy
    
    # Clamp to signed byte range for HID
    dx = max(-127, min(127, dx))
    dy = max(-127, min(127, dy))
    
    # Try HID write
    if _HANDLE is not None:
        if _write_hid_report(dx, dy):
            _OK_COUNT += 1
            _METHOD = "hid_write"
            return True
        
        if _hid_set_feature(dx, dy):
            _OK_COUNT += 1
            _METHOD = "hid_output"
            return True
    
    # Try SendInput
    if _sendinput_move(dx, dy):
        _OK_COUNT += 1
        _METHOD = "sendinput"
        return True
    
    # Fallback
    _mouse_event_move(dx, dy)
    _METHOD = "mouse_event"
    return True


def _get_raw_input_mice() -> list[str]:
    """Get mouse device paths from raw input API."""
    user32 = ctypes.windll.user32
    
    class RAWINPUTDEVICELIST(ctypes.Structure):
        _fields_ = [('hDevice', ctypes.c_void_p), ('dwType', wintypes.DWORD)]
    
    num = ctypes.c_uint(0)
    user32.GetRawInputDeviceList(None, ctypes.byref(num), ctypes.sizeof(RAWINPUTDEVICELIST))
    
    if num.value == 0:
        return []
    
    devices = (RAWINPUTDEVICELIST * num.value)()
    user32.GetRawInputDeviceList(devices, ctypes.byref(num), ctypes.sizeof(RAWINPUTDEVICELIST))
    
    mice = []
    for dev in devices:
        if dev.dwType == 0:  # RIM_TYPEMOUSE
            name_len = ctypes.c_uint(0)
            user32.GetRawInputDeviceInfoW(dev.hDevice, 0x20000007, None, ctypes.byref(name_len))
            if name_len.value > 0:
                name_buf = ctypes.create_unicode_buffer(name_len.value)
                user32.GetRawInputDeviceInfoW(dev.hDevice, 0x20000007, name_buf, ctypes.byref(name_len))
                mice.append(name_buf.value)
    
    return mice


def init() -> bool:
    """Initialize - find and open mouse HID device."""
    global _HANDLE, _METHOD, _DEVICE_PATH
    
    print("Searching for mouse devices via Raw Input API...")
    mice = _get_raw_input_mice()
    print(f"Found {len(mice)} mouse device(s)")
    
    # Prioritize Razer devices (VID 1532)
    razer = [m for m in mice if "1532" in m]
    other = [m for m in mice if "1532" not in m]
    ordered = razer + other
    
    for path in ordered:
        short = path.split("#")[1] if "#" in path else path[:50]
        print(f"Trying: {short}...")
        
        # Try write access
        h = kernel32.CreateFileW(
            path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None
        )
        if h and h != INVALID_HANDLE and int(h) > 0:
            _HANDLE = h
            _DEVICE_PATH = path
            _METHOD = "hid_write"
            print(f"Opened for write!")
            return True
        
        # Try read/write access
        h = kernel32.CreateFileW(
            path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None
        )
        if h and h != INVALID_HANDLE and int(h) > 0:
            _HANDLE = h
            _DEVICE_PATH = path
            _METHOD = "hid_rw"
            print(f"Opened for read/write!")
            return True
    
    print("No writable mouse found, using SendInput fallback")
    _METHOD = "sendinput"
    return True


def packet_count() -> int:
    return _PKT_COUNT


def ok_count() -> int:
    return _OK_COUNT


def last_move() -> tuple[int, int]:
    return _LAST_DX, _LAST_DY


def label() -> str:
    return f"{_METHOD} · {_OK_COUNT}/{_PKT_COUNT}"


def ready() -> bool:
    return True  # Always ready with fallback
