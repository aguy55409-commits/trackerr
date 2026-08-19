from ctypes import (
    POINTER,
    Structure,
    addressof,
    byref,
    c_byte,
    c_int32,
    c_ulong,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    memset,
    pointer,
    sizeof,
)
from ctypes.wintypes import HANDLE, ULONG, USHORT

ntdll = __import__("ctypes").windll.ntdll
kernel32 = __import__("ctypes").windll.kernel32

NTSTATUS = c_int32
STATUS_SUCCESS = 0
STATUS_BUFFER_TOO_SMALL = 0xC0000023
PVOID = c_void_p
PWSTR = c_wchar_p
DIRECTORY_QUERY = 0x0001
OBJ_CASE_INSENSITIVE = 0x00000040
INVALID_HANDLE_VALUE = -1
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3


class UNICODE_STRING(Structure):
    _fields_ = [("Length", USHORT), ("MaximumLength", USHORT), ("Buffer", PWSTR)]


class OBJECT_ATTRIBUTES(Structure):
    _fields_ = [
        ("Length", ULONG),
        ("RootDirectory", HANDLE),
        ("ObjectName", POINTER(UNICODE_STRING)),
        ("Attributes", ULONG),
        ("SecurityDescriptor", PVOID),
        ("SecurityQualityOfService", PVOID),
    ]


class OBJECT_DIRECTORY_INFORMATION(Structure):
    _fields_ = [("Name", UNICODE_STRING), ("TypeName", UNICODE_STRING)]


def InitializeObjectAttributes(
    InitializedAttributes, ObjectName, Attributes, RootDirectory, SecurityDescriptor
):
    memset(addressof(InitializedAttributes), 0, sizeof(InitializedAttributes))
    InitializedAttributes.Length = sizeof(InitializedAttributes)
    InitializedAttributes.ObjectName = ObjectName
    InitializedAttributes.Attributes = Attributes
    InitializedAttributes.RootDirectory = RootDirectory
    InitializedAttributes.SecurityDescriptor = SecurityDescriptor
    InitializedAttributes.SecurityQualityOfService = None


def RtlInitUnicodeString(DestinationString, Src):
    memset(addressof(DestinationString), 0, sizeof(DestinationString))
    DestinationString.Buffer = cast(Src, PWSTR)
    DestinationString.Length = sizeof(Src) - 2
    DestinationString.MaximumLength = DestinationString.Length
    return STATUS_SUCCESS


def open_directory(root_handle, dir_name, desired_access):
    dir_handle = HANDLE()
    p_us_dir = None
    if dir_name:
        w_dir = create_unicode_buffer(dir_name)
        us_dir = UNICODE_STRING()
        RtlInitUnicodeString(us_dir, w_dir)
        p_us_dir = pointer(us_dir)
    obj_attr = OBJECT_ATTRIBUTES()
    InitializeObjectAttributes(
        obj_attr, p_us_dir, OBJ_CASE_INSENSITIVE, root_handle, None
    )
    status = ntdll.NtOpenDirectoryObject(
        byref(dir_handle), desired_access, byref(obj_attr)
    )
    if status != STATUS_SUCCESS:
        raise OSError("NtOpenDirectoryObject failed")
    return dir_handle


def _read_unicode_string(us: UNICODE_STRING) -> str:
    if not us.Buffer or us.Length <= 0:
        return ""
    return us.Buffer[: us.Length // 2]


def find_sym_link(_dir: str, name: str) -> tuple[bool, str | None]:
    dir_handle = open_directory(None, "\\GLOBAL??", DIRECTORY_QUERY)
    try:
        query_context = ULONG(0)
        length = ULONG(0)
        while True:
            status = ntdll.NtQueryDirectoryObject(
                dir_handle,
                None,
                0,
                True,
                False,
                byref(query_context),
                byref(length),
            )
            if (status & 0xFFFFFFFF) != STATUS_BUFFER_TOO_SMALL:
                break

            buffer = (c_byte * length.value)()
            status = ntdll.NtQueryDirectoryObject(
                dir_handle,
                byref(buffer),
                length,
                True,
                False,
                byref(query_context),
                byref(length),
            )
            if (status & 0xFFFFFFFF) != STATUS_SUCCESS:
                break

            entry = cast(buffer, POINTER(OBJECT_DIRECTORY_INFORMATION)).contents
            link_name = _read_unicode_string(entry.Name)
            if name in link_name:
                return True, link_name
    finally:
        ntdll.NtClose(dir_handle)
    return False, None
