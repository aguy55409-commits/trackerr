from __future__ import annotations

import ctypes
import os
from ctypes import Structure, c_char, c_float, c_int, c_char_p
from pathlib import Path

from exoai_tracker.config import AppConfig
from exoai_tracker.paths import app_dir, resolve_data_path


class ExoaiConfig(Structure):
    _fields_ = [
        ("model_path", c_char * 512),
        ("bind_key", c_char * 32),
        ("mouse_method", c_char * 16),
        ("track_speed", c_float),
        ("aim_lead", c_float),
        ("aim_smooth", c_float),
        ("confidence", c_float),
        ("aim_offset_y", c_float),
        ("fov_idle", c_int),
        ("fov_held", c_int),
        ("dynamic_fov", c_int),
        ("show_overlay", c_int),
        ("target_class", c_int),
        ("capture_size", c_int),
    ]


class ExoaiStats(Structure):
    _fields_ = [
        ("detect_hz", c_float),
        ("infer_ms", c_float),
        ("target_conf", c_float),
        ("state", c_char * 16),
        ("detail", c_char * 128),
        ("running", c_int),
    ]


def _dll_path() -> Path:
    here = Path(__file__).resolve().parent.parent / "native" / "out" / "exoai_core.dll"
    if here.exists():
        return here
    return app_dir() / "exoai_core.dll"


def resolve_onnx(config: AppConfig) -> Path:
    if config.active_model == "AL":
        for name in ("best.fp16.onnx", "best.onnx", "fast.fp16.onnx", "fast.onnx"):
            p = resolve_data_path(Path("models") / "al" / name)
            if p.exists():
                return p
    else:
        for name in ("best.fp16.onnx", "best.onnx"):
            p = resolve_data_path(Path("models") / "fn" / name)
            if p.exists():
                return p
    return Path(config.resolve_model_path())


class NativeEngine:
    def __init__(self) -> None:
        self._dll = None
        path = _dll_path()
        if not path.exists():
            raise FileNotFoundError(f"Native core missing: {path}\nBuild with native\\build.bat")
        capi = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\Lib\site-packages\onnxruntime\capi"))
        torch_lib = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\Lib\site-packages\torch\lib"))
        for folder in (path.parent, capi, torch_lib):
            if folder.exists():
                os.add_dll_directory(str(folder))
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        if torch_lib.exists():
            for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
                dep = torch_lib / name
                if dep.exists():
                    ctypes.WinDLL(str(dep))
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            pass
        self._dll = ctypes.WinDLL(str(path))
        self._dll.exoai_start.argtypes = [ctypes.POINTER(ExoaiConfig)]
        self._dll.exoai_start.restype = c_int
        self._dll.exoai_stop.argtypes = []
        self._dll.exoai_stop.restype = None
        self._dll.exoai_apply.argtypes = [ctypes.POINTER(ExoaiConfig)]
        self._dll.exoai_apply.restype = c_int
        self._dll.exoai_stats.argtypes = [ctypes.POINTER(ExoaiStats)]
        self._dll.exoai_stats.restype = c_int
        self._dll.exoai_running.restype = c_int
        self._dll.exoai_last_error.restype = c_char_p

    def _pack(self, config: AppConfig) -> ExoaiConfig:
        onnx = resolve_onnx(config)
        packed = ExoaiConfig()
        packed.model_path = str(onnx).encode("utf-8")[:511]
        packed.bind_key = config.bind_key.encode("utf-8")[:31]
        packed.mouse_method = config.mouse_method.encode("utf-8")[:15]
        packed.track_speed = float(config.track_speed)
        packed.aim_lead = float(config.aim_lead)
        packed.aim_smooth = float(config.aim_smooth)
        packed.confidence = float(config.confidence)
        packed.aim_offset_y = float(config.aim_offset_y)
        packed.fov_idle = int(config.fov_radius)
        packed.fov_held = int(config.fov_radius_held)
        packed.dynamic_fov = 1 if config.dynamic_fov_enabled else 0
        packed.show_overlay = 1 if config.show_overlay else 0
        packed.target_class = 1 if config.active_model == "AL" else 0
        packed.capture_size = 320 if config.active_model == "AL" else int(config.capture_size)
        return packed

    def start(self, config: AppConfig) -> None:
        cfg = self._pack(config)
        if not resolve_onnx(config).exists():
            raise FileNotFoundError(
                f"ONNX model not found for {config.active_model}. "
                "AL uses models/al/best.fp16.onnx"
            )
        ok = self._dll.exoai_start(ctypes.byref(cfg))
        if not ok:
            err = self._dll.exoai_last_error()
            raise RuntimeError(err.decode("utf-8", "replace") if err else "native start failed")

    def stop(self) -> None:
        self._dll.exoai_stop()

    def apply(self, config: AppConfig) -> None:
        cfg = self._pack(config)
        self._dll.exoai_apply(ctypes.byref(cfg))

    def running(self) -> bool:
        return bool(self._dll.exoai_running())

    def stats(self) -> dict:
        s = ExoaiStats()
        self._dll.exoai_stats(ctypes.byref(s))
        return {
            "state": s.state.decode("utf-8", "replace").rstrip("\x00") or "IDLE",
            "detect_hz": float(s.detect_hz),
            "infer_ms": float(s.infer_ms),
            "target_conf": float(s.target_conf) or None,
            "detail": s.detail.decode("utf-8", "replace").rstrip("\x00"),
            "running": bool(s.running),
        }
