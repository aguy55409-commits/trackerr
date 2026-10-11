from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from ctypes import wintypes

import cv2
import mss
import numpy as np

from exoai_tracker.paths import resolve_data_path

user32 = ctypes.windll.user32
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32 = ctypes.windll.gdi32
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.GetDIBits.argtypes = [
    wintypes.HDC,
    wintypes.HBITMAP,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.UINT,
]
gdi32.GetDIBits.restype = ctypes.c_int
SRCCOPY = 0x00CC0020

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
_ENUM_CB = None


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def _find_apex_hwnd() -> int:
    for title in ("Apex Legends", "r5apex"):
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return int(hwnd)
    found: list[int] = []

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        title = buf.value.lower()
        if "apex legends" in title and "anti-cheat" not in title:
            found.append(int(hwnd))
            return False
        return True

    global _ENUM_CB
    _ENUM_CB = WNDENUMPROC(_cb)
    user32.EnumWindows(_ENUM_CB, 0)
    return found[0] if found else 0


@dataclass
class Detection:
    x: float
    y: float
    confidence: float
    width: float
    height: float
    cls: int = 0
    kind: str = "other"
    track_id: int = -1


@dataclass
class _InferSettings:
    confidence: float
    iou: float
    capture_size: int
    fov_radius_sq: int
    min_box_ratio: float
    max_box_ratio: float


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        capture_size: int,
        target_class_ids: tuple[int, ...] | None = None,
        iou: float = 0.5,
        min_box_ratio: float = 0.005,
        max_box_ratio: float = 0.70,
        capture_method: str = "dxcam",
        tracker: str | None = None,
    ) -> None:
        self._capture_method = capture_method
        self._tracker_cfg = self.resolve_tracker_cfg(tracker)
        self.target_class_ids = set(target_class_ids or ())
        suffix = Path(model_path).suffix.lower()
        self._device = 0 if suffix in {".onnx", ".engine"} else self._resolve_device()
        self._debug_flag = False
        self._settings = _InferSettings(
            confidence=confidence,
            iou=iou,
            capture_size=capture_size,
            fov_radius_sq=0,
            min_box_ratio=min_box_ratio,
            max_box_ratio=max_box_ratio,
        )
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self._imgsz = self._read_net_imgsz(model_path, capture_size)
        self._infer_buf: np.ndarray | None = None
        if suffix == ".pt":
            try:
                self.model.fuse()
            except Exception:
                pass
        self._class_roles = self._parse_class_roles()
        self.target_class_ids = self._resolve_target_classes(self.target_class_ids)

        self.screen_left = 0
        self.screen_top = 0
        self.screen_w = int(user32.GetSystemMetrics(0))
        self.screen_h = int(user32.GetSystemMetrics(1))
        self.monitor = {
            "left": self.screen_left,
            "top": self.screen_top,
            "width": self.screen_w,
            "height": self.screen_h,
        }
        self.center_x = self.screen_left + self.screen_w // 2
        self.center_y = self.screen_top + self.screen_h // 2
        self._region = self._build_region()
        h, w = self._region["height"], self._region["width"]
        self._bufs = [
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
        ]
        self._write_i = 0
        self._chw_buf = np.empty((3, h, w), dtype=np.uint8)
        self._host_staging: torch.Tensor | None = None
        self._class_ids_np: np.ndarray | None = None

        self._latest_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._capture_stop = threading.Event()
        self._capture_error: Exception | None = None
        self._capture_fell_back = False
        self._frame_ready = threading.Event()
        self._cached_detect: tuple[list[Detection], np.ndarray | None, dict[str, int] | None] | None = None
        self.frame_id = 0
        self._capture_seq = 0
        self._last_infer_seq = -1
        self._input_tensor: torch.Tensor | None = None
        self._host_hwc: torch.Tensor | None = None
        self._gpu_hwc: torch.Tensor | None = None
        self._cuda_stream: torch.cuda.Stream | None = None
        self._bgr_buf: np.ndarray | None = None
        self._want_frames = threading.Event()
        self._cam_lock = threading.Lock()
        self._camera = None
        self._sct = None
        self._dxgi_running = False
        self.last_info = {"raw": 0, "keep": 0, "cap": "none", "mean": 0.0, "err": ""}
        self._is_engine = suffix == ".engine"
        self._is_onnx = suffix == ".onnx"
        self._use_half = self._device != "cpu" and not self._is_engine and not self._is_onnx
        self._tune_runtime()
        self._sync_class_filter()
        self._predict_kw = {
            "conf": min(float(self._settings.confidence), 0.15),
            "iou": max(0.25, self._settings.iou),
            "imgsz": self._imgsz,
            "device": self._device,
            "verbose": False,
            "max_det": 20,
            "augment": False,
            "agnostic_nms": True,
            "workers": 0,
            "save": False,
            "stream": False,  # Non-streaming is faster for single images
            "embed": None,
            "visualize": False,
            "classes": None,
        }
        if self._use_half:
            self._predict_kw["half"] = True
        if not self._is_onnx and not self._is_engine:
            self._init_host_staging()

    def _read_net_imgsz(self, model_path: str, fallback: int) -> int:
        try:
            session = getattr(getattr(self.model, "model", None), "session", None)
            if session is not None:
                dims = session.get_inputs()[0].shape
                hw = [int(d) for d in dims if isinstance(d, (int, float)) and int(d) > 3]
                if hw:
                    return int(hw[-1])
        except Exception:
            pass
        try:
            import onnx

            graph = onnx.load(model_path, load_external_data=False).graph
            dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
            if len(dims) >= 4 and dims[2] > 0:
                return int(dims[2])
        except Exception:
            pass
        name = Path(model_path).name.lower()
        if "fast" in name:
            return 512
        return int(fallback)

    def _infer_image(self, frame: np.ndarray) -> np.ndarray:
        bgr = self._frame_for_track(frame)
        net = int(self._imgsz)
        if bgr.shape[0] == net and bgr.shape[1] == net:
            return bgr
        if self._infer_buf is None or self._infer_buf.shape[0] != net:
            self._infer_buf = np.empty((net, net, 3), dtype=np.uint8)
        cv2.resize(bgr, (net, net), dst=self._infer_buf, interpolation=cv2.INTER_LINEAR)
        return self._infer_buf

    @staticmethod
    def _resolve_device() -> str | int:
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def resolve_tracker_cfg(tracker: str | None = None) -> str:
        """Resolve an AL-tuned YAML, then an Ultralytics builtin name."""
        if tracker:
            raw = Path(tracker)
            if raw.exists():
                return str(raw.resolve())
            resolved = resolve_data_path(tracker)
            if resolved.exists():
                return str(resolved)
            return tracker
        al = resolve_data_path("models/al/tracktrack.yaml")
        if al.exists():
            return str(al)
        return "tracktrack.yaml"

    def set_tracker(self, tracker: str) -> None:
        self._tracker_cfg = self.resolve_tracker_cfg(tracker)
        self.reset_tracker()

    def reset_tracker(self) -> None:
        """Drop persist IDs so the next track() starts a new session."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        # Ultralytics skips re-init when persist=True AND hasattr(trackers).
        # An empty list still counts, so delete the attr instead of clearing it.
        for attr in ("trackers", "vid_path", "_feats"):
            if hasattr(predictor, attr):
                delattr(predictor, attr)

    @staticmethod
    def resolve_fastest_engine(model_dir: Path, preferred: str = "best.engine") -> str | None:
        """Pick smallest TensorRT engine in folder (320 beats 512 for FPS)."""
        if not model_dir.is_dir():
            return None
        engines = sorted(model_dir.glob("*.engine"), key=lambda p: p.stat().st_size)
        if not engines:
            return None
        preferred_path = model_dir / preferred
        if preferred_path.exists():
            return str(preferred_path)
        return str(engines[0])

    @staticmethod
    def _role_from_name(name: str) -> str:
        n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if n in {"0", "body", "player", "person", "operator"}:
            return "enemy"
        if n in {"1", "head", "heads"}:
            return "head"
        if n in {"knocked", "downed", "down"}:
            return "other"
        if (
            n in {"head", "player_head", "enemy_head", "heads"}
            or n.endswith("_head")
            or n.startswith("head_")
        ):
            return "head"
        if (
            n in {"ally", "allies", "teammate", "teammates", "friendly", "team"}
            or "teammate" in n
            or n.startswith("ally")
        ):
            return "ally"
        if (
            n in {"enemy", "enemies", "body", "person", "operator"}
            or n.startswith("enemy_")
            or n.endswith("_enemy")
        ):
            return "enemy"
        return "other"

    def _parse_class_roles(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        return {int(class_id): self._role_from_name(name) for class_id, name in names.items()}

    def _resolve_target_classes(self, fallback: set[int]) -> set[int]:
        roles = getattr(self, "_class_roles", None) or self._parse_class_roles()
        self._class_roles = roles
        # 4-class AL: teammate/enemy/knocked/other. Do not pin YOLO classes —
        # a tight filter was dropping every box. Allies are rejected in scan.
        return set()

    def _sync_class_filter(self) -> None:
        self._class_ids_np = (
            np.fromiter(self.target_class_ids, dtype=np.int32)
            if self.target_class_ids
            else None
        )

    def _init_host_staging(self) -> None:
        if self._device == "cpu" or self._is_onnx or self._is_engine:
            self._host_staging = None
            return
        import torch

        h, w = self._chw_buf.shape[1], self._chw_buf.shape[2]
        self._host_staging = torch.empty((3, h, w), dtype=torch.uint8, pin_memory=True)

    def _tune_runtime(self) -> None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        if self._is_onnx or self._is_engine:
            return
        try:
            import torch

            torch.set_num_threads(2)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
            if self._use_half and hasattr(self.model, "model"):
                self.model.model.half()
        except Exception:
            self._use_half = False

    def _build_region(self) -> dict[str, int]:
        half = self._settings.capture_size // 2
        left = max(self.screen_left, self.center_x - half)
        top = max(self.screen_top, self.center_y - half)
        width = min(self._settings.capture_size, self.screen_left + self.screen_w - left)
        height = min(self._settings.capture_size, self.screen_top + self.screen_h - top)
        return {"left": left, "top": top, "width": width, "height": height}

    def set_tracking_active(self, active: bool) -> None:
        if active:
            self._want_frames.set()
            self._start_dxgi()
            return
        self._want_frames.clear()
        self._stop_dxgi()

    def capture_fell_back(self) -> bool:
        return self._capture_fell_back

    def _release_capture(self) -> None:
        self._dxgi_running = False
        with self._cam_lock:
            camera = self._camera
            self._camera = None
            sct = self._sct
            self._sct = None
        if camera is not None:
            for name in ("stop", "release"):
                fn = getattr(camera, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass

    def _start_dxgi(self) -> None:
        if self._capture_method == "mss":
            return
        region = self._region
        bounds = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        with self._cam_lock:
            if self._camera is None:
                try:
                    import dxcam

                    self._camera = dxcam.create(
                        output_idx=0, output_color="RGB", max_buffer_len=2
                    )
                except Exception as exc:
                    self._capture_error = exc
                    self._capture_method = "mss"
                    self._capture_fell_back = True
                    return
                if self._camera is None:
                    self._capture_method = "mss"
                    self._capture_fell_back = True
                    return
            if not self._dxgi_running:
                try:
                    self._camera.start(region=bounds, target_fps=0)
                except TypeError:
                    try:
                        self._camera.start(region=bounds)
                    except Exception as exc:
                        self._capture_error = exc
                        return
                except Exception as exc:
                    self._capture_error = exc
                    return
                self._dxgi_running = True

    def _stop_dxgi(self) -> None:
        with self._cam_lock:
            if self._camera is not None and self._dxgi_running:
                try:
                    self._camera.stop()
                except Exception:
                    pass
            self._dxgi_running = False

    def _grab(self) -> np.ndarray | None:
        frame = None
        cap = "none"
        if self._capture_method != "mss":
            try:
                frame = self._grab_dxgi()
                if frame is not None:
                    cap = "dxgi"
            except Exception as exc:
                self._capture_error = exc
                self._stop_dxgi()
                self._release_capture()
                self._capture_method = "mss"
                self._capture_fell_back = True
        if frame is None:
            frame = self._grab_mss()
            if frame is not None:
                cap = "mss"
        if frame is not None and float(frame.mean()) < 3.0:
            hwnd_frame = self._grab_hwnd()
            if hwnd_frame is not None and float(hwnd_frame.mean()) >= 3.0:
                frame = hwnd_frame
                cap = "hwnd"
        if frame is not None:
            self.last_info["cap"] = cap
            self.last_info["mean"] = float(frame.mean())
        else:
            self.last_info["cap"] = "none"
            self.last_info["mean"] = 0.0
        return frame

    def _grab_hwnd(self) -> np.ndarray | None:
        hwnd = _find_apex_hwnd()
        if not hwnd:
            return None
        region = self._region
        w, h = int(region["width"]), int(region["height"])
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        src_x = int(region["left"] - pt.x)
        src_y = int(region["top"] - pt.y)
        hdc = user32.GetDC(hwnd)
        if not hdc:
            return None
        mem = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
        old = gdi32.SelectObject(mem, bmp)
        try:
            if not gdi32.BitBlt(mem, 0, 0, w, h, hdc, src_x, src_y, SRCCOPY):
                return None
            info = _BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            info.bmiHeader.biWidth = w
            info.bmiHeader.biHeight = -h
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0
            dest = self._bufs[self._write_i]
            if dest.shape[0] != h or dest.shape[1] != w:
                dest = np.empty((h, w, 3), dtype=np.uint8)
                self._bufs[self._write_i] = dest
            bgra = np.empty((h, w, 4), dtype=np.uint8)
            if gdi32.GetDIBits(mem, bmp, 0, h, bgra.ctypes.data, ctypes.byref(info), 0) == 0:
                return None
            cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=dest)
            self._write_i ^= 1
            return dest
        finally:
            gdi32.SelectObject(mem, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem)
            user32.ReleaseDC(hwnd, hdc)

    def _grab_dxgi(self) -> np.ndarray | None:
        if not self._dxgi_running:
            self._start_dxgi()
        region = self._region
        bounds = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        with self._cam_lock:
            if self._camera is None:
                return None
            frame = None
            if self._dxgi_running:
                try:
                    frame = self._camera.get_latest_frame()
                except Exception:
                    frame = None
            if frame is None:
                frame = self._camera.grab(region=bounds)
        if frame is None:
            return None
        return self._copy_frame(frame)

    def _grab_mss(self) -> np.ndarray | None:
        region = self._region
        with self._cam_lock:
            if self._sct is None:
                self._sct = mss.mss()
            try:
                shot = self._sct.grab(region)
            except Exception:
                return None
        dest = self._bufs[self._write_i]
        h, w = shot.height, shot.width
        if dest.shape[0] != h or dest.shape[1] != w or dest.shape[2] != 3:
            dest = np.empty((h, w, 3), dtype=np.uint8)
            self._bufs[self._write_i] = dest
        bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(h, w, 4)
        cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=dest)
        self._write_i ^= 1
        return dest

    def _copy_frame(self, source: np.ndarray) -> np.ndarray:
        dest = self._bufs[self._write_i]
        if dest.shape != source.shape:
            dest = np.empty(source.shape, dtype=np.uint8)
            self._bufs[self._write_i] = dest
            h, w = dest.shape[:2]
            self._chw_buf = np.empty((3, h, w), dtype=np.uint8)
            self._init_host_staging()
        np.copyto(dest, source)
        self._write_i ^= 1
        return dest

    def _warmup(self) -> None:
        dummy = np.zeros(
            (self._settings.capture_size, self._settings.capture_size, 3), dtype=np.uint8
        )
        self.model.predict(source=dummy, **self._predict_kw)

    def _frame_for_track(self, frame: np.ndarray) -> np.ndarray:
        """Ultralytics treats numpy HWC as BGR; capture is RGB."""
        if self._bgr_buf is None or self._bgr_buf.shape != frame.shape:
            self._bgr_buf = np.empty_like(frame)
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, dst=self._bgr_buf)
        return np.ascontiguousarray(self._bgr_buf)

    def _prepare_source(self, frame: np.ndarray) -> torch.Tensor:
        if self._device == "cpu":
            chw = np.ascontiguousarray(frame.transpose(2, 0, 1))
            tensor = torch.from_numpy(chw).unsqueeze(0).float()
            return tensor.mul_(1.0 / 255.0)

        h, w = frame.shape[0], frame.shape[1]
        dev = int(self._device)
        if (
            self._host_hwc is None
            or self._host_hwc.shape[0] != h
            or self._host_hwc.shape[1] != w
        ):
            self._host_hwc = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
            self._gpu_hwc = torch.empty((h, w, 3), dtype=torch.uint8, device=dev)
            dtype = torch.float16 if self._use_half else torch.float32
            self._input_tensor = torch.empty((1, 3, h, w), device=dev, dtype=dtype)
            self._cuda_stream = torch.cuda.Stream(device=dev)

        # Direct copy to pinned memory
        self._host_hwc.copy_(torch.from_numpy(frame))
        
        # Async GPU transfer with stream
        with torch.cuda.stream(self._cuda_stream):
            self._gpu_hwc.copy_(self._host_hwc, non_blocking=True)
            self._input_tensor[0] = self._gpu_hwc.permute(2, 0, 1).float().mul_(1.0 / 255.0)
        self._cuda_stream.synchronize()
        return self._input_tensor

    def update_settings(
        self,
        confidence: float,
        capture_size: int,
        target_class_ids: tuple[int, ...] | None = None,
        iou: float | None = None,
        min_box_ratio: float | None = None,
        max_box_ratio: float | None = None,
    ) -> None:
        self._settings.confidence = confidence
        self._predict_kw["conf"] = confidence
        if iou is not None:
            self._settings.iou = iou
            self._predict_kw["iou"] = max(0.25, iou)
        if min_box_ratio is not None:
            self._settings.min_box_ratio = min_box_ratio
        if max_box_ratio is not None:
            self._settings.max_box_ratio = max_box_ratio
        if capture_size != self._settings.capture_size:
            self._settings.capture_size = capture_size
            self._predict_kw["imgsz"] = self._imgsz
            self._region = self._build_region()
            self._input_tensor = None
            self._host_hwc = None
            self._gpu_hwc = None
            self._bgr_buf = None
            h, w = self._region["height"], self._region["width"]
            self._chw_buf = np.empty((3, h, w), dtype=np.uint8)
            self._init_host_staging()
            self.reset_tracker()
        self.target_class_ids = self._resolve_target_classes(set(target_class_ids or ()))
        self._sync_class_filter()
        self._predict_kw["classes"] = sorted(self.target_class_ids) or None

    def load_model(
        self,
        model_path: str,
        target_class_ids: tuple[int, ...] | None = None,
    ) -> None:
        suffix = Path(model_path).suffix.lower()
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self._imgsz = self._read_net_imgsz(model_path, self._settings.capture_size)
        self._predict_kw["imgsz"] = self._imgsz
        self._infer_buf = None
        self._is_engine = suffix == ".engine"
        self._is_onnx = suffix == ".onnx"
        self._use_half = self._device != "cpu" and not self._is_engine and not self._is_onnx
        if self._use_half:
            self._predict_kw["half"] = True
        else:
            self._predict_kw.pop("half", None)
        if suffix == ".pt":
            try:
                self.model.fuse()
            except Exception:
                pass
        self._class_roles = self._parse_class_roles()
        self.target_class_ids = self._resolve_target_classes(set(target_class_ids or ()))
        self._sync_class_filter()
        self._predict_kw["classes"] = sorted(self.target_class_ids) or None
        self._input_tensor = None
        self._host_hwc = None
        self._gpu_hwc = None
        self._bgr_buf = None
        self.reset_tracker()

    def set_fov_radius(self, fov_radius: int) -> None:
        self._settings.fov_radius_sq = fov_radius * fov_radius

    def close(self) -> None:
        self.reset_tracker()
        self._capture_stop.set()
        self._want_frames.clear()
        self._release_capture()
        self._cached_detect = None
        self._last_infer_seq = -1
        with self._latest_lock:
            self._latest_frame = None

    def _get_frame(self, wait_for_new: bool = False) -> tuple[np.ndarray | None, int | None]:
        self._want_frames.set()
        frame = self._grab()
        if frame is None:
            return None, None
        self._capture_seq += 1
        with self._latest_lock:
            self._latest_frame = frame
        return frame, self._capture_seq

    def detect(
        self,
        fov_radius: int,
        *,
        wait_for_new: bool = False,
    ) -> tuple[list[Detection], np.ndarray | None, dict[str, int] | None]:
        if fov_radius * fov_radius != self._settings.fov_radius_sq:
            self.set_fov_radius(fov_radius)

        frame, capture_seq = self._get_frame(wait_for_new=wait_for_new)
        if frame is None:
            self.last_info["err"] = str(self._capture_error or "no frame")
            raise RuntimeError(f"Capture produced no frames: {self.last_info['err']}")
        self._last_infer_seq = capture_seq
        self.frame_id += 1
        debug_frame = frame[:, :, ::-1].copy() if self._debug_flag else None
        try:
            results = self.model.predict(source=self._infer_image(frame), **self._predict_kw)
            if hasattr(results, "__iter__") and not hasattr(results, "boxes"):
                results = list(results)
        except Exception as exc:
            self.last_info["err"] = f"{type(exc).__name__}: {exc}"
            return self._pack_detect([], debug_frame)
        self.last_info["err"] = ""
        return self._boxes_to_detections(frame, results)

    def _boxes_to_detections(self, frame: np.ndarray, results) -> tuple[list[Detection], np.ndarray | None, dict[str, int] | None]:
        detections: list[Detection] = []
        debug_frame = frame[:, :, ::-1].copy() if self._debug_flag else None
        if not results:
            return self._pack_detect(detections, debug_frame)

        first = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(first, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self.last_info["raw"] = 0
            self.last_info["keep"] = 0
            return self._pack_detect(detections, debug_frame)

        xyxy_np = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        infer_h = float(self._imgsz)
        infer_w = float(self._imgsz)
        if infer_w > 0 and frame.shape[1] != infer_w:
            xyxy_np[:, [0, 2]] *= frame.shape[1] / infer_w
            xyxy_np[:, [1, 3]] *= frame.shape[0] / infer_h
        conf_np = boxes.conf.detach().cpu().numpy()
        cls_np = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
        id_tensor = getattr(boxes, "id", None)
        if id_tensor is not None:
            id_np = id_tensor.detach().cpu().numpy().reshape(-1).astype(np.int32, copy=False)
            if id_np.size < xyxy_np.shape[0]:
                id_np = np.pad(id_np, (0, xyxy_np.shape[0] - id_np.size), constant_values=-1)
        else:
            id_np = np.full(xyxy_np.shape[0], -1, dtype=np.int32)

        region_left = self._region["left"]
        region_top = self._region["top"]
        cx0 = self.center_x
        cy0 = self.center_y
        fov_sq = self._settings.fov_radius_sq
        capture_area = self._region["width"] * self._region["height"]
        self.last_info["raw"] = int(xyxy_np.shape[0])
        min_area = capture_area * min(self._settings.min_box_ratio, 0.001)
        max_area = capture_area * self._settings.max_box_ratio
        fov_r = (max(fov_sq, 1.0) ** 0.5)
        max_in_capture = min(self._region["width"], self._region["height"]) * 0.5
        keep = min(max_in_capture, fov_r + 64.0)
        keep_sq = keep * keep

        bw = xyxy_np[:, 2] - xyxy_np[:, 0]
        bh = xyxy_np[:, 3] - xyxy_np[:, 1]
        tx = region_left + (xyxy_np[:, 0] + xyxy_np[:, 2]) * 0.5
        ty = region_top + (xyxy_np[:, 1] + xyxy_np[:, 3]) * 0.5
        dist_sq = (tx - cx0) ** 2 + (ty - cy0) ** 2
        area = bw * bh
        mask = (
            (area >= min_area)
            & (area <= max_area)
            & (bh >= bw * 0.28)
            & (dist_sq <= keep_sq)
            & ~((bw > self._region["width"] * 0.90) & (bh > self._region["height"] * 0.90))
        )
        class_ids = self._class_ids_np
        if class_ids is not None:
            mask &= np.isin(cls_np, class_ids)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            self.last_info["keep"] = 0
            return self._pack_detect(detections, debug_frame)

        rank = dist_sq[idx] - conf_np[idx] * 900.0 - (bh[idx] / np.maximum(bw[idx], 1.0)) * 220.0
        keep_n = min(16, idx.size)
        if idx.size <= keep_n:
            order = np.argsort(rank)
        else:
            top = np.argpartition(rank, keep_n - 1)[:keep_n]
            order = top[np.argsort(rank[top])]
        roles = getattr(self, "_class_roles", {})
        raw = [
            Detection(
                x=float(tx[idx[i]]),
                y=float(ty[idx[i]]),
                confidence=float(conf_np[idx[i]]),
                width=float(bw[idx[i]]),
                height=float(bh[idx[i]]),
                cls=int(cls_np[idx[i]]),
                kind=roles.get(int(cls_np[idx[i]]), "other"),
                track_id=int(id_np[idx[i]]),
            )
            for i in order
        ]
        has_ids = any(d.track_id >= 0 for d in raw)
        detections = raw if len(raw) <= 1 or has_ids else self._merge_overlaps(raw)
        self.last_info["keep"] = len(detections)
        return self._pack_detect(detections, debug_frame)

    def _pack_detect(
        self,
        detections: list[Detection],
        debug_frame: np.ndarray | None,
    ) -> tuple[list[Detection], np.ndarray | None, dict[str, int] | None]:
        packed = (detections, debug_frame, self._region)
        self._cached_detect = packed
        return packed

    @staticmethod
    def _merge_overlaps(dets: list[Detection]) -> list[Detection]:
        if len(dets) <= 1:
            return dets
        kept: list[Detection] = []
        used = [False] * len(dets)
        for i, a in enumerate(dets):
            if used[i]:
                continue
            ax1 = a.x - a.width * 0.5
            ay1 = a.y - a.height * 0.5
            ax2 = a.x + a.width * 0.5
            ay2 = a.y + a.height * 0.5
            best = a
            used[i] = True
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                b = dets[j]
                bx1 = b.x - b.width * 0.5
                by1 = b.y - b.height * 0.5
                bx2 = b.x + b.width * 0.5
                by2 = b.y + b.height * 0.5
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                union = a.width * a.height + b.width * b.height - inter
                if union > 0 and inter / union >= 0.35 and a.kind == b.kind:
                    used[j] = True
                    if b.confidence > best.confidence:
                        best = b
            kept.append(best)
        return kept[:16]

    def set_debug(self, enabled: bool) -> None:
        self._debug_flag = enabled

    def debug_enabled(self) -> bool:
        return self._debug_flag
