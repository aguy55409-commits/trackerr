from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import dxcam
import mss
import numpy as np
import torch
from ultralytics import YOLO

# Lower capture thread priority for less system impact
def _lower_thread_priority() -> None:
    try:
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), -1  # BELOW_NORMAL
        )
    except Exception:
        pass


@dataclass
class Detection:
    x: float
    y: float
    confidence: float
    width: float
    height: float
    cls: int = 0
    kind: str = "other"


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
    ) -> None:
        self._capture_method = capture_method
        self.target_class_ids = set(target_class_ids or ())
        self._device = self._resolve_device()
        self._debug_flag = False
        self._settings = _InferSettings(
            confidence=confidence,
            iou=iou,
            capture_size=capture_size,
            fov_radius_sq=0,
            min_box_ratio=min_box_ratio,
            max_box_ratio=max_box_ratio,
        )
        self._imgsz = capture_size
        self.model = YOLO(model_path, task="detect")
        if Path(model_path).suffix.lower() != ".engine":
            try:
                self.model.fuse()
            except Exception:
                pass
        self._class_roles = self._parse_class_roles()
        self.target_class_ids = self._resolve_target_classes(self.target_class_ids)

        with mss.mss() as sct:
            self.monitor = dict(sct.monitors[1])
        self.screen_left = self.monitor["left"]
        self.screen_top = self.monitor["top"]
        self.screen_w = self.monitor["width"]
        self.screen_h = self.monitor["height"]
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
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="exoai-capture", daemon=True
        )
        self._capture_thread.start()
        self._is_engine = Path(model_path).suffix.lower() == ".engine"
        self._use_half = self._device != "cpu" and not self._is_engine
        self._tune_runtime()
        self._sync_class_filter()
        self._predict_kw = {
            "conf": self._settings.confidence,
            "iou": max(0.25, self._settings.iou),
            "imgsz": self._imgsz,
            "device": self._device,
            "verbose": False,
            "max_det": 3,  # Minimal detections = fastest NMS
            "augment": False,
            "agnostic_nms": True,
            "workers": 0,
            "save": False,
            "stream": False,  # Non-streaming is faster for single images
            "embed": None,
            "visualize": False,
            "classes": sorted(self.target_class_ids) or None,
        }
        if self._use_half:
            self._predict_kw["half"] = True
        self._init_host_staging()
        self._warmup()

    @staticmethod
    def _resolve_device() -> str | int:
        try:
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return "cpu"

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
        if (
            n in {"head", "player_head", "enemy_head", "heads"}
            or n.endswith("_head")
            or n.startswith("head_")
        ):
            return "head"
        if (
            n in {"ally", "allies", "teammate", "teammates", "friendly", "team", "knocked"}
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
        enemy_ids = {cid for cid, role in roles.items() if role == "enemy"}
        ally_ids = {cid for cid, role in roles.items() if role == "ally"}
        head_ids = {cid for cid, role in roles.items() if role == "head"}
        # Keep ally boxes in the model so we can reject ally heads at scan time.
        if ally_ids and head_ids:
            return set()
        if enemy_ids:
            return enemy_ids | head_ids
        person_ids = {cid for cid, role in roles.items() if role == "enemy"}
        return person_ids or fallback

    def _sync_class_filter(self) -> None:
        self._class_ids_np = (
            np.fromiter(self.target_class_ids, dtype=np.int32)
            if self.target_class_ids
            else None
        )

    def _init_host_staging(self) -> None:
        if self._device == "cpu":
            self._host_staging = None
            return
        h, w = self._chw_buf.shape[1], self._chw_buf.shape[2]
        self._host_staging = torch.empty((3, h, w), dtype=torch.uint8, pin_memory=True)

    def _tune_runtime(self) -> None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            torch.set_num_threads(2)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
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
        """Kept for API compatibility; capture always runs while detector is open."""
        _ = active

    def _capture_loop(self) -> None:
        _lower_thread_priority()
        if self._capture_method == "mss":
            self._capture_mss_loop()
            return

        try:
            self._capture_dxgi_loop()
        except Exception as exc:
            self._capture_error = exc
            if not self._capture_stop.is_set():
                self._capture_method = "mss"
                self._capture_fell_back = True
                self._capture_error = None
                try:
                    self._capture_mss_loop()
                except Exception as fallback_exc:
                    self._capture_error = fallback_exc

    def capture_fell_back(self) -> bool:
        return self._capture_fell_back

    def _capture_dxgi_loop(self) -> None:
        camera = None
        try:
            camera = dxcam.create(output_idx=0, output_color="RGB", max_buffer_len=1)
        except Exception:
            raise RuntimeError("DXGI capture blocked - falling back to mss")
        
        if camera is None:
            raise RuntimeError("DXGI camera creation failed - falling back to mss")

        fail_count = 0
        max_fails = 60
        last_success = time.perf_counter()
        region = self._region
        bounds = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        try:
            while not self._capture_stop.is_set():
                try:
                    frame = camera.grab(region=bounds, copy=False)
                except Exception:
                    fail_count += 1
                    if fail_count >= max_fails or (time.perf_counter() - last_success) > 2.0:
                        raise RuntimeError("DXGI grab failing repeatedly - falling back to mss")
                    time.sleep(0.005)
                    continue

                if frame is None:
                    fail_count += 1
                    if fail_count >= max_fails or (time.perf_counter() - last_success) > 2.0:
                        raise RuntimeError("DXGI returning no frames - falling back to mss")
                    time.sleep(0.001)
                    continue

                fail_count = 0
                last_success = time.perf_counter()
                self._publish_frame(frame)
        finally:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass

    def _capture_mss_loop(self) -> None:
        region = self._region
        with mss.mss() as sct:
            while not self._capture_stop.is_set():
                try:
                    shot = sct.grab(region)
                    dest = self._bufs[self._write_i]
                    h, w = shot.height, shot.width
                    if dest.shape[0] != h or dest.shape[1] != w or dest.shape[2] != 3:
                        dest = np.empty((h, w, 3), dtype=np.uint8)
                        self._bufs[self._write_i] = dest
                    bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(h, w, 4)
                    cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=dest)
                    self._publish_ready(dest)
                except Exception:
                    time.sleep(0.005)

    def _publish_ready(self, frame: np.ndarray) -> None:
        self._write_i ^= 1
        self._capture_seq += 1
        with self._latest_lock:
            self._latest_frame = frame
        self._frame_ready.set()

    def _publish_frame(self, source: np.ndarray) -> None:
        dest = self._bufs[self._write_i]
        if dest.shape != source.shape:
            dest = np.empty(source.shape, dtype=np.uint8)
            self._bufs[self._write_i] = dest
            h, w = dest.shape[:2]
            self._chw_buf = np.empty((3, h, w), dtype=np.uint8)
            self._init_host_staging()
        np.copyto(dest, source)
        self._publish_ready(dest)

    def _warmup(self) -> None:
        dummy = np.zeros(
            (self._settings.capture_size, self._settings.capture_size, 3), dtype=np.uint8
        )
        with torch.inference_mode():
            self.model.predict(source=self._prepare_source(dummy), **self._predict_kw)

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
            self._imgsz = capture_size
            self._predict_kw["imgsz"] = capture_size
            self._region = self._build_region()
            self._input_tensor = None
            self._host_hwc = None
            self._gpu_hwc = None
            h, w = self._region["height"], self._region["width"]
            self._chw_buf = np.empty((3, h, w), dtype=np.uint8)
            self._init_host_staging()
        self.target_class_ids = self._resolve_target_classes(set(target_class_ids or ()))
        self._sync_class_filter()
        self._predict_kw["classes"] = sorted(self.target_class_ids) or None

    def load_model(
        self,
        model_path: str,
        target_class_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.model = YOLO(model_path, task="detect")
        self._is_engine = Path(model_path).suffix.lower() == ".engine"
        self._use_half = self._device != "cpu" and not self._is_engine
        if self._use_half:
            self._predict_kw["half"] = True
        else:
            self._predict_kw.pop("half", None)
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
        self._warmup()

    def set_fov_radius(self, fov_radius: int) -> None:
        self._settings.fov_radius_sq = fov_radius * fov_radius

    def close(self) -> None:
        self._capture_stop.set()
        self._frame_ready.set()
        self._cached_detect = None
        self._last_infer_seq = -1
        with self._latest_lock:
            self._latest_frame = None
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1)

    def _get_frame(self, wait_for_new: bool = False) -> tuple[np.ndarray | None, int | None]:
        boot_timeout = 0.5 if self._capture_seq == 0 else 0.05

        with self._latest_lock:
            frame = self._latest_frame
            capture_seq = self._capture_seq

        if frame is not None:
            if wait_for_new and capture_seq == self._last_infer_seq:
                pass
            else:
                return frame, capture_seq

        if wait_for_new:
            deadline = time.perf_counter() + boot_timeout
            while time.perf_counter() < deadline:
                if self._frame_ready.wait(timeout=0.002):
                    self._frame_ready.clear()
                    with self._latest_lock:
                        frame = self._latest_frame
                        capture_seq = self._capture_seq
                    if frame is not None and capture_seq != self._last_infer_seq:
                        return frame, capture_seq
            return None, None

        if not self._frame_ready.wait(timeout=boot_timeout):
            with self._latest_lock:
                frame = self._latest_frame
                capture_seq = self._capture_seq
            return (frame, capture_seq) if frame is not None else (None, None)

        self._frame_ready.clear()
        with self._latest_lock:
            frame = self._latest_frame
            capture_seq = self._capture_seq
        if frame is None:
            return None, None
        return frame, capture_seq

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
            if wait_for_new:
                return [], None, None
            if self._cached_detect is not None:
                return self._cached_detect
            detail = f": {self._capture_error}" if self._capture_error is not None else ""
            raise RuntimeError(f"Capture produced no frames{detail}")
        if capture_seq == self._last_infer_seq:
            if wait_for_new:
                return [], None, None
            if self._cached_detect is not None:
                return self._cached_detect
        self._last_infer_seq = capture_seq
        self.frame_id += 1
        with torch.inference_mode():
            results = self.model.predict(source=self._prepare_source(frame), **self._predict_kw)
            # Handle streaming mode - get first result
            if hasattr(results, '__iter__') and not hasattr(results, 'boxes'):
                results = list(results)
        return self._boxes_to_detections(frame, results)

    def _boxes_to_detections(self, frame: np.ndarray, results) -> tuple[list[Detection], np.ndarray | None, dict[str, int] | None]:
        detections: list[Detection] = []
        debug_frame = frame[:, :, ::-1].copy() if self._debug_flag else None
        if not results:
            return self._pack_detect(detections, debug_frame)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._pack_detect(detections, debug_frame)

        box_data = boxes.data[:, :6].detach().cpu().numpy()
        xyxy_np = box_data[:, :4]
        conf_np = box_data[:, 4]
        cls_np = box_data[:, 5].astype(np.int32, copy=False)

        region_left = self._region["left"]
        region_top = self._region["top"]
        cx0 = self.center_x
        cy0 = self.center_y
        fov_sq = self._settings.fov_radius_sq
        capture_area = self._region["width"] * self._region["height"]
        min_area = capture_area * self._settings.min_box_ratio
        max_area = capture_area * self._settings.max_box_ratio
        keep = min(
            self._region["width"] * 0.5,
            self._region["height"] * 0.5,
            (max(fov_sq, 1.0) ** 0.5) + 180.0,
        )
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
            & (bh >= bw * 0.32)
            & (dist_sq <= keep_sq)
            & ~((bw > self._region["width"] * 0.90) & (bh > self._region["height"] * 0.90))
        )
        class_ids = self._class_ids_np
        if class_ids is not None:
            mask &= np.isin(cls_np, class_ids)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return self._pack_detect(detections, debug_frame)

        rank = dist_sq[idx] - conf_np[idx] * 900.0
        keep_n = min(6, idx.size)
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
            )
            for i in order
        ]
        detections = raw if len(raw) == 1 else self._merge_overlaps(raw)
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
        return kept[:3]

    def set_debug(self, enabled: bool) -> None:
        self._debug_flag = enabled

    def debug_enabled(self) -> bool:
        return self._debug_flag
