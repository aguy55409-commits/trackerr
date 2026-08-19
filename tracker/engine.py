from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Det:
    x: float  # screen coords
    y: float
    w: float
    h: float
    conf: float
    cls: int
    name: str = ""


class TensorRTDetector:
    """YOLO detect via Ultralytics — prefers .engine (TensorRT)."""

    def __init__(self, model_path: Path, conf: float = 0.45, iou: float = 0.45, imgsz: int = 320) -> None:
        self.path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = YOLO(str(self.path), task="detect")
        self._names = self.model.names if isinstance(self.model.names, dict) else {}
        self._is_engine = self.path.suffix.lower() == ".engine"
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._kw = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self._device,
            max_det=10,
        )
        if not self._is_engine and self._device.startswith("cuda"):
            self._kw["half"] = True

    def predict(self, frame_rgb: np.ndarray, origin_left: int, origin_top: int) -> list[Det]:
        with torch.inference_mode():
            results = self.model.predict(source=frame_rgb, **self._kw)
        if not results:
            return []
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[Det] = []
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (cx, cy, w, h), conf, cls in zip(xywh, confs, clss):
            name = str(self._names.get(int(cls), cls))
            out.append(
                Det(
                    x=float(cx) + origin_left,
                    y=float(cy) + origin_top,
                    w=float(w),
                    h=float(h),
                    conf=float(conf),
                    cls=int(cls),
                    name=name,
                )
            )
        return out
