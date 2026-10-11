"""Ultra-fast YOLO inference — all GPU, minimal copies.

Target: <1ms total inference by:
1. Resize on GPU not CPU
2. Keep tensors on GPU, only copy final boxes
3. Use CUDA streams for async
4. Smaller input when possible
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

MAX_DET = 8

_KINDS = {0: "ally", 1: "enemy", 2: "knocked", 3: "other"}


@dataclass
class Det:
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int
    name: str = ""
    kind: str = "enemy"


class Infer:
    def __init__(
        self,
        model_path: Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 320,
        warmup: int = 8,
    ) -> None:
        self.path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.backend = "cpu"
        self.fallback_reason = ""
        self._end2end = False

        self._cuda = torch is not None and torch.cuda.is_available()
        self._yolo = None
        self._graph = None
        self._net = None

        if self.path.suffix.lower() == ".engine":
            sized_engine = self.path.with_name(f"best_{self.imgsz}.engine")
            if "combined" not in self.path.stem and sized_engine.exists():
                self.path = sized_engine
            try:
                self._init_engine(warmup)
                return
            except Exception as exc:
                self.fallback_reason = f"TensorRT failed: {type(exc).__name__}: {exc}"
                onnx_fallback = self.path.with_name("best.onnx")
                pt_fallback = self.path.with_name("best.pt")
                fallback = (
                    onnx_fallback
                    if not self._cuda and onnx_fallback.exists()
                    else pt_fallback
                )
                if not fallback.exists():
                    raise RuntimeError(self.fallback_reason) from exc
                self.path = fallback
                print(f"[Infer] {self.fallback_reason}; using {fallback.name}")

        if self.path.suffix.lower() == ".onnx" or (
            not self._cuda and self.path.with_name("best.onnx").exists()
        ):
            if self.path.suffix.lower() != ".onnx":
                self.path = self.path.with_name("best.onnx")
                self.fallback_reason = "CUDA unavailable; using ONNX fallback"
            self._init_onnx()
            return

        from ultralytics import YOLO

        yolo = YOLO(str(self.path))
        self.names = yolo.names or {}
        self._yolo = yolo
        if not self._cuda:
            yolo.predict(
                np.zeros((imgsz, imgsz, 3), np.uint8), verbose=False, imgsz=imgsz
            )
            print(f"[Infer] {self.path.name} on CPU (no CUDA)")
            return

        torch.backends.cudnn.benchmark = True

        # Fuse conv+bn for speed
        net = yolo.model.eval().cuda().half()
        net = net.fuse().eval()
        self._net = net

        # Pre-allocate GPU input tensor
        self._static_in = torch.zeros(
            (1, 3, imgsz, imgsz), dtype=torch.half, device="cuda"
        )
        # Resize into page-locked host memory. This reduces the upload from the
        # full capture region to imgsz and makes the host-to-device copy truly
        # asynchronous.
        self._host_u8 = torch.empty(
            (imgsz, imgsz, 3), dtype=torch.uint8, pin_memory=True
        )
        self._host_np = self._host_u8.numpy()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                net(self._static_in)
        torch.cuda.synchronize()

        # Capture graph
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                self._static_out = net(self._static_in)
            self._graph = graph
            self.backend = "cuda-graph"
        except Exception as exc:
            self.backend = "cuda-eager"
            print(f"[Infer] graph capture failed ({exc})")

        print(f"[Infer] {self.path.name} via {self.backend} @ {imgsz}px")

    def _init_onnx(self) -> None:
        try:
            import onnx

            model = onnx.load(str(self.path), load_external_data=False)
            dims = model.graph.input[0].type.tensor_type.shape.dim
            height = int(dims[-2].dim_value)
            width = int(dims[-1].dim_value)
            if height > 0 and height == width:
                self.imgsz = height
        except Exception:
            pass
        self.names = {0: "ally", 1: "enemy", 2: "knocked", 3: "other"}
        self._onnx_net = cv2.dnn.readNetFromONNX(str(self.path))
        self.backend = "opencv-onnx"
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._predict_onnx(dummy, 0, 0)
        print(f"[Infer] {self.path.name} via OpenCV ONNX @ {self.imgsz}px")

    def _init_engine(self, warmup: int) -> None:
        import tensorrt as trt

        cp = None
        if not self._cuda:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                import cupy as cp

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if engine is None:
            raise RuntimeError("engine deserialization failed")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("execution context creation failed")

        self.names = {0: "ally", 1: "enemy", 2: "knocked", 3: "other"}
        self._trt_logger = logger
        self._trt_runtime = runtime
        self._trt_engine = engine
        self._trt_context = context
        input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise RuntimeError(f"expected one engine input, found {len(input_names)}")
        input_trt_dtype = engine.get_tensor_dtype(input_names[0])
        self._trt_use_torch = self._cuda
        if self._trt_use_torch:
            dtype_map = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT64: torch.int64,
                trt.DataType.INT8: torch.int8,
                trt.DataType.BOOL: torch.bool,
            }
            input_dtype = dtype_map.get(input_trt_dtype)
            if input_dtype not in (torch.float16, torch.float32):
                raise RuntimeError(f"unsupported engine input dtype: {input_trt_dtype}")
            self._prepare_buffers(dtype=input_dtype)
            engine_input = self._static_in
        else:
            self._cp = cp
            dtype_map = {
                trt.DataType.FLOAT: cp.float32,
                trt.DataType.HALF: cp.float16,
                trt.DataType.INT32: cp.int32,
                trt.DataType.INT64: cp.int64,
                trt.DataType.INT8: cp.int8,
                trt.DataType.BOOL: cp.bool_,
            }
            input_dtype = dtype_map.get(input_trt_dtype)
            if input_dtype not in (cp.float16, cp.float32):
                raise RuntimeError(f"unsupported engine input dtype: {input_trt_dtype}")
            self._engine_np_dtype = (
                np.float16 if input_dtype == cp.float16 else np.float32
            )
            pinned = cp.cuda.alloc_pinned_memory(self.imgsz * self.imgsz * 3)
            self._host_pinned = pinned
            self._host_np = np.frombuffer(
                pinned, dtype=np.uint8, count=self.imgsz * self.imgsz * 3
            ).reshape(self.imgsz, self.imgsz, 3)
            chw_bytes = (
                self.imgsz
                * self.imgsz
                * 3
                * np.dtype(self._engine_np_dtype).itemsize
            )
            pinned_chw = cp.cuda.alloc_pinned_memory(chw_bytes)
            self._host_chw_pinned = pinned_chw
            self._host_chw = np.frombuffer(
                pinned_chw,
                dtype=self._engine_np_dtype,
                count=self.imgsz * self.imgsz * 3,
            ).reshape(1, 3, self.imgsz, self.imgsz)
            self._cp_input = cp.zeros(
                (1, 3, self.imgsz, self.imgsz), dtype=input_dtype
            )
            engine_input = self._cp_input
        outputs: list[object] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                expected = tuple(engine_input.shape)
                if shape != expected:
                    raise RuntimeError(
                        f"engine expects {shape}, configured input is {expected}"
                    )
                pointer = (
                    engine_input.data_ptr()
                    if self._trt_use_torch
                    else engine_input.data.ptr
                )
                context.set_tensor_address(name, pointer)
            else:
                dtype = dtype_map.get(engine.get_tensor_dtype(name))
                if dtype is None:
                    raise RuntimeError(
                        f"unsupported TensorRT dtype: {engine.get_tensor_dtype(name)}"
                    )
                tensor = (
                    torch.empty(shape, dtype=dtype, device="cuda")
                    if self._trt_use_torch
                    else cp.empty(shape, dtype=dtype)
                )
                pointer = (
                    tensor.data_ptr()
                    if self._trt_use_torch
                    else tensor.data.ptr
                )
                context.set_tensor_address(name, pointer)
                outputs.append(tensor)
        if not outputs:
            raise RuntimeError("engine exposes no output tensors")
        self._trt_outputs = outputs
        for _ in range(max(1, warmup)):
            self._execute_engine()
        if self._trt_use_torch:
            torch.cuda.synchronize()
        else:
            cp.cuda.get_current_stream().synchronize()
        self.backend = "tensorrt"
        print(f"[Infer] {self.path.name} via TensorRT @ {self.imgsz}px")

    def _execute_engine(self):
        if self._trt_use_torch:
            stream_ptr = torch.cuda.current_stream().cuda_stream
        else:
            stream_ptr = self._cp.cuda.get_current_stream().ptr
        if not self._trt_context.execute_async_v3(stream_ptr):
            raise RuntimeError("TensorRT execution failed")
        return self._trt_outputs[0] if len(self._trt_outputs) == 1 else self._trt_outputs

    def _prepare_buffers(self, dtype=None) -> None:
        dtype = dtype or torch.float16
        self._static_in = torch.zeros(
            (1, 3, self.imgsz, self.imgsz), dtype=dtype, device="cuda"
        )
        self._host_u8 = torch.empty(
            (self.imgsz, self.imgsz, 3), dtype=torch.uint8, pin_memory=True
        )
        self._host_np = self._host_u8.numpy()

    def update(self, conf: float | None = None, iou: float | None = None) -> None:
        if conf is not None:
            self.conf = float(conf)
        if iou is not None:
            self.iou = float(iou)

    def predict(self, bgr: np.ndarray, ox: int, oy: int) -> list[Det]:
        if self.backend == "tensorrt":
            return self._predict_engine(bgr, ox, oy)
        if self.backend == "opencv-onnx":
            return self._predict_onnx(bgr, ox, oy)
        if not self._cuda:
            return self._predict_cpu(bgr, ox, oy)

        h, w = bgr.shape[:2]
        scale_x, scale_y = w / self.imgsz, h / self.imgsz

        with torch.inference_mode():
            if h != self.imgsz or w != self.imgsz:
                cv2.resize(
                    bgr,
                    (self.imgsz, self.imgsz),
                    dst=self._host_np,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                np.copyto(self._host_np, bgr)

            cv2.cvtColor(self._host_np, cv2.COLOR_BGR2RGB, dst=self._host_np)
            source = self._host_u8.permute(2, 0, 1).unsqueeze(0)
            self._static_in.copy_(source, non_blocking=True)
            self._static_in.mul_(1.0 / 255.0)

            # Run inference
            if self._graph is not None:
                self._graph.replay()
                out = self._static_out
            elif self.backend == "tensorrt":
                out = self._execute_engine()
            else:
                out = self._net(self._static_in)

            raw = out[0] if isinstance(out, (list, tuple)) else out
            if raw.ndim == 3 and raw.shape[-1] == 6:
                self._end2end = True
                det = raw[0]
                det = det[det[:, 4] > self.conf][:MAX_DET]
            else:
                det = self._fast_nms(raw)
            if not det.numel():
                return []
            
            # Only copy the boxes we need
            n = min(det.shape[0], MAX_DET)
            boxes = det[:n, :6].float().cpu().numpy()

        # Build detections
        out_list: list[Det] = []
        for x1, y1, x2, y2, conf, cls in boxes:
            x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            cls = int(cls)
            out_list.append(
                Det(
                    x=(x1 + x2) * 0.5 + ox,
                    y=(y1 + y2) * 0.5 + oy,
                    w=x2 - x1,
                    h=y2 - y1,
                    conf=float(conf),
                    cls=cls,
                    name=self.names.get(cls, ""),
                    kind=_KINDS.get(cls, "enemy"),
                )
            )
        return out_list

    def _predict_onnx(self, bgr: np.ndarray, ox: int, oy: int) -> list[Det]:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr,
            scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            swapRB=True,
            crop=False,
        )
        self._onnx_net.setInput(blob)
        raw = self._onnx_net.forward()
        rows = self._fast_nms_numpy(raw)
        return self._to_dets(rows, ox, oy, (w / self.imgsz, h / self.imgsz))

    def _predict_engine(self, bgr: np.ndarray, ox: int, oy: int) -> list[Det]:
        h, w = bgr.shape[:2]
        if h != self.imgsz or w != self.imgsz:
            cv2.resize(
                bgr,
                (self.imgsz, self.imgsz),
                dst=self._host_np,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            np.copyto(self._host_np, bgr)
        cv2.cvtColor(self._host_np, cv2.COLOR_BGR2RGB, dst=self._host_np)

        if self._trt_use_torch:
            source = self._host_u8.permute(2, 0, 1).unsqueeze(0)
            self._static_in.copy_(source, non_blocking=True)
            self._static_in.mul_(1.0 / 255.0)
            raw = self._execute_engine()
            raw_np = raw.float().cpu().numpy()
        else:
            cp = self._cp
            np.multiply(
                self._host_np.transpose(2, 0, 1)[None],
                self._engine_np_dtype(1.0 / 255.0),
                out=self._host_chw,
                casting="unsafe",
            )
            self._cp_input.set(
                self._host_chw, stream=cp.cuda.get_current_stream()
            )
            raw = self._execute_engine()
            raw_np = cp.asnumpy(raw)
        rows = self._fast_nms_numpy(raw_np)
        return self._to_dets(rows, ox, oy, (w / self.imgsz, h / self.imgsz))

    def _fast_nms_numpy(self, raw: np.ndarray) -> np.ndarray:
        pred = raw[0].T
        classes = pred[:, 4:].argmax(axis=1)
        scores = pred[np.arange(len(pred)), classes + 4]
        keep = scores > self.conf
        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32)
        xywh = pred[keep, :4].astype(np.float32, copy=False)
        scores = scores[keep].astype(np.float32, copy=False)
        classes = classes[keep]
        if len(scores) > 256:
            top = np.argpartition(scores, -256)[-256:]
            xywh, scores, classes = xywh[top], scores[top], classes[top]
        half = xywh[:, 2:] * 0.5
        boxes = np.concatenate((xywh[:, :2] - half, xywh[:, :2] + half), axis=1)
        nms_boxes = boxes + classes[:, None].astype(np.float32) * 4096.0
        order = scores.argsort()[::-1]
        selected: list[int] = []
        while order.size and len(selected) < MAX_DET:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            other = nms_boxes[rest]
            box = nms_boxes[current]
            x1 = np.maximum(box[0], other[:, 0])
            y1 = np.maximum(box[1], other[:, 1])
            x2 = np.minimum(box[2], other[:, 2])
            y2 = np.minimum(box[3], other[:, 3])
            inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            area_b = np.maximum(0.0, other[:, 2] - other[:, 0]) * np.maximum(
                0.0, other[:, 3] - other[:, 1]
            )
            overlap = inter / (area_a + area_b - inter + 1e-7)
            order = rest[overlap <= self.iou]
        idx = np.asarray(selected, dtype=np.int64)
        return np.concatenate(
            (
                boxes[idx],
                scores[idx, None],
                classes[idx, None].astype(np.float32),
            ),
            axis=1,
        )

    def _fast_nms(self, raw: "torch.Tensor") -> "torch.Tensor":
        """NMS specialized for this model's [1, 4 + classes, anchors] output."""
        pred = raw[0].transpose(0, 1)
        scores, classes = pred[:, 4:].max(dim=1)
        keep = scores > self.conf
        if not keep.any():
            return pred.new_empty((0, 6))

        boxes_xywh = pred[keep, :4]
        scores = scores[keep]
        classes = classes[keep]

        # Bound pathological scenes before NMS. Only eight final detections are
        # consumed, so hundreds of low-ranked candidates add latency, not value.
        if scores.numel() > 256:
            scores, top = scores.topk(256, sorted=False)
            boxes_xywh = boxes_xywh[top]
            classes = classes[top]

        half = boxes_xywh[:, 2:] * 0.5
        boxes = torch.cat(
            (boxes_xywh[:, :2] - half, boxes_xywh[:, :2] + half), dim=1
        )
        nms_boxes = boxes + classes[:, None].to(boxes.dtype) * 4096.0
        selected = self._nms(nms_boxes, scores, self.iou, MAX_DET)
        return torch.cat(
            (
                boxes[selected],
                scores[selected, None],
                classes[selected, None].to(boxes.dtype),
            ),
            dim=1,
        )

    @staticmethod
    def _nms(
        boxes: "torch.Tensor",
        scores: "torch.Tensor",
        threshold: float,
        max_det: int,
    ) -> "torch.Tensor":
        """Small GPU NMS without a binary torchvision dependency."""
        order = scores.argsort(descending=True)
        selected: list["torch.Tensor"] = []
        while order.numel() and len(selected) < max_det:
            current = order[0]
            selected.append(current)
            if order.numel() == 1:
                break
            rest = order[1:]
            box = boxes[current]
            others = boxes[rest]
            x1 = torch.maximum(box[0], others[:, 0])
            y1 = torch.maximum(box[1], others[:, 1])
            x2 = torch.minimum(box[2], others[:, 2])
            y2 = torch.minimum(box[3], others[:, 3])
            inter = (x2 - x1).clamp_(min=0) * (y2 - y1).clamp_(min=0)
            area_a = (box[2] - box[0]).clamp_(min=0) * (
                box[3] - box[1]
            ).clamp_(min=0)
            area_b = (others[:, 2] - others[:, 0]).clamp_(min=0) * (
                others[:, 3] - others[:, 1]
            ).clamp_(min=0)
            iou = inter / (area_a + area_b - inter + 1e-7)
            order = rest[iou <= threshold]
        if not selected:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        return torch.stack(selected)

    def _predict_cpu(self, bgr: np.ndarray, ox: int, oy: int) -> list[Det]:
        res = self._yolo.predict(
            bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False
        )
        if not res or res[0].boxes is None or not len(res[0].boxes):
            return []
        b = res[0].boxes
        rows = np.concatenate(
            [
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy()[:, None],
                b.cls.cpu().numpy()[:, None],
            ],
            axis=1,
        )
        return self._to_dets(rows, ox, oy, (1.0, 1.0))

    def _to_dets(
        self, rows: np.ndarray, ox: int, oy: int, scale: tuple[float, float]
    ) -> list[Det]:
        sx, sy = scale
        out: list[Det] = []
        for x1, y1, x2, y2, conf, cls in rows[:MAX_DET]:
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            cls = int(cls)
            out.append(
                Det(
                    x=(x1 + x2) * 0.5 + ox,
                    y=(y1 + y2) * 0.5 + oy,
                    w=x2 - x1,
                    h=y2 - y1,
                    conf=float(conf),
                    cls=cls,
                    name=self.names.get(cls, ""),
                    kind=_KINDS.get(cls, "enemy"),
                )
            )
        return out
