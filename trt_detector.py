import atexit
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except Exception as exc:  # pragma: no cover - runtime dependency
    trt = None
    cuda = None
    _TRT_IMPORT_ERROR = exc
else:
    _TRT_IMPORT_ERROR = None

from onnx_detector import (
    IMAGE_EXTS,
    _decode_predictions,
    _letterbox,
    _nms,
    _normalize_output,
    _scale_boxes,
    draw_detections,
    load_class_names,
)


def parse_imgsz(value):
    if value is None:
        return None
    text = str(value).lower().replace("x", ",")
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    return int(text)


def collect_images(path):
    path = Path(path)
    if path.is_dir():
        return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


class TrtYoloDetector:
    def __init__(
        self,
        engine_path,
        class_names=None,
        conf_thres=0.25,
        iou_thres=0.45,
        input_size=None,
        box_format="auto",
        device_id=0,
        trt_logger_severity=None,
    ):
        if trt is None or cuda is None:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "TensorRT + pycuda are required for TRT engine inference."
            ) from _TRT_IMPORT_ERROR

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        cuda.init()
        if device_id < 0 or device_id >= cuda.Device.count():
            raise ValueError(f"Invalid CUDA device id: {device_id}")
        self._cuda_context = cuda.Device(device_id).make_context()
        atexit.register(self._cleanup)

        self.class_names = load_class_names(class_names)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.box_format = box_format

        severity = (
            trt_logger_severity
            if trt_logger_severity is not None
            else trt.Logger.WARNING
        )
        self._logger = trt.Logger(severity)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._load_engine()
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        if self._engine.has_implicit_batch_dimension:
            raise RuntimeError(
                "This script expects an explicit-batch TensorRT engine. "
                "Rebuild the engine with explicit batch (default for ONNX)."
            )

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._stream = cuda.Stream()
        self._input_index = self._find_input_index()
        self.input_dtype = trt.nptype(self._engine.get_binding_dtype(self._input_index))

        self.input_size = self._resolve_input_size(input_size)
        self._input_shape = (1, 3, self.input_size[0], self.input_size[1])
        self._context.set_binding_shape(self._input_index, self._input_shape)
        if not self._context.all_binding_shapes_specified:
            raise RuntimeError(
                f"Failed to set input binding shape to {self._input_shape}."
            )

        self._scale = np.array(1.0 / 255.0, dtype=self.input_dtype)
        self._allocate_buffers()
        self.last_inference_time = None

    def _cleanup(self):
        if self._cuda_context is None:
            return
        try:
            self._cuda_context.pop()
        except Exception:
            pass
        try:
            self._cuda_context.detach()
        except Exception:
            pass
        self._cuda_context = None

    def close(self):
        self._cleanup()

    def _load_engine(self):
        with self.engine_path.open("rb") as f:
            return self._runtime.deserialize_cuda_engine(f.read())

    def _find_input_index(self):
        for i in range(self._engine.num_bindings):
            if self._engine.binding_is_input(i):
                return i
        raise RuntimeError("TensorRT engine has no input bindings.")

    def _resolve_input_size(self, input_size):
        if input_size is not None:
            if isinstance(input_size, int):
                return (input_size, input_size)
            return tuple(input_size)
        shape = self._engine.get_binding_shape(self._input_index)
        if len(shape) == 4 and shape[2] > 0 and shape[3] > 0:
            return (int(shape[2]), int(shape[3]))
        return (640, 640)

    def _allocate_buffers(self):
        self._bindings = [0] * self._engine.num_bindings
        self._host_inputs = []
        self._device_inputs = []
        self._host_outputs = []
        self._device_outputs = []
        self._output_shapes = []

        for idx in range(self._engine.num_bindings):
            dtype = trt.nptype(self._engine.get_binding_dtype(idx))
            shape = self._context.get_binding_shape(idx)
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Binding shape unresolved for index {idx}: {shape}"
                )
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings[idx] = int(device_mem)
            if self._engine.binding_is_input(idx):
                self._host_inputs.append(host_mem)
                self._device_inputs.append(device_mem)
            else:
                self._host_outputs.append(host_mem)
                self._device_outputs.append(device_mem)
                self._output_shapes.append(tuple(int(v) for v in shape))

    def preprocess(self, image_bgr):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, ratio, pad = _letterbox(image, self.input_size)
        blob = resized.astype(self.input_dtype, copy=False)
        blob *= self._scale
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        return blob, ratio, pad

    def predict(self, image_bgr, conf_thres=None, iou_thres=None, measure_time=False):
        conf_thres = self.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.iou_thres if iou_thres is None else iou_thres

        blob, ratio, pad = self.preprocess(image_bgr)
        flat = blob.ravel()
        if flat.size != self._host_inputs[0].size:
            raise RuntimeError(
                f"Input size mismatch: expected {self._host_inputs[0].size}, "
                f"got {flat.size}"
            )
        np.copyto(self._host_inputs[0], flat)
        cuda.memcpy_htod_async(self._device_inputs[0], self._host_inputs[0], self._stream)

        if measure_time:
            start = time.perf_counter()
            self._context.execute_async_v2(
                bindings=self._bindings, stream_handle=self._stream.handle
            )
            for host, device in zip(self._host_outputs, self._device_outputs):
                cuda.memcpy_dtoh_async(host, device, self._stream)
            self._stream.synchronize()
            self.last_inference_time = time.perf_counter() - start
        else:
            self._context.execute_async_v2(
                bindings=self._bindings, stream_handle=self._stream.handle
            )
            for host, device in zip(self._host_outputs, self._device_outputs):
                cuda.memcpy_dtoh_async(host, device, self._stream)
            self._stream.synchronize()
            self.last_inference_time = None

        outputs = [
            np.asarray(host).reshape(shape)
            for host, shape in zip(self._host_outputs, self._output_shapes)
        ]
        preds = _normalize_output(outputs).astype(np.float32, copy=False)

        num_classes = len(self.class_names) if self.class_names else None
        boxes, scores, class_ids = _decode_predictions(
            preds,
            conf_thres=conf_thres,
            num_classes=num_classes,
            box_format=self.box_format,
        )
        if len(boxes) == 0:
            return []

        boxes = _scale_boxes(boxes, ratio, pad, image_bgr.shape)

        detections = []
        if len(boxes):
            if class_ids.size == 0:
                class_ids = np.zeros(len(boxes), dtype=int)

            for class_id in np.unique(class_ids):
                idxs = np.where(class_ids == class_id)[0]
                keep = _nms(boxes[idxs], scores[idxs], iou_thres)
                for k in keep:
                    i = idxs[k]
                    detections.append(
                        {
                            "box": boxes[i].tolist(),
                            "conf": float(scores[i]),
                            "class_id": int(class_ids[i]),
                        }
                    )

        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections

    def visualize(self, image_bgr, detections, class_names=None):
        return draw_detections(image_bgr, detections, class_names or self.class_names)
