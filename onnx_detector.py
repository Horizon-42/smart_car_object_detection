import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - runtime dependency
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None

PALETTE = [
    (255, 82, 82),
    (0, 189, 214),
    (65, 179, 123),
    (255, 196, 0),
    (146, 104, 255),
    (255, 112, 67),
    (38, 198, 218),
    (156, 204, 101),
    (171, 71, 188),
    (66, 165, 245),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names(source=None):
    if source is None:
        return []
    if isinstance(source, (list, tuple)):
        return list(source)
    path = Path(source)
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _color_for_class(class_id):
    return PALETTE[class_id % len(PALETTE)]


def _text_color_for_bg(bgr):
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _draw_text_with_bg(image, text, origin, bg_color, font_scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - text_w - 1))
    y = max(text_h + baseline + 1, min(y, image.shape[0] - 1))
    top_left = (x, y - text_h - baseline - 1)
    bottom_right = (x + text_w + 2, y + baseline + 1)
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    text_color = _text_color_for_bg(bg_color)
    cv2.putText(image, text, (x + 1, y - 1), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _letterbox(image, new_shape, color=(114, 114, 114)):
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def _xywh_to_xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


def _clip_boxes(boxes, shape):
    h, w = shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def _scale_boxes(boxes, ratio, pad, shape):
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    return _clip_boxes(boxes, shape)


def _nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]
    return keep


def _normalize_output(outputs):
    if isinstance(outputs, (list, tuple)):
        output = outputs[0]
    else:
        output = outputs

    output = np.asarray(output)
    if output.ndim == 3:
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))
        output = output[0]
    elif output.ndim == 2:
        output = output
    else:
        output = output.reshape(-1, output.shape[-1])
    return output


def _decode_predictions(preds, conf_thres, num_classes=None, box_format="auto"):
    if preds.size == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    dim = preds.shape[1]

    if dim in (6, 7):
        coords = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
        if box_format == "xywh":
            coords = _xywh_to_xyxy(coords)
    else:
        coords = preds[:, :4]
        scores_raw = preds[:, 4:]
        if num_classes is not None:
            if dim == 4 + num_classes:
                scores = scores_raw[:, :num_classes]
            elif dim >= 5 + num_classes:
                obj = scores_raw[:, 0:1]
                scores = obj * scores_raw[:, 1 : 1 + num_classes]
            else:
                scores = scores_raw
        else:
            if dim >= 5 and scores_raw.shape[1] >= 2:
                obj = scores_raw[:, 0:1]
                scores = obj * scores_raw[:, 1:]
            else:
                scores = scores_raw

        class_ids = scores.argmax(axis=1)
        scores = scores.max(axis=1)

        if box_format in ("auto", "xywh"):
            coords = _xywh_to_xyxy(coords)

    mask = scores >= conf_thres
    return coords[mask], scores[mask], class_ids[mask]


class OnnxYoloDetector:
    def __init__(
        self,
        model_path,
        class_names=None,
        conf_thres=0.25,
        iou_thres=0.45,
        input_size=None,
        providers=None,
        box_format="auto",
    ):
        if ort is None:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "onnxruntime is required (pip install onnxruntime)"
            ) from _ORT_IMPORT_ERROR

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.class_names = load_class_names(class_names)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.box_format = box_format

        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self._resolve_input_size(input_size)
        self.last_inference_time = None

    def _resolve_input_size(self, input_size):
        if input_size is not None:
            return input_size
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            return (shape[2], shape[3])
        return 640

    def preprocess(self, image_bgr):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, ratio, pad = _letterbox(image, self.input_size)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, ratio, pad

    def predict(self, image_bgr, conf_thres=None, iou_thres=None, measure_time=False):
        conf_thres = self.conf_thres if conf_thres is None else conf_thres
        iou_thres = self.iou_thres if iou_thres is None else iou_thres

        blob, ratio, pad = self.preprocess(image_bgr)
        if measure_time:
            start = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: blob})
            end = time.perf_counter()
            self.last_inference_time = end - start
        else:
            outputs = self.session.run(None, {self.input_name: blob})
            self.last_inference_time = None
        preds = _normalize_output(outputs)

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


def draw_detections(image_bgr, detections, class_names=None):
    output = image_bgr.copy()
    h, w = output.shape[:2]
    thickness = max(1, int(round(min(h, w) / 400)))
    font_scale = max(0.5, min(h, w) / 1200)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        conf = det["conf"]
        class_id = det["class_id"]

        color = _color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        name = (
            class_names[class_id]
            if class_names and 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        label_text = f"{name} {conf:.2f}"
        _draw_text_with_bg(output, label_text, (x1, y1 - 4), color, font_scale, thickness)

    return output


def _collect_images(path):
    path = Path(path)
    if path.is_dir():
        return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


def _build_trt_providers(
    enable_trt,
    trt_fp16=False,
    trt_cache_path=None,
    trt_workspace_mb=None,
    cuda_device_id=0,
    extra_providers=None,
):
    if not enable_trt:
        return extra_providers

    trt_options = {}
    if trt_fp16:
        trt_options["trt_fp16_enable"] = True
    if trt_cache_path:
        cache_dir = Path(trt_cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options["trt_engine_cache_enable"] = True
        trt_options["trt_engine_cache_path"] = str(cache_dir)
    if trt_workspace_mb is not None:
        trt_options["trt_max_workspace_size"] = int(trt_workspace_mb) * 1024 * 1024

    providers = [
        ("TensorrtExecutionProvider", trt_options),
        ("CUDAExecutionProvider", {"device_id": int(cuda_device_id)}),
    ]

    if extra_providers:
        existing = {p[0] if isinstance(p, (list, tuple)) else p for p in providers}
        for provider in extra_providers:
            name = provider[0] if isinstance(provider, (list, tuple)) else provider
            if name not in existing:
                providers.append(provider)
                existing.add(name)

    return providers


def run_inference_on_path(
    model_path,
    image_path,
    classes_path=None,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    providers=None,
    save_dir=None,
    show=False,
    measure_time=False,
):
    detector = OnnxYoloDetector(
        model_path,
        class_names=classes_path,
        conf_thres=conf,
        iou_thres=iou,
        input_size=imgsz,
        providers=providers,
    )

    images = _collect_images(image_path)
    if not images:
        raise FileNotFoundError(f"No images found at {image_path}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        detections = detector.predict(image, measure_time=measure_time)
        if measure_time and detector.last_inference_time is not None:
            inf_ms = detector.last_inference_time * 1000.0
            fps = (
                1.0 / detector.last_inference_time
                if detector.last_inference_time > 0
                else float("inf")
            )
            print(f"{img_path.name}: inference {inf_ms:.2f} ms | fps {fps:.2f}")
        vis = detector.visualize(image, detections)

        if save_dir is not None:
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), vis)
        if show:
            cv2.imshow("onnx detections", vis)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break

    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ONNX YOLO inference on images and visualize results."
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model.")
    parser.add_argument("--image", help="Single image path.")
    parser.add_argument("--folder", help="Folder of images.")
    parser.add_argument(
        "--classes",
        default=None,
        help="Path to classes.txt (optional).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--imgsz",
        default=None,
        help="Input size override (e.g. 640 or 640,480).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="ONNX Runtime provider (can repeat).",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/onnx_detect",
        help="Directory to save visualized results.",
    )
    parser.add_argument("--show", action="store_true", help="Show images in a window.")
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Print inference time and FPS for each image.",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Enable TensorRT EP (with CUDA fallback).",
    )
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Enable TensorRT FP16 mode.",
    )
    parser.add_argument(
        "--trt-cache",
        default=None,
        help="TensorRT engine cache directory.",
    )
    parser.add_argument(
        "--trt-workspace-mb",
        type=int,
        default=None,
        help="TensorRT max workspace size in MB.",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        default=0,
        help="CUDA device id for TensorRT/CUDA EP.",
    )

    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Provide --image or --folder", file=sys.stderr)
        sys.exit(1)

    imgsz = None
    if args.imgsz is not None:
        text = str(args.imgsz).lower().replace("x", ",")
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) == 2:
                imgsz = (int(parts[0]), int(parts[1]))
            else:
                imgsz = int(parts[0])
        else:
            imgsz = int(text)

    target = args.image or args.folder
    save_dir = None if args.no_save else args.save_dir

    providers = _build_trt_providers(
        enable_trt=args.tensorrt,
        trt_fp16=args.trt_fp16,
        trt_cache_path=args.trt_cache,
        trt_workspace_mb=args.trt_workspace_mb,
        cuda_device_id=args.cuda_device_id,
        extra_providers=args.provider,
    )

    run_inference_on_path(
        args.model,
        target,
        classes_path=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=imgsz,
        providers=providers,
        save_dir=save_dir,
        show=args.show,
        measure_time=args.measure_time,
    )
