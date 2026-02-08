import argparse
import sys
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency
    torch = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(path, max_images=None):
    target = Path(path)
    if target.is_dir():
        images = [p for p in sorted(target.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [target] if target.suffix.lower() in IMAGE_EXTS else []
    if max_images is not None and max_images > 0:
        images = images[:max_images]
    return images


def _sync_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_yolo_model(model_arg):
    model_path = Path(model_arg)
    if model_path.exists():
        if model_path.suffix.lower() != ".pt":
            raise ValueError("Only .pt YOLO models are supported.")
        return YOLO(str(model_path.resolve()))

    lowered = str(model_arg).lower()
    if lowered.endswith((".onnx", ".engine", ".tflite", ".mlmodel")):
        raise ValueError("Only .pt YOLO models are supported.")
    return YOLO(model_arg)


def _draw_metrics(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    min_dim = min(image.shape[:2])
    scale = min(1.0, max(0.35, min_dim / 900))
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = max(3, int(round(scale * 8)))
    x, y = 10, 10 + text_h
    cv2.rectangle(
        image,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO .pt inference on images and save visualizations with timing."
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path to YOLO .pt model (or model name if available).",
    )
    parser.add_argument(
        "--source",
        default="test_images",
        help="Folder of images or a single image path.",
    )
    parser.add_argument(
        "--output",
        default="runs/yolo_infer_vis",
        help="Directory to save visualized results.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu.")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process.")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations.")
    parser.add_argument("--half", action="store_true", help="Use FP16 if supported.")
    parser.add_argument("--show", action="store_true", help="Show images in a window.")
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    args = parser.parse_args()

    if YOLO is None:  # pragma: no cover - runtime dependency
        print("Ultralytics is required: pip install ultralytics", file=sys.stderr)
        raise _YOLO_IMPORT_ERROR

    images = _collect_images(args.source, max_images=args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found at {args.source}")

    try:
        model = _load_yolo_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    save_outputs = not args.no_save
    output_dir = Path(args.output)
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    predict_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if args.half:
        predict_kwargs["half"] = True

    warmup = max(args.warmup, 0)
    if warmup > 0:
        sample = str(images[0])
        for _ in range(warmup):
            model.predict(source=sample, **predict_kwargs)
            _sync_cuda()

    total_time = 0.0
    processed = 0

    for img_path in images:
        start = time.perf_counter()
        results = model.predict(source=str(img_path), **predict_kwargs)
        _sync_cuda()
        elapsed = time.perf_counter() - start

        if not results:
            continue
        result = results[0]
        orig = getattr(result, "orig_img", None)
        if orig is None:
            orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        min_dim = min(orig.shape[:2])
        scale = 1.0
        if min_dim < 480:
            scale = min(3.0, max(1.5, 480 / min_dim))

        plot_line_width = args.line_width
        if scale > 1.0:
            plot_line_width = max(1, int(round(args.line_width / scale)))

        vis = result.plot(line_width=plot_line_width)
        if scale > 1.0:
            vis = cv2.resize(
                vis,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        inf_ms = elapsed * 1000.0
        fps = 1.0 / elapsed if elapsed > 0 else float("inf")
        _draw_metrics(vis, f"{inf_ms:.2f} ms | {fps:.2f} FPS")

        if save_outputs:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), vis)
        if args.show:
            cv2.imshow("yolo detections", vis)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break

        print(f"{img_path.name}: inference {inf_ms:.2f} ms | fps {fps:.2f}")
        total_time += elapsed
        processed += 1

    if args.show:
        cv2.destroyAllWindows()

    if processed:
        avg_ms = (total_time / processed) * 1000.0
        avg_fps = processed / total_time if total_time > 0 else float("inf")
        print(f"Processed {processed} images")
        print(f"Average latency: {avg_ms:.2f} ms/image")
        print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
