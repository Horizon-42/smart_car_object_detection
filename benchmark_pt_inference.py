import argparse
from pathlib import Path
import sys
import time

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


def _load_images(paths):
    images = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        images.append(img)
    return images


def _sync_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_benchmark(model, source, iters, warmup, predict_kwargs):
    for _ in range(warmup):
        model.predict(source=source, **predict_kwargs)
        _sync_cuda()

    _sync_cuda()
    start = time.perf_counter()
    for _ in range(iters):
        model.predict(source=source, **predict_kwargs)
        _sync_cuda()
    end = time.perf_counter()

    total_time = end - start
    total_images = len(source) * iters if isinstance(source, list) else 0
    avg_ms = (total_time / total_images) * 1000 if total_images else 0.0
    fps = (total_images / total_time) if total_time > 0 and total_images else 0.0
    return total_time, total_images, avg_ms, fps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ultralytics .pt model inference speed."
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Path to .pt model (or model name if already available).",
    )
    parser.add_argument(
        "--images",
        default="object_detection/data/combined_dataset_640X480/images",
        help="Image folder or single image path.",
    )
    parser.add_argument("--max-images", type=int, default=16, help="Max images to use.")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 if supported.")
    parser.add_argument(
        "--include-io",
        action="store_true",
        help="Include image loading by passing file paths to the model.",
    )
    args = parser.parse_args()

    if YOLO is None:  # pragma: no cover - runtime dependency
        print("Ultralytics is required: pip install ultralytics", file=sys.stderr)
        raise _YOLO_IMPORT_ERROR

    model_path = Path(args.model)
    model = YOLO(str(model_path.resolve())) if model_path.exists() else YOLO(args.model)

    image_paths = _collect_images(args.images, max_images=args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {args.images}")

    if args.include_io:
        source = [str(p) for p in image_paths]
    else:
        source = _load_images(image_paths)
        if not source:
            raise RuntimeError("Failed to load any images into memory.")

    predict_kwargs = {
        "verbose": False,
        "save": False,
        "batch": args.batch,
    }
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if args.half:
        predict_kwargs["half"] = True

    total_time, total_images, avg_ms, fps = run_benchmark(
        model,
        source,
        iters=args.iters,
        warmup=args.warmup,
        predict_kwargs=predict_kwargs,
    )

    mode = "include-io" if args.include_io else "preloaded"
    print("PT inference benchmark")
    print(f"  model: {args.model}")
    print(f"  images: {len(image_paths)} ({mode})")
    print(f"  batch: {args.batch} | imgsz: {args.imgsz} | device: {args.device} | half: {args.half}")
    print(f"  warmup: {args.warmup} | iters: {args.iters}")
    print(f"  total images: {total_images}")
    print(f"  total time: {total_time:.4f} s")
    print(f"  avg latency: {avg_ms:.3f} ms/image")
    print(f"  throughput: {fps:.2f} images/s")


if __name__ == "__main__":
    main()
