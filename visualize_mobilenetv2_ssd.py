from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from train_mobilenetv2_ssd import build_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_lines(path: Path) -> List[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]


def _load_class_names(data_root: Path) -> List[str]:
    classes_path = data_root / "classes.txt"
    return _read_lines(classes_path)


def _resolve_image_path(image_path: str, data_root: Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate
    fallback = data_root / "images" / candidate.name
    if fallback.is_file():
        return fallback
    return candidate


def _collect_images_from_input(
    inputs: Iterable[str], data_root: Path
) -> List[Path]:
    images: List[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in _read_lines(path):
                resolved = _resolve_image_path(line, data_root)
                if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTS:
                    images.append(resolved)
            continue
        if path.is_dir():
            for item in sorted(path.iterdir()):
                if item.suffix.lower() in IMAGE_EXTS:
                    images.append(item)
            continue
        resolved = _resolve_image_path(entry, data_root)
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTS:
            images.append(resolved)
    if not images:
        raise RuntimeError("No images found for the given input.")
    return images


def _color_for_class(class_id: int) -> Tuple[int, int, int]:
    random.seed(class_id + 12345)
    return (
        random.randint(30, 225),
        random.randint(30, 225),
        random.randint(30, 225),
    )


@contextmanager
def _maybe_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda"):
            yield
    else:
        yield


def _draw_detections(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: List[str],
    score_threshold: float,
    max_detections: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    if scores.numel() == 0:
        return image

    sorted_scores, order = scores.sort(descending=True)
    if max_detections > 0:
        order = order[:max_detections]

    for idx in order.tolist():
        score = scores[idx].item()
        if score < score_threshold:
            continue
        box = boxes[idx].tolist()
        label_id = int(labels[idx])
        color = _color_for_class(label_id)
        name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        label = f"{name} {score:.2f}"

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if font is not None:
            text_size = draw.textbbox((x1, y1), label, font=font)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]
            text_bg = [x1, max(0, y1 - text_h - 2), x1 + text_w + 4, y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1 + 2, max(0, y1 - text_h - 1)), label, fill="black", font=font)
    return image


def _show_with_opencv(
    image: Image.Image, window_name: str, title: str, delay: int
) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"OpenCV not available ({exc}). Install opencv-python."
        ) from exc

    frame = np.array(image)[:, :, ::-1]  # RGB -> BGR
    cv2.imshow(window_name, frame)
    if hasattr(cv2, "setWindowTitle"):
        try:
            cv2.setWindowTitle(window_name, title)
        except Exception:
            pass
    key = cv2.waitKey(delay)
    if key in (27, ord("q")):
        cv2.destroyAllWindows()
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MobileNetV2 SSD and visualize detections."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pt) saved by training script.",
    )
    parser.add_argument(
        "--data-root",
        default="finetune_dataset",
        help="Dataset root containing classes.txt and images/.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Image path(s), directory, or split .txt file.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save visualizations (used with --save/--save-json).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input size used by the SSD model.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.4,
        help="Score threshold for drawing detections.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections to draw per image (0 = no limit).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save rendered images to disk.",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show images in a window (default).",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable window display.",
    )
    parser.add_argument(
        "--show-delay",
        type=int,
        default=0,
        help="Delay per image in ms for OpenCV display (0 waits for key).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable autocast for CUDA inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images processed (0 = all).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detections as JSON alongside images.",
    )
    parser.set_defaults(show=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = Path(args.data_root).resolve()
    class_names = _load_class_names(data_root)
    if not class_names:
        raise SystemExit("classes.txt not found or empty; provide class names.")
    class_names = ["__background__"] + class_names
    num_classes = len(class_names)

    image_paths = _collect_images_from_input(args.input, data_root)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    device = torch.device(args.device)
    model = build_model(num_classes, args.image_size, weights_backbone="none")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    output_dir: Path | None = None
    if args.save or args.save_json:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path("runs/mobilenetv2_ssd/visualizations")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    window_name = "Detections"
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = F.to_tensor(image)
        with torch.no_grad():
            with _maybe_autocast(device, args.amp):
                outputs = model([tensor.to(device)])
        output = outputs[0]
        boxes = output.get("boxes", torch.empty((0, 4))).cpu()
        labels = output.get("labels", torch.empty((0,), dtype=torch.int64)).cpu()
        scores = output.get("scores", torch.empty((0,))).cpu()

        rendered = _draw_detections(
            image,
            boxes,
            labels,
            scores,
            class_names,
            args.score_threshold,
            args.max_detections,
        )

        if args.save and output_dir is not None:
            out_path = output_dir / image_path.name
            rendered.save(out_path)

        if args.save_json and output_dir is not None:
            json_path = output_dir / f"{image_path.stem}.json"
            detections = []
            for box, label, score in zip(boxes, labels, scores):
                if score.item() < args.score_threshold:
                    continue
                detections.append(
                    {
                        "label": class_names[int(label)] if int(label) < len(class_names) else int(label),
                        "score": float(score.item()),
                        "box": [float(v) for v in box.tolist()],
                    }
                )
            json_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

        if args.show:
            keep_going = _show_with_opencv(
                rendered, window_name, image_path.name, args.show_delay
            )
            if not keep_going:
                break

    if args.save or args.save_json:
        print(f"Saved results to: {output_dir}")
    else:
        print(f"Processed {len(image_paths)} images (no files saved).")


if __name__ == "__main__":
    main()
