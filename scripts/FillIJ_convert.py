from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def _parse_size(value: str) -> tuple[int, int]:
    text = value.lower().replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"Invalid size: {value} (expected W,H)")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise SystemExit("Target size must be positive.")
    return height, width


def _parse_map(text: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"Invalid map entry: {item} (expected src:dst)")
        src_text, dst_text = item.split(":", 1)
        mapping[int(src_text.strip())] = int(dst_text.strip())
    if not mapping:
        raise SystemExit("Class map is empty.")
    return mapping


def _collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


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

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return out, r, (left, top)


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError:
        return []
    labels: list[tuple[int, float, float, float, float]] = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        labels.append((class_id, x, y, w, h))
    return labels


def _convert_labels(
    labels: list[tuple[int, float, float, float, float]],
    mapping: dict[int, int],
    orig_shape: tuple[int, int],
    ratio: float,
    pad: tuple[int, int],
    target_shape: tuple[int, int],
) -> list[str]:
    if not labels:
        return []
    orig_h, orig_w = orig_shape
    tgt_h, tgt_w = target_shape
    pad_x, pad_y = pad
    converted: list[str] = []
    for class_id, xc, yc, w, h in labels:
        if class_id not in mapping:
            continue
        abs_x = xc * orig_w
        abs_y = yc * orig_h
        abs_w = w * orig_w
        abs_h = h * orig_h

        abs_x = abs_x * ratio + pad_x
        abs_y = abs_y * ratio + pad_y
        abs_w = abs_w * ratio
        abs_h = abs_h * ratio

        new_x = abs_x / tgt_w
        new_y = abs_y / tgt_h
        new_w = abs_w / tgt_w
        new_h = abs_h / tgt_h

        new_class = mapping[class_id]
        converted.append(
            f"{new_class} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        )
    return converted


def _read_gt_annotations(
    gt_path: Path,
) -> dict[str, list[tuple[int, float, float, float, float]]]:
    try:
        content = gt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read gt file: {gt_path}") from exc

    entries: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            parts = [p.strip() for p in line.split(";") if p.strip()]
        elif "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = line.split()
        if len(parts) < 6:
            continue
        name = parts[0]
        try:
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            class_id = int(float(parts[5]))
        except ValueError:
            continue
        entries.setdefault(name, []).append((class_id, x1, y1, x2, y2))
    return entries


def _convert_gt_labels(
    labels: list[tuple[int, float, float, float, float]],
    mapping: dict[int, int],
    orig_shape: tuple[int, int],
    ratio: float,
    pad: tuple[int, int],
    target_shape: tuple[int, int],
) -> list[str]:
    if not labels:
        return []
    orig_h, orig_w = orig_shape
    tgt_h, tgt_w = target_shape
    pad_x, pad_y = pad
    converted: list[str] = []
    for class_id, x1, y1, x2, y2 in labels:
        if class_id not in mapping:
            continue
        x1 = max(0.0, min(float(x1), orig_w - 1))
        y1 = max(0.0, min(float(y1), orig_h - 1))
        x2 = max(0.0, min(float(x2), orig_w - 1))
        y2 = max(0.0, min(float(y2), orig_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        x1 = x1 * ratio + pad_x
        y1 = y1 * ratio + pad_y
        x2 = x2 * ratio + pad_x
        y2 = y2 * ratio + pad_y

        xc = (x1 + x2) / 2.0 / tgt_w
        yc = (y1 + y2) / 2.0 / tgt_h
        w = (x2 - x1) / tgt_w
        h = (y2 - y1) / tgt_h

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        new_class = mapping[class_id]
        converted.append(
            f"{new_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        )
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert and sample FillIJ dataset into a YOLO-ready addition set."
    )
    parser.add_argument(
        "--src-root",
        default="FullIJCNN2013",
        help="Source dataset root containing images and gt.txt.",
    )
    parser.add_argument(
        "--gt-file",
        default="gt.txt",
        help="Ground truth file (relative to src-root or absolute).",
    )
    parser.add_argument(
        "--src-images",
        default=None,
        help="Source image directory (ppm/jpg/png) for YOLO-label mode.",
    )
    parser.add_argument(
        "--src-labels",
        default=None,
        help="Source label directory (YOLO .txt) for YOLO-label mode.",
    )
    parser.add_argument(
        "--out-dir",
        default="original_dataset_addition",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--target-size",
        default="640,320",
        help="Target size W,H (keeps ratio with letterbox).",
    )
    parser.add_argument(
        "--class-map",
        default="14:2,1:3,33:5,12:7,13:6,18:8",
        help="Class remap list like '14:2,1:3'.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Random samples per class to copy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-ext",
        default=".jpg",
        help="Output image extension (default: .jpg).",
    )
    parser.add_argument(
        "--output-prefix",
        default="real_frame_",
        help="Prefix for output image/label basenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target_shape = _parse_size(args.target_size)
    mapping = _parse_map(args.class_map)
    rng = random.Random(args.seed)

    class_to_images: dict[int, list[Path]] = {dst: [] for dst in mapping.values()}
    image_to_labels: dict[Path, list[str]] = {}

    if args.src_images and args.src_labels:
        src_images_dir = Path(args.src_images).expanduser().resolve()
        src_labels_dir = Path(args.src_labels).expanduser().resolve()
        if not src_labels_dir.is_dir():
            raise SystemExit(f"Labels directory not found: {src_labels_dir}")

        images = _collect_images(src_images_dir)
        if not images:
            raise SystemExit(f"No images found in {src_images_dir}")

        for image_path in images:
            label_path = src_labels_dir / f"{image_path.stem}.txt"
            labels = _read_yolo_labels(label_path)
            if not labels:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            _, ratio, pad = _letterbox(image, target_shape)
            converted = _convert_labels(
                labels,
                mapping,
                image.shape[:2],
                ratio,
                pad,
                target_shape,
            )
            if not converted:
                continue

            image_to_labels[image_path] = converted
            for line in converted:
                class_id = int(line.split()[0])
                class_to_images.setdefault(class_id, []).append(image_path)
    else:
        src_root = Path(args.src_root).expanduser().resolve()
        gt_path = Path(args.gt_file)
        if not gt_path.is_absolute():
            gt_path = (src_root / gt_path).resolve()
        gt_entries = _read_gt_annotations(gt_path)
        if not gt_entries:
            raise SystemExit(f"No entries found in {gt_path}")

        for name, labels in gt_entries.items():
            image_path = (src_root / name).resolve()
            if not image_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            _, ratio, pad = _letterbox(image, target_shape)
            converted = _convert_gt_labels(
                labels,
                mapping,
                image.shape[:2],
                ratio,
                pad,
                target_shape,
            )
            if not converted:
                continue
            image_to_labels[image_path] = converted
            for line in converted:
                class_id = int(line.split()[0])
                class_to_images.setdefault(class_id, []).append(image_path)

    selected: set[Path] = set()
    for class_id, image_list in class_to_images.items():
        unique = list(dict.fromkeys(image_list))
        if not unique:
            continue
        if args.per_class > 0 and len(unique) > args.per_class:
            picks = rng.sample(unique, args.per_class)
        else:
            picks = unique
        selected.update(picks)
        print(f"class {class_id}: selected {len(picks)} / {len(unique)}")

    if not selected:
        raise SystemExit("No images selected for output.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_path in sorted(selected):
        labels = image_to_labels.get(image_path)
        if not labels:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        resized, _, _ = _letterbox(image, target_shape)

        out_ext = args.output_ext if args.output_ext else ".jpg"
        if not out_ext.startswith("."):
            out_ext = f".{out_ext}"
        out_stem = f"{args.output_prefix}{image_path.stem}"
        out_image_path = out_images / f"{out_stem}{out_ext}"
        cv2.imwrite(str(out_image_path), resized)

        out_label_path = out_labels / f"{out_stem}.txt"
        out_label_path.write_text("\n".join(labels), encoding="utf-8")
        written += 1

    print(f"Wrote {written} images to {out_dir}")


if __name__ == "__main__":
    main()
