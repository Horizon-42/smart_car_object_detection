from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset_split import split_dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(images_dir: Path) -> list[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _has_label(labels_dir: Path, image_path: Path) -> bool:
    return (labels_dir / f"{image_path.stem}.txt").exists()


def _sanitize_label_file(label_path: Path) -> list[str]:
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError:
        return []
    cleaned: list[str] = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        cleaned.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return cleaned


def _ensure_images_dir(src_images: Path, dst_images: Path, copy_images: bool) -> None:
    if dst_images.exists():
        return
    if copy_images:
        dst_images.mkdir(parents=True, exist_ok=True)
        for image in _collect_images(src_images):
            shutil.copy2(image, dst_images / image.name)
        return
    try:
        dst_images.symlink_to(src_images, target_is_directory=True)
    except OSError:
        dst_images.mkdir(parents=True, exist_ok=True)
        for image in _collect_images(src_images):
            shutil.copy2(image, dst_images / image.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset, sanitize labels, and split into train/val/test."
    )
    parser.add_argument(
        "--data",
        default="finetune_dataset",
        help="Dataset directory containing images/ and labels/",
    )
    parser.add_argument(
        "--ratios",
        default="0.7,0.2,0.1",
        help="Train/val/test ratios, e.g. 0.7,0.2,0.1",
    )
    parser.add_argument(
        "--finetune-dir",
        default="finetune_dataset",
        help="Directory to write sanitized labels and splits (relative to data dir).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the finetune dataset instead of symlinking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=5,
        help=(
            "Max frame index gap to keep frames in the same split "
            "(per collection prefix)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data).resolve()

    try:
        ratios = tuple(float(x.strip()) for x in args.ratios.split(","))
    except ValueError:
        raise SystemExit("Invalid ratios format. Use something like 0.7,0.2,0.1")

    if len(ratios) != 3:
        raise SystemExit("Ratios must have three values (train,val,test).")
    if any(r < 0 for r in ratios):
        raise SystemExit("Ratios must be non-negative.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Ratios must sum to 1.0")
    if args.frame_gap < 0:
        raise SystemExit("--frame-gap must be >= 0")

    finetune_dir = Path(args.finetune_dir).expanduser()
    if not finetune_dir.is_absolute():
        finetune_dir = data_dir / finetune_dir
    finetune_dir = finetune_dir.resolve()

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    finetune_images_dir = finetune_dir / "images"
    finetune_labels_dir = finetune_dir / "labels"
    _ensure_images_dir(images_dir, finetune_images_dir, args.copy_images)
    finetune_labels_dir.mkdir(parents=True, exist_ok=True)
    classes_src = data_dir / "classes.txt"
    classes_dst = finetune_dir / "classes.txt"
    if classes_src.exists() and not classes_dst.exists():
        shutil.copy2(classes_src, classes_dst)

    images = _collect_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    labeled_images = [img for img in images if _has_label(labels_dir, img)]
    if not labeled_images:
        raise SystemExit("No labeled images found (missing label files).")

    image_names: list[str] = []
    total_label_lines = 0
    for image_path in labeled_images:
        src_label = labels_dir / f"{image_path.stem}.txt"
        cleaned_lines = _sanitize_label_file(src_label)
        total_label_lines += len(cleaned_lines)
        dst_label = finetune_labels_dir / f"{image_path.stem}.txt"
        dst_label.write_text("\n".join(cleaned_lines), encoding="utf-8")
        image_names.append(image_path.name)

    if total_label_lines == 0:
        raise SystemExit(
            "No valid labels found after sanitization. "
            "If your labels include a confidence column, this script removes it, "
            "but invalid rows are discarded."
        )

    splits_dir = finetune_dir / "splits"
    split_dataset(
        image_names,
        ratios,
        args.seed,
        finetune_dir,
        splits_dir,
        frame_gap=args.frame_gap,
    )


if __name__ == "__main__":
    main()
