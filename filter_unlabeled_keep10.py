from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(images_dir: Path) -> list[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _has_label_file(labels_dir: Path, image_path: Path) -> bool:
    return (labels_dir / f"{image_path.stem}.txt").exists()


def filter_unlabeled_keep(
    dataset_dir: Path,
    keep_ratio: float,
    seed: int,
    move_dir: Path,
) -> None:
    dataset_dir = dataset_dir.expanduser()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}")
        return
    if not labels_dir.is_dir():
        print(f"Labels directory not found: {labels_dir}")
        return

    images = _collect_images(images_dir)
    if not images:
        print(f"No images found in: {images_dir}")
        return

    unlabeled = [img for img in images if not _has_label_file(labels_dir, img)]
    if not unlabeled:
        print("No unlabeled images found.")
        return

    keep_ratio = max(0.0, min(1.0, keep_ratio))
    keep_count = int(len(unlabeled) * keep_ratio)
    if keep_ratio > 0 and keep_count == 0:
        keep_count = 1

    rng = random.Random(seed)
    rng.shuffle(unlabeled)

    keep_set = set(unlabeled[:keep_count])
    drop_list = unlabeled[keep_count:]

    if drop_list:
        move_dir.mkdir(parents=True, exist_ok=True)

    for image_path in drop_list:
        target = move_dir / image_path.name
        shutil.move(str(image_path), str(target))

    for image_path in keep_set:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("")

    print(f"Total images: {len(images)}")
    print(f"Unlabeled images: {len(unlabeled)}")
    print(f"Kept unlabeled (empty labels created): {len(keep_set)}")
    print(f"Moved unlabeled images: {len(drop_list)}")
    if drop_list:
        print(f"Moved to: {move_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter out unlabeled images, keep a ratio of them, and create empty labels."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory containing images/ and labels/",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.1,
        help="Fraction of unlabeled images to keep (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selection (default: 42)",
    )
    parser.add_argument(
        "--move-dir",
        default="images_unlabeled_filtered",
        help="Folder to move filtered unlabeled images into (relative to dataset).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_dir = Path(args.data)
    move_dir = Path(args.move_dir)
    if not move_dir.is_absolute():
        move_dir = data_dir / move_dir
    filter_unlabeled_keep(data_dir, args.keep_ratio, args.seed, move_dir)
