from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _has_labels(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    try:
        return label_path.stat().st_size > 0
    except OSError:
        return False


def _collect_images(src_dir: Path) -> list[Path]:
    return [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def filter_labeled_dataset(src_dir: Path, dst_dir: Path, move: bool = False) -> None:
    src_dir = src_dir.expanduser()
    src_images_dir = src_dir / 'images'
    if not src_images_dir.is_dir():
        print(f"Source images directory does not exist: {src_images_dir}")
        return
    src_labels_dir = src_dir / 'labels'
    if not src_labels_dir.is_dir():
        print(f"Source labels directory does not exist: {src_labels_dir}")
        return
    
    dst_dir = dst_dir.expanduser()
    images_dir = dst_dir / 'images'
    labels_dir = dst_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(src_images_dir)
    if not images:
        print(f"No images found in source directory: {src_images_dir}")
        return

    for image_path in images:
        label_path = src_labels_dir / (image_path.stem + '.txt')
        if not _has_labels(label_path):
            continue
        print(f"Processing: {image_path.name} with labels: {label_path.name}")

        dst_image = images_dir / image_path.name
        dst_label = labels_dir / label_path.name
        if move:
            shutil.move(image_path, dst_image)
            shutil.move(label_path, dst_label)
        else:
            shutil.copy2(image_path, dst_image)
            shutil.copy2(label_path, dst_label)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Copy (or move) only images that have non-empty YOLO labels.'
    )
    parser.add_argument(
        '--src',
        required=True,
        help='Source directory that contains images and .txt label files.',
    )
    parser.add_argument(
        '--dst',
        required=True,
        help='Destination directory to write filtered images and labels.',
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    filter_labeled_dataset(Path(args.src), Path(args.dst), move=args.move)
