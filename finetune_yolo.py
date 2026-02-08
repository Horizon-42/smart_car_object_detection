from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(images_dir: Path) -> list[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _load_class_names(data_dir: Path) -> list[str]:
    classes_path = data_dir / "classes.txt"
    try:
        content = classes_path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]


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


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def _check_leak(train: list[str], val: list[str], test: list[str]) -> None:
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("Data leak detected between splits:")
        if overlap_train_val:
            print(f"  Train ∩ Val: {len(overlap_train_val)}")
        if overlap_train_test:
            print(f"  Train ∩ Test: {len(overlap_train_test)}")
        if overlap_val_test:
            print(f"  Val ∩ Test: {len(overlap_val_test)}")
        sys.exit(1)


def split_dataset(
    image_names: list[str],
    ratios: tuple[float, float, float],
    seed: int,
    data_root: Path,
    splits_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    if not image_names:
        raise RuntimeError("No images available for splitting.")

    rng = random.Random(seed)
    rng.shuffle(image_names)

    train_ratio, val_ratio, test_ratio = ratios
    n_total = len(image_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        print(
            "Warning: one of the splits is empty. "
            "Adjust ratios or ensure more data."
        )

    train_imgs = image_names[:n_train]
    val_imgs = image_names[n_train : n_train + n_val]
    test_imgs = image_names[n_train + n_val :]

    # Use absolute paths so Ultralytics resolves images/labels correctly.
    images_root = data_root / "images"
    train_list = [str((images_root / name).resolve()) for name in train_imgs]
    val_list = [str((images_root / name).resolve()) for name in val_imgs]
    test_list = [str((images_root / name).resolve()) for name in test_imgs]

    _check_leak(train_list, val_list, test_list)

    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"
    test_txt = splits_dir / "test.txt"
    _write_list(train_txt, train_list)
    _write_list(val_txt, val_list)
    _write_list(test_txt, test_list)

    data_yaml = splits_dir / "data.yaml"
    names = _load_class_names(data_root)
    yaml_lines = [
        f"path: {data_root}",
        f"train: {train_txt.relative_to(data_root)}",
        f"val: {val_txt.relative_to(data_root)}",
        f"test: {test_txt.relative_to(data_root)}",
    ]
    if names:
        yaml_lines.append("names:")
        yaml_lines.extend([f"  {idx}: {name}" for idx, name in enumerate(names)])
    else:
        yaml_lines.append("names: []")
    data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")

    print(f"Total labeled images: {n_total}")
    print(f"Train: {len(train_list)}  Val: {len(val_list)}  Test: {len(test_list)}")
    print(f"Splits written to: {splits_dir}")

    return train_txt, val_txt, test_txt, data_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split dataset, check data leak, and finetune a YOLO model using Ultralytics."
        )
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
        "--model",
        default="yolo26n.pt",
        help="Model weights to finetune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--project",
        default="runs/finetune",
        help="Ultralytics project directory.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name (defaults to <model-stem>_signs).",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only create splits and check leak, skip training.",
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
    _, _, _, data_yaml = split_dataset(
        image_names, ratios, args.seed, finetune_dir, splits_dir
    )

    if args.no_train:
        return

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime dependency
        print("Ultralytics is required: pip install ultralytics", file=sys.stderr)
        raise SystemExit(1) from exc

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    run_name = args.name or f"{model_path.stem}_signs"
    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=run_name,
    )


if __name__ == "__main__":
    main()
