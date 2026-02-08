from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _parse_data_yaml(data_yaml: Path) -> dict[str, str]:
    try:
        content = data_yaml.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read data.yaml: {data_yaml}") from exc

    data: dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if line[:1].isspace():
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in {"path", "train", "val", "test"}:
            continue
        value = value.split("#", 1)[0].strip()
        data[key] = value

    missing = [k for k in ("path", "train", "val") if k not in data]
    if missing:
        raise SystemExit(f"data.yaml missing keys: {', '.join(missing)}")
    return data


def _resolve_data_root(data_yaml: Path, root_entry: str) -> Path:
    root_path = Path(root_entry).expanduser()
    if not root_path.is_absolute():
        root_path = (data_yaml.parent / root_path).resolve()
    return root_path


def _resolve_list_path(data_root: Path, entry: str) -> Path:
    list_path = Path(entry).expanduser()
    if not list_path.is_absolute():
        list_path = (data_root / list_path).resolve()
    return list_path


def _load_train_images(train_list_path: Path, data_root: Path) -> list[Path]:
    try:
        content = train_list_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read train list: {train_list_path}") from exc

    images: list[Path] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = (data_root / image_path).resolve()
        images.append(image_path)
    if not images:
        raise SystemExit(f"No training images listed in {train_list_path}")
    return images


def _collect_class_images(
    train_images: list[Path], labels_dir: Path
) -> dict[int, list[Path]]:
    class_to_images: dict[int, list[Path]] = {}
    for image_path in train_images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        try:
            content = label_path.read_text(encoding="utf-8")
        except OSError:
            continue
        class_ids: set[int] = set()
        for line in content.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            class_ids.add(class_id)
        for class_id in class_ids:
            class_to_images.setdefault(class_id, []).append(image_path)
    return class_to_images


def _build_balanced_train_list(
    class_to_images: dict[int, list[Path]], seed: int
) -> list[Path]:
    if not class_to_images:
        return []
    max_len = max(len(images) for images in class_to_images.values() if images)
    if max_len == 0:
        return []
    rng = random.Random(seed)
    balanced: list[Path] = []
    for class_id in sorted(class_to_images):
        images = class_to_images[class_id]
        if not images:
            continue
        if len(images) < max_len:
            images = images + [rng.choice(images) for _ in range(max_len - len(images))]
        balanced.extend(images)
    rng.shuffle(balanced)
    return balanced


def _write_balanced_data_yaml(
    src_yaml: Path, dst_yaml: Path, train_entry: str
) -> None:
    content = src_yaml.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines: list[str] = []
    replaced = False
    for line in lines:
        if line[:1].isspace():
            new_lines.append(line)
            continue
        if line.lstrip().startswith("#"):
            new_lines.append(line)
            continue
        if line.strip().startswith("train:"):
            new_lines.append(f"train: {train_entry}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"train: {train_entry}")
    dst_yaml.write_text("\n".join(new_lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a YOLO model using Ultralytics."
    )
    parser.add_argument(
        "--data-yaml",
        default="finetune_dataset/splits/data.yaml",
        help="Path to data.yaml produced by split_dataset_cli.py.",
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
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Random seed for class-balanced resampling.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_yaml = Path(args.data_yaml).expanduser().resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    data_cfg = _parse_data_yaml(data_yaml)
    data_root = _resolve_data_root(data_yaml, data_cfg["path"])
    train_list_path = _resolve_list_path(data_root, data_cfg["train"])
    labels_dir = data_root / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    train_images = _load_train_images(train_list_path, data_root)
    class_to_images = _collect_class_images(train_images, labels_dir)
    balanced_images = _build_balanced_train_list(
        class_to_images, args.balance_seed
    )
    if not balanced_images:
        raise SystemExit("Unable to build a balanced training list (no labels found).")

    balanced_train_path = train_list_path.with_name("train_balanced.txt")
    balanced_train_path.write_text(
        "\n".join(str(path) for path in balanced_images), encoding="utf-8"
    )
    try:
        balanced_train_entry = str(balanced_train_path.relative_to(data_root))
    except ValueError:
        balanced_train_entry = str(balanced_train_path)

    balanced_data_yaml = data_yaml.with_name("data_balanced.yaml")
    _write_balanced_data_yaml(data_yaml, balanced_data_yaml, balanced_train_entry)

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
    augment_args = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 2.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "erasing": 0.2,
    }
    model.train(
        data=str(balanced_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=run_name,
        **augment_args,
    )


if __name__ == "__main__":
    main()
