from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not images_dir.is_dir():
        return index
    for path in images_dir.iterdir():
        if path.suffix.lower() in IMAGE_EXTS:
            index[path.stem] = path
    return index


def _label_has_class_above(label_path: Path, min_class_id: int) -> bool:
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            current = int(float(parts[0]))
        except ValueError:
            continue
        if current >= min_class_id:
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find images containing labels with class IDs above a threshold."
    )
    parser.add_argument(
        "--data",
        default="finetune_dataset",
        help="Dataset directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--min-class-id",
        type=int,
        default=9,
        help="Minimum class ID to flag (>=). Default finds IDs > 8.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Print absolute paths instead of dataset-relative.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the excess image list.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.min_class_id < 0:
        raise SystemExit("--min-class-id must be >= 0")

    data_dir = Path(args.data).expanduser().resolve()
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    image_index = _build_image_index(images_dir)
    matches: list[Path] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        if not _label_has_class_above(label_path, args.min_class_id):
            continue
        stem = label_path.stem
        image_path = image_index.get(stem)
        if image_path is None:
            # Fallback to label stem when image is missing.
            image_path = images_dir / f"{stem}"
        matches.append(image_path)

    matches.sort(key=lambda path: path.name)
    print(
        f"min_class_id>={args.min_class_id} total={len(matches)}"
    )

    output_lines: list[str] = []
    for path in matches:
        if args.absolute:
            output_lines.append(str(path.resolve()))
        else:
            try:
                output_lines.append(str(path.resolve().relative_to(data_dir)))
            except ValueError:
                output_lines.append(str(path))

    if output_lines:
        print("\n".join(output_lines))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text("\n".join(output_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
