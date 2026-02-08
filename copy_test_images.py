from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy test images listed in splits/test.txt to a target folder."
    )
    parser.add_argument(
        "--splits-dir",
        default="original_dataset/finetune_dataset/splits",
        help="Directory containing test.txt (and data.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="original_dataset/finetune_dataset/test_images",
        help="Destination directory for copied test images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination.",
    )
    return parser.parse_args()


def _read_test_list(test_list_path: Path) -> list[Path]:
    try:
        content = test_list_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read {test_list_path}") from exc

    items: list[Path] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        items.append(Path(line))
    if not items:
        raise SystemExit(f"No entries found in {test_list_path}")
    return items


def main() -> None:
    args = _parse_args()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    test_list_path = splits_dir / "test.txt"
    if not test_list_path.exists():
        raise SystemExit(f"test.txt not found: {test_list_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_items = _read_test_list(test_list_path)
    copied = 0
    skipped = 0
    missing = 0

    for item in test_items:
        src = item.expanduser()
        if not src.is_absolute():
            src = (splits_dir / src).resolve()
        if not src.exists():
            missing += 1
            continue
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(
        f"Copied {copied} images to {output_dir} "
        f"(skipped {skipped}, missing {missing})."
    )


if __name__ == "__main__":
    main()
