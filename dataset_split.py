from __future__ import annotations

import random
import re
import sys
from pathlib import Path

_FRAME_RE = re.compile(r"^(?P<prefix>.+?)_frame[_-]?(?P<index>\d+)", re.IGNORECASE)


def _load_class_names(data_dir: Path) -> list[str]:
    classes_path = data_dir / "classes.txt"
    try:
        content = classes_path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]


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


def _parse_collection_frame(image_name: str) -> tuple[str, int | None]:
    stem = Path(image_name).stem
    match = _FRAME_RE.match(stem)
    if not match:
        return stem, None
    return match.group("prefix"), int(match.group("index"))


def _cluster_by_collection(
    image_names: list[str], frame_gap: int
) -> tuple[list[list[str]], list[str]]:
    grouped: dict[str, list[tuple[int | None, str]]] = {}
    for name in image_names:
        prefix, index = _parse_collection_frame(name)
        grouped.setdefault(prefix, []).append((index, name))

    clusters: list[list[str]] = []
    debug_lines: list[str] = []
    for prefix in sorted(grouped):
        items = grouped[prefix]
        with_index: list[tuple[int, str]] = []
        without_index: list[str] = []
        for index, name in items:
            if index is None:
                without_index.append(name)
            else:
                with_index.append((index, name))

        for name in without_index:
            clusters.append([name])

        cluster_ranges: list[tuple[int, int, int]] = []

        if with_index:
            with_index.sort(key=lambda item: item[0])
            current: list[str] = [with_index[0][1]]
            start_index = with_index[0][0]
            last_index = with_index[0][0]
            for index, name in with_index[1:]:
                if index - last_index <= frame_gap:
                    current.append(name)
                else:
                    clusters.append(current)
                    cluster_ranges.append((start_index, last_index, len(current)))
                    current = [name]
                    start_index = index
                last_index = index
            clusters.append(current)
            cluster_ranges.append((start_index, last_index, len(current)))

        frame_count = len(with_index)
        cluster_count = len(cluster_ranges)
        no_index_count = len(without_index)
        preview = ", ".join(
            f"{start}-{end}({count})" for start, end, count in cluster_ranges[:5]
        )
        if cluster_count > 5:
            preview = f"{preview}, ..."
        debug_lines.append(
            f"{prefix}: frames={frame_count} clusters={cluster_count} "
            f"no_index={no_index_count} ranges=[{preview}]"
        )

    return clusters, debug_lines


def split_dataset(
    image_names: list[str],
    ratios: tuple[float, float, float],
    seed: int,
    data_root: Path,
    splits_dir: Path,
    frame_gap: int = 5,
) -> tuple[Path, Path, Path, Path]:
    if not image_names:
        raise RuntimeError("No images available for splitting.")
    if frame_gap < 0:
        raise ValueError("frame_gap must be >= 0")

    train_ratio, val_ratio, test_ratio = ratios
    clusters, debug_lines = _cluster_by_collection(image_names, frame_gap)

    rng = random.Random(seed)
    rng.shuffle(clusters)

    n_total = sum(len(cluster) for cluster in clusters)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs: list[str] = []
    val_imgs: list[str] = []
    test_imgs: list[str] = []

    for cluster in clusters:
        if len(train_imgs) < n_train:
            train_imgs.extend(cluster)
        elif len(val_imgs) < n_val:
            val_imgs.extend(cluster)
        else:
            test_imgs.extend(cluster)

    if not train_imgs or not val_imgs or not test_imgs:
        print(
            "Warning: one of the splits is empty. "
            "Adjust ratios, frame_gap, or ensure more data."
        )

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
    print(
        f"Grouped into {len(clusters)} clusters "
        f"using frame_gap={frame_gap}."
    )
    if debug_lines:
        print("Cluster debug (prefix summary):")
        max_lines = 10
        for line in debug_lines[:max_lines]:
            print(f"  {line}")
        if len(debug_lines) > max_lines:
            print(f"  ... ({len(debug_lines) - max_lines} more prefixes)")
    print(f"Splits written to: {splits_dir}")

    return train_txt, val_txt, test_txt, data_yaml
