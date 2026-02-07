from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple
from collections import Counter, OrderedDict

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.transforms import functional as F


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _resolve_label_path(image_path: Path, data_root: Path) -> Path:
    default = data_root / "labels" / f"{image_path.stem}.txt"
    if default.is_file():
        return default

    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        candidate = Path(*parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    return default


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        image_paths: Iterable[str],
        data_root: Path,
        num_classes: int,
        transforms=None,
        filter_empty: bool = False,
        strict: bool = False,
        strict_labels: bool = False,
    ) -> None:
        self.data_root = data_root
        self.transforms = transforms
        self.num_classes = num_classes
        self.strict_labels = strict_labels
        self.unknown_label_counts: Counter[int] = Counter()
        self.samples: List[Tuple[Path, Path]] = []

        for raw_path in image_paths:
            resolved = _resolve_image_path(raw_path, data_root)
            if not resolved.is_file():
                if strict:
                    raise FileNotFoundError(f"Image not found: {raw_path}")
                continue
            if resolved.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = _resolve_label_path(resolved, data_root)
            if filter_empty:
                boxes, labels = self._load_labels(
                    label_path, resolved, count_unknown=True
                )
                if len(boxes) == 0:
                    continue
            self.samples.append((resolved, label_path))

        if not self.samples:
            raise RuntimeError("No usable samples found.")
        if self.unknown_label_counts:
            max_valid = self.num_classes - 2
            counts = ", ".join(
                f"{cls_id}:{count}"
                for cls_id, count in sorted(self.unknown_label_counts.items())
            )
            print(
                "Warning: skipped labels with class ids outside "
                f"[0, {max_valid}]: {counts}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._load_labels(
            label_path, image_path, image, count_unknown=False
        )

        target = self._make_target(idx, boxes, labels)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def _load_labels(
        self,
        label_path: Path,
        image_path: Path,
        image: Image.Image | None = None,
        count_unknown: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image is None:
            image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []

        if label_path.is_file():
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                except ValueError:
                    continue

                if class_id < 0 or class_id >= self.num_classes - 1:
                    if self.strict_labels:
                        raise ValueError(
                            f"Class id {class_id} out of range for {label_path}"
                        )
                    if count_unknown:
                        self.unknown_label_counts[class_id] += 1
                    continue

                x_min = (x_center - box_w / 2.0) * width
                y_min = (y_center - box_h / 2.0) * height
                x_max = (x_center + box_w / 2.0) * width
                y_max = (y_center + box_h / 2.0) * height

                x_min = max(0.0, min(x_min, width - 1))
                y_min = max(0.0, min(y_min, height - 1))
                x_max = max(0.0, min(x_max, width - 1))
                y_max = max(0.0, min(y_max, height - 1))

                if x_max <= x_min or y_max <= y_min:
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id + 1)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return boxes_tensor, labels_tensor

    @staticmethod
    def _make_target(idx: int, boxes: torch.Tensor, labels: torch.Tensor) -> dict:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return target


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            width = image.shape[-1] if torch.is_tensor(image) else image.width
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def _collate_fn(batch):
    return tuple(zip(*batch))


class MobileNetV2SSDBackbone(nn.Module):
    def __init__(self, weights: str = "none") -> None:
        super().__init__()
        if weights == "imagenet":
            try:
                backbone = mobilenet_v2(weights="DEFAULT")
            except TypeError:
                backbone = mobilenet_v2(pretrained=True)
        else:
            try:
                backbone = mobilenet_v2(weights=None)
            except TypeError:
                backbone = mobilenet_v2(pretrained=False)
        self.features = backbone.features

        self.out_channels = [32, 96, 1280, 512, 256, 256]

        self.extra = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1280, 512, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features: List[torch.Tensor] = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 6:  # stride 8
                features.append(x)
            elif idx == 13:  # stride 16
                features.append(x)
            elif idx == 18:  # stride 32
                features.append(x)

        for layer in self.extra:
            x = layer(x)
            features.append(x)
        return OrderedDict((str(i), feat) for i, feat in enumerate(features))


def build_model(num_classes: int, image_size: int, weights_backbone: str) -> SSD:
    backbone = MobileNetV2SSDBackbone(weights=weights_backbone)

    scales = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    aspect_ratios = [
        [2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2],
        [2],
    ]
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=aspect_ratios,
        scales=scales,
    )

    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(image_size, image_size),
    )
    return model


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    amp: bool,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    running_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in target.items()} for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss detected: {losses}")

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    print(f"Epoch {epoch:03d} | train loss: {avg_loss:.4f}")
    return avg_loss


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 SSD on finetune_dataset using PyTorch."
    )
    parser.add_argument(
        "--data-root",
        default="finetune_dataset",
        help="Dataset root containing images/, labels/, splits/.",
    )
    parser.add_argument(
        "--train-split",
        default="finetune_dataset/splits/train.txt",
        help="Path to train split list.",
    )
    parser.add_argument(
        "--val-split",
        default="finetune_dataset/splits/val.txt",
        help="Path to val split list.",
    )
    parser.add_argument("--image-size", type=int, default=320, help="SSD input size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="SGD weight decay."
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--output-dir",
        default="runs/mobilenetv2_ssd",
        help="Base output directory for checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run subdirectory name (defaults to timestamp).",
    )
    parser.add_argument(
        "--weights-backbone",
        choices=["none", "imagenet"],
        default="none",
        help="Backbone weights (imagenet downloads weights).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Fail on labels with class ids outside classes.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    train_split = Path(args.train_split).resolve()
    val_split = Path(args.val_split).resolve()

    class_names = _load_class_names(data_root)
    if not class_names:
        raise SystemExit("classes.txt not found or empty; provide class names.")
    num_classes = len(class_names) + 1

    train_paths = _read_lines(train_split)
    if not train_paths:
        raise SystemExit(f"Train split is empty or missing: {train_split}")
    val_paths = _read_lines(val_split) if val_split.exists() else []

    train_transforms = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    val_transforms = Compose([ToTensor()])

    train_dataset = YoloDetectionDataset(
        train_paths,
        data_root,
        num_classes=num_classes,
        transforms=train_transforms,
        filter_empty=True,
        strict_labels=args.strict_labels,
    )
    val_dataset = (
        YoloDetectionDataset(
            val_paths,
            data_root,
            num_classes=num_classes,
            transforms=val_transforms,
            strict_labels=args.strict_labels,
        )
        if val_paths
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=_collate_fn,
        )
        if val_dataset is not None
        else None
    )

    device = torch.device(args.device)
    model = build_model(num_classes, args.image_size, args.weights_backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"mobilenetv2_ssd_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({"num_classes": num_classes, "class_names": class_names})
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    metrics_path = output_dir / "metrics.jsonl"

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", best_loss)

    if val_loader is not None:
        print(
            f"Val split loaded with {len(val_loader.dataset)} samples. "
            "No loss computed; SSD losses require train mode."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = _train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.amp
        )
        lr_scheduler.step()

        best_updated = False
        if train_loss < best_loss:
            best_loss = train_loss
            best_updated = True

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }
        _save_checkpoint(output_dir / "checkpoint_last.pt", checkpoint)
        if best_updated:
            _save_checkpoint(output_dir / "checkpoint_best.pt", checkpoint)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "num_batches": len(train_loader),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

    print(f"Training complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
