from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_mobilenetv2_ssd import build_model


def _read_lines(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]


def _load_class_names(data_root: Path) -> list[str]:
    classes_path = data_root / "classes.txt"
    return _read_lines(classes_path)


class SSDExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, detections_per_img: int) -> None:
        super().__init__()
        self.model = model
        self.detections_per_img = detections_per_img

    def forward(self, images: torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        outputs = self.model(list(images))
        out = outputs[0]
        boxes = out["boxes"]
        scores = out["scores"]
        labels = out["labels"]

        max_det = self.detections_per_img
        count = boxes.shape[0]
        if count < max_det:
            pad = max_det - count
            boxes = torch.cat([boxes, boxes.new_zeros((pad, 4))], dim=0)
            scores = torch.cat([scores, scores.new_zeros((pad,))], dim=0)
            labels = torch.cat([
                labels,
                labels.new_zeros((pad,), dtype=labels.dtype),
            ])
        elif count > max_det:
            boxes = boxes[:max_det]
            scores = scores[:max_det]
            labels = labels[:max_det]

        return boxes, scores, labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MobileNetV2 SSD checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--data-root",
        default="finetune_dataset",
        help="Dataset root containing classes.txt.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input image size used by the SSD model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output ONNX path (defaults to checkpoint path with .onnx).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export (cpu or cuda).",
    )
    parser.add_argument(
        "--detections-per-img",
        type=int,
        default=100,
        help="Override detections per image (0 keeps model default).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph using onnxsim.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    data_root = Path(args.data_root).resolve()
    class_names = _load_class_names(data_root)
    if not class_names:
        raise SystemExit("classes.txt not found or empty; provide class names.")
    num_classes = len(class_names) + 1

    model = build_model(num_classes, args.image_size, weights_backbone="none")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    if args.detections_per_img > 0:
        if hasattr(model, "detections_per_img"):
            model.detections_per_img = args.detections_per_img
        detections_per_img = args.detections_per_img
    else:
        detections_per_img = getattr(model, "detections_per_img", 200)

    model.eval()

    device = torch.device(args.device)
    model.to(device)

    wrapper = SSDExportWrapper(model, detections_per_img=detections_per_img)
    wrapper.to(device)
    wrapper.eval()

    dummy = torch.zeros((1, 3, args.image_size, args.image_size), device=device)

    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.with_suffix(".onnx")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        do_constant_folding=True,
    )

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
        except Exception as exc:
            raise SystemExit(
                "onnx and onnxsim are required for --simplify"
            ) from exc
        onnx_model = onnx.load(str(output_path))
        simplified, check = simplify(onnx_model)
        if not check:
            raise SystemExit("onnxsim check failed; export left unchanged.")
        onnx.save(simplified, str(output_path))

    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
