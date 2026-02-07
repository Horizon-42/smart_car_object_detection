import argparse
import shutil
from pathlib import Path
import sys


def _parse_imgsz(value):
    if value is None:
        return None
    text = str(value).lower().replace('x', ',')
    if ',' in text:
        parts = [p.strip() for p in text.split(',') if p.strip()]
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    return int(text)


def export_pt_to_onnx(
    model_path,
    imgsz=640,
    opset=12,
    dynamic=False,
    simplify=False,
    half=False,
    device=None,
    output=None,
):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime dependency
        print("Ultralytics is required: pip install ultralytics", file=sys.stderr)
        raise exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PT model not found: {model_path}")

    model = YOLO(str(model_path.resolve()))
    export_kwargs = {
        "format": "onnx",
        "opset": opset,
        "dynamic": dynamic,
        "simplify": simplify,
        "half": half,
    }
    if imgsz is not None:
        export_kwargs["imgsz"] = imgsz
    if device is not None:
        export_kwargs["device"] = device

    onnx_path = Path(model.export(**export_kwargs))

    if output:
        output_path = Path(output)
        if output_path.is_dir():
            output_path = output_path / onnx_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, output_path)
        onnx_path = output_path

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to ONNX using Ultralytics."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .pt model (e.g. yolo26n.pt).",
    )
    parser.add_argument(
        "--imgsz",
        default=640,
        help="Image size (e.g. 640 or 640,480).",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes in ONNX export.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph after export.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 weights (if supported).",
    )
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output ONNX path or directory.",
    )

    args = parser.parse_args()

    imgsz = _parse_imgsz(args.imgsz)
    onnx_path = export_pt_to_onnx(
        args.model,
        imgsz=imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
        output=args.output,
    )
    print(f"Exported ONNX: {onnx_path}")
