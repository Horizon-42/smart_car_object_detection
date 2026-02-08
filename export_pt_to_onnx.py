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


def _trt_fix_resize_scales(onnx_path):
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except Exception:
        print("ONNX/numpy not available; skipping TensorRT Resize fix.", file=sys.stderr)
        return False

    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    const_tensors = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                const_tensors[node.output[0]] = attr.t
                break

    output_to_node = {}
    used_names = set(initializers)
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node
            used_names.add(out)
        for inp in node.input:
            used_names.add(inp)

    def _unique_name(base):
        name = base
        index = 0
        while name in used_names:
            index += 1
            name = f"{base}_{index}"
        used_names.add(name)
        return name

    modified = False
    for node in graph.node:
        if node.op_type != "Resize" or len(node.input) < 3:
            continue
        scales_name = node.input[2]
        if scales_name in initializers:
            continue

        tensor = None
        if scales_name in const_tensors:
            tensor = const_tensors[scales_name]
        else:
            src_node = output_to_node.get(scales_name)
            if src_node and src_node.op_type == "Cast" and src_node.input:
                src_name = src_node.input[0]
                if src_name in initializers:
                    tensor = initializers[src_name]
                elif src_name in const_tensors:
                    tensor = const_tensors[src_name]

        if tensor is None:
            continue

        scales = numpy_helper.to_array(tensor).astype(np.float32)
        init_name = _unique_name(f"{scales_name}_init")
        graph.initializer.append(numpy_helper.from_array(scales, name=init_name))
        node.input[2] = init_name
        modified = True

    if modified:
        onnx.save(model, str(onnx_path))
    return modified


def export_pt_to_onnx(
    model_path,
    imgsz=320,
    opset=11,
    dynamic=False,
    simplify=False,
    half=False,
    device=None,
    output=None,
    trt_fix=True,
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

    if trt_fix:
        if _trt_fix_resize_scales(onnx_path):
            print(
                "Patched Resize scales to constant initializers for TensorRT.",
                file=sys.stderr,
            )

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
        default=320,
        help="Image size (e.g. 640 or 640,480).",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
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
        default=True,
        action="store_true",
        help="Export with FP16 weights (if supported).",
    )
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output ONNX path or directory.",
    )
    parser.add_argument(
        "--no-trt-fix",
        action="store_true",
        help="Disable TensorRT Resize scales fix.",
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
        trt_fix=not args.no_trt_fix,
    )
    print(f"Exported ONNX: {onnx_path}")
