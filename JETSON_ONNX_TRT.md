# Jetson (JetRacer) ONNX Runtime + TensorRT EP Guide

This guide shows how to run `yolo26n.onnx` on a Jetson-based JetRacer using ONNX Runtime (ORT) with the TensorRT Execution Provider (EP). The goal is to keep deployment simple while getting most of the TensorRT speedup.

## 1) Prerequisites

- Jetson device with JetPack installed (JetRacer typically uses Jetson Nano/Xavier NX/Orin).
- Python 3 and pip available on the Jetson.
- ONNX model ready: `yolo26n.onnx`.

## 2) Put the Jetson in a performance mode (optional but recommended)

These commands increase clocks and set a higher power mode. They can improve FPS for real-time inference.

```bash
sudo /usr/sbin/nvpmodel -q
sudo /usr/sbin/nvpmodel -m <mode_id>
sudo /usr/bin/jetson_clocks --fan
```

Notes:
- Use `nvpmodel -q` to see the current mode.
- Mode IDs depend on your Jetson model.
- `jetson_clocks --fan` locks clocks and (on some releases) sets max fan speed.

## 2.1) Identify Jetson model and JetPack version

Run these commands on the Jetson:

```bash
# Jetson model
cat /proc/device-tree/model

# L4T (JetPack base) version
cat /etc/nv_tegra_release

# Installed JetPack meta-package version (if present)
dpkg-query -W nvidia-jetpack
```

Notes:
- The L4T version in `/etc/nv_tegra_release` maps to a JetPack release.
- If `dpkg-query` doesn't find `nvidia-jetpack`, JetPack may have been installed via a different method.

## 3) Install ONNX Runtime with TensorRT EP

The Jetson build must match your JetPack, CUDA, and TensorRT versions. ORT provides Jetson packages via Jetson Zoo.

Steps:
1. Open the Jetson Zoo page and download the correct ORT GPU package for your JetPack and Python version.
2. Install the wheel locally:

```bash
pip3 install /path/to/onnxruntime_gpu-<version>-<python>-linux_aarch64.whl
```

If you prefer containers, Jetson Zoo also provides Docker images with ORT GPU preinstalled.

## 4) Verify that ORT sees TensorRT

Run this on the Jetson:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

You should see `TensorrtExecutionProvider` and `CUDAExecutionProvider` in the list.

## 5) Use TensorRT EP in code

ORT requires you to register TensorRT explicitly. It is recommended to register CUDA as a fallback for unsupported ops.

```python
import onnxruntime as ort

providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
    }),
    ("CUDAExecutionProvider", {
        "device_id": 0,
    }),
]

sess = ort.InferenceSession("yolo26n.onnx", providers=providers)
```

Notes:
- `trt_engine_cache_*` reduces session build time after the first run.
- `trt_fp16_enable` can improve speed if your Jetson supports fast FP16.

## 6) Use TensorRT EP in this repo

`onnx_detector.py` already accepts providers. Example:

```bash
python3 onnx_detector.py \
  --model yolo26n.onnx \
  --image test_images/collect1_frame_000861.jpg \
  --provider TensorrtExecutionProvider \
  --provider CUDAExecutionProvider \
  --measure-time
```

If you are running the notebook, set `measure_time = True` to print per-frame timing.

## 7) Recommended input size

YOLO models are typically exported for fixed shapes (often 640x640). You can override the size with:

```bash
--imgsz 640
```

Keep the size aligned to the model export or a multiple of the network stride (usually 32). If you export a dynamic-shape model, you can set TensorRT profile options (advanced).

## 8) Troubleshooting

- If `TensorrtExecutionProvider` is missing:
  - The ORT build you installed likely does not include TensorRT EP. Install the Jetson GPU build from Jetson Zoo.
- If TensorRT fails to build engines:
  - Keep CUDA as fallback so ORT can run unsupported ops.
- If performance is still low:
  - Ensure power mode is high, clocks are locked, and use FP16.

## 9) References

- ONNX Runtime TensorRT EP documentation: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- ONNX Runtime execution providers overview: https://onnxruntime.ai/docs/execution-providers/
- Jetson Zoo (ORT GPU packages / containers): https://elinux.org/Jetson_Zoo
- NVIDIA Jetson power management (nvpmodel, jetson_clocks): https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3276/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html
