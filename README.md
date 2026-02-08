# Smart Car Object Detection

End-to-end utilities for dataset prep, YOLO finetuning, and inference/export (PyTorch, ONNX, TensorRT).

## Quick start

1) Install deps:
```
pip install -r requirements.txt
```

2) Prepare dataset splits (with leak-safe frame clustering):
```
python3 split_dataset_cli.py --data finetune_dataset --ratios 0.7,0.2,0.1 --frame-gap 5
```

3) Finetune (object-level balancing enabled by default):
```
python3 finetune_yolo.py --data-yaml finetune_dataset/splits/data.yaml --model models/yolo11n.pt
```

4) Visualize inference (PT):
```
python3 yolo_infer_vis.py --model models/yolo11n.pt --source test_images
```

## Dataset layout

Expected YOLO format:
```
finetune_dataset/
  images/
    collect1_frame_000123.jpg
  labels/
    collect1_frame_000123.txt
  classes.txt
```

Label files should be standard YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

`split_dataset_cli.py` sanitizes labels (keeps the first 5 columns, drops invalid rows).

## Leak-safe splitting (frame clustering)

`dataset_split.py` groups images by collection prefix and keeps nearby frames in the same split.
It uses the filename pattern:
```
<prefix>_frame_<index>.<ext>
```

Example:
```
collect1_frame_000123.jpg
collect1_frame_000130.jpg
```

Frames within `--frame-gap` (default `5`) stay together, preventing train/val/test leakage.
The splitter prints a per-prefix debug summary so you can verify clustering.

## Training (finetune_yolo.py)

Key flags:
- `--data-yaml`: path to `splits/data.yaml`
- `--model`: .pt weights
- `--epochs`, `--imgsz`, `--batch`
- `--no-balance`: disable object-level balancing
- `--balance-seed`: RNG seed for balancing

Balancing behavior:
- Counts objects per class across training labels.
- Oversamples images containing underrepresented classes until object counts match the max class.
- Writes `train_balanced.txt` and `data_balanced.yaml`, and trains with the balanced list.

Augmentations are set inside `finetune_yolo.py` (HSV, scale/translate, mosaic, mixup, copy-paste, erasing).

## Inference utilities

- `yolo_infer_vis.py`: visualize .pt inference with timing overlays.
- `benchmark_pt_inference.py`: benchmark .pt inference speed.
- `trt_detector.py`: TensorRT engine inference class (standalone, no ONNX helper dependency).

## Export

### Export .pt to ONNX:
```
python3 export_pt_to_onnx.py --model models/yolo11n.pt --imgsz 640 --output models/yolo11n.onnx
```

The exporter can optionally patch Resize scales for TensorRT compatibility.

### Export .onnx to .engine
```
onnx2_engine.sh
```

## Scripts (misc)

- `scripts/pre_labeling.py`: pre-label images using a YOLO model.
- `scripts/copy_test_images.py`: copy test split images from `splits/test.txt` into a folder.
- `scripts/data_preprocess.py`: dataset-specific merge helper (update paths as needed).
- `scripts/add_prefix.py`: dataset-specific renamer (update paths/prefix as needed).

## Notes

- TensorRT/pycuda are required for `trt_detector.py`.
- CUDA/torch are needed for GPU training/inference.
