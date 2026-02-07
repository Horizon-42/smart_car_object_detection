# MobileNetV2 SSD Finetune Dev Doc

## Goal
Train a MobileNetV2-SSD object detector on the local `finetune_dataset` using PyTorch.

## Dataset layout
Expected structure (already present in this repo):

```
finetune_dataset/
  images/            # image files (.jpg/.png/...)
  labels/            # YOLO txt labels (class x_center y_center w h)
  classes.txt        # one class name per line
  splits/
    train.txt        # list of image paths (absolute or relative)
    val.txt
    test.txt
```

Notes:
- Label format is YOLO normalized values in `[0,1]`.
- `train.txt` and `val.txt` may contain absolute paths that do not exist locally. The training script remaps each entry to `finetune_dataset/images/<basename>` if the original path is missing.
- Class ids from labels are shifted by +1 for TorchVision (0 is background).

## Training script
File: `train_mobilenetv2_ssd.py`

What it does:
- Builds a MobileNetV2 backbone and an SSD head with 6 feature maps.
- Uses TorchVision's `SSD` + `DefaultBoxGenerator` for detection.
- Loads YOLO labels, converts to absolute XYXY boxes.
- Trains with standard SGD, logs losses, and saves checkpoints.

Key arguments (defaults shown):
- `--data-root finetune_dataset`
- `--train-split finetune_dataset/splits/train.txt`
- `--val-split finetune_dataset/splits/val.txt`
- `--image-size 320`
- `--epochs 50`
- `--batch-size 8`
- `--lr 0.002`
- `--output-dir runs/mobilenetv2_ssd`
- `--weights-backbone none | imagenet` (default: `none` to avoid downloads)

Outputs:
- Run folder under `runs/mobilenetv2_ssd/` with:
  - `checkpoint_last.pt`
  - `checkpoint_best.pt`
  - `metrics.jsonl`
  - `config.json`

## Install prerequisites
This repo does not include Torch in `requirements.txt`. Install a compatible build:

```
pip install torch torchvision
```

(Choose the correct CUDA/CPU build for your environment.)

## Example run
```
python train_mobilenetv2_ssd.py \
  --data-root finetune_dataset \
  --train-split finetune_dataset/splits/train.txt \
  --val-split finetune_dataset/splits/val.txt \
  --epochs 50 \
  --batch-size 8
```

## Implementation notes
- Train set drops images with empty/invalid label files.
- Labels with class ids outside `classes.txt` are skipped with a warning (use `--strict-labels` to fail).
- Val set keeps all images but loss is not computed (SSD only returns losses in train mode).
- If you want evaluation (mAP), add a COCO-style evaluator later.
