# Convert class with this map
14 to 2
1 to 3
33 to 5
12 to 7 
13 to 6
18 to 8
# Image Copy and resizeing
take the ppm image, resize to 640 to 320, but keep the ratio
Create labels fit to yolo format
# Copy data to original_dataset_addition
For each class, random copy 50 images

## Script
Use `scripts/FillIJ_convert.py` to perform the conversion and sampling.

Example:
```
python3 scripts/FillIJ_convert.py \
  --src-root FullIJCNN2013 \
  --gt-file gt.txt \
  --out-dir original_dataset_addition \
  --target-size 640,320 \
  --class-map 14:2,1:3,33:5,12:7,13:6,18:8 \
  --per-class 50
```

Notes:
- Default mode expects `gt.txt` in the same folder as images (FullIJCNN2013 format).
- You can also use YOLO label folders with `--src-images` and `--src-labels`.
- Images are letterboxed to the target size; labels are adjusted accordingly.
- Output images are converted to `.jpg` and prefixed with `real_frame_` (configurable).
