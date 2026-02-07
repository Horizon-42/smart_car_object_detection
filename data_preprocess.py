import os
import shutil
import cv2
from tqdm import tqdm

dir1 = "object_detection/data/dataset_object_detection/20260206_060944/images"
dir2 = "object_detection/data/dataset_object_detection/20260206_061544/images"

ims1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.jpg')]
ims2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith('.jpg')]

dataset_dir = "object_detection/data/combined_dataset/images"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

for im in tqdm(ims1):
    # ignore non-jpg files
    if not im.endswith('.jpg'):
        continue
    dst_filename = os.path.join(dataset_dir, "collect1_"+os.path.basename(im))
    shutil.copy(im, dst_filename)
    # cv2.imshow("Image", cv2.imread(im))
    # cv2.waitKey(1)

for im in tqdm(ims2):
    if not im.endswith('.jpg'):
        continue
    dst_filename = os.path.join(dataset_dir, "collect2_"+os.path.basename(im))
    shutil.copy(im, dst_filename)
    # cv2.imshow("Image", cv2.imread(im))
    # cv2.waitKey(1)
cv2.destroyAllWindows()