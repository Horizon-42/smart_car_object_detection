import shutil
import tqdm
import random
import os

no_obj_dir = "/home/supercomputing/studys/smart_car_object_detection/no_obj_dataset"
nbj_images = [os.path.join(no_obj_dir, f) for f in os.listdir(no_obj_dir) if f.endswith(".jpg")]
random.shuffle(nbj_images)

# copy 1000 images to original_dataset
dst_dir = "/home/supercomputing/studys/smart_car_object_detection/original_dataset"
imgs_dst_dir = os.path.join(dst_dir, "images")
labels_dst_dir = os.path.join(dst_dir, "labels")
os.makedirs(imgs_dst_dir, exist_ok=True)
os.makedirs(labels_dst_dir, exist_ok=True)

for img_path in tqdm.tqdm(nbj_images[:1000], desc="Copying no-object images"):
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(no_obj_dir, label_name)

    # Copy image
    shutil.copy2(img_path, os.path.join(imgs_dst_dir, img_name))

    # Create empty label file if it doesn't exist
    if not os.path.exists(label_path):
        open(os.path.join(labels_dst_dir, label_name), "w").close()
    else:
        raise FileNotFoundError(f"Should not have label files for no-object images, but found: {label_path}")
