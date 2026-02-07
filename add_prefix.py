import shutil

data_dir = "object_detection/data/combined_dataset_labeled"

prefix = "l2_"
for subdir in ["images", "labels"]:
    dir_path = f"{data_dir}/{subdir}"
    for filename in shutil.os.listdir(dir_path):
        old_path = f"{dir_path}/{filename}"
        new_path = f"{dir_path}/{prefix}{filename}"
        shutil.move(old_path, new_path)