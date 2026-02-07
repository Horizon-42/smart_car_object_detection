import argparse
from pathlib import Path
from tqdm import tqdm

from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _collect_images(image_folder, max_images=None):
    path = Path(image_folder)
    if path.is_dir():
        images = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [path] if path.suffix.lower() in IMAGE_EXTS else []
    if max_images is not None and max_images > 0:
        images = images[:max_images]
    return [str(p) for p in images]


def pre_label_images(
    image_folder,
    model_name='yolo26n.pt',
    conf_threshold=0.25,
    save_conf=True,
    max_images=None,
    batch_size=16,
    imgsz=None,
    device=None,
    half=False,
    chunk_size=100,
):
    # Load the YOLOv8 model
    model_path = Path(model_name)
    if model_path.exists():
        model = YOLO(str(model_path.resolve()))
    else:
        model = YOLO(model_name)

    pareent_folder = Path(image_folder).parent
    labels_folder = pareent_folder / 'labels'
    if not labels_folder.exists():
        labels_folder.mkdir(parents=True, exist_ok=True)

    # Run inference on the images in the specified folder (optionally limited)
    image_paths = _collect_images(image_folder, max_images=max_images)
    if not image_paths:
        return
    predict_kwargs = {
        'conf': conf_threshold,
        'stream': True,
        'verbose': False,
    }
    if batch_size is not None:
        predict_kwargs['batch'] = batch_size
    if imgsz is not None:
        predict_kwargs['imgsz'] = imgsz
    if device is not None:
        predict_kwargs['device'] = device
    if half:
        predict_kwargs['half'] = True

    total_images = len(image_paths)
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_images, desc='pre-label')
    else:
        print(f'pre-label: {total_images} images')

    processed = 0
    chunk_size = chunk_size if chunk_size and chunk_size > 0 else total_images
    for start in range(0, total_images, chunk_size):
        chunk = image_paths[start : start + chunk_size]
        results = model(chunk, **predict_kwargs)

        for result in results:
            image_path = Path(result.path)
            
            pareent_folder = image_path.parent
            label_path = labels_folder / (image_path.stem + '.txt')
            if not label_path.parent.exists():
                label_path.parent.mkdir(parents=True, exist_ok=True)

            if label_path.exists():
                label_path.unlink()
            if result.boxes is None or len(result.boxes) == 0:
                label_path.write_text('')
            else:
                result.save_txt(label_path, save_conf=save_conf)
            processed += 1
            if progress is not None:
                progress.update(1)
            elif processed % 25 == 0:
                print(f'  processed {processed}/{total_images}')

    if progress is not None:
        progress.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-label images with a YOLO model.')
    parser.add_argument(
        '--images',
        default='object_detection/data/combined_dataset_640X480/images',
        help='Image folder or single image path.',
    )
    parser.add_argument(
        '--model',
        default='yolo26m.pt',
        help='Model file path or model name (e.g. yolov8n.pt).',
    )
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size.')
    parser.add_argument('--device', default=None, help='Device, e.g. 0 or cpu.')
    parser.add_argument('--half', action='store_true', help='Use FP16 if supported.')
    parser.add_argument('--chunk-size', type=int, default=100, help='Chunk size for streaming.')
    parser.add_argument('--no-save-conf', action='store_true', help='Do not save confidences.')
    args = parser.parse_args()

    pre_label_images(
        args.images,
        model_name=args.model,
        conf_threshold=args.conf,
        save_conf=not args.no_save_conf,
        max_images=args.max_images,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        chunk_size=args.chunk_size,
    )
