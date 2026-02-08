import argparse
import sys
from pathlib import Path

import cv2

from trt_detector import TrtYoloDetector, collect_images, parse_imgsz


def _run_on_images(
    detector,
    image_paths,
    save_dir=None,
    show=False,
    measure_time=False,
    max_images=None,
):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        detections = detector.predict(image, measure_time=measure_time)
        if measure_time and detector.last_inference_time is not None:
            inf_ms = detector.last_inference_time * 1000.0
            fps = (
                1.0 / detector.last_inference_time
                if detector.last_inference_time > 0
                else float("inf")
            )
            print(f"{img_path.name}: inference {inf_ms:.2f} ms | fps {fps:.2f}")

        vis = detector.visualize(image, detections)

        if save_dir is not None:
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), vis)
        if show:
            cv2.imshow("trt detections", vis)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break

        processed += 1
        if max_images is not None and processed >= max_images:
            break

    if show:
        cv2.destroyAllWindows()


def _run_camera(detector, num_frames, width, height, fps, measure_time=False):
    try:
        from jetcam.csi_camera import CSICamera
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "jetcam is required for --camera (pip install jetcam)."
        ) from exc

    camera = CSICamera(width=width, height=height, fps=fps)
    try:
        for _ in range(num_frames):
            frame = camera.read()
            if frame is None:
                continue
            detections = detector.predict(frame, measure_time=measure_time)
            if measure_time and detector.last_inference_time is not None:
                inf_ms = detector.last_inference_time * 1000.0
                fps_val = (
                    1.0 / detector.last_inference_time
                    if detector.last_inference_time > 0
                    else float("inf")
                )
                print(f"Inference: {inf_ms:.2f} ms | FPS: {fps_val:.2f}")

            vis = detector.visualize(frame, detections)
            cv2.imshow("trt camera", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if hasattr(camera, "stop"):
            camera.stop()
        if hasattr(camera, "release"):
            camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run TensorRT engine YOLO inference on images."
    )
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine.")
    parser.add_argument("--image", help="Single image path.")
    parser.add_argument("--folder", help="Folder of images.")
    parser.add_argument(
        "--classes",
        default=None,
        help="Path to classes.txt (optional).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--imgsz",
        default=None,
        help="Input size override (e.g. 640 or 640,480).",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/trt_detect_demo",
        help="Directory to save visualized results.",
    )
    parser.add_argument("--show", action="store_true", help="Show images in a window.")
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs.")
    parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Print inference time and FPS for each image.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images processed from a folder.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Run live inference from CSI camera (JetCam).",
    )
    parser.add_argument("--camera-frames", type=int, default=50, help="Camera frames.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS.")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id.")

    args = parser.parse_args()

    if not args.image and not args.folder and not args.camera:
        print("Provide --image, --folder, or --camera.", file=sys.stderr)
        sys.exit(1)

    imgsz = parse_imgsz(args.imgsz)
    detector = TrtYoloDetector(
        args.engine,
        class_names=args.classes,
        conf_thres=args.conf,
        iou_thres=args.iou,
        input_size=imgsz,
        device_id=args.device_id,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        save_dir = None if args.no_save else args.save_dir
        _run_on_images(
            detector,
            [image_path],
            save_dir=save_dir,
            show=args.show,
            measure_time=args.measure_time,
        )

    if args.folder:
        images = collect_images(args.folder)
        if not images:
            raise FileNotFoundError(f"No images found at {args.folder}")
        save_dir = None if args.no_save else args.save_dir
        _run_on_images(
            detector,
            images,
            save_dir=save_dir,
            show=args.show,
            measure_time=args.measure_time,
            max_images=args.max_images,
        )

    if args.camera:
        _run_camera(
            detector,
            num_frames=args.camera_frames,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            measure_time=args.measure_time,
        )


if __name__ == "__main__":
    main()
