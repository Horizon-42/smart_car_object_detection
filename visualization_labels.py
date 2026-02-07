import cv2
import os
from collections import Counter

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PALETTE = [
    (255, 82, 82),
    (0, 189, 214),
    (65, 179, 123),
    (255, 196, 0),
    (146, 104, 255),
    (255, 112, 67),
    (38, 198, 218),
    (156, 204, 101),
    (171, 71, 188),
    (66, 165, 245),
]


def _parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    conf = float(parts[5]) if len(parts) >= 6 else None
    return class_id, x_center, y_center, width, height, conf


def _color_for_class(class_id):
    return PALETTE[class_id % len(PALETTE)]


def _text_color_for_bg(bgr):
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _draw_text_with_bg(image, text, origin, bg_color, font_scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - text_w - 1))
    y = max(text_h + baseline + 1, min(y, image.shape[0] - 1))
    top_left = (x, y - text_h - baseline - 1)
    bottom_right = (x + text_w + 2, y + baseline + 1)
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    text_color = _text_color_for_bg(bg_color)
    cv2.putText(image, text, (x + 1, y - 1), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _format_class_name(class_id, class_names):
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f'class_{class_id}'


# visualize yolo labels
def visualize_labels(image_folder, class_names=None):
    image_files = sorted(
        f for f in os.listdir(image_folder) if f.lower().endswith(IMAGE_EXTS)
    )
    if not image_files:
        return

    index = 0
    while 0 <= index < len(image_files):
        filename = image_files[index]
        image_path = os.path.join(image_folder, filename)
        parent_folder = os.path.dirname(image_folder)
        label_path = os.path.join(parent_folder, 'labels', os.path.splitext(filename)[0] + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            index += 1
            continue

        h, w = image.shape[:2]
        thickness = max(1, int(round(min(h, w) / 400)))
        font_scale = max(0.5, min(h, w) / 1200)

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parsed = _parse_label_line(line)
                    if parsed is not None:
                        labels.append(parsed)

        class_counts = Counter()
        for class_id, x_center, y_center, width, height, conf in labels:
            class_counts[class_id] += 1
            px_center = x_center * w
            py_center = y_center * h
            box_w = width * w
            box_h = height * h

            x1 = int(round(px_center - box_w / 2))
            y1 = int(round(py_center - box_h / 2))
            x2 = int(round(px_center + box_w / 2))
            y2 = int(round(py_center + box_h / 2))

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            color = _color_for_class(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(image, (int(round(px_center)), int(round(py_center))), thickness + 1, color, -1)

            name = _format_class_name(class_id, class_names)
            if conf is None:
                label_text = f'{name} ({class_id}) {int(box_w)}x{int(box_h)}'
            else:
                label_text = f'{name} ({class_id}) {conf:.2f} {int(box_w)}x{int(box_h)}'
            _draw_text_with_bg(image, label_text, (x1, y1 - 4), color, font_scale, thickness)

        summary_lines = [
            filename,
            f'{w}x{h} | dets: {len(labels)}',
        ]
        if class_counts:
            counts_text = ', '.join(
                f'{_format_class_name(class_id, class_names)}:{count}'
                for class_id, count in sorted(class_counts.items())
            )
            summary_lines.append(counts_text)
        else:
            summary_lines.append('no labels')

        x_offset = 8
        y_offset = 20
        for line in summary_lines:
            _draw_text_with_bg(image, line, (x_offset, y_offset), (30, 30, 30), font_scale, thickness)
            y_offset += int(20 * font_scale + 10)

        _draw_text_with_bg(
            image,
            'keys: q/esc quit, n/space next, p prev',
            (x_offset, h - 10),
            (30, 30, 30),
            font_scale,
            thickness,
        )

        cv2.imshow('Image with Labels', image)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        if key in (ord('p'), ord('b')):
            index = max(0, index - 1)
            continue
        index += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize YOLO format labels on images in a folder.'
    )
    parser.add_argument(
        '--image_folder',
        required=True,
        help='Path to the folder containing images.',
    )
    parser.add_argument(
        '--class_names',
        nargs='*',
        help='Optional list of class names corresponding to class IDs.',
    )
    args = parser.parse_args()
    visualize_labels(args.image_folder, class_names=args.class_names)