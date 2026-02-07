from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
except Exception as exc:  # pragma: no cover - runtime dependency
    print("Pillow is required: pip install pillow", file=sys.stderr)
    raise

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
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


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None


def _load_class_names(model_path: str) -> list[str] | None:
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        model = YOLO(model_path)
        names = model.names
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, (list, tuple)):
            return list(names)
    except Exception:
        return None
    return None


def _resolve_dirs(data_dir: Path) -> tuple[Path, Path]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if images_dir.exists() and labels_dir.exists():
        return images_dir, labels_dir
    return data_dir, data_dir


def _collect_images(images_dir: Path) -> list[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _color_for_class(class_id: int) -> str:
    b, g, r = PALETTE[class_id % len(PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


class LabelTool:
    def __init__(self, root: tk.Tk, data_dir: Path, model_path: str, save_conf: bool):
        self.root = root
        self.data_dir = data_dir
        self.images_dir, self.labels_dir = _resolve_dirs(data_dir)
        self.image_paths = _collect_images(self.images_dir)
        self.index = 0
        self.model_path = model_path
        self.save_conf = save_conf

        self.class_names = _load_class_names(model_path)
        if not self.class_names:
            self.class_names = []
            messagebox.showwarning(
                "Label Tool",
                "Could not load class names from YOLO model. "
                "Class labels will show as class_<id>.",
            )

        self.boxes: list[Box] = []
        self.selected_idx: int | None = None
        self.current_class_id = 0
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.draw_mode = False
        self.drag_start = None
        self.drag_rect_id = None

        self._build_ui()
        if not self.image_paths:
            messagebox.showinfo("Label Tool", "No images found.")
            return
        self._load_image()

    def _build_ui(self):
        self.root.title("YOLO Label Tool")
        self.root.geometry("1200x800")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self.root, padding=8)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)

        self.info_label = ttk.Label(side, text="0/0", anchor="center")
        self.info_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        nav = ttk.Frame(side)
        nav.grid(row=1, column=0, sticky="ew")
        nav.columnconfigure((0, 1), weight=1)
        ttk.Button(nav, text="Prev", command=self.prev_image).grid(row=0, column=0, sticky="ew")
        ttk.Button(nav, text="Next", command=self.next_image).grid(row=0, column=1, sticky="ew")

        jump = ttk.Frame(side)
        jump.grid(row=2, column=0, sticky="ew", pady=(8, 8))
        ttk.Label(jump, text="Go to index").grid(row=0, column=0, sticky="w")
        self.jump_entry = ttk.Entry(jump, width=8)
        self.jump_entry.grid(row=0, column=1, padx=(6, 0))
        ttk.Button(jump, text="Go", command=self.jump_to).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(side, text="Boxes").grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.listbox = tk.Listbox(side, height=16)
        self.listbox.grid(row=4, column=0, sticky="ew")
        self.listbox.bind("<<ListboxSelect>>", self.on_select_box)

        class_frame = ttk.Frame(side)
        class_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(class_frame, text="Class").grid(row=0, column=0, sticky="w")
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(
            class_frame, textvariable=self.class_var, state="readonly"
        )
        self.class_combo.grid(row=0, column=1, padx=(6, 0))
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_change)
        self._refresh_class_combo()

        action = ttk.Frame(side)
        action.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        action.columnconfigure((0, 1), weight=1)
        ttk.Button(action, text="Save", command=self.save_labels).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(action, text="Delete", command=self.delete_box).grid(
            row=0, column=1, sticky="ew"
        )

        draw = ttk.Frame(side)
        draw.grid(row=7, column=0, sticky="ew", pady=(8, 0))
        draw.columnconfigure((0, 1), weight=1)
        ttk.Button(draw, text="Draw Box", command=self.toggle_draw).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(draw, text="Clear", command=self.clear_boxes).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(
            side,
            text="Shortcuts: n/p next/prev, s save, d delete, a draw",
            wraplength=220,
        ).grid(row=8, column=0, sticky="w", pady=(10, 0))

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.root.bind("n", lambda _: self.next_image())
        self.root.bind("p", lambda _: self.prev_image())
        self.root.bind("s", lambda _: self.save_labels())
        self.root.bind("d", lambda _: self.delete_box())
        self.root.bind("a", lambda _: self.toggle_draw())
        self.root.bind("<Left>", lambda _: self.prev_image())
        self.root.bind("<Right>", lambda _: self.next_image())

    def _refresh_class_combo(self):
        if self.class_names:
            labels = [f"{i}: {name}" for i, name in enumerate(self.class_names)]
        else:
            labels = [str(i) for i in range(80)]
        self.class_combo["values"] = labels
        if labels:
            self.class_combo.current(self.current_class_id)

    def _load_image(self):
        if not (0 <= self.index < len(self.image_paths)):
            return
        image_path = self.image_paths[self.index]
        label_path = _label_path(image_path, self.labels_dir)

        self.image = Image.open(image_path).convert("RGB")
        self._load_labels(label_path)
        self._render_image()
        self._refresh_listbox()
        self._update_info()

    def _load_labels(self, label_path: Path):
        self.boxes = []
        if not label_path.exists():
            return
        try:
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        conf = float(parts[5]) if len(parts) >= 6 else None
                    except ValueError:
                        continue
                    w, h = self.image.size
                    x1 = (x_center - width / 2) * w
                    y1 = (y_center - height / 2) * h
                    x2 = (x_center + width / 2) * w
                    y2 = (y_center + height / 2) * h
                    self.boxes.append(Box(class_id, x1, y1, x2, y2, conf))
        except OSError:
            return

    def _render_image(self):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w, img_h = self.image.size
        if canvas_w < 2 or canvas_h < 2:
            canvas_w, canvas_h = img_w, img_h
        self.scale = min(canvas_w / img_w, canvas_h / img_h)
        display_w = int(img_w * self.scale)
        display_h = int(img_h * self.scale)
        display_w = max(1, display_w)
        display_h = max(1, display_h)
        self.offset_x = (canvas_w - display_w) // 2
        self.offset_y = (canvas_h - display_h) // 2

        resized = self.image.resize((display_w, display_h), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=self.photo
        )
        self._draw_boxes()

    def on_canvas_resize(self, _event=None):
        if self.image is None:
            return
        self._render_image()

    def _draw_boxes(self):
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._to_canvas(box.x1, box.y1, box.x2, box.y2)
            color = _color_for_class(box.class_id)
            width = 3 if idx == self.selected_idx else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            label = self._label_text(box.class_id, box.conf)
            self.canvas.create_text(
                x1 + 3, y1 + 3, anchor="nw", text=label, fill=color, font=("Arial", 10)
            )

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for idx, box in enumerate(self.boxes):
            label = self._label_text(box.class_id, box.conf)
            self.listbox.insert(tk.END, f"{idx + 1:02d} {label}")
        if self.selected_idx is not None and self.selected_idx < len(self.boxes):
            self.listbox.selection_set(self.selected_idx)

    def _label_text(self, class_id: int, conf: float | None) -> str:
        if self.class_names and 0 <= class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = f"class_{class_id}"
        if conf is None:
            return f"{name} ({class_id})"
        return f"{name} ({class_id}) {conf:.2f}"

    def _update_info(self):
        text = f"{self.index + 1}/{len(self.image_paths)}"
        if self.image_paths:
            text += f"  {self.image_paths[self.index].name}"
        self.info_label.configure(text=text)

    def _to_canvas(self, x1, y1, x2, y2):
        return (
            x1 * self.scale + self.offset_x,
            y1 * self.scale + self.offset_y,
            x2 * self.scale + self.offset_x,
            y2 * self.scale + self.offset_y,
        )

    def _to_image(self, x, y):
        return (
            (x - self.offset_x) / self.scale,
            (y - self.offset_y) / self.scale,
        )

    def _clamp_box(self, x1, y1, x2, y2):
        w, h = self.image.size
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        return x1, y1, x2, y2

    def on_select_box(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            self.selected_idx = None
        else:
            self.selected_idx = selection[0]
            box = self.boxes[self.selected_idx]
            if box.class_id < len(self.class_combo["values"]):
                self.class_combo.current(box.class_id)
                self.current_class_id = box.class_id
        self._render_image()

    def on_class_change(self, _event=None):
        if not self.class_combo.get():
            return
        class_id = int(self.class_combo.get().split(":")[0])
        self.current_class_id = class_id
        if self.selected_idx is not None and self.selected_idx < len(self.boxes):
            self.boxes[self.selected_idx].class_id = class_id
            self._refresh_listbox()
            self._render_image()

    def on_canvas_click(self, event):
        if self.draw_mode:
            self.drag_start = (event.x, event.y)
            self.drag_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#ffffff", width=2, dash=(4, 2)
            )
            return

        # Select box by click
        img_x, img_y = self._to_image(event.x, event.y)
        for idx in reversed(range(len(self.boxes))):
            box = self.boxes[idx]
            if box.x1 <= img_x <= box.x2 and box.y1 <= img_y <= box.y2:
                self.selected_idx = idx
                self._refresh_listbox()
                self._render_image()
                return
        self.selected_idx = None
        self._render_image()

    def on_canvas_drag(self, event):
        if not self.draw_mode or self.drag_rect_id is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.canvas.coords(self.drag_rect_id, x0, y0, event.x, event.y)

    def on_canvas_release(self, event):
        if not self.draw_mode or self.drag_rect_id is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.canvas.delete(self.drag_rect_id)
        self.drag_rect_id = None
        self.drag_start = None

        x1_img, y1_img = self._to_image(x0, y0)
        x2_img, y2_img = self._to_image(event.x, event.y)
        x1_img, x2_img = sorted([x1_img, x2_img])
        y1_img, y2_img = sorted([y1_img, y2_img])
        x1_img, y1_img, x2_img, y2_img = self._clamp_box(
            x1_img, y1_img, x2_img, y2_img
        )
        if (x2_img - x1_img) < 2 or (y2_img - y1_img) < 2:
            return
        self.boxes.append(Box(self.current_class_id, x1_img, y1_img, x2_img, y2_img))
        self.selected_idx = len(self.boxes) - 1
        self._refresh_listbox()
        self._render_image()

    def toggle_draw(self):
        self.draw_mode = not self.draw_mode
        state = "ON" if self.draw_mode else "OFF"
        self.root.title(f"YOLO Label Tool - Draw {state}")

    def save_labels(self):
        if not self.image_paths:
            return
        image_path = self.image_paths[self.index]
        label_path = _label_path(image_path, self.labels_dir)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        w, h = self.image.size
        for box in self.boxes:
            x_center = ((box.x1 + box.x2) / 2) / w
            y_center = ((box.y1 + box.y2) / 2) / h
            width = (box.x2 - box.x1) / w
            height = (box.y2 - box.y1) / h
            if self.save_conf:
                conf = box.conf if box.conf is not None else 1.0
                lines.append(
                    f"{box.class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f} {conf:.4f}"
                )
            else:
                lines.append(
                    f"{box.class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )
        label_path.write_text("\n".join(lines))

    def delete_box(self):
        if self.selected_idx is None:
            return
        if 0 <= self.selected_idx < len(self.boxes):
            self.boxes.pop(self.selected_idx)
            self.selected_idx = None
            self._refresh_listbox()
            self._render_image()

    def clear_boxes(self):
        if not self.boxes:
            return
        if not messagebox.askyesno("Label Tool", "Clear all boxes for this image?"):
            return
        self.boxes = []
        self.selected_idx = None
        self._refresh_listbox()
        self._render_image()

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.selected_idx = None
            self._load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.selected_idx = None
            self._load_image()

    def jump_to(self):
        try:
            idx = int(self.jump_entry.get()) - 1
        except ValueError:
            return
        if 0 <= idx < len(self.image_paths):
            self.index = idx
            self.selected_idx = None
            self._load_image()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple YOLO label editor.")
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory (either contains images/labels or images+txt in one folder).",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO model weights for loading official class names.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence column (default: off).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    root = tk.Tk()
    LabelTool(root, Path(args.data), args.model, args.save_conf)
    root.mainloop()


if __name__ == "__main__":
    main()
