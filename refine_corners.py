"""Interactive corner refinement tool.

Loads a trained PaperMask checkpoint, predicts the 4 page corners for each
image, and lets you drag them into place. Refined corners are written as
normalized [0, 1] coordinates -- the same format the dataset generator
produces -- so the output can be used directly for retraining.

The architecture always comes from ``checkpoint["config"]``. There is no
``--config`` flag on purpose: a mismatched YAML used to silently rebuild a
different model.

Usage:
    python refine_corners.py --checkpoint runs/papermask/best.pth \
        --images-dir data/nexadoc-50k/papermask/val/images --output refined_labels.json

    # Predicted labels without the GUI
    python refine_corners.py --checkpoint runs/papermask/best.pth \
        --images-dir data/nexadoc-50k/papermask/val/images --auto

Controls:
    Left-click: Select nearest corner (radius 20 px), drag to move
    n: Save and next image
    b: Back to previous image
    r: Reset to model prediction
    u: Undo last adjustment
    s: Skip image without saving
    w: Write current progress to JSON
    q: Quit (saves progress)
    ESC: Quit without saving current image
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models.paper_corner import build_model
from src.utils.corners import decode_corners
from src.utils.preprocess import letterbox, unletterbox_points

ARCH_VERSION = 2
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_checkpoint(path: str) -> Tuple[torch.nn.Module, dict]:
    """Rebuild the model described by the checkpoint and load its weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch_version") != ARCH_VERSION:
        raise SystemExit(
            f"checkpoint arch_version {checkpoint.get('arch_version')!r} is not "
            f"{ARCH_VERSION}; retrain with src/train.py"
        )
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def square_view(img: np.ndarray, img_size: int):
    """Letterbox ``img`` into ``img_size``; returns ``(view, scale, pad_x, pad_y)``.

    Letterbox, not centre-crop: cropping threw away up to half of a tall photo
    and made the model fail on exactly the photos users send. This mirrors the
    training and .NET preprocessing.
    """
    return letterbox(img, img_size)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()


def predict_corners(model, img: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Predict corners in original-image pixels.

    Returns ``(corners (4, 2), scores (4,))``.
    """
    img_size = config["data"]["img_size"]
    view, scale, pad_x, pad_y = square_view(img, img_size)

    with torch.no_grad():
        outputs = model(to_tensor(view))
        corners, scores = decode_corners(
            outputs["heatmaps"], outputs["offsets"], img_size
        )

    corners = corners[0].numpy().astype(np.float32)
    scores = scores[0].numpy().astype(np.float32)
    corners = unletterbox_points(corners, scale, pad_x, pad_y)
    return corners, scores


class CornerRefinementTool:
    """Interactive tool for refining corner predictions."""

    def __init__(
        self,
        model,
        config: dict,
        images_dir: Path,
        output_path: Path,
        window_size: Tuple[int, int] = (1200, 900),
    ):
        self.model = model
        self.config = config
        self.images_dir = Path(images_dir)
        self.output_path = Path(output_path)
        self.window_size = window_size

        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

        self.refined_labels = {}
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                self.refined_labels = json.load(f)
            print(f"Loaded {len(self.refined_labels)} existing labels")

        self.current_idx = 0
        self.current_image = None
        self.current_corners = None      # display coordinates
        self.original_corners = None     # original image pixels
        self.predicted_corners = None    # reset target, original pixels
        self.selected_corner = None
        self.dragging = False
        self.undo_stack = []
        self.display_image = None
        self.scale_factor = 1.0

        self.color_corner = (0, 255, 0)
        self.color_text = (255, 255, 255)
        self.color_line = (0, 255, 0)

        for i, img_file in enumerate(self.image_files):
            if img_file.name not in self.refined_labels:
                self.current_idx = i
                break

    def load_existing(self, name: str, shape) -> np.ndarray:
        """Stored labels are normalized; convert to original pixels."""
        h, w = shape[:2]
        pts = np.array(self.refined_labels[name], dtype=np.float32).reshape(4, 2)
        return pts * np.array([w, h], dtype=np.float32)

    def load_image(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.image_files):
            return False

        img_path = self.image_files[idx]
        self.current_idx = idx
        self.current_image = cv2.imread(str(img_path))
        if self.current_image is None:
            print(f"Error: cannot read {img_path}")
            return False

        if img_path.name in self.refined_labels:
            self.original_corners = self.load_existing(
                img_path.name, self.current_image.shape
            )
            print(f"Loaded existing corners for {img_path.name}")
        else:
            print(f"Predicting {img_path.name}...")
            self.original_corners, scores = predict_corners(
                self.model, self.current_image, self.config
            )
            print("  confidence: " + " ".join(f"{s:.3f}" for s in scores))

        self.predicted_corners = self.original_corners.copy()

        orig_h, orig_w = self.current_image.shape[:2]
        max_h, max_w = self.window_size
        self.scale_factor = min(max_h / orig_h, max_w / orig_w, 1.0)
        self.display_image = cv2.resize(
            self.current_image,
            (int(orig_w * self.scale_factor), int(orig_h * self.scale_factor)),
        )
        self.current_corners = self.original_corners * self.scale_factor

        self.undo_stack = []
        self.selected_corner = None
        return True

    def find_nearest_corner(self, x: int, y: int, threshold: int = 20) -> Optional[int]:
        if self.current_corners is None:
            return None
        distances = np.sqrt(((self.current_corners - np.array([x, y])) ** 2).sum(axis=1))
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] < threshold else None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_corner = self.find_nearest_corner(x, y)
            if self.selected_corner is not None:
                self.dragging = True
                self.undo_stack = [self.current_corners.copy()]
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_corner is not None:
            self.current_corners[self.selected_corner] = [x, y]
            self.original_corners = self.current_corners / self.scale_factor
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def draw_corners(self):
        if self.display_image is None or self.current_corners is None:
            return None

        display = self.display_image.copy()
        quad = self.current_corners.astype(np.int32)
        cv2.polylines(display, [quad], True, self.color_line, 2)

        for i, (x, y) in enumerate(self.current_corners):
            x, y = int(x), int(y)
            selected = i == self.selected_corner
            color = (0, 255, 255) if selected else self.color_corner
            radius = 9 if selected else 7
            cv2.circle(display, (x, y), radius, color, 2)
            cv2.line(display, (x - 12, y), (x + 12, y), color, 1)
            cv2.line(display, (x, y - 12), (x, y + 12), color, 1)
            cv2.circle(display, (x, y), 2, color, -1)
            cv2.putText(
                display, str(i + 1), (x + 12, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        name = self.image_files[self.current_idx].name
        status = "REFINED" if name in self.refined_labels else "NEW"
        info = f"[{self.current_idx + 1}/{len(self.image_files)}] {name} ({status})"
        cv2.putText(display, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.color_text, 2)
        cv2.putText(
            display,
            "n:Next b:Back r:Reset u:Undo s:Skip w:Write q:Quit ESC:Quit(no save)",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_text, 1,
        )
        return display

    def save_current(self):
        name = self.image_files[self.current_idx].name
        h, w = self.current_image.shape[:2]
        normalized = self.original_corners / np.array([w, h], dtype=np.float32)
        self.refined_labels[name] = np.clip(normalized, 0.0, 1.0).round(6).tolist()
        print(f"[OK] Saved {name}")

    def write_json(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.refined_labels, f, indent=2)
        print(f"[OK] Wrote {len(self.refined_labels)} labels to {self.output_path}")

    def reset_to_prediction(self):
        if self.predicted_corners is not None:
            self.original_corners = self.predicted_corners.copy()
            self.current_corners = self.original_corners * self.scale_factor
            self.undo_stack = []

    def undo(self):
        if self.undo_stack:
            self.current_corners = self.undo_stack.pop()
            self.original_corners = self.current_corners / self.scale_factor

    def run(self):
        if not self.image_files:
            print(f"No images found in {self.images_dir}")
            return
        if not self.load_image(self.current_idx):
            return

        window_name = "Corner Refinement - PaperMask"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print(f"{len(self.image_files)} images, output {self.output_path}")

        while True:
            cv2.imshow(window_name, self.draw_corners())
            key = cv2.waitKey(10) & 0xFF

            if key == ord("n"):
                self.save_current()
                self.write_json()
                if not self.load_image(self.current_idx + 1):
                    break
            elif key == ord("b"):
                self.load_image(self.current_idx - 1)
            elif key == ord("r"):
                self.reset_to_prediction()
            elif key == ord("u"):
                self.undo()
            elif key == ord("s"):
                if not self.load_image(self.current_idx + 1):
                    break
            elif key == ord("w"):
                self.write_json()
            elif key == ord("q"):
                self.save_current()
                self.write_json()
                break
            elif key == 27:
                self.write_json()
                break

        cv2.destroyAllWindows()
        print(f"[OK] {len(self.refined_labels)} labels in {self.output_path}")


def auto_generate_predictions(model, config, images_dir: Path, output_path: Path):
    tool = CornerRefinementTool(model, config, images_dir, output_path)

    for i, img_path in enumerate(tool.image_files):
        if img_path.name in tool.refined_labels:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot read {img_path.name}")
            continue
        corners, _ = predict_corners(model, img, config)
        h, w = img.shape[:2]
        tool.refined_labels[img_path.name] = np.clip(
            corners / np.array([w, h], dtype=np.float32), 0.0, 1.0
        ).round(6).tolist()
        print(f"[{i + 1}/{len(tool.image_files)}] {img_path.name}")

    tool.write_json()


def main():
    parser = argparse.ArgumentParser(description="Interactive corner refinement tool")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained checkpoint (architecture is read from it)")
    parser.add_argument("--images-dir", type=str,
                        default="data/nexadoc-50k/papermask/val/images",
                        help="Directory of images to refine")
    parser.add_argument("--output", type=str, default="refined_labels.json",
                        help="Output JSON (normalized corners, same shape as labels.json)")
    parser.add_argument("--auto", action="store_true",
                        help="Predict everything without the GUI")
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint)
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"images directory not found: {images_dir}")

    if args.auto:
        auto_generate_predictions(model, config, images_dir, Path(args.output))
    else:
        CornerRefinementTool(model, config, images_dir, Path(args.output)).run()


if __name__ == "__main__":
    main()
