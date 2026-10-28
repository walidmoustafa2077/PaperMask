"""Interactive corner refinement tool with model predictions.

Loads trained heatmap model, predicts corners for each image, and provides
an interactive OpenCV GUI for manual correction. Saves refined annotations
to a new JSON file for retraining.

Features:
- Auto-predict corners using trained model
- Drag-and-drop corner adjustment
- Navigation: next/back/skip
- Undo/reset functionality
- Resume from existing refined_labels.json
- Optional ONNX export

Usage:
    # Interactive refinement
    python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth
    
    # Auto-generate predictions only (no GUI)
    python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth --auto
    
    # Export ONNX while running
    python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth --export-onnx exports/heatmap.onnx

Controls:
    Left-click: Select nearest corner (radius 20px)
    Drag: Move selected corner
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
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.heatmap_regression import create_heatmap_model
from src.utils.postprocess import heatmap_to_points


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
        
        # Get all images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                   list(self.images_dir.glob('*.png')))
        
        # Load existing refined labels if available
        self.refined_labels = {}
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                self.refined_labels = json.load(f)
            print(f"Loaded {len(self.refined_labels)} existing refined labels")
        
        # State
        self.current_idx = 0
        self.current_image = None
        self.current_corners = None  # In display coordinates
        self.original_corners = None  # In original image coordinates
        self.predicted_corners = None  # Backup for reset
        self.selected_corner = None
        self.dragging = False
        self.undo_stack = []  # Single-level undo
        self.display_image = None
        self.scale_factor = 1.0
        
        # Colors
        self.color_corner = (0, 255, 0)  # Green
        self.color_selected = (0, 255, 255)  # Yellow
        self.color_line = (0, 255, 0)  # Green
        self.color_text = (255, 255, 255)  # White
        self.color_target = (255, 0, 255)  # Magenta for target crosshair
        
        # Find first unprocessed image
        for i, img_file in enumerate(self.image_files):
            if img_file.name not in self.refined_labels:
                self.current_idx = i
                break
        
        if self.current_idx == 0 and self.image_files[0].name in self.refined_labels:
            print("All images already processed! Use 'b' to review.")
    
    def predict_corners(self, image_path: Path) -> np.ndarray:
        """Predict corners using the model.
        
        Returns:
            corners: (4, 2) array in original image pixel coordinates
        """
        # Load image
        img = cv2.imread(str(image_path))
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess (same as training)
        img_resized = cv2.resize(img, tuple(self.config['img_size']))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            heatmaps = outputs['heatmaps'].numpy()
        
        # Convert heatmaps to corners
        corners_heatmap = heatmap_to_points(
            heatmaps,
            method=self.config.get('postprocess_method', 'argmax'),
            threshold=self.config.get('postprocess_threshold', 0.3),
        )
        
        if corners_heatmap.ndim == 3:
            corners_heatmap = corners_heatmap[0]  # (4, 2)
        
        # Scale to original image size
        heatmap_h, heatmap_w = self.config['heatmap_size']
        corners_normalized = corners_heatmap.astype(np.float32)
        corners_normalized[:, 0] /= heatmap_w
        corners_normalized[:, 1] /= heatmap_h
        
        corners_original = corners_normalized.copy()
        corners_original[:, 0] *= orig_w
        corners_original[:, 1] *= orig_h
        
        return corners_original
    
    def load_image(self, idx: int) -> bool:
        """Load image and prepare for display."""
        if idx < 0 or idx >= len(self.image_files):
            return False
        
        self.current_idx = idx
        img_path = self.image_files[idx]
        
        # Load image
        self.current_image = cv2.imread(str(img_path))
        if self.current_image is None:
            print(f"Error loading {img_path}")
            return False
        
        orig_h, orig_w = self.current_image.shape[:2]
        
        # Get corners (from refined labels or predict)
        if img_path.name in self.refined_labels:
            # Load existing refined labels
            self.original_corners = np.array(self.refined_labels[img_path.name], dtype=np.float32)
            print(f"Loaded existing refinement for {img_path.name}")
        else:
            # Predict corners
            print(f"Predicting corners for {img_path.name}...")
            self.original_corners = self.predict_corners(img_path)
        
        # Backup for reset
        self.predicted_corners = self.original_corners.copy()
        
        # Calculate display scale
        max_h, max_w = self.window_size
        scale_h = max_h / orig_h
        scale_w = max_w / orig_w
        self.scale_factor = min(scale_h, scale_w, 1.0)
        
        # Resize for display
        display_h = int(orig_h * self.scale_factor)
        display_w = int(orig_w * self.scale_factor)
        self.display_image = cv2.resize(self.current_image, (display_w, display_h))
        
        # Scale corners to display coordinates
        self.current_corners = self.original_corners * self.scale_factor
        
        # Clear undo stack
        self.undo_stack = []
        self.selected_corner = None
        
        return True
    
    def find_nearest_corner(self, x: int, y: int, threshold: int = 20) -> Optional[int]:
        """Find nearest corner within threshold distance."""
        if self.current_corners is None:
            return None
        
        distances = np.sqrt(((self.current_corners - np.array([x, y])) ** 2).sum(axis=1))
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < threshold:
            return min_idx
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Select nearest corner
            self.selected_corner = self.find_nearest_corner(x, y)
            if self.selected_corner is not None:
                self.dragging = True
                # Save for undo
                self.undo_stack = [self.current_corners.copy()]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag selected corner
            if self.dragging and self.selected_corner is not None:
                self.current_corners[self.selected_corner] = [x, y]
                # Update original coordinates
                self.original_corners = self.current_corners / self.scale_factor
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Release drag
            self.dragging = False
    
    def draw_corners(self):
        """Draw corners and quadrilateral on display image."""
        if self.display_image is None or self.current_corners is None:
            return
        
        # Create fresh copy
        display = self.display_image.copy().astype(np.float32)
        overlay = display.copy()
        
        # Draw quadrilateral on overlay (will be semi-transparent)
        quad = self.current_corners.astype(np.int32)
        cv2.polylines(overlay, [quad], True, self.color_line, 3)
        
        # Blend overlay for transparency
        alpha = 0.3
        display = cv2.addWeighted(display, 1 - alpha, overlay, alpha, 0)
        
        # Draw corners with improved alignment aids
        for i, (x, y) in enumerate(self.current_corners):
            x, y = int(x), int(y)
            is_selected = (i == self.selected_corner)
            
            # Choose colors based on selection
            if is_selected:
                circle_color = (0, 255, 255)  # Bright yellow
                crosshair_color = (0, 255, 255)
                circle_size = 12
                circle_thickness = 3
                text_color = (0, 255, 255)
            else:
                circle_color = (0, 200, 0)  # Green
                crosshair_color = (0, 200, 0)
                circle_size = 10
                circle_thickness = 2
                text_color = (255, 255, 255)
            
            # Draw large semi-transparent circle
            circle_overlay = display.copy()
            cv2.circle(circle_overlay, (x, y), circle_size + 3, circle_color, -1)
            display = cv2.addWeighted(display, 0.7, circle_overlay, 0.3, 0)
            
            # Draw solid outline circle
            cv2.circle(display, (x, y), circle_size, circle_color, circle_thickness)
            
            # Draw crosshair for precise alignment
            crosshair_size = 15
            line_thickness = 2
            # Vertical line
            cv2.line(display, (x, y - crosshair_size), (x, y + crosshair_size), 
                    crosshair_color, line_thickness)
            # Horizontal line
            cv2.line(display, (x - crosshair_size, y), (x + crosshair_size, y), 
                    crosshair_color, line_thickness)
            
            # Draw small center dot for precise targeting
            cv2.circle(display, (x, y), 3, (255, 255, 0), -1)
            
            # Draw corner number with background
            label = f'{i+1}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Background for text
            text_x = x - 15
            text_y = y - 20
            cv2.rectangle(display, (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(display, label, (text_x, text_y),
                       font, font_scale, text_color, thickness)
        
        # Draw info text
        img_name = self.image_files[self.current_idx].name
        status = "REFINED" if img_name in self.refined_labels else "NEW"
        info_text = f"[{self.current_idx+1}/{len(self.image_files)}] {img_name} ({status})"
        cv2.putText(display, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
        
        # Draw controls
        controls = [
            "n:Next | b:Back | r:Reset | u:Undo | s:Skip | w:Write | q:Quit | ESC:Quit(no save)"
        ]
        y_pos = display.shape[0] - 20
        for text in controls:
            cv2.putText(display, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_text, 1)
            y_pos += 20
        
        return display.astype(np.uint8)
    
    def save_current(self):
        """Save current corners to refined labels."""
        img_name = self.image_files[self.current_idx].name
        self.refined_labels[img_name] = self.original_corners.tolist()
        print(f"[OK] Saved {img_name}")
    
    def write_json(self):
        """Write refined labels to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.refined_labels, f, indent=2)
        print(f"[OK] Wrote {len(self.refined_labels)} labels to {self.output_path}")
    
    def reset_to_prediction(self):
        """Reset current corners to model prediction."""
        if self.predicted_corners is not None:
            self.original_corners = self.predicted_corners.copy()
            self.current_corners = self.original_corners * self.scale_factor
            self.undo_stack = []
            print("Reset to model prediction")
    
    def undo(self):
        """Undo last corner adjustment."""
        if self.undo_stack:
            self.current_corners = self.undo_stack.pop()
            self.original_corners = self.current_corners / self.scale_factor
            print("Undo")
    
    def run(self):
        """Run interactive refinement loop."""
        if not self.image_files:
            print("No images found in directory!")
            return
        
        # Load first image
        if not self.load_image(self.current_idx):
            print("Failed to load first image")
            return
        
        # Create window
        window_name = 'Corner Refinement - PaperMask'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("INTERACTIVE CORNER REFINEMENT")
        print("="*60)
        print(f"Images: {len(self.image_files)}")
        print(f"Already refined: {len(self.refined_labels)}")
        print(f"Output: {self.output_path}")
        print("\nControls:")
        print("  Left-click: Select corner (drag to move)")
        print("  n: Save and next")
        print("  b: Back to previous")
        print("  r: Reset to model prediction")
        print("  u: Undo last adjustment")
        print("  s: Skip without saving")
        print("  w: Write progress to JSON")
        print("  q: Quit (saves progress)")
        print("  ESC: Quit without saving current")
        print("="*60 + "\n")
        
        # Main loop
        while True:
            # Draw
            display = self.draw_corners()
            cv2.imshow(window_name, display)
            
            # Handle keys
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('n'):  # Next
                self.save_current()
                self.write_json()
                if not self.load_image(self.current_idx + 1):
                    print("Reached end of images")
                    break
            
            elif key == ord('b'):  # Back
                if not self.load_image(self.current_idx - 1):
                    print("Already at first image")
            
            elif key == ord('r'):  # Reset
                self.reset_to_prediction()
            
            elif key == ord('u'):  # Undo
                self.undo()
            
            elif key == ord('s'):  # Skip
                print(f"Skipped {self.image_files[self.current_idx].name}")
                if not self.load_image(self.current_idx + 1):
                    print("Reached end of images")
                    break
            
            elif key == ord('w'):  # Write
                self.write_json()
            
            elif key == ord('q'):  # Quit
                self.save_current()
                self.write_json()
                print("Quit - progress saved")
                break
            
            elif key == 27:  # ESC
                self.write_json()  # Save previous progress
                print("Quit without saving current image")
                break
        
        cv2.destroyAllWindows()
        print(f"\n[OK] Refinement complete: {len(self.refined_labels)} images labeled")
        print(f"[OK] Saved to: {self.output_path}")


def auto_generate_predictions(
    model,
    config: dict,
    images_dir: Path,
    output_path: Path,
):
    """Auto-generate predictions without GUI."""
    tool = CornerRefinementTool(model, config, images_dir, output_path)
    
    print("\n" + "="*60)
    print("AUTO-GENERATING PREDICTIONS")
    print("="*60)
    print(f"Images: {len(tool.image_files)}")
    print(f"Output: {output_path}")
    print()
    
    for i, img_path in enumerate(tool.image_files):
        # Skip if already processed
        if img_path.name in tool.refined_labels:
            print(f"[{i+1}/{len(tool.image_files)}] Skipping {img_path.name} (already exists)")
            continue
        
        print(f"[{i+1}/{len(tool.image_files)}] Predicting {img_path.name}...", end=' ')
        corners = tool.predict_corners(img_path)
        tool.refined_labels[img_path.name] = corners.tolist()
        print("[OK]")
    
    # Save
    tool.write_json()
    print(f"\n[OK] Generated predictions for {len(tool.refined_labels)} images")


def export_onnx(model, config: dict, output_path: Path):
    """Export model to ONNX format."""
    print(f"\nExporting model to ONNX: {output_path}")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, *config['img_size'])
    
    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['heatmaps'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'}
        }
    )
    
    print(f"[OK] ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Interactive corner refinement tool')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/heatmap_base.yaml',
                       help='Config file used for training')
    parser.add_argument('--images-dir', type=str, default='dataset',
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='phase2_labels.json',
                       help='Output JSON file for refined labels')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-generate predictions without GUI')
    parser.add_argument('--export-onnx', type=str,
                       help='Export model to ONNX (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    model = create_heatmap_model(
        backbone=config['backbone'],
        pretrained=False,
        with_classification=config.get('with_classification', False),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val MAE: {checkpoint['val_metrics']['pixel_mae']:.2f}px")
    print(f"  Val IoU: {checkpoint['val_metrics']['quad_iou']:.4f}")
    
    # Export ONNX if requested
    if args.export_onnx:
        export_onnx(model, config, Path(args.export_onnx))
    
    # Run refinement
    if args.auto:
        auto_generate_predictions(model, config, Path(args.images_dir), Path(args.output))
    else:
        tool = CornerRefinementTool(
            model=model,
            config=config,
            images_dir=Path(args.images_dir),
            output_path=Path(args.output),
        )
        tool.run()


if __name__ == '__main__':
    main()
