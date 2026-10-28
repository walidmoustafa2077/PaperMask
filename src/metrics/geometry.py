"""Geometry metrics for corner detection evaluation."""
import torch
import numpy as np
from typing import Tuple, List
from shapely.geometry import Polygon


class CornerMetrics:
    """Metrics for evaluating corner detection quality.
    
    Metrics:
    - Pixel MAE/RMSE: Average L1/L2 distance per corner
    - Normalized MAE/RMSE: Distance normalized by image diagonal
    - PCK (Percentage of Correct Keypoints): % corners within threshold
    - Quad IoU: Intersection over Union of predicted/GT quadrilaterals
    """
    
    def __init__(self, img_size: Tuple[int, int] = (256, 256)):
        self.img_size = img_size
        self.diagonal = np.sqrt(img_size[0]**2 + img_size[1]**2)
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.pixel_errors = []
        self.normalized_errors = []
        self.ious = []
    
    def update(
        self,
        pred_corners: torch.Tensor,
        gt_corners: torch.Tensor,
        orig_size: torch.Tensor = None,
    ):
        """Update metrics with a batch of predictions.
        
        Args:
            pred_corners: Predicted corners (B, 8) in normalized [0, 1] range
            gt_corners: Ground truth corners (B, 8) in normalized [0, 1] range
            orig_size: Original image sizes (B, 2) [H, W] for scaling
        """
        B = pred_corners.size(0)
        
        for i in range(B):
            pred = pred_corners[i].detach().cpu().numpy().reshape(4, 2)
            gt = gt_corners[i].detach().cpu().numpy().reshape(4, 2)
            
            # Scale to image size (use model input size, not original size)
            # Predictions are trained on resized images (e.g., 256x256)
            h, w = self.img_size
            
            pred_scaled = pred.copy()
            pred_scaled[:, 0] *= w
            pred_scaled[:, 1] *= h
            
            gt_scaled = gt.copy()
            gt_scaled[:, 0] *= w
            gt_scaled[:, 1] *= h
            
            # Pixel error
            pixel_error = np.sqrt(np.sum((pred_scaled - gt_scaled)**2, axis=1))
            self.pixel_errors.extend(pixel_error.tolist())
            
            # Normalized error
            diagonal = np.sqrt(h**2 + w**2)
            normalized_error = pixel_error / diagonal
            self.normalized_errors.extend(normalized_error.tolist())
            
            # Quad IoU
            try:
                pred_poly = Polygon(pred_scaled)
                gt_poly = Polygon(gt_scaled)
                
                if pred_poly.is_valid and gt_poly.is_valid:
                    intersection = pred_poly.intersection(gt_poly).area
                    union = pred_poly.union(gt_poly).area
                    iou = intersection / (union + 1e-6)
                    self.ious.append(iou)
            except:
                # Invalid polygon
                self.ious.append(0.0)
    
    def compute(self) -> dict:
        """Compute accumulated metrics.
        
        Returns:
            Dictionary of metric values
        """
        # Handle empty arrays gracefully
        if len(self.pixel_errors) == 0:
            return {
                'pixel_mae': 0.0,
                'pixel_rmse': 0.0,
                'normalized_mae': 0.0,
                'normalized_rmse': 0.0,
                'pck_3px': 0.0,
                'pck_5px': 0.0,
                'pck_10px': 0.0,
                'quad_iou': 0.0,
            }
        
        pixel_errors = np.array(self.pixel_errors)
        normalized_errors = np.array(self.normalized_errors)
        ious = np.array(self.ious) if len(self.ious) > 0 else np.array([0.0])
        
        metrics = {
            'pixel_mae': float(np.mean(pixel_errors)),
            'pixel_rmse': float(np.sqrt(np.mean(pixel_errors**2))),
            'normalized_mae': float(np.mean(normalized_errors)),
            'normalized_rmse': float(np.sqrt(np.mean(normalized_errors**2))),
            'pck_3px': float(np.mean(pixel_errors < 3)),
            'pck_5px': float(np.mean(pixel_errors < 5)),
            'pck_10px': float(np.mean(pixel_errors < 10)),
            'quad_iou': float(np.mean(ious)),
        }
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.compute()
        lines = [
            f"Pixel MAE: {metrics['pixel_mae']:.2f}",
            f"Pixel RMSE: {metrics['pixel_rmse']:.2f}",
            f"Normalized MAE: {metrics['normalized_mae']:.4f}",
            f"Normalized RMSE: {metrics['normalized_rmse']:.4f}",
            f"PCK@3px: {metrics['pck_3px']*100:.1f}%",
            f"PCK@5px: {metrics['pck_5px']*100:.1f}%",
            f"PCK@10px: {metrics['pck_10px']*100:.1f}%",
            f"Quad IoU: {metrics['quad_iou']:.4f}",
        ]
        return "\n".join(lines)


def calculate_quad_iou(
    pred_corners: np.ndarray,
    gt_corners: np.ndarray
) -> float:
    """Calculate IoU between two quadrilaterals.
    
    Args:
        pred_corners: Predicted corners (4, 2)
        gt_corners: Ground truth corners (4, 2)
        
    Returns:
        IoU value
    """
    try:
        pred_poly = Polygon(pred_corners)
        gt_poly = Polygon(gt_corners)
        
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0
        
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        
        return intersection / (union + 1e-6)
    except:
        return 0.0
