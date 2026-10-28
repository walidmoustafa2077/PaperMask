"""Postprocessing utilities for heatmap and corner predictions."""
import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional


def heatmap_to_points(
    heatmaps: np.ndarray,
    method: str = 'centroid',
    threshold: float = 0.3,
    sigma: float = 2.0,
) -> np.ndarray:
    """Convert heatmaps to corner points.
    
    Args:
        heatmaps: Heatmaps (4, H, W) or (B, 4, H, W)
        method: 'argmax', 'gaussian', or 'centroid'
        threshold: Minimum peak value for centroid method
        sigma: Gaussian sigma for refinement
        
    Returns:
        Corner points (4, 2) or (B, 4, 2) in heatmap coordinates
    """
    single_image = heatmaps.ndim == 3
    if single_image:
        heatmaps = heatmaps[np.newaxis]  # (1, 4, H, W)
    
    B, C, H, W = heatmaps.shape
    corners = np.zeros((B, C, 2), dtype=np.float32)
    
    for b in range(B):
        for c in range(C):
            heatmap = heatmaps[b, c]
            
            if method == 'argmax':
                # Simple argmax
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                corners[b, c] = [x, y]
            
            elif method == 'gaussian':
                # Argmax + Gaussian refinement
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # 3x3 neighborhood
                y_min, y_max = max(0, y-1), min(H, y+2)
                x_min, x_max = max(0, x-1), min(W, x+2)
                
                window = heatmap[y_min:y_max, x_min:x_max]
                if window.sum() > 0:
                    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                    x_refined = (xx * window).sum() / window.sum()
                    y_refined = (yy * window).sum() / window.sum()
                    corners[b, c] = [x_refined, y_refined]
                else:
                    corners[b, c] = [x, y]
            
            elif method == 'centroid':
                # Contour centroid
                # Threshold heatmap
                binary = (heatmap > threshold).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Use largest contour
                    contour = max(contours, key=cv2.contourArea)
                    
                    # Calculate moments
                    M = cv2.moments(contour)
                    
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        corners[b, c] = [cx, cy]
                    else:
                        # Fallback to argmax
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        corners[b, c] = [x, y]
                else:
                    # Fallback to argmax
                    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    corners[b, c] = [x, y]
            
            else:
                raise ValueError(f"Unknown method: {method}")
    
    if single_image:
        return corners[0]  # (4, 2)
    return corners  # (B, 4, 2)


def normalize_corner_order(corners: np.ndarray) -> np.ndarray:
    """Normalize corner order to canonical sequence.
    
    Canonical order: top-left, top-right, bottom-right, bottom-left (clockwise from top-left)
    
    Args:
        corners: Corner points (4, 2) [x, y]
        
    Returns:
        Reordered corners (4, 2)
    """
    # Sort by y-coordinate
    sorted_idx = np.argsort(corners[:, 1])
    
    # Top two corners
    top_corners = corners[sorted_idx[:2]]
    top_sorted = top_corners[np.argsort(top_corners[:, 0])]  # Left to right
    
    # Bottom two corners
    bottom_corners = corners[sorted_idx[2:]]
    bottom_sorted = bottom_corners[np.argsort(bottom_corners[:, 0])]  # Left to right
    
    # Canonical order: TL, TR, BR, BL
    ordered = np.array([
        top_sorted[0],     # Top-left
        top_sorted[1],     # Top-right
        bottom_sorted[1],  # Bottom-right
        bottom_sorted[0],  # Bottom-left
    ])
    
    return ordered


def scale_corners(
    corners: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> np.ndarray:
    """Scale corner coordinates between different image sizes.
    
    Args:
        corners: Corner points (4, 2) or (B, 4, 2)
        from_size: Source size (H, W)
        to_size: Target size (H, W)
        
    Returns:
        Scaled corners
    """
    scale_x = to_size[1] / from_size[1]
    scale_y = to_size[0] / from_size[0]
    
    corners_scaled = corners.copy()
    corners_scaled[..., 0] *= scale_x
    corners_scaled[..., 1] *= scale_y
    
    return corners_scaled


def denormalize_corners(
    corners: torch.Tensor,
    img_size: Tuple[int, int],
) -> np.ndarray:
    """Convert normalized [0, 1] corners to pixel coordinates.
    
    Args:
        corners: Normalized corners (B, 8) or (8,)
        img_size: Image size (H, W)
        
    Returns:
        Pixel coordinates (B, 4, 2) or (4, 2)
    """
    corners_np = corners.detach().cpu().numpy()
    single_sample = corners_np.ndim == 1
    
    if single_sample:
        corners_np = corners_np.reshape(4, 2)
    else:
        corners_np = corners_np.reshape(-1, 4, 2)
    
    # Scale to pixels
    corners_np[..., 0] *= img_size[1]
    corners_np[..., 1] *= img_size[0]
    
    if single_sample:
        return corners_np
    return corners_np


def is_convex_quad(corners: np.ndarray) -> bool:
    """Check if quadrilateral is convex.
    
    Args:
        corners: Corner points (4, 2)
        
    Returns:
        True if convex
    """
    def cross_product_sign(p1, p2, p3):
        return np.sign((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    
    signs = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        signs.append(cross_product_sign(p1, p2, p3))
    
    # All cross products should have same sign for convex polygon
    return all(s == signs[0] for s in signs) or all(s == -signs[0] for s in signs)
