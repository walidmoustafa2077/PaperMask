"""Focal Heatmap Loss - Suppresses background noise while focusing on corners.

This loss addresses the issue of background activations in heatmaps by:
1. Heavily penalizing false positive activations in background regions
2. Focusing gradient on hard-to-learn corner locations
3. Using focal weighting to handle class imbalance (few corner pixels vs many background)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalHeatmapLoss(nn.Module):
    """Focal loss for heatmap regression with background suppression.
    
    Args:
        foreground_weight: Weight for corner regions (target > threshold)
        background_penalty: Penalty multiplier for false positive background activations
        focal_alpha: Focusing parameter (higher = more focus on hard examples)
        focal_gamma: Modulating factor (higher = down-weight easy examples more)
        threshold: Threshold to distinguish foreground from background
    """
    
    def __init__(
        self,
        foreground_weight: float = 150.0,
        background_penalty: float = 0.5,
        focal_alpha: float = 2.0,
        focal_gamma: float = 2.0,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_penalty = background_penalty
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal heatmap loss.
        
        Args:
            pred: Predicted heatmaps (B, C, H, W), values in [0, 1] after sigmoid
            target: Target heatmaps (B, C, H, W), Gaussian peaks
            
        Returns:
            Loss value
        """
        # Ensure predictions are in valid range
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Compute MSE
        mse = (pred - target) ** 2
        
        # Create weight map
        weight = torch.ones_like(target)
        
        # Foreground mask: where ground truth has corner peaks
        foreground_mask = target > self.threshold
        
        # Background mask: where ground truth is near zero
        background_mask = target < self.threshold
        
        # Foreground weighting: focus on corner regions
        weight[foreground_mask] = self.foreground_weight
        
        # Background suppression: penalize false positive activations
        # Only penalize when pred > threshold in background regions
        false_positive_mask = background_mask & (pred > self.threshold)
        weight[false_positive_mask] = weight[false_positive_mask] * (1.0 + self.background_penalty * pred[false_positive_mask])
        
        # Focal weighting: focus on hard examples
        # pt = |target - pred| indicates prediction confidence
        # Easy examples (small error) get down-weighted
        pt = torch.abs(target - pred)
        focal_weight = self.focal_alpha * (pt ** self.focal_gamma)
        
        # Combined loss
        loss = mse * weight * focal_weight
        
        return loss.mean()


class HybridHeatmapLoss(nn.Module):
    """Hybrid loss combining MSE + Background Suppression + Focal weighting.
    
    This is a simpler alternative to FocalHeatmapLoss with explicit MSE.
    
    Args:
        foreground_weight: Weight for corner peaks
        background_suppression: Penalty for background activations
    """
    
    def __init__(
        self,
        foreground_weight: float = 150.0,
        background_suppression: float = 0.5,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_suppression = background_suppression
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid heatmap loss.
        
        Args:
            pred: Predicted heatmaps (B, C, H, W)
            target: Target heatmaps (B, C, H, W)
            
        Returns:
            Loss value
        """
        # Standard MSE
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Foreground weighting
        foreground_mask = target > self.threshold
        weighted_mse = torch.where(
            foreground_mask,
            mse * self.foreground_weight,
            mse
        )
        
        # Background suppression: penalize false positive activations
        background_mask = target < self.threshold
        false_positive_mask = background_mask & (pred > self.threshold)
        
        # Extra penalty for false positives in background
        fp_penalty = torch.where(
            false_positive_mask,
            (pred - target) ** 2 * self.background_suppression,
            torch.zeros_like(pred)
        )
        
        total_loss = weighted_mse.mean() + fp_penalty.mean()
        
        return total_loss
