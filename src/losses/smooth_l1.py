"""Smooth L1 Loss for point regression."""
import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) for point regression.
    
    More robust to outliers than L2, smoother than L1.
    
    Args:
        beta: Threshold for switching between L1 and L2 (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        scale_factor: Multiplier to amplify loss (e.g., 1000 for pixel-level sensitivity)
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.scale_factor = scale_factor
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Smooth L1 Loss.
        
        Args:
            pred: Predicted corners (B, 8) in normalized [0, 1] range
            target: Target corners (B, 8) in normalized [0, 1] range
            
        Returns:
            Loss value
        """
        # Compute difference THEN scale to amplify (critical for gradients!)
        diff = torch.abs(pred - target) * self.scale_factor
        
        # Smooth L1: L2 for small errors, L1 for large errors
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PointRegressionLoss(nn.Module):
    """Combined loss for point regression with optional classification.
    
    Args:
        corner_weight: Weight for corner regression loss
        cls_weight: Weight for classification loss (document presence)
        scale_factor: Amplification factor for corner loss
    """
    
    def __init__(
        self,
        corner_weight: float = 1000.0,
        cls_weight: float = 1.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.corner_loss = SmoothL1Loss(scale_factor=scale_factor)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.corner_weight = corner_weight
        self.cls_weight = cls_weight
    
    def forward(
        self,
        pred_corners: torch.Tensor,
        target_corners: torch.Tensor,
        pred_cls: torch.Tensor = None,
        target_cls: torch.Tensor = None,
    ) -> dict:
        """Compute combined loss.
        
        Args:
            pred_corners: Predicted corners (B, 8)
            target_corners: Target corners (B, 8)
            pred_cls: Predicted classification logits (B, 1)
            target_cls: Target classification labels (B, 1)
            
        Returns:
            Dictionary with total loss and components
        """
        corner_loss = self.corner_loss(pred_corners, target_corners)
        total_loss = self.corner_weight * corner_loss
        
        losses = {
            'loss': total_loss,
            'corner_loss': corner_loss,
        }
        
        if pred_cls is not None and target_cls is not None:
            cls_loss = self.cls_loss(pred_cls, target_cls)
            total_loss = total_loss + self.cls_weight * cls_loss
            losses['loss'] = total_loss
            losses['cls_loss'] = cls_loss
        
        return losses
