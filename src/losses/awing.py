"""Adaptive Wing Loss for heatmap regression.

Reference: Wang et al., "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
https://arxiv.org/abs/1904.07399
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for heatmap regression.
    
    This loss adapts penalties for foreground (near peaks) vs background pixels,
    improving landmark localization accuracy.
    
    Args:
        alpha: Controls the shape of the loss curve
        omega: Threshold that defines the non-linear region
        epsilon: Smoothing parameter
        theta: Threshold for adaptive weighting
    """
    
    def __init__(
        self,
        alpha: float = 2.1,
        omega: float = 14.0,
        epsilon: float = 1.0,
        theta: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        
        # Pre-compute constant C
        self.C = self.omega * (1.0 - torch.log(torch.tensor(1.0 + self.omega / self.epsilon)))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute Adaptive Wing Loss.
        
        Args:
            pred: Predicted heatmaps (B, C, H, W)
            target: Target heatmaps (B, C, H, W)
            weight: Optional pixel weights (B, C, H, W)
            
        Returns:
            Loss value
        """
        delta = (target - pred).abs()
        
        # Adaptive weight based on target value
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1.0) * \
            (1.0 / self.epsilon)
        
        # Wing loss with adaptive component
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1.0 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - self.C
        )
        
        if weight is not None:
            loss = loss * weight
        
        return loss.mean()


class WeightedAdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss with automatic foreground/background weighting."""
    
    def __init__(
        self,
        alpha: float = 2.1,
        omega: float = 14.0,
        epsilon: float = 1.0,
        theta: float = 0.5,
        foreground_weight: float = 10.0,
    ):
        super().__init__()
        self.awing = AdaptiveWingLoss(alpha, omega, epsilon, theta)
        self.foreground_weight = foreground_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted Adaptive Wing Loss.
        
        Args:
            pred: Predicted heatmaps (B, C, H, W)
            target: Target heatmaps (B, C, H, W)
            
        Returns:
            Loss value
        """
        # Create weight map: higher weight for foreground (target > 0.1)
        weight = torch.ones_like(target)
        weight[target > 0.1] = self.foreground_weight
        
        return self.awing(pred, target, weight)
