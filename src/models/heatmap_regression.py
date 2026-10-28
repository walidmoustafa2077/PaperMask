"""Heatmap Regression Model with BiFPN Neck.

Architecture:
1. Backbone: MobileNetV3 or similar lightweight CNN
2. Neck: BiFPN (Bidirectional Feature Pyramid Network)
3. Head: 4-channel heatmap prediction + optional edge/classification heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from typing import List, Dict


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (mobile-friendly)."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """Single BiFPN block with weighted feature fusion.
    
    Args:
        channels: Number of channels for all feature levels
        num_levels: Number of feature pyramid levels
    """
    
    def __init__(self, channels: int, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = 1e-4
        
        # Learnable fusion weights
        self.w1 = nn.Parameter(torch.ones(num_levels - 1, 2))
        self.w2 = nn.Parameter(torch.ones(num_levels - 1, 3))
        
        # Top-down pathway
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(num_levels - 1)
        ])
    
    def _weighted_fusion(self, features: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Weighted feature fusion with normalization."""
        weights = F.relu(weights)
        weights = weights / (weights.sum(dim=0) + self.epsilon)
        
        fused = sum(w * f for w, f in zip(weights, features))
        return fused
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            features: List of feature maps [P3, P4, P5, P6, P7]
                     from shallow (high-res) to deep (low-res)
                     
        Returns:
            Enhanced feature maps
        """
        # Top-down pathway (deep to shallow)
        td_features = [features[-1]]  # Start from deepest
        
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample deeper feature
            deeper_feat = F.interpolate(
                td_features[0],
                size=features[i].shape[2:],
                mode='nearest'
            )
            
            # Weighted fusion
            fused = self._weighted_fusion(
                [features[i], deeper_feat],
                self.w1[i]
            )
            
            # Apply conv
            fused = self.td_convs[i](fused)
            td_features.insert(0, fused)
        
        # Bottom-up pathway (shallow to deep)
        bu_features = [td_features[0]]  # Start from shallowest
        
        for i in range(1, self.num_levels):
            # Downsample shallower feature
            shallower_feat = F.max_pool2d(bu_features[-1], kernel_size=2, stride=2)
            
            # Weighted fusion
            if i == self.num_levels - 1:
                # Last level: only td + bu
                fused = self._weighted_fusion(
                    [td_features[i], shallower_feat],
                    self.w1[i-1]
                )
            else:
                # Middle levels: original + td + bu
                fused = self._weighted_fusion(
                    [features[i], td_features[i], shallower_feat],
                    self.w2[i-1]
                )
            
            # Apply conv
            fused = self.bu_convs[i-1](fused)
            bu_features.append(fused)
        
        return bu_features


class BiFPN(nn.Module):
    """BiFPN (Bidirectional Feature Pyramid Network).
    
    Args:
        in_channels: List of input channel dimensions
        out_channels: Output channels for all levels
        num_blocks: Number of BiFPN blocks
        num_levels: Number of pyramid levels
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 64,
        num_blocks: int = 2,
        num_levels: int = 5,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Project input features to out_channels
        self.input_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for in_ch in in_channels
        ])
        
        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels, num_levels)
            for _ in range(num_blocks)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            features: List of feature maps from backbone
                     
        Returns:
            Enhanced multi-scale features
        """
        # Project to common channel dimension
        features = [conv(feat) for conv, feat in zip(self.input_convs, features)]
        
        # Apply BiFPN blocks
        for block in self.bifpn_blocks:
            features = block(features)
        
        return features


class HeatmapRegressionModel(nn.Module):
    """Heatmap Regression Model for document corner detection.
    
    Args:
        backbone: 'mobilenet_v3_small' or 'mobilenet_v3_large'
        pretrained: Use ImageNet pretrained weights
        fpn_channels: Number of channels in BiFPN
        num_bifpn_blocks: Number of BiFPN blocks
        heatmap_channels: Number of output heatmap channels (4 for corners)
        with_edge: Add edge prediction head
        with_classification: Add classification head
    """
    
    def __init__(
        self,
        backbone: str = 'mobilenet_v3_small',
        pretrained: bool = True,
        fpn_channels: int = 64,
        num_bifpn_blocks: int = 2,
        heatmap_channels: int = 4,
        with_edge: bool = False,
        with_classification: bool = False,
    ):
        super().__init__()
        self.with_edge = with_edge
        self.with_classification = with_classification
        
        # Backbone
        if backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            # Use 4 scales instead of 5 (layers [1,3,6,9] -> 64x64, 32x32, 16x16, 8x8)
            # Skip feat_12 (576ch, 8x8) to avoid duplicate spatial size
            self.feature_dims = [16, 24, 40, 96]  # Channels for [1, 3, 6, 9]
            self.feature_indices = [1, 3, 6, 9]
        elif backbone == 'mobilenet_v3_large':
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            self.feature_dims = [24, 40, 80, 160]
            self.feature_indices = [1, 3, 6, 9]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.features = self.backbone.features
        
        # Hook intermediate features
        self.feature_maps = {}
        self._register_hooks()
        
        # BiFPN Neck
        self.neck = BiFPN(
            in_channels=self.feature_dims,
            out_channels=fpn_channels,
            num_blocks=num_bifpn_blocks,
            num_levels=len(self.feature_dims),
        )
        
        # Heatmap head (use shallow features for high resolution)
        self.heatmap_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels),
            DepthwiseSeparableConv(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, heatmap_channels, 1),
        )
        
        # Edge head (optional)
        if with_edge:
            self.edge_head = nn.Sequential(
                DepthwiseSeparableConv(fpn_channels, fpn_channels),
                nn.Conv2d(fpn_channels, 1, 1),
            )
        
        # Classification head (optional)
        if with_classification:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(fpn_channels, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        # Register hooks for indices specified in __init__
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        for idx in self.feature_indices:
            self.features[idx].register_forward_hook(hook_fn(f"feat_{idx}"))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Dictionary with 'heatmaps' (B, 4, H/4, W/4) and optional outputs
        """
        # Extract features
        _ = self.features(x)
        
        # Gather multi-scale features (shallow to deep) based on feature_indices
        features = [self.feature_maps[f'feat_{idx}'] for idx in self.feature_indices]
        
        # Apply BiFPN
        features = self.neck(features)
        
        # Use shallowest feature for heatmaps (highest resolution)
        feat_for_heatmap = features[0]
        
        # Predict heatmaps
        heatmaps = self.heatmap_head(feat_for_heatmap)
        heatmaps = torch.sigmoid(heatmaps)
        
        output = {'heatmaps': heatmaps}
        
        # Predict edges (optional)
        if self.with_edge:
            edges = self.edge_head(feat_for_heatmap)
            edges = torch.sigmoid(edges)
            output['edges'] = edges
        
        # Predict classification (optional)
        if self.with_classification:
            cls_logits = self.cls_head(features[-1])  # Use deepest feature
            output['cls_logits'] = cls_logits
        
        return output


def create_heatmap_model(
    backbone: str = 'mobilenet_v3_small',
    pretrained: bool = True,
    with_edge: bool = False,
    with_classification: bool = False,
) -> HeatmapRegressionModel:
    """Factory function to create heatmap regression model."""
    return HeatmapRegressionModel(
        backbone=backbone,
        pretrained=pretrained,
        with_edge=with_edge,
        with_classification=with_classification,
    )
