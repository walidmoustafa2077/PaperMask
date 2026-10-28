"""Point Regression Model with Cross-Attention Neck.

Architecture:
1. Backbone: MobileNetV3 (lightweight, mobile-friendly)
2. Neck: Cross-Attention querying deep->shallow (1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2)
3. Head: Linear layer for 8-d corner prediction + optional classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from typing import List, Dict


class CrossAttentionNeck(nn.Module):
    """Cross-Attention neck querying from deep to shallow features.
    
    Args:
        feature_dims: List of feature dimensions for each scale
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.num_scales = len(feature_dims)
        
        # Project each scale to embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        # Cross-attention layers (query current scale, attend to previous scales)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(self.num_scales - 1)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(self.num_scales - 1)
        ])
        
        # FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_scales - 1)
        ])
        
        # Positional encodings (learnable)
        self.pos_encodings = nn.ParameterList([
            nn.Parameter(torch.randn(1, embed_dim, 1, 1))
            for _ in range(self.num_scales)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: List of feature maps from deep to shallow
                     [1/32, 1/16, 1/8, 1/4, 1/2] scales
                     
        Returns:
            Enhanced features (B, embed_dim)
        """
        B = features[0].size(0)
        
        # Project all features to embed_dim
        projected = []
        for i, (feat, proj, pos) in enumerate(zip(features, self.projections, self.pos_encodings)):
            feat = proj(feat) + pos
            projected.append(feat)
        
        # Start from deepest scale
        query_feat = projected[0]  # (B, C, H, W)
        
        # Sequentially attend from deep to shallow
        for i in range(1, self.num_scales):
            # Flatten spatial dimensions for attention
            B, C, H, W = query_feat.shape
            query = query_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # Key/value from current scale
            key_val_feat = projected[i]
            _, _, H_kv, W_kv = key_val_feat.shape
            key_val = key_val_feat.flatten(2).transpose(1, 2)  # (B, H_kv*W_kv, C)
            
            # Cross-attention
            attn_out, _ = self.cross_attns[i-1](query, key_val, key_val)
            
            # Residual + norm
            query = self.norms[i-1](query + attn_out)
            
            # FFN
            query = query + self.ffns[i-1](query)
            
            # Reshape back to spatial
            query_feat = query.transpose(1, 2).reshape(B, C, H, W)
            
            # Upsample to next scale's resolution (if not last)
            # Use bilinear upsampling (learned conv would add params)
            if i < self.num_scales - 1:
                query_feat = F.interpolate(
                    query_feat,
                    size=projected[i+1].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
        
        # Return full spatial features instead of pooling
        # Shape: (B, embed_dim, H, W) where H,W = final scale resolution
        return query_feat


class PointRegressionModel(nn.Module):
    """Point Regression Model for document corner detection.
    
    Args:
        backbone: 'mobilenet_v3_small' or 'mobilenet_v3_large'
        pretrained: Use ImageNet pretrained weights
        embed_dim: Embedding dimension for cross-attention
        num_heads: Number of attention heads
        with_classification: Add classification head for document presence
    """
    
    def __init__(
        self,
        backbone: str = 'mobilenet_v3_small',
        pretrained: bool = True,
        embed_dim: int = 256,
        num_heads: int = 4,
        with_classification: bool = False,
    ):
        super().__init__()
        self.with_classification = with_classification
        
        # Backbone
        if backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            # Feature dimensions at different scales (from deep to shallow)
            # Actual channels from layers [12, 9, 6, 3, 1]: [576, 96, 40, 24, 16]
            self.feature_dims = [576, 96, 40, 24, 16]
        elif backbone == 'mobilenet_v3_large':
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            # MobileNetV3-Large: [960, 160, 80, 40, 24]
            self.feature_dims = [960, 160, 80, 40, 24]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract feature extractor (remove classifier)
        self.features = self.backbone.features
        
        # Hook intermediate features
        self.feature_maps = {}
        self._register_hooks()
        
        # Cross-Attention Neck
        self.neck = CrossAttentionNeck(
            feature_dims=self.feature_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        # Regression head - spatial-aware design
        self.corner_head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Pool only at the very end
            nn.Flatten(),
            nn.Linear(64, 8),  # 4 corners * 2 coords
        )
        
        # Initialize final linear layer to output values around 0
        # sigmoid(0) = 0.5, so model starts predicting center of image
        nn.init.xavier_uniform_(self.corner_head[-1].weight, gain=0.1)
        nn.init.constant_(self.corner_head[-1].bias, 0.0)  # sigmoid(0) = 0.5
        
        # Classification head (optional)
        if with_classification:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 1),  # Binary classification
            )
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        # MobileNetV3 has 13 layers (0-12). Use [1, 3, 6, 9, 12] for multi-scale
        indices = [1, 3, 6, 9, 12]
        
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        for idx in indices:
            self.features[idx].register_forward_hook(hook_fn(f"feat_{idx}"))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Dictionary with 'corners' (B, 8) and optionally 'cls_logits' (B, 1)
        """
        # Extract features
        _ = self.features(x)
        
        # Gather multi-scale features (deep to shallow)
        features = [
            self.feature_maps['feat_12'],  # 1/32 (deepest)
            self.feature_maps['feat_9'],   # 1/16
            self.feature_maps['feat_6'],   # 1/8
            self.feature_maps['feat_3'],   # 1/4
            self.feature_maps['feat_1'],   # 1/2 (shallowest)
        ]
        
        # Apply neck
        feat = self.neck(features)
        
        # Predict corners (normalized [0, 1]) using sigmoid
        # Sigmoid has gradients everywhere, unlike clamp which has zero gradient outside [0,1]
        corners = torch.sigmoid(self.corner_head(feat))
        
        output = {'corners': corners}
        
        # Predict classification (optional)
        if self.with_classification:
            cls_logits = self.cls_head(feat)
            output['cls_logits'] = cls_logits
        
        return output


def create_point_model(
    backbone: str = 'mobilenet_v3_small',
    pretrained: bool = True,
    with_classification: bool = False,
) -> PointRegressionModel:
    """Factory function to create point regression model."""
    return PointRegressionModel(
        backbone=backbone,
        pretrained=pretrained,
        with_classification=with_classification,
    )
