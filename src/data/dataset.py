"""Dataset module for document corner detection."""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CornerDataset(Dataset):
    """Dataset for document corner detection.
    
    Args:
        labels_path: Path to labels.json
        images_dir: Directory containing images
        img_size: Target image size (height, width)
        heatmap_size: Size of output heatmap (for heatmap models)
        mode: 'point' or 'heatmap' regression
        augment: Whether to apply augmentations
        split: 'train' or 'val'
        split_ratio: Train/val split ratio (default 0.8)
        seed: Random seed for reproducible splits
    """
    
    def __init__(
        self,
        labels_path: str,
        images_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        heatmap_size: Optional[Tuple[int, int]] = (64, 64),
        mode: str = 'point',
        augment: bool = False,
        split: str = 'train',
        split_ratio: float = 0.8,
        seed: int = 42,
        sigma: float = 2.0,
    ):
        self.labels_path = Path(labels_path)
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.mode = mode
        self.augment = augment
        self.split = split
        self.sigma = sigma
        
        # Load labels
        with open(self.labels_path, 'r') as f:
            all_labels = json.load(f)
        
        # Create deterministic split
        filenames = sorted(all_labels.keys())
        np.random.seed(seed)
        indices = np.random.permutation(len(filenames))
        split_idx = int(len(filenames) * split_ratio)
        
        if split == 'train':
            self.filenames = [filenames[i] for i in indices[:split_idx]]
        else:
            self.filenames = [filenames[i] for i in indices[split_idx:]]
        
        self.labels = {fn: all_labels[fn] for fn in self.filenames}
        
        # Setup augmentations
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline."""
        if self.augment and self.split == 'train':
            return A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Perspective(scale=(0.05, 0.15), p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    p=0.5
                ),
                A.Resize(*self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Resize(*self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _generate_heatmap(
        self,
        corners: np.ndarray,
        size: Tuple[int, int],
        sigma: float
    ) -> np.ndarray:
        """Generate Gaussian heatmap for corners.
        
        Args:
            corners: Corner coordinates (4, 2) in normalized [0, 1] range
            size: Heatmap size (H, W)
            sigma: Gaussian sigma in heatmap pixels
            
        Returns:
            Heatmap array (4, H, W)
        """
        heatmaps = np.zeros((4, size[0], size[1]), dtype=np.float32)
        
        for i, (x, y) in enumerate(corners):
            # Convert normalized to heatmap coordinates
            cx, cy = int(x * size[1]), int(y * size[0])
            
            # Create coordinate grids
            x_grid = np.arange(size[1])
            y_grid = np.arange(size[0])
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Gaussian heatmap
            heatmap = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            heatmaps[i] = heatmap
        
        return heatmaps
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[idx]
        img_path = self.images_dir / filename
        
        # Load image
        image = cv2.imread(str(img_path))
        
        # Skip if image is missing or corrupted
        if image is None:
            print(f"Warning: Skipping {filename} - file not found or corrupted")
            # Return a blank image with corners at center as fallback
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Load corners (pixel coordinates)
        corners = np.array(self.labels[filename], dtype=np.float32)  # (4, 2)
        
        # Apply transforms
        transformed = self.transform(image=image, keypoints=corners)
        image = transformed['image']
        corners = np.array(transformed['keypoints'], dtype=np.float32)
        
        # Normalize corners to [0, 1]
        corners[:, 0] /= self.img_size[1]
        corners[:, 1] /= self.img_size[0]
        corners = np.clip(corners, 0, 1)
        
        result = {
            'image': image,
            'corners': torch.from_numpy(corners.flatten()),  # (8,)
            'filename': filename,
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.float32),
        }
        
        # Generate heatmaps for heatmap mode
        if self.mode == 'heatmap' and self.heatmap_size is not None:
            heatmaps = self._generate_heatmap(corners, self.heatmap_size, self.sigma)
            result['heatmaps'] = torch.from_numpy(heatmaps)
        
        return result


def create_dataloaders(
    labels_path: str,
    images_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    heatmap_size: Optional[Tuple[int, int]] = (64, 64),
    mode: str = 'point',
    split_ratio: float = 0.8,
    seed: int = 42,
    sigma: float = 2.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = CornerDataset(
        labels_path=labels_path,
        images_dir=images_dir,
        img_size=img_size,
        heatmap_size=heatmap_size,
        mode=mode,
        augment=True,
        split='train',
        split_ratio=split_ratio,
        seed=seed,
        sigma=sigma,
    )
    
    val_dataset = CornerDataset(
        labels_path=labels_path,
        images_dir=images_dir,
        img_size=img_size,
        heatmap_size=heatmap_size,
        mode=mode,
        augment=False,
        split='val',
        split_ratio=split_ratio,
        seed=seed,
        sigma=sigma,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
