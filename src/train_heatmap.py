"""Training script for Heatmap Regression Model."""
import argparse
import yaml
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.heatmap_regression import create_heatmap_model
from src.losses.awing import WeightedAdaptiveWingLoss
from src.losses.heatmap_focal import FocalHeatmapLoss, HybridHeatmapLoss
import torch.nn.functional as F
from src.metrics.geometry import CornerMetrics
from src.utils.postprocess import heatmap_to_points, denormalize_corners
import cv2


def visualize_predictions(model, loader, device, output_dir, epoch, config, images_dir='dataset'):
    """Visualize heatmap predictions vs ground truth."""
    model.eval()
    vis_dir = output_dir / 'visualizations' / f'epoch_{epoch:03d}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first batch
    batch = next(iter(loader))
    images = batch['image'].to(device)
    corners_gt = batch['corners'].to(device)
    filenames = batch['filename']
    
    # Predict
    with torch.no_grad():
        outputs = model(images)
        pred_heatmaps = outputs['heatmaps']
    
    # Convert heatmaps to corners
    heatmaps_np = pred_heatmaps.cpu().numpy()
    corners_pred = heatmap_to_points(
        heatmaps_np,
        method=config.get('postprocess_method', 'centroid'),
        threshold=config.get('postprocess_threshold', 0.3),
    )
    
    # Normalize corners
    heatmap_h, heatmap_w = heatmaps_np.shape[2:]
    corners_pred[:, :, 0] /= heatmap_w
    corners_pred[:, :, 1] /= heatmap_h
    corners_pred = torch.from_numpy(corners_pred.reshape(-1, 8))
    
    # Denormalize to 256x256 for visualization
    img_size = 256
    
    for i in range(min(14, len(images))):  # Save first 14 images
        # Get original image
        img_path = Path(images_dir) / filenames[i]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Resize to 256x256
        img_resized = cv2.resize(img, (img_size, img_size))
        
        # Get corners in pixel coordinates
        gt_pts = (corners_gt[i].cpu().numpy().reshape(4, 2) * img_size).astype(np.int32)
        pred_pts = (corners_pred[i].numpy().reshape(4, 2) * img_size).astype(np.int32)
        
        # Create side-by-side comparison: GT | Heatmaps | Prediction | Overlay
        canvas = np.zeros((img_size, img_size * 4, 3), dtype=np.uint8)
        
        # Panel 1: Ground Truth (green)
        gt_img = img_resized.copy()
        for j, (x, y) in enumerate(gt_pts):
            cv2.circle(gt_img, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(gt_img, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(gt_img, f'{j+1}', (x-5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        gt_poly = gt_pts
        cv2.polylines(gt_img, [gt_poly], True, (0, 255, 0), 2)
        cv2.putText(gt_img, 'Ground Truth', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        canvas[:, :img_size] = gt_img
        
        # Panel 2: Heatmaps visualization
        heatmap_viz = img_resized.copy().astype(np.float32)
        for j in range(4):
            hmap = cv2.resize(heatmaps_np[i, j], (img_size, img_size))
            hmap = (hmap * 255).astype(np.uint8)
            hmap_colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
            heatmap_viz = cv2.addWeighted(heatmap_viz, 0.6, hmap_colored.astype(np.float32), 0.4, 0)
        heatmap_viz = heatmap_viz.astype(np.uint8)
        cv2.putText(heatmap_viz, 'Heatmaps', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        canvas[:, img_size:img_size*2] = heatmap_viz
        
        # Panel 3: Prediction (red)
        pred_img = img_resized.copy()
        for j, (x, y) in enumerate(pred_pts):
            cv2.circle(pred_img, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(pred_img, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(pred_img, f'{j+1}', (x-5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        pred_poly = pred_pts
        cv2.polylines(pred_img, [pred_poly], True, (0, 0, 255), 2)
        cv2.putText(pred_img, 'Prediction', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        canvas[:, img_size*2:img_size*3] = pred_img
        
        # Panel 4: Overlay (GT=green, Pred=red)
        overlay = img_resized.copy()
        # GT in green
        for j, (x, y) in enumerate(gt_pts):
            cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
        cv2.polylines(overlay, [gt_poly], True, (0, 255, 0), 2)
        # Pred in red
        for j, (x, y) in enumerate(pred_pts):
            cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
            # Draw error line
            cv2.line(overlay, (x, y), (gt_pts[j][0], gt_pts[j][1]),
                    (255, 255, 0), 1)
        cv2.polylines(overlay, [pred_poly], True, (0, 0, 255), 2)
        cv2.putText(overlay, 'Overlay (GT=green, Pred=red)', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate error
        error_px = np.sqrt(((pred_pts - gt_pts)**2).sum(axis=1)).mean()
        cv2.putText(overlay, f'MAE: {error_px:.1f}px', (10, img_size-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        canvas[:, img_size*3:] = overlay
        
        # Save
        cv2.imwrite(str(vis_dir / f'{filenames[i]}'), canvas)
    
    print(f"  → Visualizations saved to {vis_dir}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, config, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics = CornerMetrics()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        heatmaps_gt = batch['heatmaps'].to(device)
        
        # Forward with mixed precision
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                pred_heatmaps = outputs['heatmaps']
                loss = criterion(pred_heatmaps, heatmaps_gt)
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            pred_heatmaps = outputs['heatmaps']
            loss = criterion(pred_heatmaps, heatmaps_gt)
            loss.backward()
            optimizer.step()
        
        # Metrics (convert heatmaps to corners)
        with torch.no_grad():
            heatmaps_np = pred_heatmaps.cpu().numpy()
            corners_pred = heatmap_to_points(
                heatmaps_np,
                method=config.get('postprocess_method', 'centroid'),
                threshold=config.get('postprocess_threshold', 0.3),
            )
            
            # Convert to normalized coordinates for metrics
            heatmap_h, heatmap_w = heatmaps_np.shape[2:]
            corners_pred[:, :, 0] /= heatmap_w
            corners_pred[:, :, 1] /= heatmap_h
            
            corners_pred_flat = torch.from_numpy(corners_pred.reshape(-1, 8))
            metrics.update(corners_pred_flat, batch['corners'], batch.get('orig_size'))
        
        total_loss += loss.item()
        
        # Update progress bar
        current_mae = metrics.compute()['pixel_mae'] if len(metrics.pixel_errors) > 0 else 0.0
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mae': f"{current_mae:.2f}",
        })
        
        # Log to tensorboard
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    avg_loss = total_loss / len(loader)
    train_metrics = metrics.compute()
    
    return avg_loss, train_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, config):
    """Validate model."""
    model.eval()
    total_loss = 0
    metrics = CornerMetrics()
    first_batch_logged = False
    
    for batch in tqdm(loader, desc="Validation"):
        images = batch['image'].to(device)
        heatmaps_gt = batch['heatmaps'].to(device)
        
        # Forward
        outputs = model(images)
        pred_heatmaps = outputs['heatmaps']
        
        # Loss
        loss = criterion(pred_heatmaps, heatmaps_gt)
        total_loss += loss.item()
        
        # Metrics
        heatmaps_np = pred_heatmaps.cpu().numpy()
        corners_pred = heatmap_to_points(
            heatmaps_np,
            method=config.get('postprocess_method', 'centroid'),
            threshold=config.get('postprocess_threshold', 0.3),
        )
        
        # Debug: Log first prediction to verify model is updating
        if not first_batch_logged:
            first_pred = corners_pred[0, :, :].flatten()[:4]
            print(f"\n[DEBUG Epoch {epoch}] First val prediction: {first_pred}...")
            first_batch_logged = True
        
        # Convert to normalized coordinates
        heatmap_h, heatmap_w = heatmaps_np.shape[2:]
        corners_pred[:, :, 0] /= heatmap_w
        corners_pred[:, :, 1] /= heatmap_h
        
        corners_pred_flat = torch.from_numpy(corners_pred.reshape(-1, 8))
        metrics.update(corners_pred_flat, batch['corners'], batch.get('orig_size'))
    
    avg_loss = total_loss / len(loader)
    val_metrics = metrics.compute()
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    for key, value in val_metrics.items():
        writer.add_scalar(f'Val/{key}', value, epoch)
    
    return avg_loss, val_metrics


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Data
    train_loader, val_loader = create_dataloaders(
        labels_path=args.labels_path,
        images_dir=args.images_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=tuple(config['img_size']),
        heatmap_size=tuple(config['heatmap_size']),
        mode='heatmap',
        split_ratio=config['split_ratio'],
        seed=config['seed'],
        sigma=config['heatmap_sigma'],
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = create_heatmap_model(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        with_edge=config.get('with_edge', False),
        with_classification=config.get('with_classification', False),
    )
    model = model.to(device)
    
    # Loss: Choose between weighted MSE or Focal Heatmap Loss
    loss_type = config.get('loss_type', 'weighted_mse')
    
    if loss_type == 'focal':
        criterion = FocalHeatmapLoss(
            foreground_weight=config.get('foreground_weight', 150.0),
            background_penalty=config.get('background_suppression', 0.5),
        )
    elif loss_type == 'hybrid':
        criterion = HybridHeatmapLoss(
            foreground_weight=config.get('foreground_weight', 150.0),
            background_suppression=config.get('background_suppression', 0.5),
        )
    else:  # weighted_mse (default)
        def weighted_mse_loss(pred, target, foreground_weight=50.0):
            """MSE with higher weight on foreground pixels."""
            mse = (pred - target) ** 2
            weight = torch.ones_like(target)
            weight[target > 0.1] = foreground_weight  # 50x weight on peaks
            return (mse * weight).mean()
        
        criterion = lambda pred, target: weighted_mse_loss(
            pred, target, foreground_weight=config.get('foreground_weight', 50.0)
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    # Mixed precision scaler for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01,
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config, scaler
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer, config
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        print(f"  Pixel MAE: {val_metrics['pixel_mae']:.2f}")
        print(f"  Pixel RMSE: {val_metrics['pixel_rmse']:.2f}")
        print(f"  PCK@5px: {val_metrics['pck_5px']*100:.1f}%")
        print(f"  Quad IoU: {val_metrics['quad_iou']:.4f}")
        
        # Visualize predictions every 5 epochs
        if epoch % 5 == 0:
            visualize_predictions(model, val_loader, device, output_dir, epoch, config, args.images_dir)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': config,
        }
        
        torch.save(checkpoint, output_dir / 'last.pth')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best.pth')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    writer.close()
    print(f"\nTraining complete! Checkpoints saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Heatmap Regression Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--labels_path', type=str, default='labels.json', help='Path to labels.json')
    parser.add_argument('--images_dir', type=str, default='dataset', help='Path to images directory')
    parser.add_argument('--output_dir', type=str, default='outputs/heatmap_regression', help='Output directory')
    
    args = parser.parse_args()
    main(args)
