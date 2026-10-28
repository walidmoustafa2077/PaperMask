"""Training script for Point Regression Model."""
import argparse
import yaml
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.point_regression import create_point_model
from src.losses.smooth_l1 import PointRegressionLoss
from src.metrics.geometry import CornerMetrics
from src.utils.postprocess import denormalize_corners


@torch.no_grad()
def visualize_predictions(model, loader, device, output_dir, epoch, images_dir='dataset'):
    """Visualize predictions on a few validation samples."""
    model.eval()
    vis_dir = output_dir / 'visualizations' / f'epoch_{epoch:03d}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first batch
    batch = next(iter(loader))
    images = batch['image'].to(device)
    corners_gt = batch['corners'].to(device)
    filenames = batch['filename']
    orig_sizes = batch['orig_size']
    
    # Predict
    outputs = model(images)
    pred_corners = outputs['corners']
    
    # Denormalize to original image size
    for i in range(min(4, len(images))):  # Save first 4 images
        # Get original image
        img_path = Path(images_dir) / filenames[i]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        orig_h, orig_w = orig_sizes[i].cpu().numpy()
        
        # Get corners in pixel coordinates (denormalized to 256x256)
        img_size = 256
        gt_pts = denormalize_corners(corners_gt[i], (img_size, img_size))  # (4, 2)
        pred_pts = denormalize_corners(pred_corners[i], (img_size, img_size))
        
        # Resize original image to 256x256 for consistent visualization
        img_resized = cv2.resize(img, (img_size, img_size))
        
        # Create side-by-side comparison: GT | Prediction | Both
        canvas = np.zeros((img_size, img_size * 3, 3), dtype=np.uint8)
        
        # Left: Ground Truth only (green)
        gt_img = img_resized.copy()
        for j, (x, y) in enumerate(gt_pts):
            cv2.circle(gt_img, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.circle(gt_img, (int(x), int(y)), 8, (255, 255, 255), 2)
            cv2.putText(gt_img, f'{j+1}', (int(x)-5, int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        gt_poly = gt_pts.astype(np.int32)
        cv2.polylines(gt_img, [gt_poly], True, (0, 255, 0), 2)
        cv2.putText(gt_img, 'Ground Truth', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        canvas[:, :img_size] = gt_img
        
        # Middle: Prediction only (red)
        pred_img = img_resized.copy()
        for j, (x, y) in enumerate(pred_pts):
            cv2.circle(pred_img, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.circle(pred_img, (int(x), int(y)), 8, (255, 255, 255), 2)
            cv2.putText(pred_img, f'{j+1}', (int(x)-5, int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        pred_poly = pred_pts.astype(np.int32)
        cv2.polylines(pred_img, [pred_poly], True, (0, 0, 255), 2)
        cv2.putText(pred_img, 'Prediction', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        canvas[:, img_size:img_size*2] = pred_img
        
        # Right: Both overlaid (green=GT, red=Pred)
        overlay = img_resized.copy()
        # GT in green
        for j, (x, y) in enumerate(gt_pts):
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.circle(overlay, (int(x), int(y)), 8, (255, 255, 255), 2)
        cv2.polylines(overlay, [gt_poly], True, (0, 255, 0), 2)
        # Pred in red
        for j, (x, y) in enumerate(pred_pts):
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (int(x), int(y)), 8, (255, 255, 255), 2)
            # Draw line from pred to GT to show error
            cv2.line(overlay, (int(x), int(y)), (int(gt_pts[j][0]), int(gt_pts[j][1])),
                    (255, 255, 0), 1)
        cv2.polylines(overlay, [pred_poly], True, (0, 0, 255), 2)
        cv2.putText(overlay, 'Overlay (GT=green, Pred=red)', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate error for this sample
        error_px = np.sqrt(((pred_pts - gt_pts)**2).sum(axis=1)).mean()
        cv2.putText(overlay, f'MAE: {error_px:.1f}px', (10, img_size-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        canvas[:, img_size*2:] = overlay
        canvas[:, img_size*2:] = overlay
        
        # Save composite image
        cv2.imwrite(str(vis_dir / f'{filenames[i]}'), canvas)
    
    print(f"  → Visualizations saved to {vis_dir}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics = CornerMetrics()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        corners_gt = batch['corners'].to(device)
        
        # Forward with mixed precision
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                pred_corners = outputs['corners']
                loss_dict = criterion(pred_corners, corners_gt)
                loss = loss_dict['loss']
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            pred_corners = outputs['corners']
            loss_dict = criterion(pred_corners, corners_gt)
            loss = loss_dict['loss']
            loss.backward()
            
            # Gradient clipping to prevent divergence
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        metrics.update(pred_corners, corners_gt, batch.get('orig_size'))
        
        # Update progress bar
        current_mae = metrics.compute()['pixel_mae'] if len(metrics.pixel_errors) > 0 else 0.0
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mae': f"{current_mae:.2f}",
        })
        
        # Log to tensorboard
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/CornerLoss', loss_dict['corner_loss'].item(), global_step)
    
    avg_loss = total_loss / len(loader)
    train_metrics = metrics.compute()
    
    return avg_loss, train_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    """Validate model."""
    model.eval()
    total_loss = 0
    metrics = CornerMetrics()
    first_batch_logged = False
    
    for batch in tqdm(loader, desc="Validation"):
        images = batch['image'].to(device)
        corners_gt = batch['corners'].to(device)
        
        # Forward
        outputs = model(images)
        pred_corners = outputs['corners']
        
        # Debug: Log first prediction to verify model is updating
        if not first_batch_logged:
            first_pred = pred_corners[0].cpu().numpy()
            print(f"\n[DEBUG Epoch {epoch}] First val prediction: {first_pred[:4]}...")
            first_batch_logged = True
        
        # Loss
        loss_dict = criterion(pred_corners, corners_gt)
        loss = loss_dict['loss']
        
        total_loss += loss.item()
        metrics.update(pred_corners, corners_gt, batch.get('orig_size'))
    
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
        mode='point',
        split_ratio=config['split_ratio'],
        seed=config['seed'],
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = create_point_model(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        with_classification=config.get('with_classification', False),
    )
    model = model.to(device)
    
    # Loss
    criterion = PointRegressionLoss(
        corner_weight=config['corner_weight'],
        cls_weight=config.get('cls_weight', 1.0),
        scale_factor=config.get('scale_factor', 1.0),
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
            model, train_loader, criterion, optimizer, device, epoch, writer, scaler
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
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
            visualize_predictions(model, val_loader, device, output_dir, epoch, args.images_dir)
        
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
    parser = argparse.ArgumentParser(description='Train Point Regression Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--labels_path', type=str, default='labels.json', help='Path to labels.json')
    parser.add_argument('--images_dir', type=str, default='dataset', help='Path to images directory')
    parser.add_argument('--output_dir', type=str, default='outputs/point_regression', help='Output directory')
    
    args = parser.parse_args()
    main(args)
