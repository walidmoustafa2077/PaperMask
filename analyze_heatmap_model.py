"""Analyze heatmap model learning quality and generalization."""
import argparse
import torch
import numpy as np
import yaml
import sys
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import create_dataloaders
from src.models.heatmap_regression import create_heatmap_model
from src.utils.postprocess import heatmap_to_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/heatmap_base.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("HEATMAP MODEL ANALYSIS: Learning vs Memorization")
    print("=" * 70)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = create_heatmap_model(
        backbone=config['backbone'],
        pretrained=False,
        with_classification=config.get('with_classification', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Val MAE: {checkpoint['val_metrics']['pixel_mae']:.2f}px")
    print(f"   Val IoU: {checkpoint['val_metrics']['quad_iou']:.4f}")
    
    # Get validation data
    _, val_loader = create_dataloaders(
        labels_path='labels.json',
        images_dir='dataset',
        batch_size=1,
        num_workers=0,
        img_size=tuple(config['img_size']),
        heatmap_size=tuple(config['heatmap_size']),
        mode='heatmap',
        sigma=config['heatmap_sigma'],
    )
    
    # Collect all predictions
    all_predictions = []
    all_gt = []
    all_errors = []
    
    print(f"\n📊 Analyzing {len(val_loader)} validation samples...")
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            corners_gt = batch['corners'].numpy()[0].reshape(4, 2)  # Get first (and only) sample
            
            # Predict
            outputs = model(images)
            heatmaps = outputs['heatmaps'].numpy()
            
            # Convert to corners
            corners_pred = heatmap_to_points(
                heatmaps,
                method=config.get('postprocess_method', 'argmax'),
                threshold=config.get('postprocess_threshold', 0.3),
            )
            
            # heatmap_to_points returns (B, 4, 2) for batch or (4, 2) for single
            if corners_pred.ndim == 3:
                corners_pred = corners_pred[0]  # Get first sample (4, 2)
            
            # Normalize to [0, 1]
            heatmap_h, heatmap_w = heatmaps.shape[2:]
            corners_pred = corners_pred.astype(np.float32)
            corners_pred[:, 0] /= heatmap_w
            corners_pred[:, 1] /= heatmap_h
            
            # Store
            all_predictions.append(corners_pred)
            all_gt.append(corners_gt)
            
            # Calculate error
            error = np.sqrt(((corners_pred - corners_gt)**2).sum(axis=1)).mean() * 256
            all_errors.append(error)
    
    all_predictions = np.array(all_predictions)  # (N, 1, 4, 2) or (N, 4, 2)
    all_gt = np.array(all_gt)  # (N, 4, 2)
    all_errors = np.array(all_errors)
    
    # Squeeze extra dimensions
    if all_predictions.ndim == 4:
        all_predictions = all_predictions.squeeze(1)  # (N, 4, 2)
    
    print("\n" + "=" * 70)
    print("1. PREDICTION DIVERSITY ANALYSIS")
    print("=" * 70)
    print("\n📍 Per-corner prediction statistics:")
    print(f"{'Corner':<10} {'Mean (x,y)':<25} {'Std (x,y)':<25} {'GT Std (x,y)'}")
    print("-" * 70)
    
    diversity_healthy = True
    for i in range(4):
        pred_mean = all_predictions[:, i, :].mean(axis=0)
        pred_std = all_predictions[:, i, :].std(axis=0)
        gt_std = all_gt[:, i, :].std(axis=0)
        
        corner_num = i + 1
        pm_x, pm_y = pred_mean[0], pred_mean[1]
        ps_x, ps_y = pred_std[0], pred_std[1]
        gs_x, gs_y = gt_std[0], gt_std[1]
        
        print(f"Corner {corner_num}   "
              f"({pm_x:.3f}, {pm_y:.3f})    "
              f"({ps_x:.3f}, {ps_y:.3f})    "
              f"({gs_x:.3f}, {gs_y:.3f})")
        
        # Check if model predictions have reasonable variance
        if pred_std[0] < gt_std[0] * 0.3 or pred_std[1] < gt_std[1] * 0.3:
            diversity_healthy = False
    
    if diversity_healthy:
        print("\n✅ GOOD: Predictions show healthy diversity (similar to ground truth)")
    else:
        print("\n⚠️  WARNING: Predictions less diverse than ground truth (possible memorization)")
    
    print("\n" + "=" * 70)
    print("2. MEMORIZATION TEST: Constant vs Adaptive Predictions")
    print("=" * 70)
    
    # Calculate variance of predictions
    pred_variance = all_predictions.reshape(len(all_predictions), -1).var(axis=0).mean()
    gt_variance = all_gt.reshape(len(all_gt), -1).var(axis=0).mean()
    
    print(f"\nPrediction variance: {pred_variance:.6f}")
    print(f"Ground truth variance: {gt_variance:.6f}")
    print(f"Variance ratio: {pred_variance/gt_variance:.2%}")
    
    if pred_variance / gt_variance > 0.5:
        print("✅ GOOD: Model predictions adapt to different images")
    elif pred_variance / gt_variance > 0.2:
        print("⚠️  MODERATE: Some adaptation but could be better")
    else:
        print("❌ BAD: Model predicting nearly constant values (memorization)")
    
    print("\n" + "=" * 70)
    print("3. ERROR DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    print(f"\nError statistics (pixels @ 256×256):")
    print(f"  Mean: {all_errors.mean():.2f}px")
    print(f"  Median: {np.median(all_errors):.2f}px")
    print(f"  Std: {all_errors.std():.2f}px")
    print(f"  Min: {all_errors.min():.2f}px")
    print(f"  Max: {all_errors.max():.2f}px")
    print(f"  25th percentile: {np.percentile(all_errors, 25):.2f}px")
    print(f"  75th percentile: {np.percentile(all_errors, 75):.2f}px")
    
    # Check for outliers (possible failure cases)
    q3 = np.percentile(all_errors, 75)
    q1 = np.percentile(all_errors, 25)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = all_errors > outlier_threshold
    
    if outliers.sum() > 0:
        print(f"\n⚠️  Found {outliers.sum()} outlier samples (error > {outlier_threshold:.1f}px)")
        print(f"   These may be difficult images or edge cases")
    else:
        print(f"\n✅ No major outliers detected")
    
    print("\n" + "=" * 70)
    print("4. STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)
    
    # Compare predictions vs average baseline
    average_corners = all_gt.mean(axis=0)
    baseline_errors = []
    
    for gt in all_gt:
        error = np.sqrt(((average_corners - gt)**2).sum(axis=1)).mean() * 256
        baseline_errors.append(error)
    
    baseline_errors = np.array(baseline_errors)
    
    print(f"\nBaseline (predict average): {baseline_errors.mean():.2f}px")
    print(f"Model performance: {all_errors.mean():.2f}px")
    print(f"Improvement: {(1 - all_errors.mean()/baseline_errors.mean())*100:.1f}%")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_errors, all_errors)
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.2e}")
    
    if p_value < 0.001:
        print("  ✅ HIGHLY SIGNIFICANT: Model clearly better than baseline")
    elif p_value < 0.05:
        print("  ✅ SIGNIFICANT: Model better than baseline")
    else:
        print("  ❌ NOT SIGNIFICANT: Model not clearly better than baseline")
    
    print("\n" + "=" * 70)
    print("5. HEATMAP QUALITY CHECK")
    print("=" * 70)
    
    # Check heatmap sharpness
    heatmap_maxes = []
    heatmap_means = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            outputs = model(images)
            heatmaps = outputs['heatmaps'].numpy()[0]  # (4, H, W)
            
            for i in range(4):
                hmap = heatmaps[i]
                heatmap_maxes.append(hmap.max())
                heatmap_means.append(hmap.mean())
    
    heatmap_maxes = np.array(heatmap_maxes)
    heatmap_means = np.array(heatmap_means)
    
    print(f"\nHeatmap peak values:")
    print(f"  Mean max: {heatmap_maxes.mean():.3f}")
    print(f"  Min max: {heatmap_maxes.min():.3f}")
    print(f"  Max max: {heatmap_maxes.max():.3f}")
    
    print(f"\nHeatmap background values:")
    print(f"  Mean avg: {heatmap_means.mean():.6f}")
    
    print(f"\nPeak-to-background ratio: {heatmap_maxes.mean() / heatmap_means.mean():.1f}×")
    
    if heatmap_maxes.mean() > 0.7:
        print("✅ GOOD: Heatmaps have sharp, confident peaks")
    elif heatmap_maxes.mean() > 0.5:
        print("⚠️  MODERATE: Heatmap peaks are somewhat weak")
    else:
        print("❌ BAD: Heatmap peaks are too diffuse")
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    score = 0
    
    # Scoring criteria
    if diversity_healthy:
        score += 25
        print("✅ [25/25] Prediction diversity is healthy")
    else:
        print("❌ [0/25] Predictions lack diversity")
    
    if pred_variance / gt_variance > 0.5:
        score += 25
        print("✅ [25/25] Model adapts to different images")
    elif pred_variance / gt_variance > 0.2:
        score += 15
        print("⚠️  [15/25] Model shows moderate adaptation")
    else:
        print("❌ [0/25] Model not adapting to images")
    
    if all_errors.mean() < baseline_errors.mean() * 0.5:
        score += 25
        print("✅ [25/25] Performance significantly better than baseline")
    elif all_errors.mean() < baseline_errors.mean() * 0.8:
        score += 15
        print("⚠️  [15/25] Performance moderately better than baseline")
    else:
        print("❌ [0/25] Performance not much better than baseline")
    
    if heatmap_maxes.mean() > 0.7:
        score += 25
        print("✅ [25/25] Heatmaps have sharp peaks")
    elif heatmap_maxes.mean() > 0.5:
        score += 15
        print("⚠️  [15/25] Heatmap peaks are moderate")
    else:
        print("❌ [0/25] Heatmap peaks too weak")
    
    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {score}/100")
    print(f"{'='*70}")
    
    if score >= 80:
        print("\n🎉 EXCELLENT: Model is learning well and generalizing!")
    elif score >= 60:
        print("\n✅ GOOD: Model is learning but has room for improvement")
    elif score >= 40:
        print("\n⚠️  FAIR: Model learning but with significant issues")
    else:
        print("\n❌ POOR: Model not learning effectively")
    
    print("\n")


if __name__ == '__main__':
    main()
