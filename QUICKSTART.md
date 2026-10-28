# PaperMask Quick Start Guide

## Installation Complete! ✓

All dependencies are installed and models are working correctly.

## Quick Start

### 1. Verify Installation
```powershell
.\torch_env\Scripts\activate
python test_installation.py
```

### 2. Train Point Regression Model
```powershell
python src/train_point.py --config configs/point_base.yaml
```

Training will:
- Use 80/20 train/val split
- Save checkpoints to `outputs/point_regression/`
- Log to TensorBoard at `outputs/point_regression/logs/`

Monitor progress:
```powershell
tensorboard --logdir outputs/point_regression/logs
```

### 3. Train Heatmap Regression Model
```powershell
python src/train_heatmap.py --config configs/heatmap_base.yaml
```

Training will:
- Use 80/20 train/val split
- Save checkpoints to `outputs/heatmap_regression/`
- Log to TensorBoard at `outputs/heatmap_regression/logs/`

### 4. Visualize Predictions
```powershell
# Visualize both models
python debug_visualize.py

# Visualize specific model
python debug_visualize.py --which point
python debug_visualize.py --which heatmap
```

Results saved to `debug_outputs/point/` and `debug_outputs/heatmap/`

### 5. Export for Deployment
```powershell
# Export point model
python src/export.py --checkpoint outputs/point_regression/best.pth --model_type point

# Export heatmap model
python src/export.py --checkpoint outputs/heatmap_regression/best.pth --model_type heatmap
```

Exported models saved to `exports/` directory.

## Configuration

Edit `configs/point_base.yaml` or `configs/heatmap_base.yaml` to adjust:
- `backbone`: 'mobilenet_v3_small' (2.6M params) or 'mobilenet_v3_large' (faster but larger)
- `img_size`: Input resolution (higher = more accurate but slower)
- `batch_size`: Batch size (reduce if OOM)
- `epochs`: Training duration
- `lr`: Learning rate

## Current Dataset

Your dataset has **61 images** in `images/` with labels in `labels.json`.

**Recommendation**: With this small dataset:
1. Use strong augmentations (already enabled)
2. Train longer (increase `epochs` to 150-200)
3. Use smaller backbone to avoid overfitting
4. Consider data augmentation or synthetic data generation

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config (try 4 or 2)
- Reduce `img_size` to `[224, 224]`
- Train on CPU (slower): add `--device cpu` flag (if implemented)

### Poor Accuracy
- Check labels are correct: `python image_labling_validation.py`
- Increase training epochs
- Try different learning rates (0.0005, 0.001, 0.002)
- Use larger backbone (`mobilenet_v3_large`)

### Import Errors
- Ensure virtual environment is activated: `.\torch_env\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Next Steps

1. **Collect more data**: 61 images is quite small; aim for 200-500+ for better generalization
2. **Tune hyperparameters**: Experiment with learning rates, loss weights, augmentations
3. **Try different backbones**: Install `timm` and use EfficientNet or GhostNet
4. **Add edge supervision**: Enable `with_edge: true` in heatmap config for better boundary detection
5. **Quantize models**: Export with INT8 quantization for mobile deployment

## Model Comparison

| Model | Params | Speed | Accuracy | Best For |
|-------|--------|-------|----------|----------|
| Point (small) | 5.9M | Fast | Medium | Real-time, mobile |
| Heatmap (small) | 2.6M | Medium | High | Precision, document scanning |

## Support

For issues or questions, check:
- `README.md` for detailed documentation
- `plan-modelDesign.prompt.md` for architecture details
- Model implementations in `src/models/`

Happy training! 🚀
