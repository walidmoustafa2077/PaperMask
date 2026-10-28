# PaperMask: Document Corner Detection

A PyTorch implementation of two architectures for document corner detection optimized for mobile deployment:

1. **Point Regression Model**: Direct corner coordinate prediction with Cross-Attention neck
2. **Heatmap Regression Model**: Gaussian heatmap-based detection with BiFPN neck

## Features

- 🚀 **Mobile-Friendly**: Lightweight backbones (MobileNetV3) optimized for edge devices
- 📊 **Two Architectures**: Point regression (end-to-end) and heatmap regression (high precision)
- 🔧 **Flexible**: Configurable backbones, input sizes, and training parameters
- 📦 **Export-Ready**: TorchScript and ONNX export for production deployment
- 📈 **Comprehensive Metrics**: Pixel MAE/RMSE, PCK, quad IoU

## Architecture Overview

### Point Regression Model
```
Input (256x256) → MobileNetV3 → Cross-Attention Neck → Corner Head (8 coords)
```
- **Pros**: End-to-end, fast inference
- **Cons**: Amplification error on upscaling

### Heatmap Regression Model
```
Input (256x256) → MobileNetV3 → BiFPN → Heatmap Head (4 channels) → Postprocess
```
- **Pros**: High precision, resolves amplification error
- **Cons**: Requires postprocessing step

## Installation

### 1. Setup Environment

```powershell
# Activate existing virtual environment
.\torch_env\Scripts\activate

# Or create a new one
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Core dependencies:**
- PyTorch 2.0.1 with CUDA 11.7
- torchvision 0.15.2
- opencv-python
- albumentations (augmentations)
- shapely (metrics)
- timm (optional, for alternative backbones)

## Dataset Preparation

Your dataset should have:

```
PaperMask/
├── images/              # RGB images
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
└── labels.json          # Corner annotations
```

**labels.json format:**
```json
{
  "0000.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "0001.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

Coordinates should be in pixel space (not normalized).

### Create Annotations

Use the included labeling tool:

```powershell
python image_labling.py
```

- Click 4 corners per image (any order)
- Press `s` to save and move to next image
- Press `+/-` to zoom

Validate annotations:

```powershell
python image_labling_validation.py
```

## Training

### Point Regression Model

```powershell
python src/train_point.py `
  --config configs/point_base.yaml `
  --labels_path labels.json `
  --images_dir images `
  --output_dir outputs/point_regression
```

**Key hyperparameters** (edit `configs/point_base.yaml`):
- `backbone`: `mobilenet_v3_small` or `mobilenet_v3_large`
- `img_size`: Input resolution (e.g., `[256, 256]`)
- `corner_weight`: Loss amplification (default: 1000)
- `lr`: Learning rate (default: 0.001)
- `epochs`: Training epochs (default: 100)

### Heatmap Regression Model

```powershell
python src/train_heatmap.py `
  --config configs/heatmap_base.yaml `
  --labels_path labels.json `
  --images_dir images `
  --output_dir outputs/heatmap_regression
```

**Key hyperparameters** (edit `configs/heatmap_base.yaml`):
- `heatmap_size`: Output heatmap resolution (e.g., `[64, 64]` for stride 4)
- `heatmap_sigma`: Gaussian sigma for target heatmaps (default: 2.0)
- `postprocess_method`: `'centroid'`, `'gaussian'`, or `'argmax'`
- `awing_*`: Adaptive Wing Loss parameters

### Monitor Training

```powershell
tensorboard --logdir outputs/point_regression/logs
```

## Inference

### Point Model

```python
import torch
from src.models.point_regression import create_point_model
from src.utils.postprocess import denormalize_corners

# Load model
model = create_point_model(backbone='mobilenet_v3_small')
checkpoint = torch.load('outputs/point_regression/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
image = torch.randn(1, 3, 256, 256)  # Preprocess your image
with torch.no_grad():
    outputs = model(image)
    corners = outputs['corners']  # (1, 8) normalized [0, 1]

# Convert to pixels
corners_px = denormalize_corners(corners, img_size=(orig_h, orig_w))
print(corners_px)  # (4, 2)
```

### Heatmap Model

```python
import torch
from src.models.heatmap_regression import create_heatmap_model
from src.utils.postprocess import heatmap_to_points, scale_corners

# Load model
model = create_heatmap_model(backbone='mobilenet_v3_small')
checkpoint = torch.load('outputs/heatmap_regression/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
image = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    outputs = model(image)
    heatmaps = outputs['heatmaps']  # (1, 4, 64, 64)

# Postprocess
corners_hm = heatmap_to_points(
    heatmaps.cpu().numpy(),
    method='centroid',
    threshold=0.3
)  # (4, 2) in heatmap coordinates

# Scale to original image
corners_px = scale_corners(
    corners_hm,
    from_size=(64, 64),
    to_size=(orig_h, orig_w)
)
print(corners_px)
```

## Corner Refinement & Retraining

After initial training, you can use the model to predict corners on new images, manually refine them, and retrain for improved accuracy.

### Interactive Refinement

```powershell
# Load model and interactively refine predictions
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth

# Resume refinement (skips already processed images)
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth --output refined_labels.json
```

**Interactive Controls:**
- **Left-click**: Select corner (click-and-drag to move)
- **n**: Save current image and move to next
- **b**: Go back to previous image
- **r**: Reset to model prediction
- **u**: Undo last adjustment
- **s**: Skip image without saving
- **w**: Write progress checkpoint
- **q**: Quit and save all progress
- **ESC**: Quit without saving current image

### Auto-Generate Predictions

Generate predictions for all images without manual review:

```powershell
python refine_corners.py --checkpoint outputs/heatmap_regression/best.pth --auto --output auto_labels.json
```

### Retrain with Refined Labels

Once you've refined corners for new images, retrain the model:

```powershell
python src/train_heatmap.py `
  --config configs/heatmap_base.yaml `
  --labels_path refined_labels.json `
  --images_dir images `
  --output_dir outputs/heatmap_regression_v2
```

This iterative process (train → refine → retrain) progressively improves model accuracy.

## Export for Deployment

Export trained models to TorchScript or ONNX:

```powershell
# Point model
python src/export.py `
  --checkpoint outputs/point_regression/best.pth `
  --model_type point `
  --output_dir exports/point `
  --img_size 256 256

# Heatmap model
python src/export.py `
  --checkpoint outputs/heatmap_regression/best.pth `
  --model_type heatmap `
  --output_dir exports/heatmap `
  --img_size 256 256
```

Exported models:
- `exports/point/point_model.pt` (TorchScript)
- `exports/point/point_model.onnx` (ONNX)

### Mobile Deployment

**Android/iOS (TorchScript):**
```java
// Android example
Module module = Module.load("point_model.pt");
Tensor input = Tensor.fromBlob(data, shape);
Tensor output = module.forward(IValue.from(input)).toTensor();
```

**ONNX Runtime:**
```python
import onnxruntime as ort

session = ort.InferenceSession("point_model.onnx")
outputs = session.run(None, {"image": image_np})
corners = outputs[0]
```

## Metrics

Models are evaluated with:

| Metric | Description |
|--------|-------------|
| **Pixel MAE** | Mean absolute error per corner (pixels) |
| **Pixel RMSE** | Root mean squared error per corner (pixels) |
| **Normalized MAE** | MAE normalized by image diagonal |
| **PCK@Npx** | % corners within N pixels of ground truth |
| **Quad IoU** | Intersection over Union of quadrilaterals |

## Configuration Tips

### Small Dataset (< 100 images)
- Use strong augmentations (`augment: true`)
- Lower learning rate (`lr: 0.0005`)
- More epochs (`epochs: 150`)
- Consider `mobilenet_v3_small` to reduce overfitting

### High Resolution Inputs
- Increase `img_size` to `[320, 320]` or `[384, 384]`
- Adjust `heatmap_size` proportionally (stride 4-8)
- Reduce batch size if OOM

### Mobile Optimization
- Use `mobilenet_v3_small` backbone
- Keep `img_size` ≤ 256 for latency
- Export with quantization (PTQ):
  ```python
  model_quantized = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
  )
  ```

## Troubleshooting

### Training Issues

**Loss not decreasing:**
- Check label format (pixel coords, not normalized)
- Verify augmentations aren't too aggressive
- Lower learning rate or increase warmup

**Overfitting (train << val):**
- Enable augmentations
- Increase weight decay
- Use smaller backbone
- Add dropout

### Inference Issues

**Poor accuracy after export:**
- Verify preprocessing matches training (normalization, resize)
- Check coordinate scaling (normalized → pixels)
- Test corner order (canonical vs arbitrary)

**Slow inference:**
- Use TorchScript for ~2x speedup
- Quantize model (INT8)
- Reduce input resolution

## Project Structure

```
PaperMask/
├── src/
│   ├── data/
│   │   └── dataset.py          # Dataset & augmentations
│   ├── models/
│   │   ├── point_regression.py # Point model + CrossAttention
│   │   └── heatmap_regression.py # Heatmap model + BiFPN
│   ├── losses/
│   │   ├── awing.py            # Adaptive Wing Loss
│   │   └── smooth_l1.py        # Smooth L1 for point regression
│   ├── metrics/
│   │   └── geometry.py         # Evaluation metrics
│   ├── utils/
│   │   └── postprocess.py      # Heatmap → points conversion
│   ├── train_point.py          # Point model training
│   ├── train_heatmap.py        # Heatmap model training
│   └── export.py               # Model export
├── configs/
│   ├── point_base.yaml         # Point model config
│   └── heatmap_base.yaml       # Heatmap model config
├── images/                     # Dataset images
├── labels.json                 # Annotations
├── image_labling.py            # Annotation tool
├── image_labling_validation.py # Validate annotations
├── debug_visualize.py          # Inference visualization
└── requirements.txt            # Dependencies
```

## References

- **PP-LCNet**: [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) (concept; PyTorch uses MobileNetV3)
- **BiFPN**: [EfficientDet](https://arxiv.org/abs/1911.09070)
- **Adaptive Wing Loss**: [Heatmap Regression for Face Alignment](https://arxiv.org/abs/1904.07399)
- **Cross-Attention**: Feature Pyramid Transformer concepts

## License

MIT License

## Citation

If you use this work, please cite:

```bibtex
@software{papermask2025,
  title={PaperMask: Document Corner Detection for Mobile},
  year={2025},
  url={https://github.com/yourusername/PaperMask}
}
```

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional backbones (EfficientNet, GhostNet)
- [ ] Edge supervision loss
- [ ] Multi-document detection
- [ ] TensorRT/CoreML export
- [ ] Benchmark on SmartDoc 2015

---

**Happy training!** 🚀
