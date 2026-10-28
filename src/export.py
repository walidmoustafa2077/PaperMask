"""Export models to TorchScript and ONNX for deployment."""
import argparse
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.point_regression import create_point_model
from src.models.heatmap_regression import create_heatmap_model


def export_point_model(checkpoint_path, output_dir, img_size=(256, 256)):
    """Export point regression model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = create_point_model(
        backbone=config['backbone'],
        pretrained=False,
        with_classification=config.get('with_classification', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, *img_size)
    
    # Export to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = output_dir / 'point_model.pt'
        traced_model.save(str(torchscript_path))
        print(f"✓ Exported TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
    
    # Export to ONNX
    try:
        onnx_path = output_dir / 'point_model.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['corners'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'corners': {0: 'batch_size'},
            }
        )
        print(f"✓ Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")


def export_heatmap_model(checkpoint_path, output_dir, img_size=(256, 256)):
    """Export heatmap regression model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = create_heatmap_model(
        backbone=config['backbone'],
        pretrained=False,
        with_edge=config.get('with_edge', False),
        with_classification=config.get('with_classification', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, *img_size)
    
    # Export to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = output_dir / 'heatmap_model.pt'
        traced_model.save(str(torchscript_path))
        print(f"✓ Exported TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
    
    # Export to ONNX
    try:
        onnx_path = output_dir / 'heatmap_model.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['heatmaps'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'heatmaps': {0: 'batch_size'},
            }
        )
        print(f"✓ Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export models to TorchScript/ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (.pth)')
    parser.add_argument('--model_type', type=str, required=True, choices=['point', 'heatmap'])
    parser.add_argument('--output_dir', type=str, default='exports', help='Output directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256], help='Input image size')
    
    args = parser.parse_args()
    
    if args.model_type == 'point':
        export_point_model(args.checkpoint, args.output_dir, tuple(args.img_size))
    else:
        export_heatmap_model(args.checkpoint, args.output_dir, tuple(args.img_size))
    
    print("\n✓ Export complete!")


if __name__ == '__main__':
    main()
