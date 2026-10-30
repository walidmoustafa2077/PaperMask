import torch
from src.models.paper_corner import PaperCornerModel, build_model


def test_forward_shapes():
    model = PaperCornerModel(pretrained=False, fpn_channels=32).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 256))
    assert out["heatmaps"].shape == (2, 4, 64, 64)
    assert out["offsets"].shape == (2, 8, 64, 64)
    assert out["logits"].shape == (2, 4, 64, 64)
    assert out["heatmaps"].min() >= 0 and out["heatmaps"].max() <= 1


def test_heatmap_bias_starts_low():
    model = PaperCornerModel(pretrained=False)
    assert torch.allclose(model.heatmap_head.bias, torch.full((4,), -4.6), atol=1e-4)


def test_offsets_start_at_zero():
    model = PaperCornerModel(pretrained=False)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out["offsets"].abs().max() < 1e-6


def test_build_model_from_config():
    cfg = {"model": {"backbone": "mobilenet_v3_small", "pretrained": False,
                     "fpn_channels": 32, "heatmap_size": 64, "sigma": 2.0}}
    m = build_model(cfg)
    assert isinstance(m, PaperCornerModel)
    assert m.heatmap_size == 64


def test_param_count_under_budget():
    m = PaperCornerModel(pretrained=False)
    assert sum(p.numel() for p in m.parameters()) < 1_500_000
