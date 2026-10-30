import torch
from src.losses.corner_loss import CornerLoss, encode_targets


def test_target_shapes_and_values():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    hm, off, mask = encode_targets(corners, 256, 64, 2.0)
    assert hm.shape == (1, 4, 64, 64) and off.shape == (1, 8, 64, 64)
    assert torch.allclose(hm[0, 0, 16, 16], torch.tensor(1.0), atol=1e-6)
    assert mask.sum() == 4
    assert torch.allclose(off[0, 0, 16, 16], torch.tensor(0.0))


def test_offset_target_fractional():
    corners = torch.tensor([[[66.0, 64.0], [192, 64], [192, 192], [64, 192]]])
    _, off, _ = encode_targets(corners, 256, 64, 2.0)
    assert torch.allclose(off[0, 0, 16, 16], torch.tensor(0.5))


def test_loss_zero_for_perfect_prediction():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    hm, off, _ = encode_targets(corners, 256, 64, 2.0)
    logits = torch.logit(hm.clamp(1e-6, 1 - 1e-6))
    loss = CornerLoss()(logits, off, corners, 256, 64)
    assert loss["total"].item() < 1e-3


def test_loss_large_for_wrong_prediction():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    loss = CornerLoss()(torch.full((1, 4, 64, 64), -10.0), torch.zeros(1, 8, 64, 64),
                        corners, 256, 64)
    assert loss["heatmap"].item() > 1.0
