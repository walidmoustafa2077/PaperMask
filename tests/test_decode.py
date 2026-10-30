import torch
from src.losses.corner_loss import encode_targets
from src.utils.corners import decode_corners, quad_props


def test_decode_is_subpixel_exact():
    corners = torch.tensor([[[66.0, 70.5], [190.0, 62.0], [200.0, 194.0], [58.0, 186.0]]])
    hm, off, _ = encode_targets(corners, 256, 64, 2.0)
    pred, scores = decode_corners(hm, off, 256)
    assert torch.allclose(pred, corners, atol=0.5), (pred, corners)
    assert scores.min() > 0.9


def test_quad_props():
    tl, tr, br, bl = (0, 0), (10, 0), (10, 10), (0, 10)
    area, convex, cw = quad_props(torch.tensor([[tl, tr, br, bl]], dtype=torch.float32)[0])
    assert abs(area - 100) < 1e-3 and convex and cw
