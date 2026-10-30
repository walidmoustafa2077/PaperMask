import torch
from src.metrics.corners import CornerMetrics


def test_perfect_prediction():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    m = CornerMetrics()
    m.update(gt.clone(), gt)
    r = m.compute()
    assert r["mae_px"] == 0 and r["iou"] == 1.0 and r["pck1"] == 1.0


def test_offset_of_one_pixel():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    pred = gt + 1.0
    m = CornerMetrics()
    m.update(pred, gt)
    r = m.compute()
    assert abs(r["mae_px"] - 1.0) < 1e-5
    assert r["pck1"] == 1.0          # 1 px == 1 % of 100... no: threshold is 1% of the diagonal
    assert r["per_corner_mae"] == [1.0, 1.0, 1.0, 1.0]


def test_invalid_quad_counted():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    pred = gt.clone()
    pred[0, 2] = pred[0, 0]          # collapse one corner
    m = CornerMetrics()
    m.update(pred, gt)
    assert m.compute()["invalid_frac"] == 1.0
