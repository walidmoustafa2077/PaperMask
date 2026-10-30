import torch
from src.models.litefpn import DepthwiseSeparableConv, LiteFPN


def test_dwconv_shape():
    m = DepthwiseSeparableConv(16, 32)
    assert m(torch.zeros(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_litefpn_shapes_and_order():
    neck = LiteFPN([16, 24, 40, 96], 64)
    feats = [torch.zeros(2, 16, 64, 64), torch.zeros(2, 24, 32, 32),
             torch.zeros(2, 40, 16, 16), torch.zeros(2, 96, 8, 8)]
    out = neck(feats)
    assert [tuple(t.shape) for t in out] == [(2, 64, 64, 64), (2, 64, 32, 32),
                                             (2, 64, 16, 16), (2, 64, 8, 8)]


def test_top_down_path_is_used():
    """A signal in the coarsest level must reach the finest level."""
    neck = LiteFPN([8, 8, 8, 8], 16).eval()
    feats = [torch.zeros(1, 8, 64, 64), torch.zeros(1, 8, 32, 32),
             torch.zeros(1, 8, 16, 16), torch.zeros(1, 8, 8, 8)]
    feats[3][0, :, 0, 0] = 5.0
    out = neck(feats)
    assert out[0].abs().sum() > 0
