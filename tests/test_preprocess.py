import numpy as np
import pytest

from src.utils.preprocess import (
    PAD_VALUE,
    is_square,
    letterbox,
    letterbox_points,
    unletterbox_points,
)


def test_letterbox_of_square_is_a_pure_resize():
    img = np.full((300, 300, 3), 7, np.uint8)
    out, scale, pad_x, pad_y = letterbox(img, 256)
    assert out.shape == (256, 256, 3)
    assert pad_x == 0 and pad_y == 0
    assert scale == pytest.approx(256 / 300)
    assert np.all(out == 7)


def test_letterbox_pads_a_tall_image_without_stretching():
    img = np.zeros((400, 200, 3), np.uint8)
    out, scale, pad_x, pad_y = letterbox(img, 256)
    assert out.shape == (256, 256, 3)
    # 200x400 scaled to fit 256 -> 128x256, so only the sides get padding
    assert scale == pytest.approx(256 / 400)
    assert pad_y == 0
    assert pad_x == pytest.approx((256 - 128) / 2)
    # padding is flat grey, and the content block is intact (geometry kept)
    assert np.all(out[:, :64] == PAD_VALUE)
    assert np.all(out[:, 192:] == PAD_VALUE)
    assert np.all(out[:, 64:192] == 0)


def test_letterbox_points_round_trip():
    pts = np.array([[0, 0], [199, 399], [100, 200]], np.float32)
    img = np.zeros((400, 200, 3), np.uint8)
    _, scale, pad_x, pad_y = letterbox(img, 256)
    back = unletterbox_points(letterbox_points(pts, scale, pad_x, pad_y), scale, pad_x, pad_y)
    assert np.allclose(back, pts, atol=1e-3)


def test_letterbox_keeps_corner_order_and_inside_frame():
    img = np.zeros((300, 500, 3), np.uint8)
    quad = np.array([[100, 50], [400, 60], [410, 250], [90, 240]], np.float32)
    _, scale, pad_x, pad_y = letterbox(img, 256)
    out = letterbox_points(quad, scale, pad_x, pad_y)
    assert np.all(out >= 0) and np.all(out <= 256)
    # x stays left-to-right, y stays top-to-bottom
    assert out[0, 0] < out[1, 0] and out[3, 0] < out[2, 0]
    assert out[0, 1] < out[3, 1] and out[1, 1] < out[2, 1]


def test_letterbox_rejects_empty_image():
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 10, 3), np.uint8), 256)


def test_is_square():
    assert is_square(np.zeros((64, 64, 3), np.uint8))
    assert not is_square(np.zeros((64, 48, 3), np.uint8))
