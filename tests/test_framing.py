import numpy as np
import pytest

from src.utils.framing import reframe
from src.utils.preprocess import PAD_VALUE

# Unit square page in the middle of a 100x100 frame.
CORNERS = np.array([[25, 25], [75, 25], [75, 75], [25, 75]], np.float32)


def _img(n=100):
    return np.full((n, n, 3), 200, np.uint8)


def test_reframe_is_deterministic_for_a_seeded_rng():
    a_img, a_pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(7))
    b_img, b_pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(7))
    assert np.array_equal(a_img, b_img)
    assert np.array_equal(a_pts, b_pts)


def test_reframe_output_shape_and_padding():
    img, pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(0))
    assert img.shape == (256, 256, 3)
    assert pts.shape == (4, 2)


def test_reframe_shrinks_and_leaves_grey_around_the_scene():
    # scale forced below 1 so the 100x100 scene cannot fill 256x256
    img, pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(1),
                       min_scale=0.5, max_scale=0.5)
    assert np.any(img == PAD_VALUE)
    # scene is 50x50, so the page (half of it) is ~25 px wide
    span = pts[:, 0].max() - pts[:, 0].min()
    assert 20 < span < 30


def test_reframe_zooms_in():
    img, pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(2),
                       min_scale=1.35, max_scale=1.35)
    # 100px scene upscaled to 135px on a 256px canvas still fits, so the
    # canvas must show the full scene; the page grows to ~67px
    span = pts[:, 0].max() - pts[:, 0].min()
    assert 60 < span < 75
    assert not np.all(img == PAD_VALUE)


def test_reframe_maps_corners_with_the_same_affine_as_the_pixels():
    # A distinctive dot at each page corner must land on the mapped corner.
    img = np.full((100, 100, 3), 10, np.uint8)
    for x, y in CORNERS.astype(int):
        img[y - 1:y + 2, x - 1:x + 2] = 255
    out, pts = reframe(img, CORNERS, 256, rng=np.random.default_rng(3),
                       min_scale=0.8, max_scale=0.8)
    for (x, y) in pts:
        yi, xi = int(round(y)), int(round(x))
        patch = out[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
        assert patch.size and patch.max() > 200, f"no dot near mapped corner {(x, y)}"


def test_reframe_preserves_corner_order():
    _, pts = reframe(_img(), CORNERS, 256, rng=np.random.default_rng(4),
                     min_scale=0.9, max_scale=0.9)
    assert pts[0, 0] < pts[1, 0] and pts[3, 0] < pts[2, 0]   # TL/TR, BL/BR
    assert pts[0, 1] < pts[3, 1] and pts[1, 1] < pts[2, 1]   # TL/BL, TR/BR


def test_reframe_can_push_corners_outside_the_frame():
    # Deliberately aggressive zoom: the caller must be able to detect and
    # reject this, which is why reframe never clamps.
    _, pts = reframe(_img(), CORNERS, 64, rng=np.random.default_rng(5),
                     min_scale=1.34, max_scale=1.34)
    assert pts.min() < 0 or pts.max() > 64
