"""The live view's depth cleaning: pixels on a depth jump larger than the ratio of the nearer depth are dropped
on both sides of the jump; smooth surfaces and missing pixels are left as they are."""

from __future__ import annotations

import numpy as np

from ora.cell.operator.live import FLYING_PIXEL_RATIO, drop_flying_pixels


def test_a_smooth_slope_is_kept() -> None:
    depth = np.tile(np.linspace(0.50, 0.51, 10), (6, 1))
    assert np.array_equal(drop_flying_pixels(depth), depth)


def test_both_sides_of_a_depth_jump_are_dropped() -> None:
    depth = np.full((5, 6), 0.5)
    depth[:, 3:] = 1.0                                                   # a step between columns 2 and 3

    cleaned = drop_flying_pixels(depth)

    assert np.all(cleaned[:, 2:4] == 0.0)
    assert np.all(cleaned[:, :2] == 0.5) and np.all(cleaned[:, 4:] == 1.0)


def test_a_jump_between_rows_is_dropped_too() -> None:
    depth = np.full((6, 4), 0.8)
    depth[4:, :] = 0.4

    cleaned = drop_flying_pixels(depth)

    assert np.all(cleaned[3:5] == 0.0) and np.all(cleaned[:3] == 0.8) and np.all(cleaned[5:] == 0.4)


def test_missing_pixels_make_no_edge_and_the_ratio_is_relative() -> None:
    depth = np.full((3, 5), 1.0)
    depth[1, 2] = 0.0
    depth[0, 4] = 1.0 + FLYING_PIXEL_RATIO / 2                            # under the ratio: kept

    cleaned = drop_flying_pixels(depth)

    assert np.array_equal(cleaned, depth)
    assert np.all(drop_flying_pixels(np.zeros((3, 3))) == 0.0)
