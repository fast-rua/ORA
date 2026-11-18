"""What the board keeps of a completed depth frame: the sensor's depth wherever it measured, the model's only
in holes within the fill radius of a measurement."""

from __future__ import annotations

import numpy as np

from ora.cell.depth.client import fuse


def test_the_sensor_wins_wherever_it_measured() -> None:
    raw = np.full((8, 8), 1.0, np.float32)
    raw[3, 3] = 0.0
    completed = np.full((8, 8), 2.0, np.float32)

    fused = fuse(raw, completed)

    assert fused.dtype == np.float32
    assert fused[3, 3] == 2.0
    assert np.all(np.delete(fused.ravel(), 3 * 8 + 3) == 1.0)


def test_holes_far_from_any_measurement_stay_empty() -> None:
    raw = np.zeros((1, 40), np.float32)
    raw[0, 0] = 1.0
    completed = np.full((1, 40), 3.0, np.float32)

    fused = fuse(raw, completed, radius_px=10)

    assert fused[0, 0] == 1.0
    assert np.all(fused[0, 1:8] == 3.0)                                  # well inside the radius
    assert np.all(fused[0, 14:] == 0.0)                                  # well outside it


def test_a_frame_without_holes_is_the_raw_frame_and_one_without_measurements_is_empty() -> None:
    raw = np.full((4, 4), 0.7, np.float32)
    assert fuse(raw, np.ones((4, 4), np.float32)) is raw
    assert np.all(fuse(np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)) == 0.0)
