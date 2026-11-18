"""The kit's pure geometry: the dominant plane under outliers, one point per voxel, and joint readings against
the model limits (padded as i2rt pads them) and the recorded rest pose."""

from __future__ import annotations

import numpy as np

from ora.cell.geometry import JOINT_LIMIT_PAD_RAD, REST_DEVIATION_RAD, fit_plane, joint_sanity, voxel_downsample


def test_fit_plane_finds_the_table_among_clutter() -> None:
    rng = np.random.default_rng(3)
    table = np.c_[rng.uniform(-0.5, 0.5, (600, 2)), 0.02 + rng.normal(0.0, 0.001, 600)]
    clutter = rng.uniform([-0.5, -0.5, 0.1], [0.5, 0.5, 0.6], (200, 3))
    points = np.vstack([table, clutter])

    normal, offset, inliers = fit_plane(points)

    assert abs(abs(normal[2]) - 1.0) < 1e-3
    assert abs(abs(offset) - 0.02) < 1e-3
    assert inliers[:600].mean() > 0.99
    assert not inliers[600:].any()
    assert np.allclose(fit_plane(points)[0], normal)                    # seeded: the same answer every time


def test_voxel_downsample_keeps_the_first_point_of_each_cube() -> None:
    points = np.array([[0.001, 0.001, 0.0], [0.004, 0.002, 0.0], [0.012, 0.0, 0.0], [0.0, 0.0, -0.001],
                       [0.013, 0.001, 0.0]])

    kept = voxel_downsample(points, 0.01)

    assert kept.tolist() == [0, 2, 3]
    assert len(voxel_downsample(np.random.default_rng(0).uniform(0, 0.1, (5000, 3)), 0.05)) == 8


def test_joint_sanity_passes_readings_inside_the_padded_limits() -> None:
    limits = {"left": np.array([[-1.0, 1.0]] * 6)}
    q = {"left": np.array([0.0, 1.0 + JOINT_LIMIT_PAD_RAD - 0.01, 0.0, 0.0, 0.0, -1.0])}

    report = joint_sanity(q, limits, None)

    assert report == {"outside_limits": {}, "rest_deviation_rad": None, "at_rest": None, "passed": True}


def test_joint_sanity_names_joints_outside_the_padded_limits() -> None:
    limits = {"left": np.array([[-1.0, 1.0]] * 6), "right": np.array([[-1.0, 1.0]] * 6)}
    q = {"left": np.array([0.0, 1.2, 0.0, 0.0, 0.0, -3.0]), "right": np.zeros(6)}

    report = joint_sanity(q, limits, None)

    assert report["passed"] is False
    assert report["outside_limits"] == {"left": {"joint2": 1.2, "joint6": -3.0}}


def test_joint_sanity_measures_the_distance_from_rest() -> None:
    limits = {"left": np.array([[-1.0, 1.0]] * 6)}
    rest = {"left": [0.0] * 6}

    near = joint_sanity({"left": np.full(6, REST_DEVIATION_RAD / 2)}, limits, rest)
    away = joint_sanity({"left": np.array([0.0, 0.0, 2 * REST_DEVIATION_RAD, 0.0, 0.0, 0.0])}, limits, rest)

    assert near["at_rest"] is True
    assert near["rest_deviation_rad"]["left"] == [REST_DEVIATION_RAD / 2] * 6
    assert away["at_rest"] is False and away["passed"] is True


def test_wrist_table_check_takes_widened_limits() -> None:
    from ora.cell.geometry import wrist_table_check

    plane = {"tilt_deg": 3.4, "height_m": -0.115, "inlier_fraction": 0.8, "points": 900}

    assert wrist_table_check(plane, -0.09)["passed"] is False
    assert wrist_table_check(plane, -0.09, max_tilt_deg=4.5, max_height_gap_m=0.035)["passed"] is True
