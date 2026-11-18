"""Principal axes, covariances, the support plane and point transforms do not depend on the BLAS kernel the machine
runs."""

from __future__ import annotations

import numpy as np
import pytest

from ora.agent.code_mode.ora_lib import geometry as sandbox_geometry
from ora.agent.toolkit.lib import _principal_axes

IMPLEMENTATIONS = (_principal_axes.principal_axes, sandbox_geometry._principal_axes)


def _rotating_eigh(monkeypatch, angle: float) -> None:
    """eigh as a kernel that returns its repeated eigenspace's basis turned by ``angle``."""
    original = np.linalg.eigh

    def eigh(matrix):
        values, vectors = original(matrix)
        vectors = vectors.copy()
        for start in range(len(values) - 1):
            if abs(values[start + 1] - values[start]) <= 1e-12 * max(1.0, abs(values).max()):
                c, s = np.cos(angle), np.sin(angle)
                pair = vectors[:, start:start + 2].copy()
                vectors[:, start], vectors[:, start + 1] = c * pair[:, 0] + s * pair[:, 1], -s * pair[:, 0] + c * pair[:, 1]
                break
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", eigh)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("covariance", [np.diag([0.004, 0.004, 0.0]), np.diag([0.003, 0.003])], ids=["square", "disc"])
def test_a_repeated_variance_gets_the_same_axes_whichever_basis_eigh_returns(monkeypatch, implementation, covariance):
    answers = []
    for angle in (0.0, 0.4, 1.3, -2.2):
        _rotating_eigh(monkeypatch, angle)
        answers.append(implementation(covariance))
    for values, axes in answers:
        assert np.array_equal(values, answers[0][0])
        assert np.allclose(axes, answers[0][1], atol=1e-12)
    values, axes = answers[0]
    assert np.allclose(axes[:, :2], np.eye(len(covariance))[:, :2], atol=1e-12)  # x, then y
    assert values[0] == values[1]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_distinct_variances_give_eighs_axes_greatest_first(implementation):
    """A 2 x 2 keeps eigh's own axes; a 3 x 3's Jacobi axes are eigh's to rounding, up to sign."""
    rotation = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    turn = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    for covariance in (rotation @ np.diag([0.01, 0.004, 0.001]) @ rotation.T, turn @ np.diag([0.01, 0.002]) @ turn.T):
        values, axes = implementation(covariance)
        expected_values, expected_axes = np.linalg.eigh(covariance)
        assert np.allclose(values, expected_values[::-1], rtol=1e-12, atol=0)
        if len(covariance) == 2:
            assert np.array_equal(axes, expected_axes[:, ::-1])
        else:
            assert np.allclose(np.abs(np.sum(axes * expected_axes[:, ::-1], axis=0)), 1.0, rtol=0, atol=1e-12)
            assert np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-14)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_3x3_gets_the_same_axes_however_eigh_rounds(monkeypatch, implementation):
    """LAPACK reduces a 3 x 3 by Householder steps whose BLAS calls round by kernel; principal_axes does not use it."""
    rotation = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))[0]
    covariance = rotation @ np.diag([0.01, 0.004, 0.001]) @ rotation.T
    original, answers = np.linalg.eigh, []

    def rounding(ulps):
        """eigh as a kernel whose eigenvectors land ``ulps`` units in the last place from this machine's."""
        def eigh(matrix):
            values, vectors = original(matrix)
            return values, vectors + ulps * np.spacing(vectors)
        return eigh

    for ulps in (0, 1, -3):
        monkeypatch.setattr(np.linalg, "eigh", rounding(ulps))
        answers.append(implementation(covariance))
    for values, axes in answers:
        assert np.array_equal(values, answers[0][0]) and np.array_equal(axes, answers[0][1])


def test_scatter_and_covariance_sum_in_numpy_and_agree_with_blas_to_rounding():
    points = np.random.default_rng(7).normal(size=(500, 3)) * [0.05, 0.02, 0.01] + [0.3, -0.1, 0.8]
    centered = points - np.median(points, axis=0)
    scatter = _principal_axes.scatter(centered)
    assert np.array_equal(scatter, [[np.sum(centered[:, i] * centered[:, j]) for j in range(3)] for i in range(3)])
    assert np.array_equal(scatter, scatter.T)
    assert np.allclose(scatter, centered.T @ centered, rtol=1e-12, atol=0)
    assert np.allclose(_principal_axes.covariance(points), np.cov(points, rowvar=False), rtol=1e-12, atol=0)


def test_the_support_plane_is_lstsqs_plane_to_rounding():
    from ora.agent.toolkit.lib import _vision_support

    rng = np.random.default_rng(11)
    xy = rng.uniform(-0.1, 0.1, size=(200, 2)) + [0.4, -0.2]
    points = np.column_stack([xy, 0.03 * xy[:, 0] - 0.02 * xy[:, 1] + 0.75 + rng.normal(scale=1e-3, size=200)])
    line = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 2, 20), np.linspace(1, 3, 20)])
    for cloud, exact in ((points, False), (line, True)):  # collinear: lstsq's own minimum-norm answer
        design = np.column_stack([cloud[:, 0], cloud[:, 1], np.ones(len(cloud))])
        expected = np.linalg.lstsq(design, cloud[:, 2], rcond=None)[0]
        found = _vision_support._least_squares_plane(cloud)
        assert np.array_equal(found, expected) if exact else np.allclose(found, expected, rtol=1e-11, atol=0)


def test_a_mask_is_localized_without_lapack_or_a_covariance_product(monkeypatch):
    """The mask localization, its preview's calibration and the observed-extent box: no call that rounds by kernel."""
    from ora.agent.toolkit.lib import _grasp_extent, _vision_card, _vision_measurement

    depth = np.full((120, 160), 1.0)  # a table 1 m below a camera looking straight down from 1.5 m
    mask = np.zeros(depth.shape, dtype=bool)
    mask[40:70, 60:100] = True
    depth[mask] = 0.9  # a 10 cm box on it
    calibration = {"intrinsic_cv": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
                   "extrinsic_cv": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.5]]}

    def refused(*args, **kwargs):
        raise AssertionError("rounds by BLAS kernel")

    original = np.linalg.eigh
    for module, name in ((np.linalg, "inv"), (np.linalg, "lstsq"), (np, "cov")):
        monkeypatch.setattr(module, name, refused)
    monkeypatch.setattr(np.linalg, "eigh", lambda matrix: original(matrix) if len(matrix) <= 2 else refused())
    found = _vision_measurement._mask_depth_localization(mask=mask, depth_m=depth, calibration=calibration,
                                                         extended_frame=True)
    assert found["status"] == "localized" and found["footprint_extent_m"] is not None
    assert found["support_z_m"] == pytest.approx(0.5) and found["height_above_support_m"] == pytest.approx(0.1)
    assert np.allclose(_vision_card._calibration(calibration)[1][:3, 3], [0.0, 0.0, 1.5])
    box = _grasp_extent.fusion_box(sources_by_camera={"head": {"calibration": calibration, "depth": depth}},
                                   camera_names=["head"], max_depth_m=2.0)
    assert box["z"][0] < 1.5 < box["z"][1]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_tilted_repeated_plane_gets_an_orthonormal_basis_of_that_plane(implementation):
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    covariance = 0.005 * (np.eye(3) - np.outer(normal, normal))
    values, axes = implementation(covariance)
    assert values[0] == values[1] and values[2] == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    assert np.allclose(axes[:, :2].T @ normal, 0.0, atol=1e-12)


def test_the_elementwise_affine_transform_and_inverse_agree_with_blas_to_rounding():
    from scipy.spatial.transform import Rotation

    from ora.agent.toolkit.lib import _rigid

    rng = np.random.default_rng(5)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3)
    points = rng.normal(size=(50, 3))
    assert np.allclose(_rigid.apply(points, matrix), points @ matrix[:3, :3].T + matrix[:3, 3], rtol=0, atol=1e-14)
    assert np.allclose(_rigid.invert(matrix), np.linalg.inv(matrix), rtol=0, atol=1e-14)
    assert np.allclose(_rigid.apply(_rigid.apply(points, matrix), _rigid.invert(matrix)), points, rtol=0, atol=1e-13)
