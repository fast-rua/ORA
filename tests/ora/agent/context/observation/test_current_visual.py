"""The calibration numbers a current observation publishes are host-independent."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ora.agent.context.observation.current_visual import _affine_inverse, camera_geometry
from ora.agent.evidence.image_binding import PostActionObservationError


def _camera(root: Path, extrinsic: list[list[float]]) -> dict:
    calibration = root / "calibration.json"
    calibration.write_text(json.dumps({
        "intrinsic_cv": [[320.5, 0.0, 128.0], [0.0, 320.5, 96.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": extrinsic}))
    return {"files": {"calibration": {"path": "calibration.json",
                                      "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest()}}}


# A real camera pose (the robocasa golden's agentview): a rotation that is only
# orthonormal to rounding, and a translation that exercises every term.
WORLD_TO_CAMERA = [
    [0.9792973845575502, -0.2019736284348028, 0.013538316572630264, -3.3425823506234017],
    [-0.09726115801450047, -0.5281292977015201, -0.8435755520704576, 0.9135130742447641],
    [0.17752999673412637, 0.8247945794718701, -0.536839828937274, 1.5832727693498066],
    [0.0, 0.0, 0.0, 1.0]]

# Not a rigid transform: an inverse by transposition would be wrong here.
SCALED = [[2.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.25, -2.0], [0.5, 0.0, 4.0, 0.5], [0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("extrinsic", [WORLD_TO_CAMERA, SCALED, np.eye(4).tolist()])
def test_the_plain_python_inverse_is_the_inverse(extrinsic):
    inverse = _affine_inverse(np.asarray(extrinsic, dtype=float))
    assert np.allclose(np.asarray(inverse) @ np.asarray(extrinsic), np.eye(4), atol=1e-12)
    assert np.allclose(inverse, np.linalg.inv(np.asarray(extrinsic)), atol=1e-12)
    assert inverse[3] == [0.0, 0.0, 0.0, 1.0]
    assert all(type(value) is float for row in inverse for value in row)


def test_the_inverse_uses_no_linear_algebra_library(monkeypatch):
    """The digits must not depend on which BLAS/LAPACK kernel this host runs."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("camera geometry must not call numpy.linalg")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    inverse = _affine_inverse(np.asarray(WORLD_TO_CAMERA))
    assert inverse[1][2] == pytest.approx(0.8247945794718701, abs=1e-15)


def test_the_inverse_never_carries_a_negative_zero():
    """An axis-aligned camera makes zero cofactors; their sign is noise the model need not read."""
    axis_aligned = [[0.0, -1.0, 0.0, 0.2], [-1.0, 0.0, 0.0, -0.3], [0.0, 0.0, -1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
    inverse = _affine_inverse(np.asarray(axis_aligned))
    assert np.allclose(inverse, np.linalg.inv(np.asarray(axis_aligned)), atol=1e-12)
    assert "-0.0" not in json.dumps(inverse)
    assert all(math.copysign(1.0, value) > 0 for row in inverse for value in row if value == 0.0)


def test_a_singular_calibration_is_refused():
    singular = [[1.0, 2.0, 3.0, 0.0], [2.0, 4.0, 6.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    with pytest.raises(PostActionObservationError, match="singular"):
        _affine_inverse(np.asarray(singular))


def test_camera_geometry_derives_origin_and_axes_from_the_same_digits(tmp_path):
    geometry = camera_geometry(tmp_path, _camera(tmp_path, WORLD_TO_CAMERA))
    inverse = geometry["cam2world_cv"]
    assert geometry["extrinsic_cv"] == WORLD_TO_CAMERA
    assert geometry["camera_origin_world_m"] == [row[3] for row in inverse[:3]]
    assert geometry["camera_axes_world"] == {
        "right": [row[0] for row in inverse[:3]], "down": [row[1] for row in inverse[:3]],
        "forward": [row[2] for row in inverse[:3]]}
    # A 3x4 extrinsic publishes the same numbers as its 4x4 form.
    assert camera_geometry(tmp_path, _camera(tmp_path, WORLD_TO_CAMERA[:3])) == geometry
    # Rendered the way the observation renders it: nothing but plain floats.
    assert json.loads(json.dumps(geometry)) == geometry
