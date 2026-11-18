"""The station model takes a cell's measured corrections: joint values stay encoder readings,
an encoder zero offset moves the model under them, and a base correction moves only that
arm, its wrist camera with it. Needs MuJoCo and i2rt (the YAM interpreter)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("i2rt")

from ora.environment.backends.yam.kinematics import StationKinematics  # noqa: E402
from ora.environment.backends.yam.scene import build_cell_model  # noqa: E402

Q = {"left": np.array([0.1, 0.8, 0.5, 0.2, -0.3, 0.4]), "right": np.array([-0.2, 1.0, 0.6, -0.1, 0.2, -0.3])}
OFFSET = np.array([0.0, 0.04, -0.02, 0.03, 0.0, 0.0])


def _model() -> StationKinematics:
    return StationKinematics(build_cell_model(table=None))


def test_joint_offsets_shift_the_model_but_not_the_readings() -> None:
    plain, corrected = _model(), _model()
    corrected.joint_offsets = {"left": OFFSET}
    plain.set_state({"left": Q["left"] + OFFSET, "right": Q["right"]})
    corrected.set_state(Q)

    assert np.allclose(corrected.tcp("left"), plain.tcp("left"))
    assert np.allclose(corrected.arm_q("left"), Q["left"])
    assert np.allclose(corrected.arm_limits["left"], plain.arm_limits["left"] - OFFSET[:, None])
    result = corrected.ik("left", corrected.fk("left", Q["left"]), Q["left"] + 0.05)
    assert result.ok and np.allclose(result.q, Q["left"], atol=2e-3)


def test_a_base_correction_moves_only_that_arm() -> None:
    plain, corrected = _model(), _model()
    shift = np.eye(4)
    shift[:3, 3] = [0.0, 0.08, 0.02]
    corrected.set_base_corrections({"right": shift})
    corrected.set_base_corrections({"right": shift})          # applying twice is applying once
    plain.set_state(Q)
    corrected.set_state(Q)

    assert np.allclose(corrected.tcp("left"), plain.tcp("left"))
    assert np.allclose(corrected.tcp("right"), shift @ plain.tcp("right"))
    assert np.allclose(corrected.wrist_camera("right"), shift @ plain.wrist_camera("right"))
    assert np.allclose(corrected.arm_q("right"), Q["right"])
    corrected.set_base_corrections({})
    corrected.set_state(Q)
    assert np.allclose(corrected.tcp("right"), plain.tcp("right"))


def test_the_folded_zero_pose_brings_the_wrist_nearer_its_own_base_than_a_raised_elbow() -> None:
    model = _model()
    model.set_state({"left": np.zeros(6), "right": np.zeros(6)})
    folded = model.self_distances("left")
    model.set_state({"left": np.array([0.0, 0.5, 1.5, 0.0, 0.0, 0.0]), "right": np.zeros(6)})
    raised = model.self_distances("left")
    assert folded.shape == raised.shape and len(folded) > 0
    assert folded.min() < 0.03 < raised.min()


def test_the_loaded_finger_geometry_is_the_public_descriptors_and_leaves_the_state_alone() -> None:
    """The worker verifies this measurement against the bound descriptor at every round's start."""
    import hashlib
    import json
    from pathlib import Path

    from ora.contracts.robot_geometry import GEOMETRY_BINDING_SCHEMA, verify_loaded_robot_geometry

    path = Path(__file__).resolve().parents[5] / "ora/policy/embodiments/yam-station/descriptor-public-geometry-v1.json"
    descriptor = json.loads(path.read_text())
    binding = {"schema": GEOMETRY_BINDING_SCHEMA, "descriptor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "geometry": {item["end_effector_id"]: item["contact_geometry"]
                            for item in descriptor["geometry"]["grippers"]}}
    model = _model()
    model.set_state(Q, {"left": 0.01, "right": 0.02})
    before = model.tcp("left").copy()
    assert verify_loaded_robot_geometry(binding, model.contact_geometry())["passed"] is True
    np.testing.assert_array_equal(model.tcp("left"), before)
