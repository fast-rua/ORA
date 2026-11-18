"""The FK parity check, and each recorded robot's chain against what its simulator measured."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity

ROOT = Path(__file__).resolve().parents[4]


def _shift(frame, offset):
    moved = np.array(frame, dtype=float)
    moved[:3, 3] += offset
    return moved.tolist()


def test_parity_reports_the_worst_sample_and_its_verdict():
    close = {"simulator": np.eye(4).tolist(), "planner": _shift(np.eye(4), [0.001, 0.0, 0.0])}
    far = {"simulator": np.eye(4).tolist(), "planner": _shift(np.eye(4), [0.0065, 0.0, 0.0])}
    assert parity([close], *ROBOT_TOLERANCES["arx_x5"])["passed"] is True
    report = parity([close, far], *ROBOT_TOLERANCES["arx_x5"])
    assert report["passed"] is False and report["position_error_m"] == pytest.approx(0.0065)
    with pytest.raises(ValueError):
        parity([], 0.002, 0.02)


def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _chain_fk(chain, joints):
    transform = np.eye(4)
    for link, angle in zip(chain, joints):
        origin = np.eye(4)
        origin[:3, :3], origin[:3, 3] = _rpy(*link["rpy"]), link["xyz"]
        x, y, z = link["axis"]
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        turn = np.eye(4)
        turn[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew
        transform = transform @ origin @ turn
    return transform


@pytest.mark.parametrize("fixture", ["arx_x5", "aloha_agilex", "robocasa_panda_omron", "franka_robotiq",
                                     "yam_linear_4310"])
def test_the_recorded_simulator_poses_match_the_robots_description(fixture):
    """The fixture holds the description's chain and its tool link as the simulator and the planner measured it."""
    record = json.loads((ROOT / f"tests/data/fk-parity/{fixture}.json").read_text(encoding="utf-8"))
    robot = record["robot"]
    tool = np.asarray(record.get("tool", np.eye(4).tolist()))
    samples = [{"simulator": sample["simulator"],
                "planner": (_chain_fk(record["chain"], sample["joints"]) @ tool).tolist()}
               for sample in record["samples"]]
    assert parity(samples, *ROBOT_TOLERANCES[robot])["passed"]
    assert parity(record["samples"], *ROBOT_TOLERANCES[robot])["passed"]
