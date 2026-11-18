"""The host's copy of the shared planner's Panda spheres (``robot_spheres.py``): the planner's set, on its chain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.environment.controllers import robot_spheres as R
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity
from ora.environment.planners.curobo_v2 import robots

ROOT = Path(__file__).resolve().parents[4]
#: The ``collision_spheres_sha256`` the cuRobo V2 worker named in its ready answer for ``franka_panda`` (RoboCasa smoke
#: rc-ab-952eeed23, ``motion-planner/binding.json``, V2 revision d17b54ce, ``franka.yml`` with ORA's overrides).
PLANNER_SPHERES_SHA256 = "c4e353852f62d2736547dce201cf741f74926664efb0524165405b6acf89ec18"


def test_the_table_is_the_set_the_planner_names():
    assert R.SPHERE_SET_SHA256 == PLANNER_SPHERES_SHA256
    kinematics = {"collision_link_names": list(R.COLLISION_LINKS),
                  "collision_spheres": {link: [{"center": list(row[:3]), "radius": row[3]} for row in rows]
                                        for link, rows in R.SPHERES.items()},
                  "collision_sphere_buffer": R.BUFFER_M,
                  "extra_collision_spheres": {"attached_object": R.HELD_SLOTS}}
    assert robots.sphere_set_sha256(kinematics) == R.SPHERE_SET_SHA256, "hashed as the planner hashes its own"
    assert R.BUFFER_M == robots.SPHERE_BUFFER_M
    for link, spheres in robots.OVERRIDES["franka_panda"]["collision_spheres"].items():
        assert [(*sphere["center"], sphere["radius"]) for sphere in spheres] == list(R.SPHERES[link])


def test_the_chain_lands_on_the_planner_s_hand():
    """The recorded parity samples hold the planner's own ``panda_hand`` at each joint vector."""
    record = json.loads((ROOT / "tests/data/fk-parity/robocasa_panda_omron.json").read_text(encoding="utf-8"))
    samples = [{"simulator": R.hand(sample["joints"][:7]).tolist(), "planner": sample["planner"]}
               for sample in record["samples"]]
    report = parity(samples, *ROBOT_TOLERANCES[R.ROBOT])
    assert report["passed"] and report["position_error_m"] < 1e-6 and report["rotation_error"] < 1e-6


def test_spheres_carry_the_buffer_and_follow_the_base():
    joints = [0.0, -1.0, 0.0, -2.2, 0.0, 1.5, 0.7]
    local = R.spheres(joints)
    assert local.shape == (sum(len(rows) for rows in R.SPHERES.values()), 4)
    assert local[:2, 3] == pytest.approx([0.03 + R.BUFFER_M] * 2) and local[:2, 2] == pytest.approx([0.085] * 2)
    base = np.eye(4)
    base[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    base[:3, 3] = (1.0, 2.0, 0.8)
    world = R.spheres(joints, base)
    assert world[:, :3] == pytest.approx(local[:, :3] @ base[:3, :3].T + base[:3, 3]) and (world[:, 3] == local[:, 3]).all()
    frames = R.link_frames(joints)
    fingers = frames["panda_leftfinger"][:3, 3] - frames["panda_rightfinger"][:3, 3]
    assert np.linalg.norm(fingers) == pytest.approx(2 * R.FINGER_LOCK_M), "the fingers locked open, as the planner does"
    with pytest.raises(ValueError):
        R.spheres(joints[:6])
