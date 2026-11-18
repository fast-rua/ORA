"""ORA's changes to a robot's V2 file, without V2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.environment.controllers.depth_world import ROBOT_WORLDS, tool_from_hand, world_parameters
from ora.environment.controllers.motion_planner import IMPLEMENTATIONS, ROBOT_FILE_ROBOTS, ROBOTS
from ora.environment.planners.curobo_v2 import robots, yam

ROOT = Path(__file__).resolve().parents[5]


def _sdk_file():
    return {"planner": {"frame_bias": [0.0, 0.0, 0.0]}, "robot_cfg": {"kinematics": {
        "format_version": 2.0, "base_link": "base_link", "tool_frames": ["link6"], "urdf_path": "/a/X5A.urdf",
        "collision_link_names": ["base_link", "link6"], "debug": None,
        "collision_spheres": {"base_link": [{"center": [0, 0, 0], "radius": 0.04}],
                              "link6": [{"center": [0, 0, 0], "radius": 0.03}]},
        "collision_sphere_buffer": 0.0, "self_collision_ignore": {"link6": ["link5"]},
        "self_collision_buffer": {"link6": -0.01},
        "cspace": {"joint_names": ["joint1", "joint8", "joint7"], "default_joint_position": [0.0, 0.02, 0.02],
                   "max_jerk": [500.0] * 3, "unknown": 1},
    }}}


def test_the_x5_gets_its_fingers_locked_open_and_a_slot_for_the_load():
    data = robots.robot_config("arx_x5", _sdk_file(), held_link="attached_object", held_slots=32)
    kinematics = data["robot_cfg"]["kinematics"]
    assert "debug" not in kinematics and "unknown" not in kinematics["cspace"] and "planner" not in data
    assert kinematics["collision_link_names"] == ["base_link", "link6", "link7", "link8", "attached_object"]
    assert len(kinematics["collision_spheres"]["link7"]) == 4 and len(kinematics["collision_spheres"]["link8"]) == 3
    assert kinematics["lock_joints"] == {"joint7": robots.X5_FINGER_OPEN_M, "joint8": robots.X5_FINGER_OPEN_M}
    assert kinematics["extra_links"]["attached_object"]["parent_link_name"] == "link6"
    assert kinematics["extra_collision_spheres"] == {"attached_object": 32}
    assert kinematics["collision_sphere_buffer"] == robots.SPHERE_BUFFER_M
    assert "link7" in kinematics["self_collision_ignore"]["link6"] and "link5" in kinematics["self_collision_ignore"]["link6"]
    assert robots.sphere_set_sha256(kinematics) != robots.sphere_set_sha256(_sdk_file()["robot_cfg"]["kinematics"])


def test_the_x5_plans_within_its_real_joints_not_the_descriptions_placeholder():
    cspace = robots.robot_config("arx_x5", _sdk_file(), held_link="attached_object", held_slots=32)[
        "robot_cfg"]["kinematics"]["cspace"]
    limits = [robots.X5_URDF_LIMIT_RAD - clip for clip in cspace["position_limit_clip"][:5]]
    assert limits == pytest.approx(robots.X5_JOINT_LIMITS_RAD)
    assert len(cspace["position_limit_clip"]) == 6 and cspace["position_limit_clip"][5] == 0.0  # the file's +/-3.14
    # The measured extremes the limits hold; joint 1's runs wound by a whole turn stay out (see the constant).
    assert all(abs(low) <= limit and high <= limit for (low, high), limit in zip(
        ((-3.649, 3.377), (-0.298, 3.175), (-0.044, 3.809), (-2.395, 1.877)), limits))
    assert limits[4] < 1.565  # under where joint 5 falls behind its command, near its stop at 90 degrees
    assert limits[0] < 7.617


def test_the_panda_keeps_its_bundled_file_shape():
    bundled = {"robot_cfg": {"kinematics": {
        "collision_link_names": ["panda_leftfinger", "panda_rightfinger"],
        "collision_spheres": {"panda_leftfinger": [], "panda_rightfinger": []}, "lock_joints": {"a": 1}}}}
    data = robots.robot_config("franka_panda", bundled, held_link="attached_object", held_slots=4)
    assert data["robot_cfg"]["kinematics"]["lock_joints"] == {"a": 1}
    assert data["robot_cfg"]["kinematics"]["collision_link_names"] == ["panda_leftfinger", "panda_rightfinger"]


def _yam_kinematics():
    data = robots.robot_config(yam.ROBOT, robots.robot_file(yam.ROBOT, "/i2rt/yam_station_linear_4310_d405.urdf"),
                               held_link="attached_object", held_slots=32)
    assert data["load_dynamics"] is False
    return data["robot_cfg"]["kinematics"]


def test_the_yam_is_one_arm_of_the_station_urdf_with_its_public_tcp():
    kinematics = _yam_kinematics()
    assert kinematics["urdf_path"] == "/i2rt/yam_station_linear_4310_d405.urdf"
    assert kinematics["base_link"] == "left_base" and kinematics["tool_frames"] == ["left_tcp"]
    assert ROBOTS[yam.ROBOT] == {"joint_names": yam.JOINT_NAMES, "tool_frame": yam.TOOL_FRAME}
    assert kinematics["cspace"]["joint_names"] == [*yam.JOINT_NAMES, "left_joint7", "left_joint8"]
    assert yam.ROBOT in IMPLEMENTATIONS["curobo_v2"].robots
    assert yam.ROBOT in ROBOT_FILE_ROBOTS and yam.ROBOT in robots.ROBOT_FILE_ROBOTS and yam.ROBOT in robots.URDF_ROBOTS
    tcp, held = kinematics["extra_links"]["left_tcp"], kinematics["extra_links"]["attached_object"]
    assert tcp["parent_link_name"] == held["parent_link_name"] == "left_gripper"
    assert tcp["fixed_transform"] == list(yam.TCP_IN_GRIPPER)
    record = json.loads((ROOT / "tests/data/fk-parity/yam_linear_4310.json").read_text(encoding="utf-8"))
    x, y, z, qw, qx, qy, qz = yam.TCP_IN_GRIPPER
    assert (qw, qx, qy, qz) == (0.0, 0.0, 0.0, 1.0), "the flange's axes turned a half turn about z"
    assert np.allclose(record["tool"], [[-1, 0, 0, x], [0, -1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], atol=1e-6)
    # The hand convention's jaw line (y) is the TCP's closing line (x), its approach the TCP's z.
    turn = tool_from_hand(yam.ROBOT)[:3, :3]
    assert np.allclose(turn[:, 1], [1, 0, 0]) and np.allclose(turn[:, 2], [0, 0, 1])
    assert world_parameters("none", yam.ROBOT) is None
    assert world_parameters("public_depth_voxel_v1", yam.ROBOT) is ROBOT_WORLDS[yam.ROBOT][0]


def test_the_yam_spheres_cover_every_link_and_name_their_hash():
    kinematics = _yam_kinematics()
    assert kinematics["collision_link_names"] == [*yam.COLLISION_LINKS, "attached_object"]
    assert all(kinematics["collision_spheres"][link] for link in yam.COLLISION_LINKS)
    assert kinematics["collision_sphere_buffer"] == robots.SPHERE_BUFFER_M
    assert kinematics["extra_collision_spheres"] == {"attached_object": 32}
    assert robots.sphere_set_sha256(kinematics) == robots.YAM_SPHERE_SET_SHA256
    assert sum(len(spheres) for spheres in kinematics["collision_spheres"].values()) == 190
    assert kinematics["lock_joints"] == {"left_joint7": yam.FINGER_OPEN_M, "left_joint8": yam.FINGER_OPEN_M}
    assert yam.ROBOT in robots.FINGERS_PLANNED_OPEN
    assert {link for link, part in robots.LINK_PARTS[yam.ROBOT].items() if part == "fingers"} == {
        "left_tip_left", "left_tip_right"}


def test_the_yam_self_collision_pairs_and_padding():
    kinematics = _yam_kinematics()
    ignored = kinematics["self_collision_ignore"]
    links = set(kinematics["collision_link_names"])
    assert all(link in links and set(others) <= links for link, others in ignored.items())
    pairs = {frozenset((link, other)) for link, others in ignored.items() for other in others}
    chain = ["left_base", *(f"left_link{index}" for index in range(1, 6)), "left_gripper"]
    assert all(frozenset(pair) in pairs for pair in zip(chain, chain[1:])), "neighbours are not checked"
    for wrist in ("left_link5", "left_gripper", "left_tip_left", "left_tip_right", "left_camera"):
        for low in ("left_base", "left_link1", "left_link2"):
            assert frozenset((wrist, low)) not in pairs, "the wrist is checked against the shoulder, as the station is"
    assert set(ignored["attached_object"]) == set(yam.HELD_IGNORE)
    padding = kinematics["self_collision_buffer"]
    assert padding["attached_object"] == 0.0
    for link in yam.COLLISION_LINKS:  # the world buffer comes back off; the fold's on link1 and link2
        fold = yam.FOLD_BUFFER_M if link in ("left_link1", "left_link2") else 0.0
        assert padding[link] == pytest.approx(fold - robots.SPHERE_BUFFER_M)
