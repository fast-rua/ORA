"""The planner's YAM arm (``yam.py``) against the station model the cell runs: names, limits, TCP, chain, spheres."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("i2rt")

from ora.environment.backends.yam.kinematics import StationKinematics, station_xml_path  # noqa: E402
from ora.environment.controllers import yam_station as Y  # noqa: E402
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity  # noqa: E402
from ora.environment.planners.curobo_v2 import yam  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
RECORD = json.loads((ROOT / "tests/data/fk-parity/yam_linear_4310.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def station():
    return StationKinematics()


@pytest.fixture(scope="module")
def urdf():
    return ET.parse(station_xml_path().with_suffix(".urdf")).getroot()


def _numbers(text):
    return [float(value) for value in text.split()]


def _mujoco_id(station, kind, name):
    return station.mujoco.mj_name2id(station.model, getattr(station.mujoco.mjtObj, kind), name)


def test_links_and_joints_are_the_station_model_s(station, urdf):
    links = {link.get("name") for link in urdf.iter("link")}
    joints = {joint.get("name"): joint for joint in urdf.iter("joint")}
    for arm in Y.ARMS:
        for link in yam.COLLISION_LINKS:
            name = link.replace("left_", f"{arm}_", 1)
            assert name in links and _mujoco_id(station, "mjOBJ_BODY", name) >= 0, name
        for joint in (*yam.JOINT_NAMES, *yam.FINGER_JOINTS):
            name = joint.replace("left_", f"{arm}_", 1)
            assert name in joints and _mujoco_id(station, "mjOBJ_JOINT", name) >= 0, name
    assert joints["left_joint1"].find("parent").get("link") == yam.BASE_LINK
    assert joints["left_joint6"].find("child").get("link") == yam.HAND_LINK
    assert tuple(yam.FINGER_JOINTS) == Y.FINGER_JOINTS["left"]
    assert [link.removeprefix("left_") for link in yam.COLLISION_LINKS if link != "left_camera"] == [
        "base", *(f"link{index}" for index in range(1, 6)), "gripper", "tip_left", "tip_right"]
    assert Y.WRIST_CAMERA_BODIES["left"] in yam.COLLISION_LINKS


def test_joint_limits_are_the_station_model_s(station, urdf):
    joints = {joint.get("name"): joint for joint in urdf.iter("joint")}
    for arm in Y.ARMS:
        limits = [[float(joints[name.replace("left_", f"{arm}_", 1)].find("limit").get(side)) for side in ("lower", "upper")]
                  for name in yam.JOINT_NAMES]
        assert np.allclose(limits, station.model_limits[arm], atol=1e-6)
        for finger in yam.FINGER_JOINTS:
            limit = joints[finger.replace("left_", f"{arm}_", 1)].find("limit")
            assert float(limit.get("lower")) == 0.0 and float(limit.get("upper")) == pytest.approx(yam.FINGER_OPEN_M)
    assert yam.FINGER_OPEN_M == Y.FINGER_OPEN_M


def test_the_tool_frame_is_the_public_tcp(station):
    site = _mujoco_id(station, "mjOBJ_SITE", Y.TCP_SITES["left"])
    assert station.model.site_bodyid[site] == _mujoco_id(station, "mjOBJ_BODY", yam.HAND_LINK)
    turn = np.zeros(9)
    station.mujoco.mju_quat2Mat(turn, station.model.site_quat[site])
    tcp = turn.reshape(3, 3) @ np.asarray(Y.TCP_AXES_IN_SITE).T
    x, y, z, qw, qx, qy, qz = yam.TCP_IN_GRIPPER
    assert np.allclose(station.model.site_pos[site], [x, y, z], atol=1e-9)
    assert np.allclose(tcp, np.asarray(RECORD["tool"])[:3, :3], atol=1e-6)
    assert np.allclose(np.asarray(RECORD["tool"])[:3, 3], [x, y, z], atol=1e-6)


def test_the_recorded_chain_is_the_urdf_s(urdf):
    joints = {joint.get("name"): joint for joint in urdf.iter("joint")}
    for entry, name in zip(RECORD["chain"], yam.JOINT_NAMES, strict=True):
        joint = joints[name]
        assert entry == {"xyz": _numbers(joint.find("origin").get("xyz")), "rpy": _numbers(joint.find("origin").get("rpy")),
                         "axis": _numbers(joint.find("axis").get("xyz"))}


def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _planner_tcp(joints):
    transform = np.eye(4)
    for link, angle in zip(RECORD["chain"], joints):
        origin = np.eye(4)
        origin[:3, :3], origin[:3, 3] = _rpy(*link["rpy"]), link["xyz"]
        axis = np.asarray(link["axis"]) / np.linalg.norm(link["axis"])
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        turn = np.eye(4)
        turn[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew
        transform = transform @ origin @ turn
    return transform @ np.asarray(RECORD["tool"])


def test_the_planner_chain_lands_on_station_kinematics_for_both_arms(station):
    """The live form of the parity check: either arm, in its own base frame, across its limits."""
    rng = np.random.default_rng(7)
    for arm in Y.ARMS:
        station.set_state({other: [0.0] * 6 for other in Y.ARMS})
        body = station.base_body[arm]
        base = np.eye(4)
        base[:3, :3] = station.data.xmat[body].reshape(3, 3)
        base[:3, 3] = station.data.xpos[body] + np.asarray(Y.STATION_TO_WORLD_XYZ_M)
        limits = station.model_limits[arm]
        rows = [np.zeros(6), *rng.uniform(limits[:, 0], limits[:, 1], size=(40, 6))]
        samples = [{"simulator": (np.linalg.inv(base) @ station.fk(arm, q)).tolist(), "planner": _planner_tcp(q).tolist()}
                   for q in rows]
        report = parity(samples, *ROBOT_TOLERANCES[yam.ROBOT])
        assert report["passed"] and report["position_error_m"] < 1e-6, (arm, report)


def test_every_mesh_vertex_lies_within_reach_of_its_link_s_spheres(station):
    """No more of a link's surface lies outside its spheres than the fit allows (``yam.py``: 7.6 mm at most)."""
    model = station.model
    spheres = yam.collision_spheres()
    for link in yam.COLLISION_LINKS:
        body = _mujoco_id(station, "mjOBJ_BODY", link)
        table = np.array([[*sphere["center"], sphere["radius"]] for sphere in spheres[link]])
        for geom in np.flatnonzero(model.geom_bodyid == body):
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            turn = np.zeros(9)
            station.mujoco.mju_quat2Mat(turn, model.geom_quat[geom])
            points = model.mesh_vert[start:start + count] @ turn.reshape(3, 3).T + model.geom_pos[geom]
            outside = (np.linalg.norm(points[:, None] - table[None, :, :3], axis=2) - table[None, :, 3]).min(axis=1)
            assert outside.max() <= 0.008, (link, float(outside.max()))
