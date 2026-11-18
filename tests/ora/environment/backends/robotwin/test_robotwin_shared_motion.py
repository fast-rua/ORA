"""RoboTwin on the shared motion base: frames, wrapped joints, plan results, reach rows and the binding.

No simulator and no planner: the vendored ``plan_path`` is stood in for by its frame
arithmetic, and the planner client by canned answers.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ora.contracts.planning_diagnostics import public_planning_diagnostics
from ora.contracts.robotwin_reach import CLAIM_BOUNDARY, IK_FAILED, IK_MODEL, SHARED_PLANNER_CHECK
from ora.environment.backends.robotwin import shared_motion
from ora.environment.backends.robotwin.common import RoboTwinLiveError
from ora.environment.backends.robotwin.planning import PLAN_RESULT_SCHEMA, planner_public_summary
from ora.environment.controllers.depth_world import ROBOT_WORLDS, pose_matrix
from ora.environment.controllers.motion_planner import ROBOTS, IMPLEMENTATIONS
from ora.environment.controllers.motion_worker import MotionPlannerError
from ora.environment.planners.curobo_v2 import robots


def _measured():
    return {"left": [0.0] * 6, "right": [0.1] * 6}


def _motion(tmp_path, client=None):
    motion = shared_motion.SharedMotion(client=client, root=tmp_path / "motion-planner")
    motion.planner_from_world = {"left": np.eye(4), "right": np.eye(4)}
    return motion


def _answer(status="planned", reason=None, trajectory=None):
    return {"status": status, "reason": reason, "trajectory": trajectory, "timing_s": {"solve": 0.1, "total": 0.2}}


# --------------------------------------------------------------------------
# Wrapped joints
# --------------------------------------------------------------------------

def test_a_wound_joint_is_planned_within_a_turn_and_the_answer_is_shifted_back():
    joints, offset = shared_motion.wrap([0.5, -8.926, 6.641, math.pi, 0.0, 3.2])
    assert all(abs(value) <= math.pi + 1e-9 for value in joints)
    assert np.allclose(np.asarray(joints) + np.asarray(offset), [0.5, -8.926, 6.641, math.pi, 0.0, 3.2])
    trajectory = {"positions": [[*joints], [*joints]], "velocities": [[0.0] * 6, [0.0] * 6], "dt_s": 0.02}
    result = shared_motion._plan_result(_answer(trajectory=trajectory), joint_names=[f"fr_joint{i}" for i in range(6)],
                                        offset=offset)
    assert result["schema"] == PLAN_RESULT_SCHEMA and result["status"] == "Success"
    assert result["planner_status"] == "planned" and result["joint_names"][0] == "fr_joint0"
    assert np.allclose(result["position"][0], [0.5, -8.926, 6.641, math.pi, 0.0, 3.2])
    assert result["interpolation_dt_s"] == 0.02


def test_a_refusal_keeps_its_neutral_reason():
    result = shared_motion._plan_result(_answer("refused", "path_blocked"), joint_names=["a"], offset=[0.0])
    assert result["status"] == "Fail" and result["planner_status"] == "path_blocked"
    assert "position" not in result and result["error"]["message"] == "path_blocked"


def test_a_refusal_tells_the_receipt_why_in_its_own_words():
    why = "the arm reaches this pose with nothing in the way, but the fingers would be 0.9 cm inside what the depth shows"
    answer = {**_answer("refused", "goal_unreachable"), "detail": {"goal_diagnosis": {"message": why}}}
    result = shared_motion._plan_result(answer, joint_names=["a"], offset=[0.0])
    public = planner_public_summary(result)
    assert public["planning_diagnostics"]["recovery"][0] == f"{why[0].upper()}{why[1:]}."
    assert public_planning_diagnostics(public["planning_diagnostics"])["outcome"] == "goal_unreachable"
    assert not re.search("curobo|cuda", json.dumps(public["planning_diagnostics"]), re.IGNORECASE)


def test_a_start_the_planner_calls_self_colliding_leaves_the_move_to_the_vendored_planner(tmp_path, monkeypatch):
    class Client:
        def plan(self, request):
            return {**_answer("refused", "start_in_collision"), "world": None, "planner": None,
                    "detail": {"start_check": {"self_collision": 0.002, "world_collision": 0.0}}}

    monkeypatch.setattr(shared_motion, "world_input", lambda *a, **k: ({"path": "/x", "sha256": "0" * 64}, [0.0] * 6, [0.0] * 6))
    target = {"kind": "joint", "joints": [0.1] * 6}
    motion = _motion(tmp_path, Client())
    assert shared_motion.plan(_session(tmp_path), None, motion, "left", target, _measured(), ["a"]) is None


def test_a_start_in_the_world_stays_a_refusal():
    blocked = {"reason": "start_in_collision", "detail": {"start_check": {"self_collision": 0.0, "world_collision": 0.3}}}
    assert shared_motion._self_colliding_start(blocked) is False
    assert shared_motion._self_colliding_start({"reason": "path_blocked", "detail": {}}) is False


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

def _session(tmp_path):
    return SimpleNamespace(physics_step_sequence=4, world_mutation_sequence=0, output_root=tmp_path)


def test_a_failed_planner_leaves_the_move_to_the_vendored_planner(tmp_path, monkeypatch):
    class Failing:
        def plan(self, request):
            raise MotionPlannerError("worker gone")

    monkeypatch.setattr(shared_motion, "world_input", lambda *a, **k: ({"path": "/x", "sha256": "0" * 64}, [0.0] * 6, [0.0] * 6))
    motion = _motion(tmp_path, Failing())
    target = {"kind": "pose", "pose_wxyz": [0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]}
    assert shared_motion.plan(_session(tmp_path), None, motion, "left", target, _measured(), ["a"]) is None
    assert json.loads((motion.root / "plans.jsonl").read_text())["fallback"] is True


def test_a_joint_goal_is_asked_the_short_way_from_the_wrapped_start(tmp_path, monkeypatch):
    asked = []

    class Client:
        def plan(self, request):
            asked.append(request)
            return {**_answer("refused", "path_blocked"), "world": None, "planner": None}

    monkeypatch.setattr(shared_motion, "world_input", lambda *a, **k: ({"path": "/x", "sha256": "0" * 64}, [3.0] * 6, [0.0] * 6))
    target = {"kind": "joint", "joints": [-3.0] * 6}
    shared_motion.plan(_session(tmp_path), None, _motion(tmp_path, Client()), "left", target, _measured(), ["a"])
    assert asked[0]["goal"]["joint_positions"] == pytest.approx([-3.0 + 2.0 * math.pi] * 6)
    assert asked[0]["held_object"] == {"source": "world"}


# --------------------------------------------------------------------------
# Reach rows
# --------------------------------------------------------------------------

def test_reach_rows_hold_the_solution_to_the_arrival_bound_and_chain(tmp_path, monkeypatch):
    blank = {"solved": False, "skipped": False, "refusal": None, "joint_positions": None}

    class Client:
        def solve_ik(self, request):
            assert len(request["queries"]) == 4 and request["world"]["model"] == "public_depth_voxel_v1"
            return {"status": "solved", "joint_limits": [[-4.2] * 6, [4.2] * 6], "results": [
                {**blank, "solved": True, "joint_positions": [0.2] * 6}, {**blank},
                {**blank, "skipped": True}, {**blank, "solved": True, "joint_positions": [0.5] * 6}]}

    monkeypatch.setattr(shared_motion, "world_input", lambda *a, **k: ({"path": "/x", "sha256": "0" * 64}, [0.0] * 6, [0.0] * 6))
    pose = [0.3, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0]
    adapter = SimpleNamespace(
        compile_public_target=lambda arm, value: {"pose_wxyz": [*value[:3], 1.0, 0.0, 0.0, 0.0]},
        fk_public_tcp_pose=lambda arm, joints: pose if joints[0] < 0.4 else [0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
    queries = [{"pose_xyzw": pose, "from_index": start} for start in (None, None, 1, 0)]
    answer = shared_motion.reach(_session(tmp_path), None, _motion(tmp_path, Client()), adapter, "left", queries,
                                 _measured(), position_tolerance_m=0.005, orientation_tolerance_rad=0.05)
    solved, failed, orphan, far = answer["results"]
    assert answer["ik_model"] == IK_MODEL and answer["claim_boundary"] == CLAIM_BOUNDARY
    assert answer["matches_move_pose_admission"] is False
    assert solved["reachable"] is True and solved["joint_delta_rad"] == pytest.approx(0.2)
    assert solved["joint_limit_margin_rad"] == pytest.approx(4.0)
    assert failed["reason"] == IK_FAILED and failed["position_error_m"] is None
    assert orphan["reason"] == "start_unreachable"
    assert far["reachable"] is False and far["reason"] == IK_FAILED and far["position_error_m"] == pytest.approx(0.1)
    assert far["joint_delta_rad"] == pytest.approx(0.3)  # from the solution it was chained onto


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

class _Planner:
    """The vendored ``plan_path``'s shape over one fixed rigid transform."""

    def __init__(self, frame):
        self.frame = frame

    def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
        w, x, y, z = target_gripper_pose[3:]
        goal = self.frame @ pose_matrix([*target_gripper_pose[:3], x, y, z, w])
        from ora.contracts.pose import matrix_quaternion
        qx, qy, qz, qw = matrix_quaternion(goal[:3, :3].tolist())

        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def detach(self):
                return self

            def to(self, _device):
                return self

            def numpy(self):
                return self.value

        return self._plan_pose_via_nearest_ik(None, None, SimpleNamespace(
            position=Tensor(goal[:3, 3]), quaternion=Tensor([qw, qx, qy, qz])))


def _robot(planner, *, shifted=False, communication=False):
    entity = SimpleNamespace(get_qpos=lambda: [0.0] * 8)

    def endlink(pose, arm_tag=None):
        return [pose[0] + (0.01 if shifted and pose[0] else 0.0), *pose[1:]]

    return SimpleNamespace(communication_flag=communication, left_planner=planner, left_entity=entity,
                           _trans_from_gripper_to_endlink=endlink)


def test_the_planner_frame_is_read_off_the_vendored_transform():
    frame = pose_matrix([0.1, 0.4, -0.78, 0.0, 0.0, 0.01, 1.0 / math.hypot(0.01, 1.0)])
    measured = shared_motion.planner_frame(_robot(_Planner(frame)), "left")
    assert measured == pytest.approx(frame, abs=1e-5)


def test_a_vendored_transform_that_is_not_one_frame_refuses_the_planner():
    with pytest.raises(RoboTwinLiveError, match="rigid frame"):
        shared_motion.planner_frame(_robot(_Planner(np.eye(4)), shifted=True), "left")
    with pytest.raises(RoboTwinLiveError, match="in this process"):
        shared_motion.planner_frame(_robot(_Planner(np.eye(4)), communication=True), "left")


# --------------------------------------------------------------------------
# The binding
# --------------------------------------------------------------------------

def _preflight(root: Path, *checks):
    (root / "preflight.json").write_text(json.dumps({"checks": list(checks)}), encoding="utf-8")


def test_a_run_binds_the_planner_its_preflight_passed(tmp_path):
    output = tmp_path / "native"
    assert shared_motion._preflight_binding(output) is None
    _preflight(tmp_path, {"check": "other", "passed": True})
    assert shared_motion._preflight_binding(output) is None
    detail = {"python": "/p", "root": "/r"}
    _preflight(tmp_path, {"check": SHARED_PLANNER_CHECK, "passed": True, "detail": detail})
    assert shared_motion._preflight_binding(output) == detail
    _preflight(tmp_path, {"check": SHARED_PLANNER_CHECK, "passed": False, "detail": "absent"})
    with pytest.raises(RoboTwinLiveError):
        shared_motion._preflight_binding(output)


def test_the_preflight_check_records_what_the_worker_binds(monkeypatch, tmp_path):
    from ora.environment.controllers import motion_planner
    from ora.orchestrator.backends.robotwin import preflight

    client = SimpleNamespace(python_executable=Path("/v2/python"), root=Path("/v2/root"),
                             binding={"python_sha256": "a" * 64})
    monkeypatch.setattr(motion_planner, "planner_client", lambda *a, **k: client)
    planner = {"implementation": "curobo_v2", "python_env": "V2_PY", "root_env": "V2_ROOT", "robot": "aloha_agilex",
               "required_revision": "r", "world_model": "public_depth_voxel_v1"}
    runtime = {"root_env": "RT_ROOT"}
    check = lambda name, passed, detail: {"check": name, "passed": passed, "detail": detail}
    missing = preflight.shared_planner_check(planner, runtime, check)
    assert missing["passed"] is False and "V2_PY" in missing["detail"]
    for name in ("V2_PY", "V2_ROOT", "RT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path))
    passed = preflight.shared_planner_check(planner, runtime, check)
    assert passed["passed"] is True and passed["check"] == SHARED_PLANNER_CHECK
    assert passed["detail"]["python"] == "/v2/python" and passed["detail"]["python_sha256"] == "a" * 64
    assert "python_env" not in passed["detail"] and passed["detail"]["robot"] == "aloha_agilex"


# --------------------------------------------------------------------------
# The robot
# --------------------------------------------------------------------------

def _aloha_file(tmp_path):
    spheres = {link: [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]
               for link in ("fl_base_link", "fl_link6", "fl_link7", "fl_link8")}
    (tmp_path / "spheres.yml").write_text(yaml.safe_dump({"collision_spheres": spheres}), encoding="utf-8")
    names = [*(f"fl_joint{i}" for i in range(1, 9))]
    return {"robot_cfg": {"kinematics": {
        "urdf_path": "/a.urdf", "base_link": "fl_base_link", "ee_link": "fl_link6",
        "collision_link_names": ["fl_base_link", "fl_link6", "fl_link7", "fl_link8"],
        "collision_spheres": str(tmp_path / "spheres.yml"), "collision_sphere_buffer": 0.004,
        "extra_collision_spheres": {}, "self_collision_ignore": {"fl_link6": ["fl_link7"]},
        "self_collision_buffer": {"fl_link6": 0.0}, "mesh_link_names": [], "lock_joints": {"fl_joint7": 0.04, "fl_joint8": 0.04},
        "extra_links": None, "use_global_cumul": True,
        "cspace": {"joint_names": names, "retract_config": [0.0] * 8, "null_space_weight": [1.0] * 8,
                   "cspace_distance_weight": [1.0] * 8, "max_jerk": 100.0, "max_acceleration": 3.0}}},
        "planner": {"frame_bias": [-0.2305, -0.297, -0.782]}}


def test_the_vendored_robot_file_becomes_a_shared_planner_robot(tmp_path):
    config = robots.robot_config("aloha_agilex", _aloha_file(tmp_path), held_link="attached_object", held_slots=32)
    kinematics = config["robot_cfg"]["kinematics"]
    assert kinematics["tool_frames"] == ["fl_link6"] and kinematics["format_version"] == 2.0
    assert "ee_link" not in kinematics and "planner" not in config
    assert kinematics["collision_spheres"]["fl_link6"][-1]["center"] == [0.07, -0.06, 0.0]
    assert kinematics["collision_link_names"][-1] == "attached_object"
    assert kinematics["extra_links"]["attached_object"]["parent_link_name"] == "fl_link6"
    assert kinematics["extra_collision_spheres"] == {"attached_object": 32}
    assert kinematics["lock_joints"] == {"fl_joint7": 0.04, "fl_joint8": 0.04}
    assert kinematics["cspace"]["position_limit_clip"] == pytest.approx(10.0 - robots.ALOHA_JOINT_LIMIT_RAD)
    assert kinematics["cspace"]["max_jerk"] == [100.0] * 8
    assert kinematics["cspace"]["default_joint_position"] == [0.0] * 8
    assert "fl_link5" not in kinematics["self_collision_ignore"].get("fl_link3", ())
    assert kinematics["self_collision_buffer"]["fl_link3"] == kinematics["self_collision_buffer"]["fl_link5"] == (
        robots.ALOHA_FOLD_BUFFER_M)
    assert kinematics["self_collision_ignore"]["fl_link2"] == ["fl_link5"]


def test_the_robot_is_registered_on_every_shared_table():
    assert ROBOTS["aloha_agilex"]["tool_frame"] == "fl_link6"
    assert "aloha_agilex" in IMPLEMENTATIONS["curobo_v2"].robots and "aloha_agilex" in robots.ROBOT_FILE_ROBOTS
    parameters, _turn = ROBOT_WORLDS["aloha_agilex"]
    assert parameters.world_model == "public_depth_voxel_v1"
    assert parameters.carve_box_tip_m < parameters.hand_box_m[2][1]
    assert parameters.fingertip_m < parameters.carve_box_tip_m
