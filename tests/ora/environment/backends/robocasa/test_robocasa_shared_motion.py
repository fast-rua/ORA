"""RoboCasa365 on the shared motion base: frames, refusals, lines, the lift, stands, the robot's own body, the base
search and the binding.

The planner is a stand-in answering the shared contract; ``kinematics`` is replaced where it needs MuJoCo, and the
robot's own body stands far from what the cameras see unless a test puts it there.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from ora.environment.backends.robocasa import kinematics, shared_motion
from ora.environment.backends.robocasa.common import RoboCasaError
from ora.environment.controllers import base_placement, robocasa_omron as C, robot_spheres
from ora.environment.controllers.depth_world import pose_matrix
from ora.environment.controllers.motion_worker import MotionPlannerError

IDENTITY = [0.0, 0.0, 0.0, 1.0]
HOME = [0.0, -1.0, 0.0, -2.2, 0.0, 1.5, 0.7]
BLOCKED_ROW = {"status": "blocked", "first_blocked_row": 1, "link": "panda_link7", "penetration_m": 0.02}
CLEAR = {"status": "clear", "first_blocked_row": None, "link": None, "penetration_m": None}


class IO:
    model = data = None
    control_steps = 7
    gripper_directive = -1.0
    torso = 0.1

    def arm_qpos(self) -> list[float]:
        return list(HOME)

    def grip_pose(self) -> list[float]:
        return [1.0, 0.0, 1.0, *IDENTITY]

    def hand_pose(self) -> list[float]:
        return [1.0, 0.0, 1.1, *IDENTITY]

    def arm_base_pose(self) -> list[float]:
        return [0.5, 0.0, 0.8, *IDENTITY]

    def finger_qpos(self) -> list[float]:
        return [0.04, -0.04]

    def torso_height(self) -> float:
        return self.torso

    def base_pose(self) -> list[float]:
        return [0.5, 0.0, 0.0]

    def anchor_offset(self) -> list[float]:
        return [-0.21, 0.0]


class Client:
    """Records requests; answers what a test set."""

    def __init__(self) -> None:
        self.plans: list[dict[str, Any]] = []
        self.iks: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.plan_answer: dict[str, Any] = {"status": "planned", "reason": None,
                                            "trajectory": {"positions": [HOME, [0.1, *HOME[1:]], [0.2, *HOME[1:]]]}}
        self.ik_rows: Any = None
        self.check_answers: list[dict[str, Any]] = []

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.plans.append(request)
        return {"world": {"sha256": "w", "summary": {}}, "planner": {}, "timing_s": {}, **self.plan_answer}

    def solve_ik(self, request: dict[str, Any]) -> dict[str, Any]:
        self.iks.append(request)
        rows = self.ik_rows(request) if callable(self.ik_rows) else self.ik_rows
        return {"status": "solved", "reason": None, "timing_s": {}, "results": rows}

    def check_path(self, request: dict[str, Any]) -> dict[str, Any]:
        self.checks.append(request)
        return {"timing_s": {}, **(self.check_answers.pop(0) if self.check_answers else CLEAR)}


class Session:
    round_index = 0
    planner_config = None

    def __init__(self, tmp_path: Any) -> None:
        self.io = IO()
        self.output_root = tmp_path
        self.motion_planner = Client()
        self.world_inputs: dict[Any, Any] = {}

    def _camera(self, _source: str) -> tuple[Any, Any, dict[str, Any]]:
        return None, np.ones((4, 4), dtype=np.float32), {
            "intrinsic": np.eye(3), "cam2world": np.eye(4), "near": 0.01, "far": 10.0}


#: The robot's own body far from what the cameras see (``Session._camera``: x and y 0 to 3 m, 1 m up).
FAR = base_placement.OwnBody(np.array([[10.0, 10.0, 1.0, 0.1]]), 0.03)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(kinematics, "grip_poses", lambda _model, _data, rows: [[1.0, 0.0, 1.0, *IDENTITY] for _ in rows])
    monkeypatch.setattr(shared_motion, "body_at", lambda _mount, _joints, _held=None: FAR)
    return Session(tmp_path)


def _solved(joints: list[float] | None = None) -> dict[str, Any]:
    return {"solved": True, "joint_positions": joints or list(HOME), "position_error_m": 0.0, "rotation_error": 0.0}


UNSOLVED = {"solved": False, "joint_positions": None, "position_error_m": None, "rotation_error": None}


def test_a_goal_is_the_hand_pose_in_the_arm_base_frame_for_the_grip_site_target(session):
    goal = shared_motion._hand_goal(session.io, [1.0, 0.2, 1.3, *IDENTITY])
    assert goal == pytest.approx([0.5, 0.2, 0.6, *IDENTITY])
    assert shared_motion._hand_goal(session.io, [1.0, 0.2, 1.3, *IDENTITY], 0.1)[2] == pytest.approx(0.5)


def test_a_snapshot_is_the_public_depth_of_this_moment_and_is_reused_until_the_robot_steps(session):
    first = shared_motion._snapshot(session)
    assert shared_motion._snapshot(session) == first
    assert shared_motion._snapshot(session, blank=True) != first
    with np.load(first["path"], allow_pickle=False) as stored:
        assert stored["base_pose_world_xyzw"].tolist() == session.io.arm_base_pose()
        assert stored["joint_positions"].tolist() == HOME and float(stored["gripper_directive"]) == -1.0
        assert stored["agentview_depth_m"].shape == (4, 4) and stored["wrist_depth_m"].shape == (4, 4)
    session.io.control_steps += 1
    assert shared_motion._snapshot(session) != first


def test_a_blank_snapshot_holds_no_depth(session):
    with np.load(shared_motion._snapshot(session, blank=True)["path"], allow_pickle=False) as stored:
        assert not stored["agentview_depth_m"].any() and not stored["wrist_depth_m"].any()


def test_a_route_is_the_planner_path_without_its_start_row(session):
    plan = shared_motion.plan_to_pose(session, [1.0, 0.1, 1.0, *IDENTITY])
    assert plan["refusal"] is None and plan["path"] == [[0.1, *HOME[1:]], [0.2, *HOME[1:]]]
    request = session.motion_planner.plans[0]
    assert request["robot"] == "franka_panda" and request["held_object"] == {"source": "world"}
    assert request["world"]["model"] == "public_depth_voxel_v1" and request["goal"]["tool_frame"] == "panda_hand"


@pytest.mark.parametrize(("reason", "code"), sorted(shared_motion.CODES.items()))
def test_every_refusal_of_the_planner_has_a_public_code(session, reason, code):
    session.motion_planner.plan_answer = {"status": "refused", "reason": reason, "trajectory": None}
    session.motion_planner.ik_rows = [UNSOLVED]
    plan = shared_motion.plan_to_pose(session, [1.0, 0.1, 1.0, *IDENTITY])
    assert plan["code"] == code and plan["path"] is None and plan["refusal"]["planner_reason"] == reason


def test_a_target_the_arm_reaches_only_through_the_world_is_in_collision(session):
    session.motion_planner.plan_answer = {"status": "refused", "reason": "goal_unreachable", "trajectory": None}
    session.motion_planner.ik_rows = [_solved()]
    plan = shared_motion.plan_to_pose(session, [1.0, 0.1, 1.0, *IDENTITY])
    assert plan["code"] == "target_in_collision_at_base_pose"
    assert plan["refusal"]["planning_diagnostics"]["outcome"] == "goal_unreachable"


def test_a_dead_planner_refuses_and_nothing_moves(session):
    def fails(_request: Any) -> dict[str, Any]:
        raise MotionPlannerError("gone")

    session.motion_planner.plan = fails
    plan = shared_motion.arm_path(session, [1.0, 0.1, 1.0, *IDENTITY], "joint", "free_space")
    assert plan["code"] == shared_motion.UNAVAILABLE and plan["path"] is None
    session.motion_planner = None
    assert shared_motion.joint_path(session, HOME)["code"] == shared_motion.UNAVAILABLE


def _line_rows(count: int, step: float = 0.01) -> list[dict[str, Any]]:
    return [_solved([step * (index + 1), *HOME[1:]]) for index in range(count)]


def test_a_line_chains_its_cuts_in_a_blank_world_and_interpolates_between_them(session):
    session.motion_planner.ik_rows = lambda request: _line_rows(len(request["queries"]))
    plan = shared_motion.line_path(session, [1.0, 0.1, 1.0, *IDENTITY], "contact_approach")
    assert plan["refusal"] is None and plan["kind"] == "line"
    queries = session.motion_planner.iks[0]["queries"]
    assert len(queries) == 5 and [query["from_index"] for query in queries] == [None, 0, 1, 2, 3]
    assert session.motion_planner.iks[0]["world"]["input"]["path"].endswith("-blank.npz")
    assert len(session.motion_planner.checks) == 1


def test_a_cut_that_does_not_solve_or_jumps_refuses_the_line(session):
    session.motion_planner.ik_rows = [_solved(), UNSOLVED]
    first = shared_motion.line_path(session, [1.0, 0.03, 1.0, *IDENTITY], "precision")
    assert first["code"] == "target_unreachable_at_base_pose" and first["refusal"]["path_fraction"] == 1.0
    session.motion_planner.ik_rows = [_solved(), _solved([C.LINE_MAX_JOINT_JUMP_RAD + 0.01, *HOME[1:]])]
    assert shared_motion.line_path(session, [1.0, 0.03, 1.0, *IDENTITY], "precision")["path"] is None


def test_a_line_keeps_the_arm_out_of_the_world_with_no_goal_clearing(session):
    session.motion_planner.ik_rows = lambda request: _line_rows(len(request["queries"]))
    session.motion_planner.check_answers = [BLOCKED_ROW]
    plan = shared_motion.line_path(session, [1.0, 0.1, 1.0, *IDENTITY], "contact_approach")
    assert plan["code"] == "path_in_collision_at_base_pose" and plan["refusal"]["robot_part"] == "arm"
    check = session.motion_planner.checks[0]
    assert check["path"][-1] == HOME, "the check's world is one a plan to the start would see"
    assert set(check["allowed_links"]) == set(shared_motion.GRIPPER_LINKS)


def test_a_precision_line_checks_the_gripper_too_and_a_contact_approach_does_not(session):
    session.motion_planner.ik_rows = lambda request: _line_rows(len(request["queries"]))
    session.motion_planner.check_answers = [CLEAR, {**BLOCKED_ROW, "link": "panda_hand"}]
    plan = shared_motion.line_path(session, [1.0, 0.1, 1.0, *IDENTITY], "precision")
    assert plan["refusal"]["robot_part"] == "gripper"
    assert session.motion_planner.checks[-1]["allowed_links"] == []


def _blank(request: dict[str, Any]) -> bool:
    return request["world"]["input"]["path"].endswith("-blank.npz")


def test_the_lift_offers_the_nearest_heights_the_arm_reaches_from_across_the_lift_s_stop_band(session):
    """Kinematics alone, nearest first; a height whose band edge does not reach is left out."""
    session.motion_planner.ik_rows = lambda request: [
        UNSOLVED if index in (0, 1, 2, 4) else _solved() for index in range(len(request["queries"]))]
    heights = shared_motion.torso_heights(session, [1.0, 0.0, 1.3, *IDENTITY])
    (request,) = session.motion_planner.iks
    assert _blank(request)
    lifts = [0.6 - query["pose_base_xyzw"][2] for query in request["queries"]]
    centres = lifts[0::3]
    assert [round(abs(lift), 6) for lift in centres] == sorted(round(abs(lift), 6) for lift in centres)
    assert all(abs(lift) >= C.TORSO_MIN_CHANGE_M for lift in centres)
    assert [lifts[3 * index + 1] - lifts[3 * index] for index in range(2)] == pytest.approx([-C.TORSO_TOLERANCE_M] * 2)
    # the first height's centre and the second's lower edge do not solve; the third is the first offered
    assert heights[0] == pytest.approx(session.io.torso + centres[2])
    assert all(C.TORSO_RANGE_M[0] - 1e-9 <= height <= C.TORSO_RANGE_M[1] + 1e-9 for height in heights)


def test_no_height_that_solves_is_none(session):
    session.motion_planner.ik_rows = lambda request: [UNSOLVED] * len(request["queries"])
    assert shared_motion.torso_heights(session, [1.0, 0.0, 1.3, *IDENTITY]) == []


def test_a_line_refusal_names_its_cut_and_whether_the_target_itself_is_reachable(session):
    goal = [1.0, 0.03, 1.0, *IDENTITY]
    answers = [[_solved(), {**UNSOLVED, "position_error_m": 0.02, "rotation_error": 0.1}], [_solved()], [_solved()]]
    session.motion_planner.ik_rows = lambda request: answers.pop(0)
    plan = shared_motion.line_path(session, goal, "precision", explain=True)
    refusal = plan["refusal"]
    assert plan["code"] == "target_unreachable_at_base_pose" and refusal["cause"] == "ik_unsolved"
    assert refusal["path_fraction"] == 1.0 and refusal["at_pose"] == pytest.approx(goal, abs=1e-4)
    assert refusal["ik_residual"] == {"translation_m": 0.02, "orientation_rad": 0.1}
    assert refusal["endpoint"] == shared_motion.ENDPOINT_REACHABLE
    blank, observed = session.motion_planner.iks[1:]
    assert _blank(blank) and not _blank(observed) and len(observed["queries"]) == 1
    answers[:] = [[_solved(), UNSOLVED], [_solved()], [UNSOLVED]]
    assert shared_motion.line_path(session, goal, "precision", explain=True)["refusal"]["endpoint"] == (
        shared_motion.ENDPOINT_IN_WORLD)
    answers[:] = [[_solved(), UNSOLVED], [UNSOLVED]]
    assert shared_motion.line_path(session, goal, "precision", explain=True)["refusal"]["endpoint"] == (
        shared_motion.ENDPOINT_UNREACHABLE)


def test_a_line_whose_ik_changes_branch_names_the_joint_and_the_jump(session):
    jump = C.LINE_MAX_JOINT_JUMP_RAD + 0.05
    session.motion_planner.ik_rows = [_solved(), _solved([*HOME[:3], HOME[3] + jump, *HOME[4:]])]
    refusal = shared_motion.line_path(session, [1.0, 0.03, 1.0, *IDENTITY], "precision")["refusal"]
    assert refusal["cause"] == "joint_branch_jump" and refusal["joint"] == 4
    assert refusal["jump_rad"] == pytest.approx(jump) and refusal["bound_rad"] == C.LINE_MAX_JOINT_JUMP_RAD
    assert "endpoint" not in refusal, "only an explained refusal asks the target on its own"


def test_a_stand_is_asked_in_its_own_frame_and_in_a_world_without_the_robot(session):
    stand = shared_motion.Stand(0.8, 0.0, 0.0, session.io.torso + 0.1)
    session.motion_planner.ik_rows = [_solved()]
    shared_motion.solve_ik(session, [[1.0, 0.2, 1.3, *IDENTITY]], stand=stand)
    (request,) = session.motion_planner.iks
    assert request["queries"][0]["pose_base_xyzw"][:3] == pytest.approx([0.2, 0.2, 0.5])
    path = request["world"]["input"]["path"]
    assert "-stand-000" in path
    with np.load(path, allow_pickle=False) as stored:
        assert stored["base_pose_world_xyzw"][:3].tolist() == pytest.approx([0.8, 0.0, 0.9])
        assert stored["joint_positions"].tolist() == HOME and stored["agentview_depth_m"].all()
    assert shared_motion.stand_world(session, stand) == request["world"]["input"], "kept for the moment"


def test_the_robot_s_own_pixels_are_dark_in_a_stand_s_world(session, monkeypatch):
    patch = np.array([[float(x), float(y), 1.0, 0.01] for y in range(4) for x in range(4)])
    monkeypatch.setattr(shared_motion, "body_at", lambda _mount, _joints, _held=None: base_placement.OwnBody(patch, 0.0))
    record = shared_motion.stand_world(session, shared_motion.Stand(0.8, 0.0, 0.0, session.io.torso))
    with np.load(record["path"], allow_pickle=False) as stored:
        assert not stored["agentview_depth_m"].any() and not stored["wrist_depth_m"].any()


def test_the_base_search_asks_past_the_kept_plans_and_answers_once_a_moment_and_move(session, monkeypatch):
    raw = session.motion_planner
    session.motion_planner = shared_motion.KeptPlans(raw)
    asked, admitted = [], []

    def search(target, here, base, points, *, solve, admits):
        asked.append(target)
        assert base.torso_range_m is None, "a stand keeps the measured torso height"
        stand = shared_motion.Stand(0.8, 0.0, 0.0, here.torso_m)
        return {"suggestions": [{"x_m": 0.8, "reached": solve([stand]), "admitted": admits(stand)}], "searched": {}}

    def admits(stand):
        admitted.append(stand)
        return shared_motion.arm_path(session, goal, "line", "contact_approach", stand, kept=False)["refusal"] is None

    monkeypatch.setattr(shared_motion.base_placement, "search", search)
    raw.ik_rows = lambda request: [_solved(list(HOME))] * len(request["queries"])
    goal = [1.4, 0.0, 1.0, *IDENTITY]
    move = ("line", "contact_approach", {"preset": "careful"})
    with shared_motion.KeptPlans.previewing(session.motion_planner):
        first = shared_motion.base_search(session, goal, move, admits)
    assert first["suggestions"][0] == {"x_m": 0.8, "reached": [True], "admitted": True}
    assert not session.motion_planner._kept, "a search's questions are not a dry run's to keep"
    assert raw.checks and all("-stand-" in check["world"]["input"]["path"] for check in raw.checks), (
        "the line itself is checked, in the stand's world")
    assert shared_motion.base_search(session, goal, move, admits) is first and len(asked) == 1
    assert shared_motion.base_search(session, goal, ("joint", "free_space", None), admits) is not first, (
        "another move is another question")
    session.io.control_steps += 1
    assert shared_motion.base_search(session, goal, move, admits) is not first
    lines = [json.loads(line) for line in
             (session.output_root / "motion-planner/base-placement.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and lines[0]["target"] == goal and lines[0]["kind"] == "line"
    assert lines[0]["interaction"] == "contact_approach" and lines[0]["speed"] == {"preset": "careful"}


def test_a_base_move_is_checked_against_the_depth_seen_now(session, monkeypatch):
    # The cameras see a 4x4 patch 1 m up at x, y in 0..3 m: a base driven under it meets nothing; the column does.
    clear = shared_motion.base_path_refusal(session, {"x_m": 0.5, "y_m": 0.0, "yaw_rad": 0.3})
    assert clear is None
    session.io.torso, session.io.control_steps = 0.34, session.io.control_steps + 1  # the column reaches 1.04 m
    blocked = shared_motion.base_path_refusal(session, {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0})
    assert blocked["path_kind"] == "base" and blocked["robot_part"] == "base"
    assert blocked["obstacle_height_m"] == pytest.approx(1.0) and 0 < blocked["path_fraction"] <= 1.0
    # A hand carried at 1 m, short of the patch: the footprint passes under it, the hand does not.
    session.io.torso, session.io.control_steps = 0.1, session.io.control_steps + 1
    hand = base_placement.OwnBody(np.array([[0.5, 1.0, 1.0, 0.05]]), 0.03)
    monkeypatch.setattr(shared_motion, "body_at", lambda _mount, _joints, _held=None: hand)
    carried = shared_motion.base_path_refusal(session, {"x_m": 1.5, "y_m": 0.0, "yaw_rad": 0.0})
    assert carried["path_kind"] == "base" and carried["robot_part"] == "arm"


def test_a_probe_answers_with_the_shared_ik_and_no_plan(session):
    session.motion_planner.ik_rows = [UNSOLVED, _solved()]
    answer = shared_motion.probe(session, {"variant_policy": "parallel_jaw_equivalents", "queries": [
        {"query_id": "q0", "arm": "right", "pose": {"position_m": [1.0, 0.0, 1.0], "orientation_xyzw": IDENTITY}}]})
    assert answer[0]["reachable"] is True and answer[0]["selected_variant_id"] == "jaw_flip_180"
    assert answer[0]["plan_ref"] is None and answer[0]["variants"][1]["planner"] == {"kind": "shared_ik", "plan": "none"}
    with pytest.raises(RoboCasaError):
        shared_motion.probe(session, {"queries": [{"query_id": "q0", "arm": "left", "pose": {}}]})


def test_the_planner_is_bound_from_the_runs_preflight(tmp_path, monkeypatch):
    native = tmp_path / "native"
    config = {"python_env": "CUROBO_V2_PYTHON", "root_env": "CUROBO_V2_ROOT"}
    monkeypatch.delenv("CUROBO_V2_PYTHON", raising=False)
    monkeypatch.delenv("CUROBO_V2_ROOT", raising=False)
    with pytest.raises(RoboCasaError):
        shared_motion.planner_binding(config, native)
    (tmp_path / "preflight.json").write_text(json.dumps({"checks": [
        {"check": C.MOTION_PLANNER_PREFLIGHT_CHECK, "passed": True, "detail": {"python": "/p", "root": "/r"}}]}),
        encoding="utf-8")
    assert shared_motion.planner_binding(config, native) == ("/p", "/r")


def test_parity_rows_stay_inside_the_joint_limits():
    limits = [(-1.0, 1.0)] * 7
    rows = shared_motion.parity_rows([0.9, *([0.0] * 6)], limits)
    assert rows[0] == [0.9, *([0.0] * 6)] and all(-1.0 <= value <= 1.0 for row in rows for value in row)
    assert len(rows) == len(shared_motion.PARITY_OFFSETS_RAD) + 6 * (len(shared_motion.PARITY_OFFSETS_RAD) - 1)


def test_the_hand_goal_round_trips_through_the_frames(session):
    goal = shared_motion._hand_goal(session.io, [0.8, -0.2, 1.2, *IDENTITY])
    world = pose_matrix(session.io.arm_base_pose()) @ pose_matrix(goal)
    assert world[:3, 3].tolist() == pytest.approx([0.8, -0.2, 1.3]) and math.isclose(world[0, 0], 1.0)


#: The recorded depth of the robot's own arm and of the scene next to its hand (``world-inputs`` of the RoboCasa runs of
#: 2026-11-06). ``step_S_*``: the motion qualification at control steps 0 and 159, where nothing but the robot lay
#: within 0.20 m of the arm: ``points_m``, the agentview points 0.08 to 0.20 m from it, the shell the 0.10 m capsules
#: left, and ``own_m``, both cameras' points within 0.20 m, one per centimetre voxel; the arm mount pose and joints.
#: ``garnish_003422_*``: smoke run cdd27bf7aa6c at step 3422, the wrist camera's points within 5 cm of the strawberry
#: target the move refused, the shelf and the fruit beside the open jaws, with the mount pose and joints there.
OWN_ARM_DEPTH = "tests/data/robocasa/own-arm-depth-20261106.npz"


def _recorded() -> Any:
    from pathlib import Path

    return np.load(Path(__file__).resolve().parents[5] / OWN_ARM_DEPTH)


def _capsules_952eeed23(chain: Any, points: Any) -> Any:
    """Which ``points`` the capsules of 952eeed23 took for the arm's own: 0.15 m about the links, 0.12 m the hand."""
    inside = np.zeros(len(points), dtype=bool)
    for index, (start, end) in enumerate(zip(chain, chain[1:])):
        radius = 0.12 if index == len(chain) - 2 else 0.15
        axis, length = end - start, float((end - start) @ (end - start))
        along = np.clip(((points - start) @ axis) / length, 0.0, 1.0) if length > 1e-12 else np.zeros(len(points))
        inside |= ((points - (start + along[:, None] * axis)) ** 2).sum(axis=1) <= radius * radius
    return inside


@pytest.mark.parametrize("step", ["000000", "000159"])
def test_the_arm_s_own_depth_leaves_a_stand_s_world_whole(step):
    """The torso raise of the 2026-11-06 qualification (RO-12): at 0.10 m capsules a shell of the arm's own depth
    stayed in every lifted stand's world and the lifted arm met it (``panda_link3``, 11-15 mm). The planner's own
    spheres, with the world's robot margin and the hand's and base link's boxes, hold every own point, the shell
    included."""
    recorded = _recorded()
    mount = recorded[f"step_{step}_mount_xyzw"].tolist()
    body = shared_motion.body_at(mount, recorded[f"step_{step}_joints"].tolist())
    assert len(body.spheres) == sum(len(rows) for rows in robot_spheres.SPHERES.values()) and body.held == 0
    # The mount stands on the base origin's vertical, 0.7 m up plus the torso, with the base's yaw.
    x, y, z, qx, qy, qz, qw = mount
    here = (x, y, math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    for name in ("points_m", "own_m"):
        points = recorded[f"step_{step}_{name}"].astype(np.float64)
        base = base_placement.footprint_depth(points, shared_motion.MOBILE_BASE, here, z - 0.7,
                                              base_placement.SEARCH.own_margin_m) >= 0.0
        assert len(points) > 1000 and (body.holds(points) | base).all(), (
            f"the robot's own depth ({name}) leaves no shell: its arm, or its base where it stands")
        assert body.holds(points[~base]).all()


def test_the_scene_next_to_the_hand_stays_in_a_stand_s_world():
    """Smoke run cdd27bf7aa6c, step 3422: the move to a strawberry deep on a fridge shelf was refused, and the search
    said a stand reaches it clear because 952eeed23's capsules had taken every one of the 2,970 points within 5 cm of
    it out of the stand's world. They lie beside the open jaws, more than the world's margin from every sphere of the
    hand and fingers: the scene, which now stays."""
    recorded = _recorded()
    mount, joints = recorded["garnish_003422_mount_xyzw"].tolist(), recorded["garnish_003422_joints"].tolist()
    scene = recorded["garnish_003422_scene_m"].astype(np.float64)
    assert len(scene) == 2970
    assert not shared_motion.body_at(mount, joints).holds(scene).any()
    base = pose_matrix(mount)
    frames = robot_spheres.link_frames(joints)
    hand = base @ frames["panda_hand"]
    chain = np.array([(base @ frames[f"panda_link{index}"])[:3, 3] for index in range(8)]
                     + [hand[:3, 3], (hand @ np.array([0.0, 0.0, 0.0965, 1.0]))[:3]])
    assert _capsules_952eeed23(chain, scene).all(), "the capsules took all of it"


def test_a_held_load_is_the_robot_s_own_too(session, monkeypatch):
    """A load between closed jaws is taken out as the planner takes it out of its world: the spheres it would attach,
    fitted to the load's voxels in this moment's world input. Nothing held, nothing added."""
    monkeypatch.undo()
    monkeypatch.setattr(kinematics, "grip_poses", lambda _model, _data, rows: [[1.0, 0.0, 1.0, *IDENTITY] for _ in rows])
    hand = pose_matrix(session.io.arm_base_pose()) @ robot_spheres.hand(HOME)
    between = (hand @ np.array([0.0, 0.0, 0.1, 1.0]))[:3]
    camera = np.eye(4)
    camera[:3, 3] = between - (0.0, 0.0, 0.5)

    def sees_the_load(_source):
        return None, np.full((4, 4), 0.5, dtype=np.float32), {
            "intrinsic": np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]), "cam2world": camera,
            "near": 0.01, "far": 10.0}

    session._camera = sees_the_load
    assert shared_motion.held_spheres(session) is None, "open jaws hold nothing"
    assert shared_motion.own_body(session).held == 0
    session.io.gripper_directive, session.io.control_steps = 1.0, session.io.control_steps + 1
    session.io.finger_qpos = lambda: [0.015, -0.015]
    body = shared_motion.own_body(session)
    assert body.held > 0 and body.holds(between[None, :]).all()
    load = base_placement.within_spheres(between[None, :], body.spheres[-body.held:], 0.0)
    assert load.all(), "the load's own spheres hold it"


def test_the_session_refuses_a_planner_whose_spheres_are_not_the_host_s(session, monkeypatch):
    """The host takes the robot's own depth out of a stand's world, and carries the arm along a base move, with its own
    copy of the planner's spheres: the planner must name the same set, and the host's chain land on its hand."""
    monkeypatch.setattr(kinematics, "joint_range", lambda _model, _name: (-3.0, 3.0))

    class Planner:
        def __init__(self, offset: float = 0.0) -> None:
            self.offset = offset

        def forward_kinematics(self, rows):
            frames = [robot_spheres.hand(row) for row in rows]
            return [[*(frame[:3, 3] + (self.offset, 0.0, 0.0)), *shared_motion.matrix_quaternion(frame[:3, :3].tolist())]
                    for frame in frames]

    def ready(named):
        return {"worker": {"ready": {"engine": {"collision_spheres_sha256": named}}}}

    passed = shared_motion.own_spheres_check(session, Planner(), ready(robot_spheres.SPHERE_SET_SHA256))
    assert passed["passed"] and passed["hand"]["position_error_m"] < 1e-9
    assert not shared_motion.own_spheres_check(session, Planner(), ready("0" * 64))["passed"]
    assert not shared_motion.own_spheres_check(session, Planner(), {})["passed"], "a planner that names no set fails"
    assert not shared_motion.own_spheres_check(session, Planner(0.01), ready(robot_spheres.SPHERE_SET_SHA256))["passed"]
