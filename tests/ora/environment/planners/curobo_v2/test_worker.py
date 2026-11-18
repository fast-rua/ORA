"""The cuRobo V2 worker's request handling over a CPU stand-in for the V2 engine."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment.controllers.depth_world import (
    PUBLIC_DEPTH_VOXEL_V1, WORLD_MODEL, build_world, curobo_v2_grid, pose_matrix, read_world_inputs,
)
from ora.environment.controllers.motion_pace import MAX_UNREQUESTED_TURN_DEG, joint_path_grid, path_turns
from ora.environment.controllers.motion_planner import (
    CHECK_REQUEST_SCHEMA, IK_REQUEST_SCHEMA, PLAN_ANSWER_SCHEMA_V2, PLAN_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA_V2, PROTOCOL,
    planned_retime_request, planning_diagnostics, refusal_message, validate_check_answer, validate_ik_answer,
    validate_plan_answer,
)
from ora.environment.planners.curobo_v2.goal_nearby import NEARBY_PLANS
from ora.environment.planners.curobo_v2.worker import Worker, compose_pose, serve
from tests.support.libero.world_inputs import TABLE_Z, overhead, write

NAMES = [f"panda_joint{index}" for index in range(1, 8)]
JOINTS = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
#: The stand-in hand: 0.5 m above the base, approach +z.
HAND = [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
SPHERES = np.array([[0.4, 0.0, 0.45, 0.04], [0.0, 0.0, 0.0, -100.0]])
BOX = ((0.6, 0.7), (-0.05, 0.05), 0.0)
#: A 3 cm object 10 cm along the hand's approach axis, between the jaws.
HELD = ((0.39, 0.41), (-0.015, 0.015), 0.6)
#: Hand pointing down (180 degrees about x), fingertips 3 cm inside the table.
INSIDE = [0.8, 0.3, TABLE_Z - 0.03 + PUBLIC_DEPTH_VOXEL_V1.fingertip_m, 1.0, 0.0, 0.0, 0.0]


class Engine:
    """Records what the worker asks of the planner; every check passes unless told otherwise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.blocked: set[str] = set()
        self.outcome: dict[str, Any] | None = None

    def identity(self) -> dict[str, Any]:
        return {"stand_in": True}

    def planner_identity(self) -> dict[str, Any]:
        return {"robot_sha256": "e" * 64, "collision_spheres_sha256": "f" * 64}

    def ik_identity(self) -> dict[str, Any]:
        return {"num_seeds": 32}

    def kinematics(self, joints: Any) -> tuple[Any, Any]:
        self.calls.append(("kinematics", list(joints)))
        return pose_matrix(HAND), SPHERES.copy()

    def tool_pose(self, joints: Any) -> list[float]:
        return list(HAND)

    def tool_poses(self, rows: Any) -> list[list[float]]:
        return [self.tool_pose(row) for row in rows]

    def path_spheres(self, rows: Any) -> Any:
        self.calls.append(("path_spheres", len(rows)))
        return np.stack([SPHERES] * len(rows))

    def sphere_links(self) -> list[str]:
        return ["link6", "attached_object"]

    def joint_limits(self) -> list[list[float]]:
        return [[-3.0] * 7, [3.0] * 7]

    def load_world(self, grid: dict[str, Any]) -> None:
        self.calls.append(("load_world", grid))

    def attach(self, spheres: Any) -> None:
        self.calls.append(("attach", None if spheres is None else np.asarray(spheres)))

    def reset_seeds(self) -> None:
        self.calls.append(("reset", None))

    def check(self, joints: Any) -> dict[str, Any]:
        self.calls.append(("check", list(joints)))
        key = "goal" if list(joints) != JOINTS else "start"
        return {"within_limits": f"{key}_limits" not in self.blocked,
                "self_collision": 0.0, "world_collision": 1.0 if f"{key}_world" in self.blocked else 0.0}

    def _outcome(self, joints: Any) -> dict[str, Any]:
        if self.outcome is not None:
            return self.outcome
        return {"trajectory": {"joint_names": NAMES, "positions": [list(joints), JOINTS], "velocities": None,
                               "dt_s": 0.02}, "ik_solved": True, "detail": {"stand_in": True},
                "ik_attempts": [{"status": "captured", "candidate_counts": {"optimized": 32, "accepted": 4}}]}

    def plan_pose(self, start: Any, pose: Any, *, max_attempts: int | None = None) -> dict[str, Any]:
        self.calls.append(("plan_pose", list(pose)))
        return self._outcome(start)

    def plan_joints(self, start: Any, goal: Any) -> dict[str, Any]:
        self.calls.append(("plan_joints", list(goal)))
        return self._outcome(start)

    def solve_ik(self, seed: Any, pose: Any) -> dict[str, Any]:
        self.calls.append(("solve_ik", list(pose)))
        solved = pose[2] > 0.3
        return {"solved": solved, "joint_positions": JOINTS if solved else None,
                "position_error_m": 0.0 if solved else 0.2, "rotation_error": 0.0 if solved else 0.1}

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _world_input(tmp_path, name="world.npz", *, footprints=(BOX,), gap=0.08, directive=-1.0):
    record = write(tmp_path / name, agentview=overhead((0.5, 0.0, 1.2), footprints, base_world=(0, 0, 0)),
                   joints=JOINTS, gap=gap, directive=directive)
    return {"path": record["path"], "sha256": record["sha256"]}


def _request(world_input=None, *, pose=None, joints=None, held=None) -> dict[str, Any]:
    goal = {"joint_positions": joints} if joints is not None else {
        "tool_frame": "panda_hand", "pose_base_xyzw": pose or [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]}
    return {"schema": PLAN_REQUEST_SCHEMA, "robot": "franka_panda",
            "start": {"joint_names": NAMES, "joint_positions": JOINTS}, "goal": goal,
            "world": {"model": WORLD_MODEL, "input": world_input} if world_input else {"model": "none"},
            "held_object": held or {"source": "world" if world_input else "none"}, "constraints": []}


def _worker(world: bool = True) -> tuple[Worker, Engine]:
    engine = Engine()
    return Worker(engine, robot="franka_panda", world_model=WORLD_MODEL if world else "none"), engine


def test_a_world_plan_loads_the_exported_world_once_per_goal_and_names_its_hash(tmp_path):
    worker, engine = _worker()
    world_input = _world_input(tmp_path)
    request = _request(world_input)
    answer = validate_plan_answer(worker.plan(request), request)
    assert answer["status"] == "planned" and answer["reason"] is None
    # The world is the planner-neutral one: same hash as depth_world builds from the same evidence.
    expected = build_world(read_world_inputs(tmp_path / "world.npz", world_input["sha256"]),
                           robot_spheres_base=SPHERES, start_hand_base=pose_matrix(HAND),
                           goal_hand_base=pose_matrix(request["goal"]["pose_base_xyzw"]))
    assert answer["world"]["sha256"] == expected.sha256 and answer["world"]["model"] == WORLD_MODEL
    # The answer names the sphere set it was planned with and keeps the IK seed populations.
    assert answer["planner"] == {"implementation": "curobo_v2", "robot_sha256": "e" * 64,
                                 "collision_spheres_sha256": "f" * 64}
    assert answer["detail"]["ik_attempts"] == [{"status": "captured",
                                                "candidate_counts": {"optimized": 32, "accepted": 4}}]
    grids = [value for name, value in engine.calls if name == "load_world"]
    assert len(grids) == 1 and grids[0]["features"].shape == (121, 121, 101)
    assert grids[0]["features"].dtype == np.float16
    assert answer["world"]["summary"]["planner_grid_sha256"] == curobo_v2_grid(expected.esdf)["sha256"]
    # Detach for the robot filter, detach before the goal, then one seed reset, the start check, the plan and its check.
    assert engine.names() == ["attach", "kinematics", "attach", "load_world", "reset", "check", "plan_pose", "path_spheres"]
    # The same snapshot reuses its scene; a joint goal loads its own world and checks the goal too.
    engine.calls.clear()
    joint_goal = _request(world_input, joints=[0.1] * 7)
    second = validate_plan_answer(worker.plan(joint_goal), joint_goal)
    assert second["world"]["inputs_sha256"] == answer["world"]["inputs_sha256"]
    assert engine.names() == ["attach", "kinematics", "load_world", "reset", "check", "check", "plan_joints",
                              "path_spheres"]


def test_a_goal_below_the_support_is_reported_in_the_world_frame_of_the_policy_s_poses(tmp_path):
    """The world refuses in the arm's base frame; the policy's poses are world, so the answer says the world height."""
    base = (0.1, -0.2, 0.9)
    record = write(tmp_path / "raised.npz", agentview=overhead((0.5, 0.0, 1.2), (BOX,), base_world=base),
                   base_world=base, joints=JOINTS)
    worker, _ = _worker()
    answer = validate_plan_answer(worker.plan(_request({"path": record["path"], "sha256": record["sha256"]},
                                                       pose=INSIDE)))
    refusal = answer["world"]["summary"]["goal_refusal"]
    assert answer["reason"] == "goal_below_support" and refusal["support_z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert refusal["in_world"] == {"frame": "world", "support_z_m": pytest.approx(TABLE_Z + 0.9, abs=1e-4),
                                   "fingertip_z_m": pytest.approx(TABLE_Z - 0.03 + 0.9, abs=1e-4)}
    assert f"at z {TABLE_Z + 0.9:.3f} m (world frame)" in refusal["message"] and "base frame" not in refusal["message"]
    assert refusal_message(answer) == refusal["message"]
    assert refusal["message"] in planning_diagnostics(answer)["recovery"][0]


def test_reasons_follow_the_world_the_start_the_goal_and_the_solver_in_that_order(tmp_path):
    worker, engine = _worker()
    world_input = _world_input(tmp_path)
    below = validate_plan_answer(worker.plan(_request(world_input, pose=INSIDE)))
    assert (below["status"], below["reason"]) == ("refused", "goal_below_support")
    assert below["world"]["summary"]["goal_refusal"]["reason"] == "fingertips_below_support"
    assert "check" not in engine.names() and "plan_pose" not in engine.names()
    cases = [({"start_limits"}, None, "start_outside_limits"), ({"start_world"}, None, "start_in_collision"),
             ({"goal_world"}, [0.1] * 7, "goal_in_collision"), ({"goal_limits"}, [0.1] * 7, "goal_unreachable")]
    for blocked, joints, reason in cases:
        engine.blocked, engine.calls = blocked, []
        answer = validate_plan_answer(worker.plan(_request(world_input, joints=joints)))
        assert answer["reason"] == reason and not {"plan_pose", "plan_joints"} & set(engine.names())
    engine.blocked = set()
    for ik_solved, reason in ((False, "goal_unreachable"), (True, "path_blocked")):
        engine.outcome = {"trajectory": None, "ik_solved": ik_solved, "detail": {}}
        answer = validate_plan_answer(worker.plan(_request(world_input)))
        assert answer["status"] == "refused" and answer["reason"] == reason
        assert answer["detail"]["solver"] == {}
    joint_goal = validate_plan_answer(worker.plan(_request(world_input, joints=[0.1] * 7)))
    assert joint_goal["reason"] == "path_blocked"


def test_a_start_resting_on_a_joint_stop_plans_from_just_inside_it(tmp_path, monkeypatch):
    worker, engine = _worker()
    world_input = _world_input(tmp_path)
    planned_from = []
    plan_pose = engine.plan_pose
    monkeypatch.setattr(engine, "plan_pose", lambda start, pose, **keywords: (
        planned_from.append(list(start)), plan_pose(start, pose, **keywords))[1])
    # Joint 6 measured 0.2 mrad past its upper limit, joint 1 exactly on its lower one.
    monkeypatch.setattr(engine, "joint_limits", lambda: [[0.0] + [-3.0] * 6, [3.0] * 5 + [1.0 - 2e-4, 3.0]])
    answer = validate_plan_answer(worker.plan(_request(world_input)))
    assert answer["status"] == "planned"
    assert planned_from[-1][0] == pytest.approx(0.001) and planned_from[-1][5] == pytest.approx(1.0 - 2e-4 - 0.001)
    assert planned_from[-1][1:5] == JOINTS[1:5]
    assert [(row["joint"], row["measured"]) for row in answer["detail"]["start_inside_limits"]] == [
        ("panda_joint1", 0.0), ("panda_joint6", 1.0)]
    # Further out than the slack the start is not moved, and stays the check's to refuse.
    monkeypatch.setattr(engine, "joint_limits", lambda: [[-3.0] * 7, [3.0] * 5 + [0.9, 3.0]])
    planned_from.clear()
    answer = validate_plan_answer(worker.plan(_request(world_input)))
    assert "start_inside_limits" not in answer["detail"] and planned_from[-1] == JOINTS


SOLVED_IN_EMPTY_SPACE = [0.1] * 7


def _unreachable(tmp_path, spheres, monkeypatch):
    """A pose goal no IK attempt solves in its world; ``spheres`` are the robot's at the solve in empty space."""
    worker, engine = _worker()
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}}
    kinematics = engine.kinematics
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: {
        "solved": True, "joint_positions": SOLVED_IN_EMPTY_SPACE, "position_error_m": 0.0, "rotation_error": 0.0})
    monkeypatch.setattr(engine, "kinematics", lambda joints: (
        (pose_matrix(HAND), np.asarray(spheres, dtype=np.float64)) if list(joints) == SOLVED_IN_EMPTY_SPACE
        else kinematics(joints)))
    return validate_plan_answer(worker.plan(_request(_world_input(tmp_path)))), engine


def test_a_goal_no_ik_reaches_with_nothing_in_the_way_is_out_of_reach(tmp_path, monkeypatch):
    worker, engine = _worker()
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}}
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: {"solved": False, "joint_positions": None})
    answer = validate_plan_answer(worker.plan(_request(_world_input(tmp_path))))
    assert answer["reason"] == "goal_unreachable"
    assert answer["detail"]["goal_diagnosis"]["cause"] == "out_of_reach"
    assert "even with nothing in the way" in refusal_message(answer)
    diagnostics = planning_diagnostics(answer)
    assert diagnostics["recovery"][0].startswith("No joint configuration reaches this pose")
    assert len(diagnostics["recovery"]) == 3


def test_a_refusal_that_names_the_fingers_says_the_plan_keeps_them_open(tmp_path, monkeypatch):
    """Closing the jaws changes no plan on a robot planned with them open, and the sentence says so: a model closed
    them twice, spent the native actions and met the same refusal."""
    from ora.environment.planners.curobo_v2 import worker as module

    monkeypatch.setattr(module, "LINK_PARTS", {"franka_panda": {"link6": "fingers"}})
    spheres = [[0.75, 0.3, TABLE_Z - 0.015, 0.01], [0.2, 0.3, 0.3, 0.01]]
    (tmp_path / "other").mkdir()
    (tmp_path / "open").mkdir()
    other = _unreachable(tmp_path / "other", spheres, monkeypatch)[0]["detail"]["goal_diagnosis"]["message"]
    assert "keeps the jaws" not in other, "only a robot planned with its jaws open"
    monkeypatch.setattr(module, "FINGERS_PLANNED_OPEN", frozenset({"franka_panda"}))
    message = _unreachable(tmp_path / "open", spheres, monkeypatch)[0]["detail"]["goal_diagnosis"]["message"]
    assert "there its fingers would be" in message
    assert "(the plan keeps the jaws fully open, so closing them changes nothing here)" in message


def test_a_goal_that_ik_reaches_in_empty_space_names_what_stands_there(tmp_path, monkeypatch):
    answer, engine = _unreachable(tmp_path, [[0.75, 0.3, TABLE_Z - 0.015, 0.01], [0.2, 0.3, 0.3, 0.01]], monkeypatch)
    diagnosis = answer["detail"]["goal_diagnosis"]
    assert answer["reason"] == "goal_unreachable" and diagnosis["cause"] == "robot_in_collision"
    assert diagnosis["link"] == "link6" and diagnosis["part"] == "arm" and diagnosis["penetration_m"] > 0
    assert "there its arm would be" in diagnosis["message"]
    assert "cm inside the supporting surface" in diagnosis["message"]
    assert refusal_message(answer) == diagnosis["message"]
    # The extra solve ran in an empty world, with nothing attached, after the goal's own.
    assert engine.names().index("plan_pose") < engine.names().index("reset", engine.names().index("plan_pose"))
    assert ([call for name, call in engine.calls if name == "load_world"][-1]["features"] > 0).all()
    assert [call for name, call in engine.calls if name == "attach"][-1] is None


def test_a_goal_clear_of_the_world_at_that_solution_says_only_what_it_knows(tmp_path, monkeypatch):
    answer, _engine = _unreachable(tmp_path, [[0.2, 0.3, 0.3, 0.01], [0.2, 0.2, 0.3, 0.01]], monkeypatch)
    assert answer["detail"]["goal_diagnosis"]["cause"] == "no_clear_solution_found"


def test_the_smallest_tilt_that_plans_is_what_the_refusal_offers(tmp_path, monkeypatch):
    worker, engine = _worker()
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}}
    plan_pose = engine.plan_pose
    requested = _request()["goal"]["pose_base_xyzw"]
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: {"solved": True, "joint_positions": JOINTS})

    def plans_only_tilted(start, pose, *, max_attempts=None):
        planned = plan_pose(start, pose, max_attempts=max_attempts)
        return planned if pose == requested else {**planned, "trajectory": {"positions": [JOINTS]}}

    monkeypatch.setattr(engine, "plan_pose", plans_only_tilted)
    answer = validate_plan_answer(worker.plan(_request(_world_input(tmp_path))))
    diagnosis = answer["detail"]["goal_diagnosis"]
    assert diagnosis["nearby"]["tilt_deg"] == 10.0 and engine.names().count("plan_pose") == 2
    assert "an approach tilted 10 degrees plans:" in refusal_message(answer) and " in world coordinates instead of (" in (
        refusal_message(answer))
    assert answer["timing_s"]["goal_diagnosis"] >= 0.0


def test_a_tilt_search_plans_a_bounded_number_of_goals_and_says_so(tmp_path, monkeypatch):
    worker, engine = _worker()
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}}
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: {"solved": True, "joint_positions": JOINTS})
    answer = validate_plan_answer(worker.plan(_request(_world_input(tmp_path))))
    assert answer["detail"]["goal_diagnosis"]["nearby"] is None
    assert engine.names().count("plan_pose") == 1 + NEARBY_PLANS
    assert "no approach tilted up to 45 degrees was found to plan here either" in refusal_message(answer)


def test_seeds_stopped_by_the_arm_itself_read_as_that_not_as_the_world(tmp_path, monkeypatch):
    worker, engine = _worker()
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}, "ik_attempts": [
        {"status": "captured", "joint_positions": [[0.2] * 7, [0.3] * 7, [0.4] * 7],
         "converged": [True, True, False]}]}
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: {"solved": True, "joint_positions": JOINTS})
    monkeypatch.setattr(engine, "kinematics", lambda joints: (pose_matrix(HAND), np.array([[0.2, 0.3, 0.3, 0.01]])))
    monkeypatch.setattr(engine, "check", lambda joints: {
        "within_limits": True, "self_collision": 1.0 if joints != JOINTS else 0.0, "world_collision": 0.0})
    answer = validate_plan_answer(worker.plan(_request(_world_input(tmp_path))))
    assert answer["detail"]["goal_diagnosis"]["cause"] == "self_collision"
    assert "the solutions found put the arm against itself" in refusal_message(answer)


def test_a_refusal_at_the_goal_names_what_stands_there_as_a_refused_path_does():
    worker, _ = _worker()
    other_arm = np.zeros(worker.parameters.grid_shape, dtype=bool)
    voxel = np.floor((np.asarray([0.4, 0.0, 0.3]) - worker.parameters.grid_low_m) / worker.parameters.voxel_size_m)
    other_arm[tuple(voxel.astype(int))] = True
    worker._scene = (None, SimpleNamespace(parameters=worker.parameters, other_arm=other_arm))
    world = SimpleNamespace(summary={"support": {"goal": {"z_m": 0.1}}})
    assert worker._stands(np.asarray([0.4, 0.0, 0.3, 0.02]), world) == "the other arm"
    assert worker._stands(np.asarray([0.7, 0.0, 0.11, 0.02]), world) == "the supporting surface"
    assert worker._stands(np.asarray([0.7, 0.0, 0.3, 0.02]), world) == "what the depth shows"


def test_the_other_arm_is_asked_only_of_an_empty_hand_on_a_two_arm_robot(monkeypatch):
    worker, engine = _worker()
    seen = {}
    monkeypatch.setattr(engine, "solve_ik", lambda seed, pose: seen.update(seed=seed, pose=pose) or {"solved": True})
    empty, holding = SimpleNamespace(held_spheres_hand=np.zeros((0, 4))), SimpleNamespace(held_spheres_hand=np.ones((1, 4)))
    worker._inputs = SimpleNamespace(other_arm_joint_positions=None)
    assert worker._other_arm_reaches(list(HAND), empty) is None
    worker._inputs = SimpleNamespace(other_arm_joint_positions=np.zeros(7), base_pose_world_xyzw=[1.0, 0, 0, 0, 0, 0, 1],
                                     other_arm_base_pose_world_xyzw=[0.0, 0, 0, 0, 0, 0, 1])
    assert worker._other_arm_reaches(list(HAND), holding) is None
    assert worker._other_arm_reaches(list(HAND), empty) is True
    assert seen["seed"] == [0.0] * 7 and seen["pose"][0] == pytest.approx(HAND[0] + 1.0)


def test_only_a_pose_goal_a_world_was_built_for_is_diagnosed(tmp_path):
    worker, engine = _worker(world=False)
    engine.outcome = {"trajectory": None, "ik_solved": False, "detail": {}}
    answer = validate_plan_answer(worker.plan(_request()))
    assert answer["reason"] == "goal_unreachable" and "goal_diagnosis" not in answer["detail"]
    assert "solve_ik" not in engine.names()


def test_the_held_load_is_attached_from_the_world_or_from_the_request(tmp_path):
    worker, engine = _worker()
    world_input = _world_input(tmp_path, footprints=(BOX, HELD), gap=0.04, directive=1.0)
    answer = validate_plan_answer(worker.plan(_request(world_input)))
    summary = answer["world"]["summary"]
    assert summary["holding"] is True and summary["held_spheres"] > 0
    attached = [value for name, value in engine.calls if name == "attach"]
    assert attached[:2] == [None, None] and attached[2].shape == (summary["held_spheres"], 4)
    robot_only, engine = _worker(world=False)
    spheres = [[0.0, 0.0, 0.1, 0.03], [0.0, 0.0, 0.14, 0.03]]
    held = {"source": "spheres", "spheres_tool_frame": spheres}
    answer = validate_plan_answer(robot_only.plan(_request(held=held)))
    assert answer["world"] is None and answer["status"] == "planned"
    assert [value for name, value in engine.calls if name == "attach"][-1].tolist() == spheres
    assert "load_world" not in engine.names()


def test_a_kitchen_has_no_support_so_a_goal_below_one_goes_to_the_solver(tmp_path):
    worker = Worker(Engine(), robot="franka_panda", world_model=WORLD_MODEL, scene="kitchen")
    assert worker.ready()["scene"] == "kitchen" and "scene" not in _worker()[0].ready()
    answer = validate_plan_answer(worker.plan(_request(_world_input(tmp_path), pose=INSIDE)))
    assert answer["reason"] != "goal_below_support"
    assert answer["world"]["summary"]["goal_refusal"] is None
    assert answer["world"]["summary"]["support"]["goal"] is not None  # its slab stays under the goal carve
    with pytest.raises(ValueError):
        Worker(Engine(), robot="franka_panda", world_model=WORLD_MODEL, scene="garden")


def test_ik_solves_each_query_in_its_plans_world_and_skips_orphans(tmp_path):
    worker, engine = _worker()
    world_input = _world_input(tmp_path)
    request = {"schema": IK_REQUEST_SCHEMA, "robot": "franka_panda",
               "start": {"joint_names": NAMES, "joint_positions": JOINTS},
               "queries": [{"tool_frame": "panda_hand", "pose_base_xyzw": HAND, "from_index": None},
                           {"tool_frame": "panda_hand", "pose_base_xyzw": INSIDE, "from_index": None},
                           {"tool_frame": "panda_hand", "pose_base_xyzw": HAND, "from_index": 1}],
               "world": {"model": WORLD_MODEL, "input": world_input}, "held_object": {"source": "world"}}
    answer = validate_ik_answer(worker.solve_ik(request), request)
    assert [row["solved"] for row in answer["results"]] == [True, False, False]
    assert [row["refusal"] for row in answer["results"]] == [None, "goal_below_support", None]
    assert answer["results"][2]["skipped"] is True and answer["world"]["queries"][2] is None
    assert answer["planner"]["collision_spheres_sha256"] == "f" * 64
    assert [name for name in engine.names() if name in {"load_world", "reset", "solve_ik"}] == [
        "load_world", "reset", "solve_ik", "load_world", "reset"]
    # The IK query's world is the world a plan to that pose is planned in.
    plan = validate_plan_answer(worker.plan(_request(world_input, pose=HAND)))
    assert plan["world"]["sha256"] == answer["world"]["queries"][0]["sha256"]


def test_errors_are_answers_and_the_protocol_serves_ready_plan_ik_and_close(tmp_path):
    worker, engine = _worker(world=False)
    engine.plan_pose = lambda *_: (_ for _ in ()).throw(RuntimeError("device lost"))
    error = validate_plan_answer(worker.plan(_request()))
    assert (error["status"], error["reason"]) == ("error", "planner_error")
    assert error["detail"]["error"] == {"type": "RuntimeError", "message": "device lost"}
    malformed = validate_plan_answer(worker.plan({"schema": PLAN_REQUEST_SCHEMA}))
    assert malformed["status"] == "error" and malformed["request"] is None
    worker, _ = _worker(world=False)
    lines = [json.dumps({"protocol": PROTOCOL, "command": "ready", "request_id": "r0"}),
             json.dumps({"protocol": PROTOCOL, "command": "plan", "request_id": "r1", "request": _request()}),
             json.dumps({"protocol": "other", "command": "plan", "request_id": "r2"}),
             json.dumps({"protocol": PROTOCOL, "command": "teleport", "request_id": "r3"}),
             json.dumps({"protocol": PROTOCOL, "command": "close", "request_id": "r4"}),
             json.dumps({"protocol": PROTOCOL, "command": "ready", "request_id": "never"})]
    written: list[dict[str, Any]] = []
    assert serve(worker, lines, lambda text: written.append(json.loads(text))) == 0
    assert [reply["request_id"] for reply in written] == ["r0", "r1", "r2", "r3", "r4"]
    assert written[0]["ready"]["implementation"] == "curobo_v2" and written[0]["ready"]["world_model"] == "none"
    assert written[1]["answer"]["status"] == "planned"
    assert "error" in written[2] and "error" in written[3] and written[4]["closed"] is True


def _check_request(world_input, *, allowed=(), tolerance=0.0, rows=None) -> dict[str, Any]:
    return {"schema": CHECK_REQUEST_SCHEMA, "robot": "franka_panda",
            "start": {"joint_names": NAMES, "joint_positions": JOINTS}, "path": rows or [JOINTS],
            "world": {"model": WORLD_MODEL, "input": world_input}, "allowed_links": list(allowed),
            "tolerance_m": tolerance}


def test_a_path_check_names_the_link_that_enters_the_world_and_spares_the_allowed_ones(tmp_path):
    worker, engine = _worker()
    # The world holds an object where the stand-in's one sphere is, past the start carve at the path's last row.
    far = np.array([[0.65, 0.0, 0.02, 0.04], [0.0, 0.0, 0.0, -100.0]])
    start = pose_matrix(HAND), SPHERES.copy()
    calls = iter([start, start, (pose_matrix(HAND), far)])
    engine.kinematics = lambda joints: next(calls)
    clear = worker.check_path(_check_request(_world_input(tmp_path, "a.npz"), allowed=["link6"]))
    assert clear["status"] == "clear" and clear["first_blocked_row"] is None
    validate_check_answer(clear)
    calls = iter([start, start, (pose_matrix(HAND), far)])
    blocked = worker.check_path(_check_request(_world_input(tmp_path, "b.npz")))
    assert blocked["status"] == "blocked" and blocked["first_blocked_row"] == 0 and blocked["link"] == "link6"
    assert blocked["penetration_m"] > 0
    validate_check_answer(blocked)


def test_a_path_check_error_is_an_answer(tmp_path):
    worker, _ = _worker()
    answer = worker.check_path({**_check_request(_world_input(tmp_path)), "tolerance_m": 1.0})
    assert answer["status"] == "error" and answer["reason"] == "planner_error"
    validate_check_answer(answer)


def test_the_fk_command_answers_the_planners_own_tool_poses():
    worker, _ = _worker(world=False)
    assert worker.fk({"joint_positions": [JOINTS]})["tool_poses_base_xyzw"] == [HAND]
    with pytest.raises(ValueError):
        worker.fk({"joint_positions": []})


def test_a_worker_refuses_another_world_model_or_robot():
    with pytest.raises(ValueError, match="unsupported"):
        Worker(Engine(), robot="aloha", world_model="none")
    worker, _ = _worker(world=False)
    answer = validate_plan_answer(worker.plan(_request({"path": "/w.npz", "sha256": "a" * 64})))
    assert answer["status"] == "error" and "none planner" in answer["detail"]["error"]["message"]


#: A plan v2's timing: the controlled point 10 cm along the tool frame's approach axis.
TIMING = {
    "controlled_point": {"tool_frame": "panda_hand", "offset_xyzw": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]},
    "speed": {"speed": "normal", "slow_approaching": False}, "slow_approach": True,
    "limits": {"joint_velocity_rad_s": [1.0] * 7, "joint_acceleration_rad_s2": None,
               "joint_jerk_rad_s3": [1000.0] * 7, "jerk_source": "unverified"},
    "tick_s": 0.05, "budget_ticks": 2000, "cut_to_budget": False,
    "wrist": {"joint_index": 6, "limits_rad": [-2.8, 2.8]},
}


def _timed_request(timing=TIMING) -> dict[str, Any]:
    return {**_request(pose=[0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), "schema": PLAN_REQUEST_SCHEMA_V2, "timing": timing}


def test_a_timed_plan_keeps_its_path_on_the_joint_grid_with_the_controlled_point_at_every_row():
    worker, engine = _worker(world=False)
    goal = [row + 0.1 for row in JOINTS]
    engine.outcome = {"trajectory": {"joint_names": NAMES, "positions": [JOINTS, JOINTS, goal, goal],
                                     "velocities": None, "dt_s": 0.02},
                      "ik_solved": True, "detail": {}, "ik_attempts": []}
    request = _timed_request()
    answer = worker.plan(request)
    checked = validate_plan_answer(answer, request, untimed=True)
    assert checked["schema"] == PLAN_ANSWER_SCHEMA_V2 and checked["status"] == "planned" and checked["timing"] is None
    path = checked["detail"]["timing_path"]
    rows = np.asarray(path["joint_positions"])
    # The repeated rows are gone and the nodes at most 0.02 rad apart, the ends exact.
    assert rows[0].tolist() == JOINTS and rows[-1].tolist() == goal
    assert len(rows) == 6 and np.abs(np.diff(rows, axis=0)).max() <= 0.02 + 1e-9
    # The stand-in hand is at HAND, so the point is 10 cm further along its z.
    assert path["tool_poses_xyzw"][0] == pytest.approx([0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0])
    retime = planned_retime_request(checked)
    assert retime["speed"] == TIMING["speed"] and retime["joint_names"] == NAMES and retime["path"] == path


def test_a_plan_v2_without_timing_or_refused_keeps_no_path():
    worker, engine = _worker(world=False)
    request = _timed_request(None)
    answer = validate_plan_answer(worker.plan(request), request)
    assert answer["schema"] == PLAN_ANSWER_SCHEMA_V2 and answer["timing"] is None
    assert "timing_path" not in answer["detail"]
    engine.outcome = {"trajectory": None, "ik_solved": True, "detail": {}, "ik_attempts": []}
    request = _timed_request()
    refused = validate_plan_answer(worker.plan(request), request, untimed=True)
    assert refused["status"] == "refused" and refused["reason"] == "path_blocked"
    assert "timing_path" not in refused["detail"]


def test_a_planner_error_answers_in_the_request_s_version():
    worker, engine = _worker(world=False)
    engine.tool_pose = lambda joints: (_ for _ in ()).throw(RuntimeError("no kinematics"))
    request = _timed_request()
    answer = validate_plan_answer(worker.plan(request), request, untimed=True)
    assert answer["schema"] == PLAN_ANSWER_SCHEMA_V2 and answer["reason"] == "planner_error"


def test_compose_pose_applies_the_offset_in_the_tool_frame():
    down = [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]  # 180 degrees about x: the tool's z points down.
    pose = compose_pose(down, [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    assert pose[:3] == pytest.approx([0.5, 0.0, 0.2]) and abs(pose[3]) == pytest.approx(1.0)


#: A pure translation's goal: the first joint moves the hand 15 cm, the wrist keeps its angle.
TRANSLATED = [0.3, *JOINTS[1:]]


def _turned_hand(joints: Any) -> list[float]:
    """The stand-in hand: 0.5 m per radian of the first joint along x, turned about its approach by the last."""
    half = float(joints[6]) / 2.0
    return [0.4 + 0.5 * float(joints[0]), 0.0, 0.5, 0.0, 0.0, float(np.sin(half)), float(np.cos(half))]


class DetourEngine(Engine):
    """Plans to its goal joints the long way: the wrist swings ``swing`` rad out and back on the way."""

    def __init__(self, goal: list[float] = TRANSLATED, swing: float = 2.5, limit: float = 3.0) -> None:
        super().__init__()
        self.goal, self.swing, self.limit, self.self_hits = list(goal), swing, limit, False

    def tool_pose(self, joints: Any) -> list[float]:
        return _turned_hand(joints)

    def joint_limits(self) -> list[list[float]]:
        return [[-self.limit] * 7, [self.limit] * 7]

    def _outcome(self, joints: Any) -> dict[str, Any]:
        middle = [(start + goal) / 2.0 for start, goal in zip(joints, self.goal)]
        middle[6] += self.swing
        rows = [list(joints), middle, list(self.goal)]
        return {"trajectory": {"joint_names": NAMES, "positions": rows, "velocities": [[0.0] * 7] * 3, "dt_s": 0.02},
                "ik_solved": True, "detail": {"stand_in": True}, "ik_attempts": []}

    def check(self, joints: Any) -> dict[str, Any]:
        found = super().check(joints)
        return {**found, "self_collision": 1.0 if self.self_hits and list(joints) != JOINTS else 0.0}


def _unrequested_turn(rows: list[list[float]]) -> float:
    """What the pacing's wrist check measures of a path (``motion_pace.check_unwind``)."""
    turns = path_turns(joint_path_grid(rows, lambda joints: [_turned_hand(row) for row in joints]).poses)
    return turns["path_turn_deg"] - turns["requested_turn_deg"]


def _from(start: list[float]) -> dict[str, Any]:
    """A robot-only pose request from ``start``."""
    return {**_request(), "start": {"joint_names": NAMES, "joint_positions": list(start)}}


def _candidate(record: dict[str, Any], source: str) -> dict[str, Any]:
    (found,) = [item for item in record["candidates"] if item["source"] == source]
    return found


@pytest.mark.parametrize("world", [True, False])
def test_a_planner_detour_the_straight_joint_line_avoids_is_answered_as_that_line(tmp_path, world):
    engine = DetourEngine()
    worker = Worker(engine, robot="franka_panda", world_model=WORLD_MODEL if world else "none")
    request = _request(_world_input(tmp_path) if world else None)
    answer = validate_plan_answer(worker.plan(request), request)
    planned = engine._outcome(JOINTS)["trajectory"]["positions"]
    assert _unrequested_turn(planned) > MAX_UNREQUESTED_TURN_DEG  # the pacing would refuse the planner's path
    rows = answer["trajectory"]["positions"]
    # The same goal, reached by a path the wrist check accepts; the planner's own checks ran on every row.
    assert answer["status"] == "planned" and rows[0] == JOINTS and rows[-1] == TRANSLATED
    assert _unrequested_turn(rows) <= MAX_UNREQUESTED_TURN_DEG
    assert len(answer["trajectory"]["velocities"]) == len(rows) and answer["trajectory"]["dt_s"] == 0.02
    assert engine.names().count("check") >= len(rows)
    detour = answer["detail"]["wrist_detour"]
    assert (detour["status"], detour["chosen"]) == ("replaced", "same_goal")
    assert detour["max_unrequested_turn_deg"] == MAX_UNREQUESTED_TURN_DEG
    assert detour["planned"]["path_turn_deg"] == pytest.approx(np.degrees(5.0), abs=0.1)
    assert detour["candidates"] == [{"source": "same_goal", "path_turn_deg": 0.0, "requested_turn_deg": 0.0,
                                     "outcome": "chosen"}]
    assert (answer["world"]["summary"]["wrist_detour"] == detour) if world else answer["world"] is None
    assert answer["timing_s"]["wrist_detour"] >= 0.0


def test_a_straight_line_that_fails_the_planner_s_checks_or_unwinds_too_leaves_the_plan(tmp_path):
    engine = DetourEngine()
    engine.self_hits = True
    worker = Worker(engine, robot="franka_panda", world_model=WORLD_MODEL)
    blocked = validate_plan_answer(worker.plan(_request(_world_input(tmp_path))))
    assert blocked["trajectory"]["positions"] == engine._outcome(JOINTS)["trajectory"]["positions"]
    record = blocked["detail"]["wrist_detour"]
    assert (record["status"], record["chosen"]) == ("kept", None)
    assert _candidate(record, "same_goal")["outcome"] == "blocked"
    assert _candidate(record, "same_goal")["blocked"]["check"] == "planner"
    # A Panda goal at the far end of its last joint, the start near the other end: the straight line turns the long
    # way as well and no 2 pi equivalent is in range, so nothing here helps (the backend's jaw twin may); the plan
    # stays for the pacing to refuse.
    start = [*JOINTS[:6], 2.5]
    far = DetourEngine(goal=[*JOINTS[:6], -2.8], swing=0.0, limit=2.8973)
    kept = validate_plan_answer(Worker(far, robot="franka_panda", world_model="none").plan(_from(start)))
    assert kept["trajectory"]["positions"] == far._outcome(start)["trajectory"]["positions"]
    record = kept["detail"]["wrist_detour"]
    assert (record["status"], [item["source"] for item in record["candidates"]]) == ("kept", ["same_goal"])
    assert _candidate(record, "same_goal")["outcome"] == "unwinds"
    # A plan within the rule is answered as it is, with no record.
    within = validate_plan_answer(_worker(world=False)[0].plan(_request()))
    assert "wrist_detour" not in within["detail"]


#: The ALOHA's planning limit: a full turn of range plus about a radian past the seam each way.
ALOHA_LIMIT = 4.2


@pytest.mark.parametrize("start_wrist,goal_wrist,refused", [(-1.1, 2.2, False), (-2.0, 3.5, True)])
def test_a_goal_a_whole_turn_away_takes_its_nearer_equivalent_in_range(start_wrist, goal_wrist, refused):
    """An IK goal more than half a turn from the wrist: the same pose lies at goal - 2 pi, inside +/-4.2 rad."""
    start = [*JOINTS[:6], start_wrist]
    engine = DetourEngine(goal=[*JOINTS[:6], goal_wrist], swing=0.0, limit=ALOHA_LIMIT)
    answer = validate_plan_answer(Worker(engine, robot="franka_panda", world_model="none").plan(_from(start)))
    planned = engine._outcome(start)["trajectory"]["positions"]
    # 3.3 rad of wrist is 18 degrees beyond the request, which the pacing accepts; 5.5 rad is 270 and refused.
    assert (_unrequested_turn(planned) > MAX_UNREQUESTED_TURN_DEG) is refused
    rows = answer["trajectory"]["positions"]
    assert rows[0] == start and rows[-1][6] == pytest.approx(goal_wrist - 2 * np.pi) and rows[-1][:6] == start[:6]
    reached, asked = _turned_hand(rows[-1]), _turned_hand(planned[-1])
    assert -ALOHA_LIMIT <= rows[-1][6] and reached[:3] == pytest.approx(asked[:3])
    assert abs(float(np.dot(reached[3:], asked[3:]))) == pytest.approx(1.0)  # the same pose
    assert _unrequested_turn(rows) <= 1e-6
    record = answer["detail"]["wrist_detour"]
    assert (record["status"], record["chosen"]) == ("replaced", "alias")
    alias = _candidate(record, "alias")
    assert alias["goal_joints"][6] == pytest.approx(goal_wrist - 2 * np.pi, abs=1e-6)
    assert alias["path_turn_deg"] < record["planned"]["path_turn_deg"]
    # Accepted by the pacing, the plan took part in the ranking and lost it; refused, it did not take part.
    assert [item["source"] for item in record["candidates"]] == (["alias", "same_goal"] if refused
                                                                 else ["planned", "alias"])


def test_a_goal_with_no_equivalent_in_range_is_left_to_the_planner():
    """A Panda's last joint (+/-2.8973 rad) has no room for a second equivalent of any goal."""
    start = [*JOINTS[:6], -0.5]
    engine = DetourEngine(goal=[*JOINTS[:6], 2.8], swing=0.0, limit=2.8973)
    answer = validate_plan_answer(Worker(engine, robot="franka_panda", world_model="none").plan(_from(start)))
    assert answer["trajectory"]["positions"] == engine._outcome(start)["trajectory"]["positions"]
    assert "wrist_detour" not in answer["detail"]  # 18 degrees within the rule, and 2.8 - 2 pi is out of range


class BranchEngine(DetourEngine):
    """A hand turned about its approach by joints 5 and 7 together, so two arm branches reach one orientation."""

    def tool_pose(self, joints: Any) -> list[float]:
        half = (float(joints[4]) + float(joints[6])) / 2.0
        return [0.4 + 0.5 * float(joints[0]), 0.0, 0.5, 0.0, 0.0, float(np.sin(half)), float(np.cos(half))]

    def _outcome(self, joints: Any) -> dict[str, Any]:
        outcome = super()._outcome(joints)
        return {**outcome, "ik_attempts": self.attempts}


def test_a_refused_plan_takes_the_ik_solution_whose_line_turns_least_not_the_one_planned(tmp_path):
    """At the Panda's last-joint limit: the planned branch winds joint 7 the long way, another keeps it in range."""
    start = [*JOINTS[:4], 0.0, JOINTS[5], 2.5]
    goal = [*JOINTS[:4], 0.0, JOINTS[5], -2.8]  # the hand at -2.8 rad: 5.3 rad of wrist from 2.5
    other = [*JOINTS[:4], 1.0, JOINTS[5], 2.5 - 0.0170]  # 1.0 + 2.483 = -2.8 + 2 pi: the same orientation
    engine = BranchEngine(goal=goal, swing=0.0, limit=2.8973)
    engine.attempts = [{"status": "captured", "joint_positions": [
        goal,  # the plan's own goal, counted once
        [*JOINTS[:4], -1.5, JOINTS[5], -1.3],  # the same orientation the long way round: unwinds too
        [*JOINTS[:4], 2.0, JOINTS[5], 1.483],  # as short a turn, ranked after the nearer ones
        other,
        [*JOINTS[:4], 0.6, JOINTS[5], 2.883],  # as short a turn and nearest, but its line is blocked below
        [*JOINTS[:4], 0.5, JOINTS[5], 2.983],  # not accepted by the IK
    ], "accepted": [True, True, True, True, True, False]}]

    def check(joints: Any) -> dict[str, Any]:
        hit = abs(float(joints[4]) - 0.6) < 1e-9 and abs(float(joints[6]) - 2.883) < 1e-9  # that branch's goal
        return {"within_limits": True, "self_collision": 1.0 if hit else 0.0, "world_collision": 0.0}

    engine.check = check
    answer = validate_plan_answer(Worker(engine, robot="franka_panda", world_model="none").plan(_from(start)))
    rows = answer["trajectory"]["positions"]
    assert rows[0] == start and rows[-1] == pytest.approx(other)
    record = answer["detail"]["wrist_detour"]
    assert (record["status"], record["chosen"]) == ("replaced", "ik_pool")
    pool = [item for item in record["candidates"] if item["source"] == "ik_pool"]
    # Nearest the start first; the three short turns tie, so the blocked nearest yields to the next.
    assert [item["goal_joints"][4] for item in pool] == pytest.approx([0.6, 1.0, 2.0, -1.5])
    assert [item["outcome"] for item in pool] == ["blocked", "chosen", "ranked_lower", "unwinds"]
    assert pool[0]["blocked"]["check"] == "planner"
    assert pool[1]["path_turn_deg"] - pool[1]["requested_turn_deg"] == pytest.approx(0.0, abs=0.05)
    assert _candidate(record, "same_goal")["outcome"] == "unwinds"


def test_one_scene_is_kept_per_arm_so_switching_arms_rebuilds_neither(tmp_path):
    from ora.environment.planners.curobo_v2.worker import SCENES_KEPT

    worker, _ = _worker()
    left, right = _world_input(tmp_path, "left.npz"), _world_input(tmp_path, "right.npz", footprints=())
    answers = [validate_plan_answer(worker.plan(_request(world)), _request(world)) for world in (left, right, left, right)]
    assert SCENES_KEPT == 2
    fresh, _ = _worker()
    expected = [validate_plan_answer(fresh.plan(_request(world)), _request(world)) for world in (left, right)]
    assert [answer["world"]["sha256"] for answer in answers] == [item["world"]["sha256"] for item in expected * 2]
    assert answers[2]["world"]["summary"] == expected[0]["world"]["summary"], "the kept scene answers as built"
    third = _world_input(tmp_path, "third.npz")
    worker.plan(_request(third))
    assert list(worker._kept)[0][0] == right["path"] and len(worker._kept) == SCENES_KEPT, "the oldest is let go"


def test_a_scene_is_built_once_however_the_arms_alternate(tmp_path, monkeypatch):
    import ora.environment.planners.curobo_v2.worker as module

    built = []
    original = module.read_world_inputs
    monkeypatch.setattr(module, "read_world_inputs", lambda path, sha: built.append(Path(path).name) or original(path, sha))
    worker, _ = _worker()
    left, right = _world_input(tmp_path, "left.npz"), _world_input(tmp_path, "right.npz")
    for world in (left, right, left, right, left):
        worker.plan(_request(world))
    assert built == ["left.npz", "right.npz"]


def test_prepare_builds_the_scenes_the_requests_then_find(tmp_path, monkeypatch):
    import ora.environment.planners.curobo_v2.worker as module
    from ora.environment.controllers.motion_planner import PREPARE_ANSWER_SCHEMA, PREPARE_REQUEST_SCHEMA

    built = []
    original = module.read_world_inputs
    monkeypatch.setattr(module, "read_world_inputs", lambda path, sha: built.append(Path(path).name) or original(path, sha))
    worker, _ = _worker()
    left, right = _world_input(tmp_path, "left.npz"), _world_input(tmp_path, "right.npz")
    start = {"joint_names": NAMES, "joint_positions": JOINTS}
    answer = worker.prepare({"schema": PREPARE_REQUEST_SCHEMA, "robot": "franka_panda", "worlds": [
        {"start": start, "world": {"model": WORLD_MODEL, "input": left}},
        {"start": start, "world": {"model": WORLD_MODEL, "input": right}}]})
    assert answer["schema"] == PREPARE_ANSWER_SCHEMA and answer["status"] == "prepared" and answer["prepared"] == 2
    worker.plan(_request(left))
    worker.plan(_request(right))
    assert built == ["left.npz", "right.npz"], "no request rebuilt what was prepared"
    refused = worker.prepare({"schema": PREPARE_REQUEST_SCHEMA, "robot": "franka_panda", "worlds": []})
    assert refused["status"] == "error" and refused["prepared"] == 0 and refused["detail"]["error"]["type"] == "MotionPlannerError"
    lines = [json.dumps({"protocol": PROTOCOL, "command": "prepare", "request_id": "p1", "request": {
        "schema": PREPARE_REQUEST_SCHEMA, "robot": "franka_panda",
        "worlds": [{"start": start, "world": {"model": WORLD_MODEL, "input": left}}]}})]
    written: list[dict[str, Any]] = []
    serve(worker, lines, lambda text: written.append(json.loads(text)))
    assert written[0]["answer"]["status"] == "prepared"
