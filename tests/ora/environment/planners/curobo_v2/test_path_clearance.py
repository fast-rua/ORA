"""A planned path against what the goal carve took: the check, the finger world and the worker's three answers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ora.environment.controllers.depth_world import (
    PUBLIC_DEPTH_VOXEL_V1, PUBLIC_DEPTH_VOXEL_X5_V1, WORLD_MODEL, Scene, pose_matrix, read_world_inputs, signed_distance,
)
from ora.environment.controllers.motion_planner import (
    IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, refusal_message, validate_ik_answer, validate_plan_answer,
)
from ora.environment.planners.curobo_v2.path_clearance import (
    FINGERS_FROM_M, TOLERANCE_M, FingerScene, is_clear, path_clearance, what_stands,
)
from ora.environment.planners.curobo_v2.worker import Worker
from tests.support.libero.world_inputs import overhead, write

NAMES = [f"panda_joint{index}" for index in range(1, 8)]
JOINTS = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
#: The hand pointing straight down at (0.5, 0, 0.1): its fingertips reach the table's height of a low object.
GOAL = [0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
#: A wall beside the wrist at that goal, inside the hand box and well behind the fingers.
WALL = ((0.52, 0.54), (-0.05, 0.05), 0.15)
LINKS = ["link6", "attached_object"]
AWAY = [0.2, 0.0, 0.4, 0.03]
AT_WALL = [0.53, 0.0, 0.14, 0.02]
AROUND = [0.45, 0.0, 0.3, 0.03]
ROWS = {"start": AWAY, "through": AT_WALL, "around": AROUND}
EMPTY = [0.0, 0.0, 0.0, -100.0]
START_HAND = [0.2, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]


def _world_input(tmp_path: Path, footprints=(WALL,)) -> dict[str, str]:
    record = write(tmp_path / "world.npz", agentview=overhead((0.5, 0.0, 1.2), footprints, base_world=(0, 0, 0)),
                   joints=JOINTS, gap=0.08, directive=-1.0)
    return {"path": record["path"], "sha256": record["sha256"]}


def _scene(world_input: dict[str, str]) -> Scene:
    inputs = read_world_inputs(Path(world_input["path"]), world_input["sha256"])
    return Scene(inputs, robot_spheres_base=[AWAY], start_hand_base=pose_matrix(START_HAND))


def _request(world_input: dict[str, str], *, joints=None) -> dict[str, Any]:
    goal = {"joint_positions": joints} if joints else {"tool_frame": "panda_hand", "pose_base_xyzw": GOAL}
    return {"schema": PLAN_REQUEST_SCHEMA, "robot": "franka_panda",
            "start": {"joint_names": NAMES, "joint_positions": JOINTS}, "goal": goal,
            "world": {"model": WORLD_MODEL, "input": world_input}, "held_object": {"source": "world"}, "constraints": []}


class PathEngine:
    """A stand-in planner whose one sphere stands where a named joint row says: the rows are joints[0] = 0, 1, 2..."""

    def __init__(self, *paths: str) -> None:
        self.paths = list(paths)
        self.names_of = ["start", "through", "around"]
        self.calls: list[str] = []

    def planner_identity(self) -> dict[str, Any]:
        return {"robot_sha256": "e" * 64, "collision_spheres_sha256": "f" * 64}

    def load_world(self, grid: Any) -> None:
        self.calls.append("load_world")

    def attach(self, spheres: Any) -> None:
        pass

    def reset_seeds(self) -> None:
        pass

    def check(self, joints: Any) -> dict[str, Any]:
        return {"within_limits": True, "self_collision": 0.0, "world_collision": 0.0}

    def joint_limits(self) -> list[list[float]]:
        return [[-3.0] * 7, [3.0] * 7]

    def sphere_links(self) -> list[str]:
        return list(LINKS)

    def tool_pose(self, joints: Any) -> list[float]:
        return [*self.kinematics(joints)[1][0][:3].tolist(), 0.0, 0.0, 0.0, 1.0]

    def tool_poses(self, rows: Any) -> list[list[float]]:
        return [self.tool_pose(row) for row in rows]

    def solve_ik(self, seed: Any, pose: Any) -> dict[str, Any]:
        return {"solved": False, "joint_positions": None, "position_error_m": 0.2, "rotation_error": 0.1}

    def plan_pose(self, start: Any, pose: Any, *, max_attempts: int | None = None) -> dict[str, Any]:
        self.calls.append("plan")
        return self._outcome(start)

    def plan_joints(self, start: Any, goal: Any) -> dict[str, Any]:
        self.calls.append("plan")
        return self._outcome(start)

    def kinematics(self, joints: Any) -> tuple[Any, Any]:
        name = "start" if list(joints) == JOINTS else self.names_of[int(joints[0])]
        return pose_matrix(START_HAND), np.array([ROWS[name], EMPTY])

    def path_spheres(self, rows: Any) -> Any:
        return np.stack([self.kinematics(row)[1] for row in rows])

    def _outcome(self, joints: Any) -> dict[str, Any]:
        rows = self.paths.pop(0)
        if rows == "blocked":
            return {"trajectory": None, "ik_solved": True, "detail": {"stand_in": True}, "ik_attempts": []}
        if rows == "no_ik":
            return {"trajectory": None, "ik_solved": False, "detail": {"stand_in": True}, "ik_attempts": []}
        positions = [JOINTS, *[[float(self.names_of.index(name)), *JOINTS[1:]] for name in rows.split("+")]]
        return {"trajectory": {"joint_names": NAMES, "positions": positions, "velocities": None, "dt_s": 0.02},
                "ik_solved": True, "detail": {"stand_in": True}, "ik_attempts": []}

    def plans(self) -> int:
        return self.calls.count("plan")


class SolutionEngine(PathEngine):
    """``PathEngine`` whose IK solves to the named rows in turn, ``None`` finding no solution."""

    def __init__(self, *solutions: str | None) -> None:
        super().__init__()
        self.solutions = list(solutions)

    def solve_ik(self, seed: Any, pose: Any) -> dict[str, Any]:
        self.calls.append("solve_ik")
        name = self.solutions.pop(0)
        if name is None:
            return {"solved": False, "joint_positions": None, "position_error_m": 0.2, "rotation_error": 0.1}
        return {"solved": True, "joint_positions": [float(self.names_of.index(name)), *JOINTS[1:]],
                "position_error_m": 0.0, "rotation_error": 0.0}

    def ik_identity(self) -> dict[str, Any]:
        return {"stand_in": True}


def _solve(tmp_path: Path, *solutions: str | None) -> tuple[dict[str, Any], SolutionEngine]:
    engine = SolutionEngine(*solutions)
    worker = Worker(engine, robot="franka_panda", world_model=WORLD_MODEL)
    request = {"schema": IK_REQUEST_SCHEMA, "robot": "franka_panda",
               "start": {"joint_names": NAMES, "joint_positions": JOINTS},
               "queries": [{"tool_frame": "panda_hand", "pose_base_xyzw": GOAL, "from_index": None}],
               "world": {"model": WORLD_MODEL, "input": _world_input(tmp_path)}, "held_object": {"source": "world"}}
    return validate_ik_answer(worker.solve_ik(request), request), engine


def _plan(tmp_path: Path, *paths: str, **keywords: Any) -> tuple[dict[str, Any], PathEngine]:
    engine = PathEngine(*paths)
    worker = Worker(engine, robot="franka_panda", world_model=WORLD_MODEL)
    answer = worker.plan(_request(_world_input(tmp_path), **keywords))
    validate_plan_answer(answer)
    return answer, engine


def test_clearance_is_the_nearest_sphere_of_the_whole_path_and_negative_inside():
    parameters = PUBLIC_DEPTH_VOXEL_V1
    occupied = np.zeros(parameters.grid_shape, dtype=bool)
    index = tuple(np.floor((np.array([0.5, 0.0, 0.1]) - parameters.grid_low_m) / parameters.voxel_size_m).astype(int))
    occupied[index] = True
    esdf = signed_distance(occupied, parameters)
    rows = [np.array([[0.2, 0.0, 0.1, 0.02], [0.3, 0.0, 0.1, -100.0]]),
            np.array([[0.5, 0.0, 0.1, 0.02], [0.3, 0.0, 0.1, -100.0]]),
            np.array([[0.4, 0.0, 0.1, 0.02], [0.3, 0.0, 0.1, -100.0]])]
    found = path_clearance(rows, ["wrist", "empty"], esdf, parameters)
    assert found["row"] == 1 and found["link"] == "wrist" and found["clearance_m"] < -0.02
    assert found["point_base_m"] == [0.5, 0.0, 0.1] and found["radius_m"] == 0.02
    assert not is_clear(found)
    clear = path_clearance(rows[:1], ["wrist", "empty"], esdf, parameters)
    assert clear["row"] == 0 and clear["clearance_m"] > 0 and is_clear(clear)
    off = path_clearance([np.array([[9.0, 9.0, 9.0, 0.1]])], ["wrist"], esdf, parameters)
    assert off["row"] is None and off["clearance_m"] == parameters.esdf_clamp_m and is_clear(off)


def test_the_tolerance_is_one_voxel_of_the_world():
    assert TOLERANCE_M == PUBLIC_DEPTH_VOXEL_V1.voxel_size_m == PUBLIC_DEPTH_VOXEL_X5_V1.voxel_size_m
    assert is_clear({"clearance_m": -TOLERANCE_M}) and not is_clear({"clearance_m": -TOLERANCE_M - 1e-6})


def test_the_finger_world_keeps_the_wrist_s_wall_and_clears_the_fingers_load_and_nothing_else(tmp_path):
    scene = _scene(_world_input(tmp_path))
    goal = pose_matrix(GOAL)
    wide, narrow = scene.world(goal), FingerScene(scene, np.zeros(0, dtype=bool)).world(goal)
    wall = (0.53, 0.0, 0.145)
    index = tuple(np.floor((np.array(wall) - scene.parameters.grid_low_m) / 0.01).astype(int))
    assert scene.occupied[index] and not wide.occupied[index] and narrow.occupied[index]
    assert narrow.sha256 != wide.sha256 and narrow.summary["carved_voxels"]["goal"] < wide.summary["carved_voxels"]["goal"]
    # The scene is shared, not changed.
    assert scene.world(goal).sha256 == wide.sha256


def test_the_finger_world_clears_the_box_where_the_fingers_are(tmp_path):
    # An object between the jaws at the goal, its top below where the fingers begin.
    between = ((0.48, 0.52), (-0.02, 0.02), 0.03)
    scene = _scene(_world_input(tmp_path, footprints=(between,)))
    narrow = FingerScene(scene, np.zeros(0, dtype=bool)).world(pose_matrix(GOAL))
    index = tuple(np.floor((np.array([0.5, 0.0, 0.025]) - scene.parameters.grid_low_m) / 0.01).astype(int))
    assert scene.occupied[index] and not narrow.occupied[index]


def test_a_joint_goal_carves_only_the_fingers_spheres(tmp_path):
    scene = _scene(_world_input(tmp_path))
    fingers = np.array([True, False])
    spheres = np.array([[0.9, 0.5, 0.0, 0.01], [0.53, 0.0, 0.145, 0.03]])
    goal = pose_matrix([0.1, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0])
    index = tuple(np.floor((np.array([0.53, 0.0, 0.145]) - scene.parameters.grid_low_m) / 0.01).astype(int))
    assert FingerScene(scene, fingers).world(goal, spheres).occupied[index]
    assert not scene.world(goal, spheres).occupied[index]


def test_a_path_that_stays_clear_is_the_plan_the_planner_gave_with_no_second_world(tmp_path):
    answer, engine = _plan(tmp_path, "around")
    assert answer["status"] == "planned" and engine.plans() == 1 and engine.calls.count("load_world") == 1
    assert answer["detail"]["path_clearance"]["status"] == "clear"
    assert answer["world"]["summary"]["path_clearance"]["status"] == "clear"
    assert answer["trajectory"]["positions"][-1][0] == 2.0
    assert "path_clearance" in answer["timing_s"]


def test_a_path_through_what_the_carve_took_is_planned_again_without_it(tmp_path):
    answer, engine = _plan(tmp_path, "through", "around")
    assert answer["status"] == "planned" and engine.plans() == 2
    assert answer["trajectory"]["positions"][-1][0] == 2.0
    record = answer["detail"]["path_clearance"]
    assert record["status"] == "replanned" and record["link"] == "link6" and record["clearance_m"] < -TOLERANCE_M
    # The world in the answer is the one the plan ran in: the first plan's carve took the wall.
    (tmp_path / "wide").mkdir()
    wide, _ = _plan(tmp_path / "wide", "around")
    assert answer["world"]["summary"]["carved_voxels"]["goal"] < wide["world"]["summary"]["carved_voxels"]["goal"]
    assert answer["world"]["sha256"] != wide["world"]["sha256"]


def test_no_clear_plan_refuses_and_says_where_and_what(tmp_path):
    answer, engine = _plan(tmp_path, "through+around", "blocked")
    assert answer["status"] == "refused" and answer["reason"] == "path_blocked" and answer["trajectory"] is None
    diagnosis = answer["detail"]["goal_diagnosis"]
    assert diagnosis["cause"] == "path_in_collision" and diagnosis["link"] == "link6" and diagnosis["penetration_m"] > 0
    message = refusal_message(answer)
    assert "inside what the depth shows at (0.53, 0.00, 0.14) in world coordinates" in message and "before the goal" in message
    assert diagnosis["before_goal_m"] > 0.01
    assert answer["detail"]["path_clearance"]["status"] == "blocked"


def test_a_replan_that_still_enters_the_world_is_no_plan(tmp_path):
    answer, _ = _plan(tmp_path, "through", "through")
    assert answer["status"] == "refused" and answer["reason"] == "path_blocked"
    assert refusal_message(answer).startswith("at the goal, its arm would be")


def test_a_goal_only_the_carve_allowed_keeps_the_goal_diagnosis(tmp_path):
    answer, _ = _plan(tmp_path, "through", "no_ik")
    assert answer["status"] == "refused" and answer["reason"] == "goal_unreachable"
    assert answer["detail"]["goal_diagnosis"]["cause"] != "path_in_collision"
    assert answer["detail"]["path_clearance"]["status"] == "blocked"


def test_a_joint_goal_that_stands_in_the_world_is_refused_there(tmp_path):
    engine = PathEngine("through")
    worker = Worker(engine, robot="franka_panda", world_model=WORLD_MODEL)
    seen: list[bool] = []

    def reason(joints: Any, *, goal: bool) -> tuple[str | None, dict[str, Any]]:
        seen.append(goal)
        return ("goal_in_collision" if len(seen) == 3 else None), {}

    worker._state_reason = reason
    answer = worker.plan(_request(_world_input(tmp_path), joints=[1.0, *JOINTS[1:]]))
    validate_plan_answer(answer)
    assert answer["status"] == "refused" and answer["reason"] == "goal_in_collision"
    assert answer["detail"]["goal_diagnosis"]["cause"] == "path_in_collision"


def test_an_ik_solution_that_stays_clear_is_the_answer_with_no_second_world(tmp_path):
    answer, engine = _solve(tmp_path, "around")
    assert answer["results"][0]["solved"] is True and answer["results"][0]["joint_positions"][0] == 2.0
    assert engine.calls.count("load_world") == 1 and engine.calls.count("solve_ik") == 1
    assert answer["world"]["queries"][0]["path_clearance"]["status"] == "clear"


def test_an_ik_solution_in_what_the_carve_took_is_solved_again_in_the_finger_world(tmp_path):
    # The goal carve took the wall beside the wrist: a plan ending on that solution would be planned again.
    answer, engine = _solve(tmp_path, "through", "around")
    assert answer["results"][0]["solved"] is True and answer["results"][0]["joint_positions"][0] == 2.0
    record = answer["world"]["queries"][0]["path_clearance"]
    assert record["status"] == "replanned" and record["link"] == "link6" and record["clearance_m"] < -TOLERANCE_M
    assert engine.calls.count("load_world") == 2 and engine.calls.count("solve_ik") == 2
    # The query's world is the finger world it was solved in, which keeps the wall.
    (tmp_path / "wide").mkdir()
    wide, _ = _solve(tmp_path / "wide", "around")
    assert answer["world"]["queries"][0]["sha256"] != wide["world"]["queries"][0]["sha256"]
    assert (answer["world"]["queries"][0]["carved_voxels"]["goal"]
            < wide["world"]["queries"][0]["carved_voxels"]["goal"])


def test_an_ik_solution_only_the_carve_allowed_is_no_solution(tmp_path):
    # A plan to that pose is refused (``test_a_goal_only_the_carve_allowed_keeps_the_goal_diagnosis``); so is the IK.
    for second in (None, "through"):
        answer, _ = _solve(tmp_path / str(second), "through", second)
        result = answer["results"][0]
        assert result["solved"] is False and result["refusal"] is None and result["joint_positions"] is None
        assert answer["world"]["queries"][0]["path_clearance"]["status"] == "blocked"


def test_what_stands_there_names_the_other_arm_then_the_support_then_the_depth(tmp_path):
    scene = _scene(_world_input(tmp_path))
    found = {"point_base_m": [0.5, 0.0, 0.1], "radius_m": 0.02}
    assert what_stands(found, scene, None) == "what the depth shows"
    assert what_stands(found, scene, {"z_m": 0.09}) == "the supporting surface"
    index = tuple(np.floor((np.array(found["point_base_m"]) - scene.parameters.grid_low_m) / 0.01).astype(int))
    scene.other_arm[index] = True
    assert what_stands(found, scene, {"z_m": 0.09}) == "the other arm"


def test_a_robot_only_plan_has_nothing_to_check():
    engine = PathEngine("around")
    worker = Worker(engine, robot="franka_panda", world_model="none")
    answer = worker.plan({"schema": PLAN_REQUEST_SCHEMA, "robot": "franka_panda",
                          "start": {"joint_names": NAMES, "joint_positions": JOINTS},
                          "goal": {"tool_frame": "panda_hand", "pose_base_xyzw": GOAL}, "world": {"model": "none"},
                          "held_object": {"source": "none"}, "constraints": []})
    assert answer["status"] == "planned" and "path_clearance" not in answer["detail"]


def test_where_the_fingers_begin_is_ahead_of_every_robot_s_wrist():
    assert 0.0 < FINGERS_FROM_M < PUBLIC_DEPTH_VOXEL_X5_V1.fingertip_m and FINGERS_FROM_M < PUBLIC_DEPTH_VOXEL_V1.fingertip_m


def test_the_recorded_carry_into_a_box_wall_is_clear_only_of_the_world_that_hid_the_wall():
    # A RoboDojo carry that pushed a box 44 mm with its wrist, no stop, a reached receipt. The spheres are its recorded
    # path and the grids the three worlds of its recorded depth.
    data = np.load(Path(__file__).parents[4] / "data/planners/carry-into-box-wall.npz")
    parameters = PUBLIC_DEPTH_VOXEL_X5_V1
    size = int(np.prod(parameters.grid_shape))

    def found(name: str) -> dict[str, Any]:
        occupied = np.unpackbits(data[name], count=size).reshape(parameters.grid_shape).astype(bool)
        return path_clearance(data["spheres"], [str(link) for link in data["links"]], signed_distance(occupied, parameters), parameters)

    assert is_clear(found("goal_carve"))
    for world in ("no_goal_carve", "finger_carve"):
        hit = found(world)
        assert not is_clear(hit) and hit["link"] == "link6" and hit["row"] == 9 and hit["clearance_m"] < -0.015


def test_the_recorded_desk_reach_holds_its_hand_only_in_the_world_that_hid_the_stapler():
    # A play-scene audit's align_plan pose: the right arm's link6 0.10 m behind a desk stapler's centre, approaching
    # from the back. The IK in the goal's carved world solved it, the only reachable answer of that audit, and a plan
    # to the pose was refused: at the IK's solution the hand stands 4.9 cm inside the stapler's depth, which only the
    # goal carve took, and the finger world holds no solution. The spheres are that solution, the grids its three worlds.
    data = np.load(Path(__file__).parents[4] / "data/planners/desk-stapler-reach.npz")
    parameters = PUBLIC_DEPTH_VOXEL_X5_V1
    size = int(np.prod(parameters.grid_shape))

    def found(name: str) -> dict[str, Any]:
        occupied = np.unpackbits(data[name], count=size).reshape(parameters.grid_shape).astype(bool)
        return path_clearance(data["spheres"], [str(link) for link in data["links"]], signed_distance(occupied, parameters), parameters)

    assert is_clear(found("goal_carve"))
    for world in ("no_goal_carve", "finger_carve"):
        hit = found(world)
        assert not is_clear(hit) and hit["link"] == "link6" and hit["clearance_m"] < -0.04
