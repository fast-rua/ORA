"""The cuRobo V2 planner worker: planner-neutral requests in, answers out, one JSON object per line.

``MotionPlannerClient`` (``controllers/motion_planner.py``) starts it as
``python -m ora.environment.planners.curobo_v2.worker`` under the V2
interpreter, with the V2 source root first on ``PYTHONPATH``. It answers
``ready``, ``plan``, ``ik``, ``check``, ``fk``, ``prepare`` and ``close``.

The goal-independent scene of a world input (its depth world at the measured start) is kept for the last
``SCENES_KEPT`` inputs, one per arm of a two-arm robot, so a request for one arm does not rebuild the other's;
``prepare`` builds the scenes of the inputs it names before any request asks, and answers nothing else. A kept
scene is the scene the same input would build again, so no answer changes, only when its scene is built.

``Worker`` holds everything but the solver calls: request validation, the
world (``depth_world.py``, built from the request's world input with the
planner's own robot spheres, then exported to V2's layout), the held object,
the seed reset before every goal, the start and goal checks, and the neutral
reasons. The solver calls sit behind ``engine.CuroboV2Engine``, so the CPU
tests drive ``Worker`` with a stand-in engine.

Per goal, in order: a goal the world refuses is ``goal_below_support`` (no
solver runs); a start at most ``START_LIMIT_SLACK_RAD`` past a joint limit is
the arm resting on its stop, and plans from ``START_LIMIT_INSET_RAD`` inside it
(``detail.start_inside_limits`` names each joint moved); a start further outside
the joint limits is ``start_outside_limits``; a
start touching the world or the robot itself is ``start_in_collision``; a
joint goal outside the limits is ``goal_unreachable`` and one in collision
``goal_in_collision``; a pose goal no IK attempt solves is
``goal_unreachable``; anything else the planner does not plan is
``path_blocked``. IK queries and plans use the same world for the same goal.

A ``goal_unreachable`` says why in ``detail.goal_diagnosis``: one more IK solve
with nothing in the way tells a pose out of reach from a pose something occupies,
that solution says what stands there (a part of the robot or the held object), and
the seeds that reached the pose say whether the arm itself stopped them. It also
says what would work: the smallest tilt of the approach that plans in the same
world (``goal_nearby.py``), and on a two-arm robot whether the other arm reaches the pose.

``SCENES`` names the world a scene builds; the client selects one through
``SCENE_ENV`` and ``ready`` names it. A kitchen has no one supporting surface:
the counter above a drawer is no floor for a hand in front of that drawer, so a
goal below a surface is judged by the voxels around it, not refused outright.

Every answer's ``planner`` names the robot configuration and the collision
sphere set it was planned with (``robot_sha256``, ``collision_spheres_sha256``);
a plan answer's ``detail.ik_attempts`` keeps each IK attempt's seed population.

A plan is then read against the world without its goal carve (``path_clearance.py``): a path that
lies in what the carve took is planned again in a world that clears only what the fingers and the
load may touch, and refused when no plan stays clear, with ``detail.goal_diagnosis`` saying where it
hits and what stands there. ``detail.path_clearance`` and the world summary keep what the check found.
An IK solution is read the same way, as the one row a plan to its pose ends on: one standing in what
the carve took is solved again in the finger world and answered unsolved when none there is clear, so
a reach check holds what a plan to the pose would; its query's world summary keeps the finding.

A pose plan the pacing would refuse for its wrist (``motion_pace.check_unwind``: the hand turns more than
``MAX_UNREQUESTED_TURN_DEG`` beyond its start-to-goal turn), or whose goal lies a whole turn away on a joint that
has room for a nearer equivalent, is answered as a straight joint line to the same pose instead (to its goal, that
goal's equivalent or, for a refused plan, another IK solution the attempts kept), when one turns the hand within the
rule, less than the plan, and passes the planner's own checks; ``detail.wrist_detour`` says what was found
(``_unwound``, ``wrist_detour.py``).

A plan v2 request with ``timing`` is planned the same way. V2's interpreter
holds no timing library, so the worker answers the planner's own trajectory and
keeps it in ``detail.timing_path`` on the joint grid, with the controlled point's
pose at every row from V2's kinematics and the request's offset from the tool
frame; ``MotionPlannerClient`` has that path timed by the timing worker before
the backend sees the answer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import OrderedDict
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.pose import matrix_quaternion, quaternion_matrix
from ora.environment.controllers.depth_world import (
    Scene, curobo_v2_grid, other_arm_spheres_base, pose_matrix, read_world_inputs, tool_from_hand, world_parameters,
)
from ora.environment.controllers.motion_pace import MAX_UNREQUESTED_TURN_DEG, joint_path_grid, path_turns
from ora.environment.controllers.motion_planner import (
    CHECK_ANSWER_SCHEMA, IK_ANSWER_SCHEMA, PLAN_ANSWER_SCHEMA, PLAN_ANSWER_SCHEMA_V2, PLAN_REQUEST_SCHEMA_V2,
    PREPARE_ANSWER_SCHEMA, PROTOCOL, ROBOTS, WORLD_MODELS, error_answer, validate_check_request, validate_ik_request,
    validate_plan_request, validate_prepare_request,
)

from .goal_nearby import BLOCKING_SEEDS, NEARBY_PLANS, nearby_sentence, tilted_goals, what_blocks
from . import wrist_detour
from .path_clearance import HELD_LINK, FingerScene, is_clear, path_clearance, what_stands
from .robots import FINGERS_PLANNED_OPEN, LINK_PARTS

IMPLEMENTATION = "curobo_v2"
GRID_ARM = "arm"
SCENE_ENV = "ORA_PLANNER_SCENE"
#: A measured start joint at most this far past its limit is resting on the stop. LIBERO's Panda parks joints 2, 5, 6
#: and 7 there (0 to 0.2 mrad past), which V2 reads as outside the limits, so every later move was refused.
START_LIMIT_SLACK_RAD = 0.005
#: Such a joint plans from this far inside the limit; with the slack it moves the path's first row at most 6 mrad.
START_LIMIT_INSET_RAD = 0.001
#: Goal-independent scenes kept: one per arm of a two-arm robot (RoboDojo, RoboTwin), the most recent first evicted
#: last. One kept scene made every arm switch rebuild the depth world (most of a 2.2 s reach check, 2026-11-06).
SCENES_KEPT = 2


def compose_pose(tool: Any, offset: Any) -> list[float]:
    """The pose ``offset`` (x, y, z, qx, qy, qz, qw) names in the frame at pose ``tool``."""
    rotation = np.asarray(quaternion_matrix(tool[3:]), dtype=np.float64)
    position = np.asarray(tool[:3], dtype=np.float64) + rotation @ np.asarray(offset[:3], dtype=np.float64)
    turned = rotation @ np.asarray(quaternion_matrix(offset[3:]), dtype=np.float64)
    return [*position.tolist(), *matrix_quaternion(turned.tolist())]


def goal_refusal_in_world(refusal: Mapping[str, Any] | None, base_pose_world_xyzw: Any,
                          at_base: Any) -> dict[str, Any] | None:
    """``refusal`` (``depth_load``'s, in the arm's base frame) as the policy reads it: in the world frame of its poses.

    ``in_world`` holds the refusal's heights in that frame (``frame: world``) and the message names the support at its
    world height. ``at_base`` is a base-frame point of the goal (its hand), whose x and y place the heights; on a
    level base they do not matter. The base-frame fields stay as the world summary keeps every other height."""
    if refusal is None:
        return None
    world_from_base = pose_matrix(base_pose_world_xyzw)
    point = np.asarray(at_base, dtype=np.float64)

    def world_z(z: float) -> float:
        return round(float(world_from_base[2, :3] @ np.array([point[0], point[1], float(z)]) + world_from_base[2, 3]), 4)

    heights = {key: world_z(refusal[key]) for key in ("support_z_m", "fingertip_z_m", "load_z_m") if key in refusal}
    message = str(refusal.get("message") or "")
    if "support_z_m" in heights:
        base_words = f"at z {refusal['support_z_m']:.3f} m (base frame)"
        world_words = f"at z {heights['support_z_m']:.3f} m (world frame)"
        message = message.replace(base_words, world_words) if base_words in message else f"{message} ({world_words})"
    return {**refusal, "message": message, "in_world": {"frame": "world", **heights}}


class KitchenScene(Scene):
    """The depth world of a kitchen: the support slab stays under the goal carve, and no goal is refused for lying below it."""

    def _goal_refusal(self, goal: Any, support: Any) -> None:
        return None


SCENES = {"tabletop": Scene, "kitchen": KitchenScene}


class Worker:
    """Answers plan and IK requests for one robot in one world model over an engine."""

    def __init__(self, engine: Any, *, robot: str, world_model: str, scene: str = "tabletop") -> None:
        if robot not in ROBOTS or world_model not in WORLD_MODELS or scene not in SCENES:
            raise ValueError(f"unsupported robot, world model or scene: {robot}, {world_model}, {scene}")
        self.engine = engine
        self.robot = robot
        self.world_model = world_model
        self.scene = scene
        self.parameters = world_parameters(world_model, robot)
        self.tool_from_hand = tool_from_hand(robot)
        self._scene: tuple[tuple[Any, ...], Any] | None = None
        #: The last ``SCENES_KEPT`` scenes by key, each with the world inputs it was built from.
        self._kept: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._goal_world: Any = None
        self._inputs: Any = None
        self._sphere_links: list[str] | None = None

    def ready(self) -> dict[str, Any]:
        return {"implementation": IMPLEMENTATION, "robot": self.robot, "world_model": self.world_model,
                "engine": self.engine.identity(), **({"scene": self.scene} if self.scene != "tabletop" else {})}

    # -- world -----------------------------------------------------------------

    def _scene_for(self, request: Mapping[str, Any]) -> Any:
        """The request's goal-independent world, or ``None`` robot-only (reused for the same snapshot)."""
        world = request["world"]
        if world["model"] != self.world_model:
            raise ValueError(f"a {self.world_model} planner received a {world['model']} request")
        if world["model"] == "none":
            return None
        start = request["start"]["joint_positions"]
        key = (world["input"]["path"], world["input"]["sha256"], tuple(start))
        if self._scene is not None and self._scene[0] == key:
            return self._scene[1]
        kept = self._kept.get(key)
        if kept is not None:  # another arm's request came between: its scene, and the inputs that built it
            self._kept.move_to_end(key)
            self._scene, self._inputs = (key, kept[0]), kept[1]
            return kept[0]
        self._scene = None
        inputs = self._inputs = read_world_inputs(Path(world["input"]["path"]), world["input"]["sha256"])
        if [float(item) for item in inputs.joint_positions] != [float(item) for item in start]:
            raise ValueError("world inputs were recorded at other joints than the request's start")
        # The robot filter must see the robot without a previous request's load.
        self.engine.attach(None)
        hand, spheres = self.engine.kinematics(start)
        other = None
        if inputs.other_arm_joint_positions is not None:
            other = other_arm_spheres_base(inputs, self.engine.kinematics(inputs.other_arm_joint_positions.tolist())[1])
        scene = SCENES[self.scene](inputs, robot_spheres_base=spheres, start_hand_base=hand @ self.tool_from_hand,
                                   parameters=self.parameters, other_arm_spheres=other)
        self._scene = (key, scene)
        self._kept[key] = (scene, inputs)
        while len(self._kept) > SCENES_KEPT:
            self._kept.popitem(last=False)
        return scene

    def prepare(self, raw: Any) -> dict[str, Any]:
        """Build the scene of each world the request names now, so the requests that follow find it kept."""
        started = time.time()
        try:
            request = validate_prepare_request(raw)
            if request["robot"] != self.robot:
                raise ValueError(f"a {self.robot} planner received a {request['robot']} prepare request")
            for item in request["worlds"]:
                self._scene_for(item)
            return {"schema": PREPARE_ANSWER_SCHEMA, "status": "prepared", "reason": None,
                    "prepared": len(request["worlds"]), "timing_s": {"total": time.time() - started}}
        except Exception as exc:  # an answer, never a broken worker: the requests that follow build as ever
            return {"schema": PREPARE_ANSWER_SCHEMA, "status": "error", "reason": "planner_error", "prepared": 0,
                    "timing_s": {"total": time.time() - started},
                    "detail": {"error": {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__}}}

    def _apply(self, scene: Any, request: Mapping[str, Any], *, pose: list[float] | None = None,
               joints: list[float] | None = None) -> dict[str, Any] | None:
        """Load one goal's world and held object and reset every seed; the world's summary or ``None``."""
        self.engine.attach(None)
        held_source = request["held_object"]["source"]
        held = request["held_object"]["spheres_tool_frame"] if held_source == "spheres" else None
        summary = None
        if scene is not None:
            hand, spheres = (pose_matrix(pose), None) if pose is not None else self.engine.kinematics(joints)
            world = self._goal_world = scene.world(hand @ self.tool_from_hand, spheres)
            grid = curobo_v2_grid(world.esdf, self.parameters)
            self.engine.load_world(grid)
            if held_source == "world" and len(world.held_spheres_hand):
                rows = world.held_spheres_hand
                turn = self.tool_from_hand[:3, :3]
                held = np.hstack([rows[:, :3] @ turn.T, rows[:, 3:]]).tolist()
            refusal = goal_refusal_in_world(world.summary["goal_refusal"], self._inputs.base_pose_world_xyzw,
                                            (hand @ self.tool_from_hand)[:3, 3])
            summary = {"model": self.world_model, "sha256": world.sha256, "inputs_sha256": scene.inputs_sha256,
                       "summary": {**world.summary, "goal_refusal": refusal, "planner_grid_sha256": grid["sha256"]}}
        if held:
            self.engine.attach(held)
        # Every goal starts from the same seeds, whatever was asked before.
        self.engine.reset_seeds()
        return summary

    def _planner(self) -> dict[str, Any]:
        return {"implementation": IMPLEMENTATION, **self.engine.planner_identity()}

    # -- checks ----------------------------------------------------------------

    def _start_inside_limits(self, names: list[str], start: list[float]) -> tuple[list[float], list[dict[str, Any]]]:
        """``start`` with each joint resting on its stop moved just inside it, and what moved."""
        low, high = self.engine.joint_limits()
        inside, moved = [float(value) for value in start], []
        for index, (value, lower, upper) in enumerate(zip(inside, low, high)):
            if upper - START_LIMIT_INSET_RAD < value <= upper + START_LIMIT_SLACK_RAD:
                inside[index] = upper - START_LIMIT_INSET_RAD
            elif lower - START_LIMIT_SLACK_RAD <= value < lower + START_LIMIT_INSET_RAD:
                inside[index] = lower + START_LIMIT_INSET_RAD
            else:
                continue
            moved.append({"joint": names[index], "measured": value, "planned_from": inside[index]})
        return inside, moved

    def _state_reason(self, joints: list[float], *, goal: bool) -> tuple[str | None, dict[str, Any]]:
        check = self.engine.check(joints)
        if not check["within_limits"]:
            return ("goal_unreachable" if goal else "start_outside_limits"), check
        if check["self_collision"] > 0 or (check["world_collision"] or 0) > 0:
            return ("goal_in_collision" if goal else "start_in_collision"), check
        return None, check

    # -- commands --------------------------------------------------------------

    def fk(self, raw: Any) -> dict[str, Any]:
        """The tool frame's base-frame pose of each joint row, for the FK parity check (``motion_planner``)."""
        rows = raw.get("joint_positions") if isinstance(raw, Mapping) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("fk takes joint_positions rows")
        return {"robot": self.robot, "tool_poses_base_xyzw": [self.engine.tool_pose(row) for row in rows]}

    def plan(self, raw: Any) -> dict[str, Any]:
        started = time.time()
        request = None
        try:
            request = validate_plan_request(raw)
            if request["robot"] != self.robot:
                raise ValueError(f"this worker plans {self.robot}")
            return self._plan(request, started)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            schema = (PLAN_ANSWER_SCHEMA_V2 if request is not None and request["schema"] == PLAN_REQUEST_SCHEMA_V2
                      else PLAN_ANSWER_SCHEMA)
            return error_answer(schema, request, "planner_error", exc,
                                implementation=IMPLEMENTATION, started=started)

    def _timing_path(self, trajectory: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
        """The planned path on the joint grid, with the controlled point's pose at every row."""
        offset = timing["controlled_point"]["offset_xyzw"]
        grid = joint_path_grid(
            trajectory["positions"], lambda rows: [compose_pose(self.engine.tool_pose(row), offset) for row in rows],
            arm=GRID_ARM, wrist_joint_index=int(timing["wrist"]["joint_index"]))
        return {"joint_positions": grid.grid.joints[GRID_ARM].tolist(),
                "tool_poses_xyzw": [list(pose) for pose in grid.poses]}

    def _plan(self, request: Mapping[str, Any], started: float) -> dict[str, Any]:
        timing: dict[str, float] = {}
        start = request["start"]["joint_positions"]
        goal = request["goal"]
        mark = time.time()
        scene = self._scene_for(request)
        world = self._apply(scene, request, pose=goal.get("pose_base_xyzw"), joints=goal.get("joint_positions"))
        timing["world"] = time.time() - mark
        detail: dict[str, Any] = {}
        start, moved = self._start_inside_limits(request["start"]["joint_names"], start)
        if moved:
            detail["start_inside_limits"] = moved
        trajectory, reason = None, None
        if world is not None and world["summary"]["goal_refusal"] is not None:
            reason = "goal_below_support"
        else:
            reason, detail["start_check"] = self._state_reason(start, goal=False)
            if reason is None and "joint_positions" in goal:
                reason, detail["goal_check"] = self._state_reason(goal["joint_positions"], goal=True)
            if reason is None:
                mark = time.time()
                outcome = (self.engine.plan_pose(start, goal["pose_base_xyzw"]) if "pose_base_xyzw" in goal
                           else self.engine.plan_joints(start, goal["joint_positions"]))
                timing["solve"] = time.time() - mark
                trajectory = outcome["trajectory"]
                detail["solver"] = outcome["detail"]
                detail["ik_attempts"] = list(outcome.get("ik_attempts") or [])
                if trajectory is None:
                    reason = ("goal_unreachable" if "pose_base_xyzw" in goal and not outcome["ik_solved"]
                              else "path_blocked")
                if reason == "goal_unreachable" and scene is not None:
                    mark = time.time()
                    detail["goal_diagnosis"] = self._goal_diagnosis(start, goal["pose_base_xyzw"], detail["ik_attempts"])
                    timing["goal_diagnosis"] = time.time() - mark
                if trajectory is not None and scene is not None:
                    mark = time.time()
                    trajectory, reason, world = self._kept_clear(scene, request, start, trajectory, world, detail)
                    timing["path_clearance"] = time.time() - mark
                if trajectory is not None and "pose_base_xyzw" in goal:
                    mark = time.time()
                    trajectory, world = self._unwound(scene, start, trajectory, world, detail)
                    timing["wrist_detour"] = time.time() - mark
        if trajectory is not None and trajectory["joint_names"] != request["start"]["joint_names"]:
            raise ValueError("the planned trajectory's joint order differs from the request's")
        version_two = request["schema"] == PLAN_REQUEST_SCHEMA_V2
        if version_two and trajectory is not None and request["timing"] is not None:
            mark = time.time()
            detail["timing_path"] = self._timing_path(trajectory, request["timing"])
            timing["timing_path"] = time.time() - mark
        timing["total"] = time.time() - started
        return {
            "schema": PLAN_ANSWER_SCHEMA_V2 if version_two else PLAN_ANSWER_SCHEMA,
            "status": "planned" if trajectory is not None else "refused",
            "reason": reason, "trajectory": trajectory, "start_tool_pose_base_xyzw": self.engine.tool_pose(start),
            "world": world, "planner": self._planner(), "timing_s": timing,
            "request": request, "detail": detail, **({"timing": None} if version_two else {}),
        }

    def _turns(self, rows: Any) -> dict[str, float]:
        """``motion_pace.path_turns`` of a joint path on the pacing's joint grid, by the tool frame's kinematics.

        A rigid offset to the controlled point turns with the tool frame, so the turns are the pacing's. The grid's
        poses come from one batched kinematics call (``engine.tool_poses``), so every pose plan pays for it once."""
        return path_turns(joint_path_grid(rows, self.engine.tool_poses, arm=GRID_ARM).poses)

    def _unwound(self, scene: Any, start: list[float], trajectory: Mapping[str, Any], world: Any,
                 detail: dict[str, Any]) -> tuple[Mapping[str, Any], Any]:
        """``trajectory``, or a straight joint line to the same pose that turns the hand less (``wrist_detour.py``).

        The lines go to the plan's goal, its nearest 2 pi equivalent in range and, for a plan the pacing would
        refuse, the other IK solutions the planner's attempts kept; they are ranked with the plan by the pacing's
        rule and then by the smallest path turn, not taken in the order the planner found them.

        The line answered passes the planner's own checks (``_line_blocked``). ``detail.wrist_detour`` (and the
        world summary's) keeps ``status`` (``replaced``, ``kept``), ``chosen``, the plan's turns and each
        candidate's turns and outcome; a plan within the rule with no nearer goal costs one kinematics pass and
        leaves no record."""
        path = trajectory["positions"]
        goal = [float(value) for value in path[-1]]
        planned, limits = self._turns(path), self.engine.joint_limits()
        alias = wrist_detour.nearest_alias(start, goal, limits)
        refused = wrist_detour.refused(planned)
        if not refused and alias is None:
            return trajectory, world
        candidates: list[dict[str, Any]] = [] if refused else [{"source": "planned", **planned}]
        lines: dict[int, Any] = {}
        pool = wrist_detour.pool_goals(detail.get("ik_attempts"), start, limits, [goal]) if refused else []
        for source, joints in (("alias", alias), ("same_goal", goal if refused else None),
                               *(("ik_pool", item) for item in pool)):
            if joints is None:
                continue
            lines[len(candidates)] = line = wrist_detour.straight_line(start, joints)
            candidates.append({"source": source, **self._turns(line.tolist()),
                               **({"goal_joints": [round(value, 6) for value in joints]} if joints is not goal
                                  else {})})
        chosen = wrist_detour.choose(candidates, lines, lambda line: self._line_blocked(scene, line))
        record = {"status": "replaced" if chosen in lines else "kept",
                  "chosen": None if chosen is None else candidates[chosen]["source"],
                  "max_unrequested_turn_deg": MAX_UNREQUESTED_TURN_DEG, "planned": planned, "candidates": candidates}
        detail["wrist_detour"] = record
        if world is not None:
            world = {**world, "summary": {**world["summary"], "wrist_detour": record}}
        if chosen not in lines:
            return trajectory, world
        return wrist_detour.as_trajectory(lines[chosen], trajectory["joint_names"], trajectory["dt_s"]), world

    def _line_blocked(self, scene: Any, line: Any) -> dict[str, Any] | None:
        """Why a straight joint line fails the planner's own checks, or None when it passes them.

        Every row gets the start and goal check (limits, the robot itself, the goal's loaded world, zero activation
        distance), and the whole line the clearance check a planned path gets after the goal carve."""
        for row, joints in enumerate(line.tolist()):
            check = self.engine.check(joints)
            if not check["within_limits"] or check["self_collision"] > 0 or (check["world_collision"] or 0) > 0:
                return {"check": "planner", "row": row, **check}
        if scene is None:
            return None
        hand, spheres = self.engine.kinematics(line[-1].tolist())
        found = self._clearance(line.tolist(), FingerScene(scene, self._fingers()).world(
            hand @ self.tool_from_hand, spheres).esdf)
        return None if is_clear(found) else {"check": "world", **found}

    def _links(self) -> list[str]:
        if self._sphere_links is None:
            self._sphere_links = self.engine.sphere_links()
        return self._sphere_links

    def _fingers(self) -> Any:
        """Which rows of the robot's spheres are its fingers."""
        return np.array([LINK_PARTS[self.robot].get(link) == "fingers" for link in self._links()], dtype=bool)

    def _clearance(self, path: list[Any], esdf: Any) -> dict[str, Any]:
        """``path_clearance`` of ``path`` with the load attached now."""
        return path_clearance(self.engine.path_spheres(path), self._links(), esdf, self.parameters)

    def _kept_clear(self, scene: Any, request: Mapping[str, Any], start: list[float], trajectory: Mapping[str, Any],
                    world: Mapping[str, Any], detail: dict[str, Any]) -> tuple[Any, str | None, dict[str, Any]]:
        """``trajectory`` if it stays out of what the goal carve took, else a plan that does, else the refusal.

        The first look is at the scene with no goal carve, which costs no world; a hand that reached into
        the box is read again in the finger world. Answers the trajectory, the reason and the world it holds for."""
        goal = request["goal"]
        path = trajectory["positions"]
        found = self._clearance(path, scene.esdf)
        if is_clear(found):
            return trajectory, None, self._with_clearance(world, detail, found, "clear")
        world = self._apply(FingerScene(scene, self._fingers()), request,
                            pose=goal.get("pose_base_xyzw"), joints=goal.get("joint_positions"))
        found = self._clearance(path, self._goal_world.esdf)
        if is_clear(found):
            return trajectory, None, self._with_clearance(world, detail, found, "clear")
        reason = None
        if "joint_positions" in goal:
            reason, detail["goal_check"] = self._state_reason(goal["joint_positions"], goal=True)
        again = None
        if reason is None:
            outcome = (self.engine.plan_pose(start, goal["pose_base_xyzw"]) if "pose_base_xyzw" in goal
                       else self.engine.plan_joints(start, goal["joint_positions"]))
            detail["solver"] = outcome["detail"]
            detail["ik_attempts"] = list(outcome.get("ik_attempts") or [])
            again = outcome["trajectory"]
            if again is None:
                reason = "goal_unreachable" if "pose_base_xyzw" in goal and not outcome["ik_solved"] else "path_blocked"
        if again is not None and is_clear(self._clearance(again["positions"], self._goal_world.esdf)):
            return again, None, self._with_clearance(world, detail, found, "replanned")
        hit = self._path_cause(found, scene, path, world)
        detail["goal_diagnosis"] = (self._goal_diagnosis(start, goal["pose_base_xyzw"], detail["ik_attempts"])
                                    if reason == "goal_unreachable" else hit)
        return None, reason or "path_blocked", self._with_clearance(world, detail, found, "blocked")

    def _solution_kept_clear(self, scene: Any, request: Mapping[str, Any], seed: list[float], pose: list[float],
                             solved: Mapping[str, Any], world: Mapping[str, Any],
                             ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``_kept_clear`` for an IK solution, the one row a plan to ``pose`` ends on.

        A solution standing in what the goal carve took is solved again in the finger world, and unsolved when that
        finds none clear: a plan to the pose would be refused there. Answers the solve and the world it holds for."""
        found = self._clearance([solved["joint_positions"]], scene.esdf)
        if is_clear(found):
            return dict(solved), self._with_clearance(world, {}, found, "clear")
        world = self._apply(FingerScene(scene, self._fingers()), request, pose=pose)
        found = self._clearance([solved["joint_positions"]], self._goal_world.esdf)
        if is_clear(found):
            return dict(solved), self._with_clearance(world, {}, found, "clear")
        again = self.engine.solve_ik(seed, pose)
        if again["solved"] and is_clear(self._clearance([again["joint_positions"]], self._goal_world.esdf)):
            return dict(again), self._with_clearance(world, {}, found, "replanned")
        return {**again, "solved": False, "joint_positions": None}, self._with_clearance(world, {}, found, "blocked")

    def _with_clearance(self, world: Mapping[str, Any], detail: dict[str, Any], found: Mapping[str, Any],
                        status: str) -> dict[str, Any]:
        """Keep the check's finding in ``detail`` and in the world summary, the evidence every backend keeps."""
        record = {"status": status, **found}
        detail["path_clearance"] = record
        return {**world, "summary": {**world["summary"], "path_clearance": record}}

    def _path_cause(self, found: Mapping[str, Any], scene: Any, path: list[Any], world: Mapping[str, Any]) -> dict[str, Any]:
        """Where ``path`` enters the world and what stands there, in the words a refusal carries."""
        link = found["link"]
        part = "held object" if link == HELD_LINK else LINK_PARTS[self.robot].get(link, "arm")
        point = np.asarray(found["point_base_m"], dtype=np.float64)
        at = (pose_matrix(self._inputs.base_pose_world_xyzw) @ np.append(point, 1.0))[:3]
        tool = np.asarray([self.engine.tool_pose(row)[:3] for row in (path[found["row"]], path[-1])], dtype=np.float64)
        before = float(np.linalg.norm(tool[1] - tool[0]))
        stands = what_stands(found, scene, world["summary"]["support"]["goal"])
        depth = -found["clearance_m"]
        where = "at the goal" if before < 0.01 else f"on the way, {before * 100:.0f} cm before the goal"
        return {"cause": "path_in_collision", "link": link, "part": part, "penetration_m": depth, "before_goal_m": before,
                "message": (f"{where}, its {part} would be {depth * 100:.1f} cm inside {stands} "
                            f"at ({at[0]:.2f}, {at[1]:.2f}, {at[2]:.2f}) in world coordinates")}

    def _goal_diagnosis(self, start: list[float], pose: list[float], attempts: list[Any]) -> dict[str, Any]:
        """Why no IK attempt solved ``pose``, and what near it would: the goal's own world first, then an empty one."""
        blocks = self._what_blocks(attempts)
        nearby = self._nearby(start, pose)
        world = self._goal_world
        parameters = self.parameters
        self.engine.attach(None)
        self.engine.load_world(curobo_v2_grid(
            np.full(parameters.grid_shape, -parameters.esdf_clamp_m, dtype=np.float32), parameters))
        self.engine.reset_seeds()
        free = self.engine.solve_ik(start, pose)
        cause = self._goal_cause(free, pose, world, blocks)
        turn = pose_matrix(self._inputs.base_pose_world_xyzw)[:3, :3]
        approach = turn @ (pose_matrix(pose) @ self.tool_from_hand)[:3, 2]
        clauses = [cause["message"], nearby_sentence(nearby, approach, turn)]
        reaches = self._other_arm_reaches(pose, world)
        if reaches is not None:
            clauses.append(f"the other arm {'reaches' if reaches else 'does not reach'} this pose "
                           "with nothing in the way")
        return {**cause, "message": "; ".join(clauses), "nearby": nearby, "other_arm_reaches": reaches}

    def _nearby(self, start: list[float], pose: list[float]) -> dict[str, Any] | None:
        """The smallest tilt of ``pose``'s approach that plans in the goal's own world, or ``None`` for none tried."""
        self.engine.reset_seeds()
        plans = 0
        for goal in tilted_goals(pose, self.tool_from_hand, self.parameters.fingertip_m):
            if not self.engine.solve_ik(start, goal["pose_base_xyzw"])["solved"]:
                continue
            if self.engine.plan_pose(start, goal["pose_base_xyzw"], max_attempts=1)["trajectory"] is not None:
                return goal
            plans += 1
            if plans == NEARBY_PLANS:
                return None
        return None

    def _what_blocks(self, attempts: list[Any]) -> str | None:
        """What stops the seeds of the last IK attempt that reached the pose, checked in the goal's own world."""
        last = attempts[-1] if attempts else None
        if not isinstance(last, Mapping) or last.get("status") != "captured":
            return None
        reached = [joints for joints, converged in zip(last["joint_positions"], last["converged"]) if converged]
        return what_blocks([self.engine.check(joints) for joints in reached[:BLOCKING_SEEDS]])

    def _goal_cause(self, free: Mapping[str, Any], pose: list[float], world: Any, blocks: str | None) -> dict[str, Any]:
        """What stands at the empty world's solution of ``pose`` (``free``), or that nothing reaches it."""
        if not free["solved"]:
            return {"cause": "out_of_reach",
                    "message": "no joint configuration reaches this pose even with nothing in the way"}
        spheres = self.engine.kinematics(free["joint_positions"])[1]
        links = self.engine.sphere_links()
        arm = self._deepest(spheres, world.esdf, 0.0)
        hand = pose_matrix(pose) @ self.tool_from_hand
        rows = world.held_spheres_hand
        held = np.hstack([rows[:, :3] @ hand[:3, :3].T + hand[:3, 3], rows[:, 3:]])
        load = self._deepest(held, world.esdf, 0.0)
        reached = "the robot reaches this pose with nothing in the way, but"
        if load is not None and (arm is None or load[1] >= arm[1]):
            return {"cause": "held_object_in_collision", "link": None, "penetration_m": load[1],
                    "message": (f"{reached} there the held object would be {load[1] * 100:.1f} cm inside "
                                f"{self._stands(held[load[0]], world)}")}
        if arm is not None:
            link = links[arm[0]]
            part = LINK_PARTS[self.robot].get(link, "arm")
            opened = " (the plan keeps the jaws fully open, so closing them changes nothing here)" if (
                part == "fingers" and self.robot in FINGERS_PLANNED_OPEN) else ""
            return {"cause": "robot_in_collision", "link": link, "part": part, "penetration_m": arm[1],
                    "message": (f"{reached} there its {part} would be {arm[1] * 100:.1f} cm inside "
                                f"{self._stands(spheres[arm[0]], world)}{opened}")}
        if blocks == "self":
            return {"cause": "self_collision",
                    "message": f"{reached} clear of what the depth shows, the solutions found put the arm against itself"}
        return {"cause": "no_clear_solution_found",
                "message": f"{reached} this bounded search found no configuration clear of the world"}

    def _stands(self, sphere: Any, world: Any) -> str:
        """``what_stands`` for one sphere reaching into a goal's world."""
        return what_stands({"point_base_m": sphere[:3], "radius_m": sphere[3]}, self._scene[1],
                           world.summary["support"]["goal"])

    def _other_arm_reaches(self, pose: list[float], world: Any) -> bool | None:
        """Whether an empty-handed two-arm robot's other arm reaches ``pose``, with nothing in the way; else ``None``."""
        inputs = self._inputs
        if inputs.other_arm_joint_positions is None or len(world.held_spheres_hand):
            return None
        goal = (np.linalg.inv(pose_matrix(inputs.other_arm_base_pose_world_xyzw))
                @ pose_matrix(inputs.base_pose_world_xyzw) @ pose_matrix(pose))
        at = [*goal[:3, 3].tolist(), *matrix_quaternion(goal[:3, :3].tolist())]
        return bool(self.engine.solve_ik([float(item) for item in inputs.other_arm_joint_positions], at)["solved"])

    def _deepest(self, spheres: Any, esdf: Any, tolerance: float) -> tuple[int, float] | None:
        """The row of ``spheres`` reaching deepest into ``esdf``, past ``tolerance``, and how far; ``None`` when clear."""
        parameters = self.parameters
        index = np.floor((spheres[:, :3] - parameters.grid_low_m) / parameters.voxel_size_m).astype(np.int64)
        inside = ((index >= 0) & (index < np.asarray(parameters.grid_shape))).all(axis=1) & (spheres[:, 3] > 0)
        depth = spheres[inside, 3] + esdf[tuple(index[inside].T)]
        if not depth.size or float(depth.max()) <= tolerance:
            return None
        worst = int(np.argmax(depth))
        return int(np.nonzero(inside)[0][worst]), float(depth[worst])

    def solve_ik(self, raw: Any) -> dict[str, Any]:
        started = time.time()
        request = None
        try:
            request = validate_ik_request(raw)
            if request["robot"] != self.robot:
                raise ValueError(f"this worker solves {self.robot}")
            return self._solve_ik(request, started)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return error_answer(IK_ANSWER_SCHEMA, request, "planner_error", exc,
                                implementation=IMPLEMENTATION, started=started)

    def check_path(self, raw: Any) -> dict[str, Any]:
        started = time.time()
        request = None
        try:
            request = validate_check_request(raw)
            if request["robot"] != self.robot:
                raise ValueError(f"this worker checks {self.robot}")
            return self._check_path(request, started)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            answer = error_answer(CHECK_ANSWER_SCHEMA, request, "planner_error", exc,
                                  implementation=IMPLEMENTATION, started=started)
            return {**answer, "first_blocked_row": None, "link": None, "penetration_m": None, "world": None}

    def _check_path(self, request: Mapping[str, Any], started: float) -> dict[str, Any]:
        """Every link but the allowed ones against the world with only the fingers' goal carve."""
        rows = request["path"]
        scene = self._scene_for(request)
        hand, spheres = self.engine.kinematics(rows[-1])
        world = FingerScene(scene, self._fingers()).world(hand @ self.tool_from_hand, spheres)
        links = self._links()
        allowed = set(request["allowed_links"])
        mine = [index for index, link in enumerate(links) if link not in allowed]
        blocked, link, deepest = None, None, 0.0
        for row, joints in enumerate(rows):
            found = self._deepest(self.engine.kinematics(joints)[1][mine], world.esdf, float(request["tolerance_m"]))
            if found is not None:
                blocked, link, deepest = row, links[mine[found[0]]], found[1]
                break
        return {
            "schema": CHECK_ANSWER_SCHEMA, "status": "clear" if blocked is None else "blocked", "reason": None,
            "first_blocked_row": blocked, "link": link, "penetration_m": deepest if blocked is not None else None,
            "world": {"model": self.world_model, "sha256": world.sha256, "inputs_sha256": scene.inputs_sha256,
                      "summary": world.summary},
            "planner": self._planner(), "timing_s": {"total": time.time() - started}, "request": request, "detail": {},
        }

    def _solve_ik(self, request: Mapping[str, Any], started: float) -> dict[str, Any]:
        """One IK solve per query, each in the world a plan to its pose would use, seeded as chained."""
        scene = self._scene_for(request)
        results: list[dict[str, Any]] = []
        ends: list[list[float] | None] = []
        worlds: list[dict[str, Any] | None] = []
        blank = {"solved": False, "skipped": False, "refusal": None, "joint_positions": None,
                 "tool_pose_base_xyzw": None, "position_error_m": None, "rotation_error": None}
        for query in request["queries"]:
            start = query["from_index"]
            seed = request["start"]["joint_positions"] if start is None else ends[start]
            if seed is None:
                results.append({**blank, "skipped": True})
                ends.append(None)
                worlds.append(None)
                continue
            pose = query["pose_base_xyzw"]
            world = self._apply(scene, request, pose=pose)
            if world is not None and world["summary"]["goal_refusal"] is not None:
                worlds.append({**world["summary"], "sha256": world["sha256"]})
                results.append({**blank, "refusal": "goal_below_support"})
                ends.append(None)
                continue
            solved = self.engine.solve_ik(seed, pose)
            if solved["solved"] and world is not None:
                solved, world = self._solution_kept_clear(scene, request, seed, pose, solved, world)
            worlds.append(None if world is None else {**world["summary"], "sha256": world["sha256"]})
            joints = solved["joint_positions"] if solved["solved"] else None
            results.append({**blank, "solved": solved["solved"], "joint_positions": joints,
                            "tool_pose_base_xyzw": self.engine.tool_pose(joints) if joints else None,
                            "position_error_m": solved["position_error_m"],
                            "rotation_error": solved["rotation_error"]})
            ends.append(joints)
        names = request["start"]["joint_names"]
        return {
            "schema": IK_ANSWER_SCHEMA, "status": "solved", "reason": None, "joint_names": list(names),
            "joint_limits": self.engine.joint_limits(),
            "start_tool_pose_base_xyzw": self.engine.tool_pose(request["start"]["joint_positions"]),
            "results": results,
            "world": None if scene is None else {"model": self.world_model, "inputs_sha256": scene.inputs_sha256,
                                                 "queries": worlds},
            "planner": self._planner(), "timing_s": {"total": time.time() - started},
            "request": request, "detail": {"solver": self.engine.ik_identity()},
        }


def serve(worker: Worker, lines: Any, write: Any) -> int:
    """Answer one JSON request per line until ``close`` or the end of input."""
    for line in lines:
        request_id: Any = None
        command: Any = None
        try:
            envelope = json.loads(line)
            request_id, command = envelope.get("request_id"), envelope.get("command")
            if envelope.get("protocol") != PROTOCOL:
                raise ValueError("protocol drift")
            reply: dict[str, Any] = {"protocol": PROTOCOL, "request_id": request_id, "command": command}
            if command == "ready":
                reply["ready"] = worker.ready()
            elif command == "plan":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = worker.plan(envelope.get("request"))
            elif command == "ik":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = worker.solve_ik(envelope.get("request"))
            elif command == "fk":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = worker.fk(envelope.get("request"))
            elif command == "check":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = worker.check_path(envelope.get("request"))
            elif command == "prepare":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = worker.prepare(envelope.get("request"))
            elif command == "close":
                write(json.dumps({**reply, "closed": True}))
                return 0
            else:
                raise ValueError("unsupported command")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {"protocol": PROTOCOL, "request_id": request_id, "command": command,
                     "error": {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__}}
        write(json.dumps(reply, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot", default="franka_panda", choices=sorted(ROBOTS))
    parser.add_argument("--world-model", default="none", choices=WORLD_MODELS)
    parser.add_argument("--robot-file", default=None)
    parser.add_argument("--scene", default=os.environ.get(SCENE_ENV, "tabletop"), choices=sorted(SCENES))
    args = parser.parse_args()
    from .engine import CuroboV2Engine

    with contextlib.redirect_stdout(sys.stderr):
        engine = CuroboV2Engine(device=args.device, robot=args.robot, world_model=args.world_model,
                                robot_file=args.robot_file)
    worker = Worker(engine, robot=args.robot, world_model=args.world_model, scene=args.scene)
    code = serve(worker, sys.stdin, lambda text: print(text, flush=True))
    engine.close()
    return code


if __name__ == "__main__":
    # Leave without interpreter teardown: freeing captured CUDA graphs' tensors
    # after the CUDA context is gone aborts the process.
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
