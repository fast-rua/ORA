"""RoboTwin's place on the shared motion base: planning, IK and the world.

``RoboTwinMotionBackendAdapter`` answers ``plan_global``, ``plan_batch_global``, the
reach check and ``align_plan`` here when the run's preflight bound a planner. The
vendored planner stays in the worker for the expert and the seed scans, and for a
session with no planner (a direct session, a qualification): ``session.py`` and
``vendored/`` feed the seed scans' hash, so nothing here edits them.

* **Binding.** The worker's config cannot carry a planner (its session fields are
  fixed), so ``runtime.motion_planner`` reaches it through the run's own
  ``preflight.json`` (``SHARED_PLANNER_CHECK``), as the timing worker does.
* **Robot.** ``aloha_agilex``: one worker plans both arms, whose chains are the same
  joint for joint, each request in the moving arm's base frame.
* **Frames.** ``planner_from_world`` asks the vendored ``plan_path`` itself what it
  would solve (:class:`_GoalCapture`), so the end-link offset, the world-to-base
  transform and the per-arm frame bias and yaw stay in one place. At start the
  planner's forward kinematics of both arms' measured joints must land on the
  simulator's link6 (``fk_parity``), or the session refuses the planner.
* **Joints.** The description's joints run to a placeholder +/-10 rad and a run can
  wind a joint past a full turn, so a start is wrapped into +/-pi for the planner
  (:func:`wrap`) and the answer shifted back: the same pose, and a legal start.
* **World.** ``public_depth_voxel_v1`` for the ALOHA: the depth of the policy's
  cameras as ``get_obs`` publishes them now, with the other arm's spheres at its
  measured joints standing in the world. A snapshot per physical state and arm.
* **A planner that fails** never stops a move: an error falls back to the vendored
  planner, and ``motion-planner/plans.jsonl`` says so. A refusal stands, but for a
  measured start the planner's own spheres call self-colliding: the arm is there, so
  that says nothing about the goal (:func:`_self_colliding_start`).

Answers carry no planner library's name: a plan comes back as the plan result the
admission reads, with a neutral ``planner_status``, and the full answers stay in
``motion-planner/`` beside the run's other private evidence.
"""
from __future__ import annotations

import atexit
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.pose import matrix_quaternion
from ora.contracts.reach_check import START_UNREACHABLE
from ora.contracts.robotwin_reach import CLAIM_BOUNDARY, IK_FAILED, IK_MODEL, ROBOT_FILE, SHARED_PLANNER_CHECK
from ora.environment.controllers.kept_plans import KeptPlans
from ora.environment.controllers.depth_world import WORLD_MODEL, pose_matrix, write_world_inputs
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity
from ora.environment.controllers.motion_planner import (
    IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, ROBOTS, planner_client, planning_diagnostics,
)
from ora.environment.controllers.motion_worker import MotionPlannerError
from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

from .common import RoboTwinLiveError
from .planning import PLAN_RESULT_SCHEMA, cartesian_pose_error_twist

ROBOT = "aloha_agilex"
TOOL_FRAME = ROBOTS[ROBOT]["tool_frame"]
JOINT_NAMES = list(ROBOTS[ROBOT]["joint_names"])
ARMS = ("left", "right")
#: How close a vendored goal must lie to ``planner_from_world`` applied to its target.
FRAME_TOLERANCE_M = 1e-5
_TWO_PI = 2.0 * math.pi


class _GoalCapture:
    """``self`` for ``CuroboPlanner.plan_path``, which stops at the solve and returns its goal."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)

    def _plan_pose_via_nearest_ik(self, _start: Any, _joint_angles: Any, goal_pose: Any) -> Any:
        return goal_pose


@dataclass
class SharedMotion:
    """One session's planner client, and what it has built for the current physical state."""

    client: Any
    root: Path
    planner_from_world: dict[str, Any] = field(default_factory=dict)
    snapshots: dict[Any, dict[str, str]] = field(default_factory=dict)
    snapshot_count: int = 0
    binding: dict[str, Any] = field(default_factory=dict)


def _vendored_goal(robot: Any, arm: str, pose_wxyz: Sequence[float]) -> Any:
    """The 4x4 base-frame goal the vendored ``plan_path`` would solve for one public-planner pose."""
    planner = getattr(robot, f"{arm}_planner")
    target = robot._trans_from_gripper_to_endlink([float(item) for item in pose_wxyz], arm_tag=arm)
    qpos = list(getattr(robot, f"{arm}_entity").get_qpos())
    goal = type(planner).plan_path(_GoalCapture(planner), qpos, target, arms_tag=arm)
    position = goal.position.detach().to("cpu").numpy().astype(np.float64).reshape(3)
    w, x, y, z = goal.quaternion.detach().to("cpu").numpy().astype(np.float64).reshape(4)
    return pose_matrix([*position, x, y, z, w])


def planner_frame(robot: Any, arm: str) -> Any:
    """The 4x4 ``planner_from_world`` of ``arm``, read off the vendored transform and checked on a second pose."""
    if getattr(robot, "communication_flag", True):
        raise RoboTwinLiveError("the shared planner's frames need the vendored planner in this process")
    frame = _vendored_goal(robot, arm, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sample = [0.3, 0.1, 0.9, 0.5, 0.5, 0.5, 0.5]
    expected = frame @ pose_matrix([*sample[:3], *sample[4:], sample[3]])
    if np.abs(_vendored_goal(robot, arm, sample) - expected).max() > FRAME_TOLERANCE_M:
        raise RoboTwinLiveError("the vendored planner's target transform is not one rigid frame")
    return frame


def _preflight_binding(output_root: Path) -> Mapping[str, Any] | None:
    """The planner this run's preflight bound, or None for a run without one."""
    record = Path(output_root).parent / "preflight.json"
    if not record.is_file():
        return None
    checks = json.loads(record.read_text(encoding="utf-8")).get("checks") or ()
    matches = [item for item in checks if isinstance(item, Mapping) and item.get("check") == SHARED_PLANNER_CHECK]
    if not matches:
        return None
    detail = matches[0].get("detail") if matches[0].get("passed") is True else None
    if len(matches) != 1 or not isinstance(detail, Mapping):
        raise RoboTwinLiveError("the motion planner preflight binding is absent or invalid")
    return detail


def _measured_link6(session: Any, environment: Any, arm: str, joints: Sequence[float]) -> Any:
    link, _chain = session._pinocchio_world_link_poses(environment, arm, joints)
    w, x, y, z = (float(value) for value in link.q)
    return pose_matrix([*(float(value) for value in link.p), x, y, z, w])


def start(session: Any, environment: Any) -> None:
    """Bind this process's planner, once, if the run's preflight named one; check it against the simulator."""
    session.shared_motion = None
    detail = _preflight_binding(session.output_root)
    if detail is None:
        return
    root = Path(session.output_root) / "motion-planner"
    client = planner_client(detail, python_executable=Path(detail["python"]), root=Path(detail["root"]),
                            output_root=root, robot_file=Path(session.robotwin_root) / ROBOT_FILE)
    if client.binding["python_sha256"] != detail["python_sha256"]:
        raise RoboTwinLiveError("the motion planner interpreter changed after preflight")
    binding = client.start()
    atexit.register(client.close)
    motion = SharedMotion(client=KeptPlans(client), root=root)
    bind_round(session, environment, motion)
    samples = []
    for arm in ARMS:
        joints, _offset = wrap(session._measured_arm_joints(environment, arm))
        samples.append({
            "joints": joints,
            "simulator": (motion.planner_from_world[arm] @ _measured_link6(session, environment, arm, joints)).tolist(),
            "planner": pose_matrix(client.forward_kinematics([joints])[0]).tolist()})
    report = parity(samples, *ROBOT_TOLERANCES[ROBOT])
    motion.binding = {**binding, "fk_parity": report}
    (root / "binding.json").write_text(json.dumps(motion.binding, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        client.close()
        raise RoboTwinLiveError(f"the planner's forward kinematics differs from the simulator's: {report}")
    session.shared_motion = motion


def bind_round(session: Any, environment: Any, motion: SharedMotion) -> None:
    """A round's frames and an empty snapshot table: a new round's physical clock starts again."""
    motion.planner_from_world = {arm: planner_frame(environment.robot, arm) for arm in ARMS}
    motion.snapshots = {}


def active(session: Any) -> SharedMotion | None:
    return getattr(session, "shared_motion", None)


def wrap(joints: Sequence[float]) -> tuple[list[float], list[float]]:
    """``joints`` in +/-pi and what was taken off each: the same pose, whatever turns a run wound up."""
    values = np.asarray([float(item) for item in joints], dtype=np.float64)
    offset = _TWO_PI * np.round(values / _TWO_PI)
    return (values - offset).tolist(), offset.tolist()


def base_pose_xyzw(motion: SharedMotion, arm: str) -> list[float]:
    """The arm's planner base in the world, ``[x, y, z, qx, qy, qz, qw]``."""
    base = np.linalg.inv(motion.planner_from_world[arm])
    return [*base[:3, 3].tolist(), *matrix_quaternion(base[:3, :3].tolist())]


def _base_pose(motion: SharedMotion, arm: str, pose_wxyz: Sequence[float]) -> list[float]:
    """One vendored-frame target (``compile_public_target``'s ``pose_wxyz``) in the arm's base frame."""
    w, x, y, z = (float(value) for value in pose_wxyz[3:])
    local = motion.planner_from_world[arm] @ pose_matrix([*(float(value) for value in pose_wxyz[:3]), x, y, z, w])
    return [*local[:3, 3].tolist(), *matrix_quaternion(local[:3, :3].tolist())]


def _cameras(session: Any, environment: Any) -> dict[str, dict[str, Any]]:
    """The published depth of every policy camera, in metres, read now without advancing physics."""
    cameras = {}
    for name, camera in session._read_observation_data(environment)["observation"].items():
        extrinsic = np.eye(4)
        extrinsic[:3, :4] = np.asarray(camera["extrinsic_cv"], dtype=np.float64)[:3, :4]
        cameras[str(name)] = {"depth_m": np.asarray(camera["depth"], dtype=np.float32) / 1000.0,
                              "intrinsic_cv": np.asarray(camera["intrinsic_cv"], dtype=np.float64),
                              "cam2world_cv": np.linalg.inv(extrinsic), "depth_range_m": (0.0, 1000.0)}
    return cameras


def _finger_travel(environment: Any, arm: str) -> list[float]:
    """Each finger's measured travel in metres; their sum is the jaw separation."""
    robot = environment.robot
    entity = getattr(robot, f"{arm}_entity")
    qpos, joints = entity.get_qpos(), list(entity.get_active_joints())
    return [max(float(qpos[joints.index(joint)]), 0.0) for joint, _multiplier, _offset in getattr(robot, f"{arm}_gripper")]


def world_input(session: Any, environment: Any, motion: SharedMotion, arm: str,
                measured: Mapping[str, Sequence[float]]) -> tuple[dict[str, str], list[float], list[float]]:
    """``arm``'s world snapshot of the current physical state, its wrapped joints and what was taken off them."""
    joints, offset = wrap(measured[arm])
    other = "right" if arm == "left" else "left"
    other_joints, _other_offset = wrap(measured[other])
    key = (arm, int(session.physics_step_sequence), int(session.world_mutation_sequence), tuple(joints),
           tuple(other_joints))
    record = motion.snapshots.get(key)
    if record is None:
        motion.snapshot_count += 1
        directive = 1.0 if session.gripper_command_targets.get(arm) == "close" else -1.0
        record = write_world_inputs(
            motion.root / "world-inputs" / f"{motion.snapshot_count:05d}-{arm}.npz",
            cameras=_cameras(session, environment), base_pose_world_xyzw=base_pose_xyzw(motion, arm),
            joint_positions=joints, gripper_finger_qpos=_finger_travel(environment, arm), gripper_directive=directive,
            other_arm={"joint_positions": other_joints, "base_pose_world_xyzw": base_pose_xyzw(motion, other)})
        motion.snapshots = {key: record, **{name: value for name, value in motion.snapshots.items()
                                            if name[1:3] == key[1:3]}}
    return record, joints, offset


def _evidence(motion: SharedMotion, kind: str, record: Mapping[str, Any]) -> None:
    motion.root.mkdir(parents=True, exist_ok=True)
    with (motion.root / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _plan_result(answer: Mapping[str, Any], *, joint_names: Sequence[str], offset: Sequence[float]) -> dict[str, Any]:
    """A planner answer as the plan result the admission reads."""
    seconds = {key: float((answer.get("timing_s") or {}).get(key) or 0.0) for key in ("world", "solve", "total")}
    result: dict[str, Any] = {
        "schema": PLAN_RESULT_SCHEMA, "status": "Success" if answer["status"] == "planned" else "Fail",
        "planner_status": str(answer["reason"] or "planned"), "valid_query": True, "attempts": 1,
        "trajopt_attempts": 0, "used_graph": False, "goalset_index": None, "path_buffer_last_tstep": None,
        "errors": {"position": None, "rotation": None, "cspace": None},
        "timings_s": {"solve": seconds["solve"], "ik": 0.0, "graph": 0.0, "trajopt": 0.0, "finetune": 0.0,
                      "total": seconds["total"]},
        "interpolation_dt_s": None, "joint_names": list(joint_names),
    }
    trajectory = answer.get("trajectory")
    if trajectory is None:
        return {**result, "error": {"stage": "plan", "type": "PlanRefused", "message": str(answer["reason"])},
                "planning_diagnostics": planning_diagnostics(answer)}
    shift = np.asarray(offset, dtype=np.float64)
    return {**result, "interpolation_dt_s": float(trajectory["dt_s"]),
            "path_buffer_last_tstep": len(trajectory["positions"]) - 1,
            "position": np.asarray(trajectory["positions"], dtype=np.float64) + shift,
            "velocity": np.asarray(trajectory["velocities"], dtype=np.float64)}


def _self_colliding_start(answer: Mapping[str, Any]) -> bool:
    """A start refused for the arm against itself alone, not for the world."""
    check = (answer.get("detail") or {}).get("start_check") or {}
    return answer.get("reason") == "start_in_collision" and not check.get("world_collision")


def plan(session: Any, environment: Any, motion: SharedMotion, arm: str, target: Mapping[str, Any],
         measured: Mapping[str, Sequence[float]], joint_names: Sequence[str]) -> dict[str, Any] | None:
    """One arm's path from ``measured`` to a ``compile_public_target`` target; None when the planner failed."""
    try:
        record, joints, offset = world_input(session, environment, motion, arm, measured)
        if target["kind"] == "joint":
            goal = np.asarray(target["joints"], dtype=np.float64)
            start = np.asarray(joints, dtype=np.float64)
            goal = {"joint_positions": (goal - _TWO_PI * np.round((goal - start) / _TWO_PI)).tolist()}
        else:
            goal = {"tool_frame": TOOL_FRAME, "pose_base_xyzw": _base_pose(motion, arm, target["pose_wxyz"])}
        answer = motion.client.plan({
            "schema": PLAN_REQUEST_SCHEMA, "robot": ROBOT,
            "start": {"joint_names": JOINT_NAMES, "joint_positions": joints}, "goal": goal,
            "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}, "constraints": []})
    except MotionPlannerError as error:
        _evidence(motion, "plans", {"arm": arm, "error": str(error), "fallback": True})
        return None
    _evidence(motion, "plans", {key: answer.get(key) for key in (
        "status", "reason", "world", "planner", "timing_s", "start_tool_pose_base_xyzw")} | {"arm": arm})
    if answer["status"] == "error" or _self_colliding_start(answer):
        _evidence(motion, "plans", {"arm": arm, "fallback": True})
        return None
    return _plan_result(answer, joint_names=joint_names, offset=offset)


def _residual(target: Sequence[float], pose: Sequence[float]) -> tuple[float, float]:
    error = cartesian_pose_error_twist(list(target), list(pose))
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def _blank(reason: str) -> dict[str, Any]:
    return {"reachable": False, "reason": reason, "route": None, "position_error_m": None,
            "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}


def reach(session: Any, environment: Any, motion: SharedMotion, adapter: Any, arm: str,
          queries: Sequence[Mapping[str, Any]], measured: Mapping[str, Sequence[float]], *,
          position_tolerance_m: float, orientation_tolerance_rad: float) -> dict[str, Any]:
    """One row per query from the planner's own IK, in the world a plan to that pose would see."""
    record, joints, offset = world_input(session, environment, motion, arm, measured)
    answer = motion.client.solve_ik({
        "schema": IK_REQUEST_SCHEMA, "robot": ROBOT,
        "start": {"joint_names": JOINT_NAMES, "joint_positions": joints},
        "queries": [{"tool_frame": TOOL_FRAME, "from_index": query["from_index"], "pose_base_xyzw": _base_pose(
            motion, arm, adapter.compile_public_target(arm, query["pose_xyzw"])["pose_wxyz"])} for query in queries],
        "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}})
    _evidence(motion, "ik", {"arm": arm, "status": answer["status"], "timing_s": answer.get("timing_s"),
                             "queries": len(queries)})
    if answer["status"] != "solved":
        raise MotionPlannerError(f"the shared IK answered {answer['status']}: {answer.get('reason')}")
    limits = list(zip(*answer["joint_limits"]))
    shift = np.asarray(offset, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for query, solved in zip(queries, answer["results"]):
        start = query["from_index"]
        if solved["skipped"]:
            rows.append(_blank(START_UNREACHABLE))
            continue
        if not solved["solved"]:
            rows.append(_blank(IK_FAILED))
            continue
        solution = solved["joint_positions"]
        pose = adapter.fk_public_tcp_pose(arm, (np.asarray(solution, dtype=np.float64) + shift).tolist())
        position, orientation = _residual(query["pose_xyzw"], pose)
        within = position <= position_tolerance_m and orientation <= orientation_tolerance_rad
        seed = joints if start is None else answer["results"][start]["joint_positions"]
        rows.append({"reachable": within, "reason": None if within else IK_FAILED, "route": None,
                     "position_error_m": position, "orientation_error_rad": orientation,
                     "joint_limit_margin_rad": joint_limit_margin_rad(solution, limits),
                     "joint_delta_rad": joint_delta_rad(solution, seed)})
    return {"ik_model": IK_MODEL, "matches_move_pose_admission": False, "claim_boundary": CLAIM_BOUNDARY,
            "results": rows}


__all__ = ["ARMS", "JOINT_NAMES", "ROBOT", "SharedMotion", "TOOL_FRAME", "active",
           "base_pose_xyzw", "bind_round", "plan", "planner_frame", "reach", "start", "world_input", "wrap"]
