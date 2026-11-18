"""The YAM station's arm motions planned by ORA's shared cuRobo V2 planner (``runtime.motion_planner``, off by default).

Without the switch the station plans by itself (``motion.py``: mink IK along its own Cartesian lines). With it, the
planning is the shared motion base's, as the simulator backends have it, and everything after the plan stays the
station's: the timing (its own, or the shared timing worker's with ``runtime.motion_timing``), the per-tick checks
of the joint limits, the table clearance and the inter-arm distance on the ticks as timed, the operator's approval,
the execution and the receipt.

* **Planner.** One ``MotionPlannerClient`` for ``yam_linear_4310`` (``planners/curobo_v2/yam.py``), a process of its
  own under ``CUROBO_V2_PYTHON``, bound from the run's preflight (``PREFLIGHT_CHECK``), else from the environment;
  its robot file is the station URDF of the i2rt this interpreter holds. With neither, the station plans by itself
  and ``motion-planner/binding.json`` says so. At start the planner's forward kinematics must land on the station
  model's TCP in each arm's base frame, around the measured joints (``fk_parity``), or the session refuses to start.
* **Frames and calibration.** One six-joint model plans either arm, each request in the moving arm's base frame:
  the station model's ``{arm}_base`` body as the cell's calibration placed it (``StationKinematics
  .set_base_corrections``), so the right arm's base correction is that frame. The planner plans model angles: each
  joint's encoder zero offset (``joint_offsets``) goes on every joint sent and comes off every row answered, so a
  row is an encoder reading again, as the station's checks and the rig take it.
* **A start on a stop.** The cell rests a little past joint limits (the measured encoder zero, 2026-11-14: joints 2
  and 3 by 0.5 to 1 degree). A start at most ``START_LIMIT_SLACK_RAD`` outside a model limit is planned from
  ``START_LIMIT_INSET_RAD`` inside it, and the path starts with the step from the measured start in: inward only.
  The station's per-tick limit check, widened to the start (``StationKinematics.widened_limits``), passes that step.
* **World.** ``public_depth_voxel_v1`` with the YAM's parameters (``depth_world.PUBLIC_DEPTH_VOXEL_YAM_V1``): the
  depth of the station's calibrated public cameras, read when the plan is made, the other arm standing in it as the
  planner's own spheres at its measured joints (a two-arm world input). The table is what the cameras see of it.
* **Routes.** A ``free_space`` target is one pose plan, collision-free in that world; the station's lift, traverse
  and descend give way to it. A ``precision`` or ``contact_approach`` target keeps the station's straight line: the
  shared IK solves it every ``LINE_STEP_M`` (or ``LINE_STEP_RAD`` of turn), each knot seeded by the one before, in
  a world without depth, as RoboCasa solves a line; the line is then checked in the real world, the gripper and
  its fingertips allowed to touch (``GRIPPER_LINKS``). The admission's reach question (does any configuration
  reach the target; how high the arm reaches here; the nearest position that solves) is the shared IK's too.
* **Refusals** keep the station's words (``CODES``): ``target_ik_unreachable`` with ``highest_reachable_z_m``,
  ``target_below_table_clearance``, ``joint_limit``, ``ik_branch_jump``; what only a world can say is
  ``target_in_collision``, ``start_in_collision`` and ``path_blocked``. A planner that fails never stops a move: an
  error is planned by the station itself, and ``motion-planner/plans.jsonl`` and the receipt's ``motion.planned_by``
  (evidence only, ``policy_receipt``) say so.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.benchmarks.yam import MOTION_PLANNER_PREFLIGHT_CHECK
from ora.environment.controllers import yam_station as Y
from ora.environment.controllers.depth_world import WORLD_MODEL, write_world_inputs
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity
from ora.environment.controllers.motion_planner import (
    CHECK_REQUEST_SCHEMA, IK_REQUEST_SCHEMA, MAXIMUM_IK_QUERIES, PLAN_REQUEST_SCHEMA, ROBOTS, planner_client,
)
from ora.environment.controllers.motion_worker import MotionPlannerError

from .common import YamError, pose_error
from .kinematics import pose_matrix, pose_xyz_xyzw, station_xml_path

ROBOT = "yam_linear_4310"
TOOL_FRAME = ROBOTS[ROBOT]["tool_frame"]
JOINT_NAMES = list(ROBOTS[ROBOT]["joint_names"])
#: The preflight record that binds the planner's interpreter and source (``orchestrator/backends/yam/preflight.py``).
PREFLIGHT_CHECK = MOTION_PLANNER_PREFLIGHT_CHECK
#: A planned row's ``motion.planned_by`` implementation when the station planned the arm itself.
STATION_IMPLEMENTATION = "yam_mink_ik"
#: How far past a model limit a start may lie and still be the arm on its stop, and how far inside it is planned from.
START_LIMIT_SLACK_RAD = math.radians(2.0)
START_LIMIT_INSET_RAD = 0.002
#: The knots of a straight line: at most this far apart along it, and this much turn.
LINE_STEP_M = 0.01
LINE_STEP_RAD = 0.05
#: Two knots further apart than this on any joint left the IK branch (``motion.plan_arm_path``'s rule).
BRANCH_JUMP_RAD = 0.35
#: The links a straight approach may bring against the world: the gripper and its two fingertips.
GRIPPER_LINKS = ("left_gripper", "left_tip_left", "left_tip_right")
#: How far another link may lie inside the world along a line (its depth noise).
LINE_TOLERANCE_M = 0.01
#: The planner's neutral refusals in the station's words. The admission has asked the same IK whether any configuration
#: reaches the target (``target_ik_unreachable`` there), so a pose goal the planner cannot reach is one the world takes.
CODES = {
    "goal_unreachable": "target_in_collision",
    "goal_below_support": "target_below_table_clearance",
    "goal_in_collision": "target_in_collision",
    "start_in_collision": "start_in_collision",
    "start_outside_limits": "joint_limit",
    "path_blocked": "path_blocked",
}
#: Each camera's published depth range for the world: a zero pixel has no depth.
DEPTH_RANGE_M = (0.0, 3.0)
#: Joint offsets of the FK parity rows around the measured joints.
PARITY_OFFSETS_RAD = (0.0, 0.3, -0.3)


class Unplanned(Exception):
    """The planner could not answer (an error, not a refusal): the station plans this arm itself."""


@dataclass
class SharedPlanner:
    """One session's planner client and the world snapshots of the current physical state."""

    client: Any
    root: Path
    binding: dict[str, Any]
    snapshots: dict[Any, dict[str, str]] = field(default_factory=dict)
    cameras: tuple[Any, dict[str, Any]] | None = None
    count: int = 0


def enabled(session: Any) -> bool:
    """Whether this session's runtime selects the shared planner (``runtime.motion_planner``)."""

    return isinstance((getattr(session, "config", None) or {}).get("motion_planner"), Mapping)


def active(session: Any) -> SharedPlanner | None:
    return getattr(session, "shared_planner", None)


def located(config: Mapping[str, Any], output_root: Path) -> tuple[str, str] | None:
    """The interpreter and source root of this run's planner: the preflight's, else the named environment, else None.

    A run whose preflight names the check binds what it recorded; a record that names it failed or malformed is
    refused, as for the timing worker."""

    record = Path(output_root).parent / "preflight.json"
    if record.is_file():
        checks = json.loads(record.read_text(encoding="utf-8")).get("checks") or ()
        matches = [item for item in checks if isinstance(item, Mapping) and item.get("check") == PREFLIGHT_CHECK]
        if matches:
            detail = matches[0].get("detail") if len(matches) == 1 and matches[0].get("passed") is True else None
            if not (isinstance(detail, Mapping) and isinstance(detail.get("python"), str)
                    and isinstance(detail.get("root"), str)):
                raise YamError("the motion planner preflight binding is absent or invalid")
            return detail["python"], detail["root"]
    python, root = os.environ.get(str(config["python_env"])), os.environ.get(str(config["root_env"]))
    if python and root:
        return os.path.abspath(os.path.expanduser(python)), root
    return None


def start(session: Any) -> dict[str, Any] | None:
    """Bind the planner and check it against the station model; the binding, or None with the switch off."""

    session.shared_planner = None
    if not enabled(session):
        return None
    config = session.config["motion_planner"]
    root = Path(session.output_root) / "motion-planner"
    root.mkdir(parents=True, exist_ok=True)
    where = located(config, session.output_root)
    if where is None:
        bound = {"implementation": STATION_IMPLEMENTATION, "reason": f"no planner interpreter "
                 f"({config['python_env']}, {config['root_env']}) and no preflight binding"}
        (root / "binding.json").write_text(json.dumps(bound, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return bound
    client = planner_client(config, python_executable=Path(where[0]), root=Path(where[1]), output_root=root,
                            robot_file=station_xml_path().with_suffix(".urdf"))
    bound = client.start()
    report = fk_parity(session.kinematics, client)
    bound = {**bound, "fk_parity": report}
    (root / "binding.json").write_text(json.dumps(bound, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        client.close()
        raise YamError(f"the planner's forward kinematics differs from the station model's: {report}")
    session.shared_planner = SharedPlanner(client=client, root=root, binding=bound)
    return bound


def close(session: Any) -> None:
    planner = active(session)
    if planner is not None:
        planner.client.close()
    session.shared_planner = None


# ---------------------------------------------------------------- frames and joints
def offsets(kin: Any, arm: str) -> np.ndarray:
    """``arm``'s encoder zero offsets (model angle = reading + offset)."""

    return np.asarray(kin.joint_offsets.get(arm, np.zeros(Y.ARM_JOINT_COUNT)), dtype=float)


def base_world(kin: Any, arm: str) -> np.ndarray:
    """The 4x4 pose of ``arm``'s base in ``yam_station_world``, as the cell's calibration placed it."""

    body = kin.base_body[arm]
    if not np.any(kin.data.xmat[body]):        # a model whose kinematics never ran
        kin.set_state({name: kin.arm_q(name) for name in Y.ARMS})
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(kin.data.xmat[body], dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(kin.data.xpos[body], dtype=float) + np.asarray(Y.STATION_TO_WORLD_XYZ_M)
    return matrix


def in_base(kin: Any, arm: str, pose_world: np.ndarray) -> list[float]:
    """A world TCP pose as a planner goal: ``[x, y, z, qx, qy, qz, qw]`` in ``arm``'s base frame."""

    return pose_xyz_xyzw(np.linalg.inv(base_world(kin, arm)) @ pose_world)


def inside_limits(kin: Any, arm: str, model: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """``model`` (model angles) with each joint just past a limit moved just inside it, and what moved.

    Raises ``Refused`` (``joint_limit``) for a joint further outside than ``START_LIMIT_SLACK_RAD``."""

    limits = kin.model_limits[arm]
    inside, moved = np.asarray(model, dtype=float).copy(), []
    for index, (value, (low, high)) in enumerate(zip(inside, limits)):
        if low + START_LIMIT_INSET_RAD <= value <= high - START_LIMIT_INSET_RAD:
            continue
        if value < low - START_LIMIT_SLACK_RAD or value > high + START_LIMIT_SLACK_RAD:
            raise Refused("joint_limit", joint=JOINT_NAMES[index].replace("left_", ""), tick=0)
        inside[index] = min(max(value, low + START_LIMIT_INSET_RAD), high - START_LIMIT_INSET_RAD)
        moved.append({"joint": index + 1, "measured": float(value), "planned_from": float(inside[index])})
    return inside, moved


class Refused(Exception):
    """A planner refusal in the station's words: ``code``, the detail ``motion.Refusal`` carries (public words only)
    and the planner's ``record`` for the receipt's evidence."""

    def __init__(self, code: str, record: Mapping[str, Any] | None = None, **detail: Any) -> None:
        super().__init__(code)
        self.code, self.record, self.detail = code, None if record is None else dict(record), detail


def fk_parity(kin: Any, client: Any) -> dict[str, Any]:
    """The planner's TCP against the station model's, each arm in its base frame, around the measured joints."""

    samples = []
    for arm in Y.ARMS:
        reading = kin.arm_q(arm)
        limits = kin.model_limits[arm]
        rows = []
        for index in range(Y.ARM_JOINT_COUNT):
            for step in PARITY_OFFSETS_RAD[1 if index else 0:]:
                row = reading + offsets(kin, arm)
                row[index] = min(max(row[index] + step, float(limits[index, 0])), float(limits[index, 1]))
                rows.append(row)
        base = np.linalg.inv(base_world(kin, arm))
        station = [base @ kin.fk(arm, row - offsets(kin, arm)) for row in rows]
        planned = client.forward_kinematics([row.tolist() for row in rows])
        samples.extend({"joints": row.tolist(), "simulator": s.tolist(), "planner": pose_matrix(p[:3], p[3:]).tolist()}
                       for row, s, p in zip(rows, station, planned))
        kin.set_state({arm: reading})
    return parity(samples, *ROBOT_TOLERANCES[ROBOT])


# ---------------------------------------------------------------- world
def _state_key(session: Any) -> tuple[Any, ...]:
    rig = session.rig
    return (session.round_index, session.control_ticks,
            *(tuple(np.round(rig.arm_q(arm), 6)) for arm in Y.ARMS),
            *(round(float(rig.finger_m(arm)), 6) for arm in Y.ARMS),
            *(round(float(session.finger_targets.get(arm, Y.FINGER_OPEN_M)), 6) for arm in Y.ARMS))


def _cameras(session: Any, planner: SharedPlanner, key: Any) -> dict[str, dict[str, Any]]:
    """The calibrated public cameras' depth at this state, read once for both arms' worlds."""

    if planner.cameras is None or planner.cameras[0] != key:
        cameras = {}
        # A real rig places its wrist cameras by the station model: at the measured joints, whatever was asked of it.
        session.kinematics.set_state({arm: session.rig.arm_q(arm) for arm in Y.ARMS}, session.finger_state())
        for name, data in session.rig.cameras().items():
            cam2world = data.get("cam2world_cv")
            if cam2world is None:      # an uncalibrated camera says nothing of where its depth is
                continue
            depth = np.asarray(data["depth"], dtype=np.float32)
            cameras[str(name)] = {"depth_m": np.where(np.isfinite(depth), depth, 0.0).astype(np.float32),
                                  "intrinsic_cv": np.asarray(data["intrinsic_cv"], dtype=np.float64),
                                  "cam2world_cv": np.asarray(cam2world, dtype=np.float64),
                                  "depth_range_m": DEPTH_RANGE_M}
        if not cameras:
            raise Unplanned("no calibrated public camera to build the world from")
        planner.cameras = (key, cameras)
    return planner.cameras[1]


def world_input(session: Any, arm: str, start: np.ndarray, *, blank: bool = False) -> dict[str, str]:
    """``arm``'s world snapshot of the current physical state, planned from ``start`` (the model angles
    ``inside_limits`` gives the measured joints); ``blank``: the same without any depth."""

    planner = active(session)
    kin, rig = session.kinematics, session.rig
    key = _state_key(session)
    record = planner.snapshots.get((key, arm, blank))
    if record is not None:
        return record
    if blank:
        cameras = {name: {"depth_m": np.zeros((1, 1), np.float32), "intrinsic_cv": np.eye(3),
                          "cam2world_cv": np.eye(4), "depth_range_m": DEPTH_RANGE_M}
                   for name in ("overhead", "front")}
    else:
        cameras = _cameras(session, planner, key)
    other = "right" if arm == "left" else "left"
    closed = float(session.finger_targets.get(arm, Y.FINGER_OPEN_M)) <= Y.FINGER_CLOSED_M + 1e-6
    planner.count += 1
    record = write_world_inputs(
        planner.root / "world-inputs" / f"{planner.count:05d}-{arm}{'-blank' if blank else ''}.npz",
        cameras=cameras, base_pose_world_xyzw=pose_xyz_xyzw(base_world(kin, arm)),
        joint_positions=[float(value) for value in start],
        gripper_finger_qpos=[float(rig.finger_m(arm))] * 2, gripper_directive=1.0 if closed else -1.0,
        other_arm={"joint_positions": (rig.arm_q(other) + offsets(kin, other)).tolist(),
                   "base_pose_world_xyzw": pose_xyz_xyzw(base_world(kin, other))})
    planner.snapshots = {name: value for name, value in planner.snapshots.items() if name[0] == key}
    planner.snapshots[(key, arm, blank)] = record
    return record


def _world(record: Mapping[str, str]) -> dict[str, Any]:
    return {"model": WORLD_MODEL, "input": dict(record)}


def _start(model: Sequence[float]) -> dict[str, Any]:
    return {"joint_names": JOINT_NAMES, "joint_positions": [float(value) for value in model]}


def _evidence(planner: SharedPlanner, kind: str, record: Mapping[str, Any]) -> None:
    with (planner.root / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=float) + "\n")


def _ask(planner: SharedPlanner, kind: str, method: str, request: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """One exchange with the worker; an error or a broken worker is ``Unplanned``, recorded either way."""

    try:
        answer = getattr(planner.client, method)(request)
    except MotionPlannerError as error:
        _evidence(planner, kind, {"arm": arm, "error": str(error)})
        raise Unplanned(str(error)) from None
    _evidence(planner, kind, {"arm": arm, **{key: answer.get(key) for key in (
        "status", "reason", "world", "planner", "timing_s", "first_blocked_row", "link", "penetration_m")
        if key in answer}, **({"queries": len(request["queries"])} if "queries" in request else {})})
    if answer["status"] == "error":
        error = ((answer.get("detail") or {}).get("error") or {})
        _evidence(planner, kind, {"arm": arm, "error": error.get("message"), "error_type": error.get("type")})
        raise Unplanned(f"the shared planner answered an error: {error.get('message') or answer.get('reason')}")
    return answer


# ---------------------------------------------------------------- reach
def solves(session: Any, arm: str, poses: Sequence[np.ndarray]) -> list[bool]:
    """Whether the shared IK reaches each world TCP pose from the measured joints, the depth left out (kinematics)."""

    planner = active(session)
    kin = session.kinematics
    start, _moved = inside_limits(kin, arm, session.rig.arm_q(arm) + offsets(kin, arm))
    record = world_input(session, arm, start, blank=True)
    out: list[bool] = []
    for first in range(0, len(poses), MAXIMUM_IK_QUERIES):
        chunk = poses[first:first + MAXIMUM_IK_QUERIES]
        answer = _ask(planner, "ik", "solve_ik", {
            "schema": IK_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(start),
            "queries": [{"tool_frame": TOOL_FRAME, "pose_base_xyzw": in_base(kin, arm, pose), "from_index": None}
                        for pose in chunk],
            "world": _world(record), "held_object": {"source": "none"}}, arm)
        out.extend(bool(result["solved"]) for result in answer["results"])
    return out


def highest_reachable_z(session: Any, arm: str, target: np.ndarray, floor: float) -> float | None:
    """``motion.highest_reachable_z`` by the shared IK: the highest height, 1 cm steps down to ``floor``, that solves."""

    heights = list(np.arange(target[2, 3] - 0.01, floor - 1e-9, -0.01))
    poses = []
    for z in heights:
        lower = target.copy()
        lower[2, 3] = z
        poses.append(lower)
    solved = solves(session, arm, poses) if poses else []
    return next((round(float(z), 3) for z, ok in zip(heights, solved) if ok), None)


def nearest_reachable_position(session: Any, arm: str, candidates: Sequence[np.ndarray]) -> list[float] | None:
    """The first of ``candidates`` (``motion.nearest_candidates``' order) the shared IK solves, as a position."""

    solved = solves(session, arm, candidates) if candidates else []
    return next(([round(float(value), 3) for value in pose[:3, 3]] for pose, ok in zip(candidates, solved) if ok),
                None)


# ---------------------------------------------------------------- paths
def _rows(kin: Any, arm: str, rows: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Planner rows (model angles) as encoder readings."""

    return [np.asarray(row, dtype=float) - offsets(kin, arm) for row in rows]


def plan_pose(session: Any, arm: str, q_start: np.ndarray, target: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """A collision-free path from ``q_start`` (readings) to the world TCP pose ``target``: the readings, the start
    first, and the planner's record for ``motion.planned_by``. Raises ``Refused`` or ``Unplanned``."""

    planner = active(session)
    kin = session.kinematics
    model = np.asarray(q_start, dtype=float) + offsets(kin, arm)
    start, moved = inside_limits(kin, arm, model)
    answer = _ask(planner, "plans", "plan", {
        "schema": PLAN_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(start),
        "goal": {"tool_frame": TOOL_FRAME, "pose_base_xyzw": in_base(kin, arm, target)},
        "world": _world(world_input(session, arm, start)), "held_object": {"source": "world"}, "constraints": []},
        arm)
    record = _record(answer, "plan", moved)
    if answer["status"] != "planned":
        raise Refused(CODES[answer["reason"]], record=record)
    rows = _rows(kin, arm, answer["trajectory"]["positions"])
    return [np.asarray(q_start, dtype=float), *rows], record


def _knots(current: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    translation, rotation = pose_error(pose_xyz_xyzw(target), pose_xyz_xyzw(current))
    count = max(1, math.ceil(max(translation / LINE_STEP_M, rotation / LINE_STEP_RAD)))
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([current[:3, :3], target[:3, :3]]))
    out = []
    for index in range(1, count + 1):
        s = index / count
        pose = np.eye(4)
        pose[:3, :3] = slerp([s]).as_matrix()[0]
        pose[:3, 3] = (1 - s) * current[:3, 3] + s * target[:3, 3]
        out.append(pose)
    return out


def plan_line(session: Any, arm: str, q_start: np.ndarray, target: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """The station's straight line from ``q_start`` (readings) to ``target`` by the shared IK, checked in the world:
    the readings, the start first, and the planner's record. Raises ``Refused`` or ``Unplanned``."""

    planner = active(session)
    kin = session.kinematics
    model = np.asarray(q_start, dtype=float) + offsets(kin, arm)
    start, moved = inside_limits(kin, arm, model)
    knots = _knots(kin.fk(arm, np.asarray(q_start, dtype=float)), target)
    if len(knots) > MAXIMUM_IK_QUERIES:
        raise Refused("ik_unreachable", waypoint_index=0)
    answer = _ask(planner, "ik", "solve_ik", {
        "schema": IK_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(start),
        "queries": [{"tool_frame": TOOL_FRAME, "pose_base_xyzw": in_base(kin, arm, pose),
                     "from_index": None if index == 0 else index - 1} for index, pose in enumerate(knots)],
        "world": _world(world_input(session, arm, start, blank=True)), "held_object": {"source": "none"}}, arm)
    record = _record(answer, "line", moved)
    results = answer["results"]
    failed = next((index for index, result in enumerate(results) if not result["solved"]), None)
    if failed is not None:
        error = results[failed]["position_error_m"]
        raise Refused("ik_unreachable", waypoint_index=0, knot=failed, position_error_m=error, record=record)
    rows = [start, *(np.asarray(result["joint_positions"], dtype=float) for result in results)]
    for index, (before, after) in enumerate(zip(rows, rows[1:])):
        step = float(np.max(np.abs(after - before)))
        if step > BRANCH_JUMP_RAD:
            raise Refused("ik_branch_jump", waypoint_index=0, knot=index, joint_step_rad=step, record=record)
    checked = _ask(planner, "checks", "check_path", {
        "schema": CHECK_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(start),
        "path": [row.tolist() for row in rows], "world": _world(world_input(session, arm, start)),
        "allowed_links": list(GRIPPER_LINKS), "tolerance_m": LINE_TOLERANCE_M}, arm)
    record["check"] = {key: checked.get(key) for key in ("status", "first_blocked_row", "link", "penetration_m")}
    if checked["status"] == "blocked":
        raise Refused("path_blocked", waypoint_index=0, knot=int(checked["first_blocked_row"]), record=record)
    return [np.asarray(q_start, dtype=float), *_rows(kin, arm, rows)], record


def _record(answer: Mapping[str, Any], route: str, moved: list[dict[str, Any]]) -> dict[str, Any]:
    """An arm's ``motion.planned_by`` entry: what planned it, where, how long, and what it said."""

    world = answer.get("world") or {}
    seconds = (answer.get("timing_s") or {}).get("total")
    return {"implementation": (answer.get("planner") or {}).get("implementation"), "route": route,
            "world_model": WORLD_MODEL, "status": answer.get("status"), "reason": answer.get("reason"),
            "world_sha256": world.get("sha256") if route == "plan" else None,
            "solve_s": None if seconds is None else float(seconds),
            **({"start_inside_limits": moved} if moved else {})}


def station_record(reason: str) -> dict[str, Any]:
    """``motion.planned_by`` of an arm the station planned itself although the switch is on."""

    return {"implementation": STATION_IMPLEMENTATION, "reason": reason}


__all__ = ["CODES", "GRIPPER_LINKS", "PREFLIGHT_CHECK", "ROBOT", "Refused", "STATION_IMPLEMENTATION",
           "SharedPlanner", "Unplanned", "active", "base_world", "close", "enabled", "fk_parity", "highest_reachable_z",
           "inside_limits", "located", "nearest_reachable_position", "plan_line", "plan_pose", "solves", "start",
           "station_record", "world_input"]
