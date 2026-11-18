"""RoboDojo's place on the shared motion base: planning, the world, timing, and the path check.

The toolkit lib plans in terms of ``ora.kinematics-request.v1``; this module
answers its ``plan_path``, ``pace_path`` and ``contact_path`` with the services
every backend shares, so RoboDojo supplies only what a backend supplies: its
robot description, its frames and its executor.

* **Planner.** One ``MotionPlannerClient`` (``runtime.motion_planner``), a
  process of its own on this worker's card, plans for the ARX X5 from the
  SDK's own robot file. The SDK's planners stay for the SDK's IK: they cannot
  update their world, and they answer ``ik_chain``. Both arms share one planner
  worker: each request is in the moving arm's base frame.
* **World.** ``public_depth_voxel_v1`` for the X5 (``depth_world.py``): the depth of
  the head and both wrist cameras exactly as the last observation published
  them, with the other arm's spheres at its measured joints standing in the
  world. A snapshot per observation and arm; an arm that moved since that
  observation has no snapshot, and the answer is ``unavailable`` (``stale_observation``).
* **FK parity.** At start, the planner's forward kinematics of both arms'
  measured joints must land on the simulator's link6 (``fk_parity``), or the
  session refuses the planner: every frame below relies on it.
* **Timing.** ``motion_pace.pace_path``, with the timing worker the run's
  preflight bound (``motion_timing.bind_preflight``), one row per native
  action. A run without it takes the documented fallback, and the evidence says so.
* **Prebuild.** When an observation is published (``prebuild``), both arms'
  world snapshots are written and the planner is asked, from a thread of its
  own and one arm at a time (the arm that just moved first), to build their
  scenes while the policy decides; the first request after the observation then
  finds its arm's scene built instead of paying the depth world (most of a 2.2 s
  reach check, 2026-11-06). The answers are the same: only when the scene is
  built moves. Each prebuild is a line of ``motion-planner/prebuild.jsonl``; a
  failed or superseded one leaves every request to build as before.

Answers carry no planner or timing library's name: what the toolkit records
holds neutral statuses and joint rows, and the full answers stay in
``motion-planner/`` beside the run's other private evidence.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers import motion_pace, motion_timing
from ora.environment.controllers.depth_world import WORLD_MODEL, pose_matrix, write_world_inputs
from ora.environment.controllers.motion_planner import IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, ROBOTS, planner_client
from ora.environment.controllers.motion_planner import CHECK_REQUEST_SCHEMA, refusal_message
from ora.environment.controllers.motion_worker import MotionPlannerError
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity

from . import reachability
from .bridge import RobodojoError

ROBOT = "arx_x5"
TOOL_FRAME = ROBOTS[ROBOT]["tool_frame"]
JOINT_NAMES = list(ROBOTS[ROBOT]["joint_names"])
WRIST_JOINT_INDEX = 5
#: X5A.urdf's joint6 limit (``lib.motion_plan.checks.JOINT6_LIMIT_RAD``).
WRIST_LIMITS_RAD = (-3.14, 3.14)
NATIVE_ACTION_S = 0.04
#: The links a deliberate contact may touch the world with: the two fingers.
FINGER_LINKS = ("link7", "link8")
#: How far any other link may lie inside the world: the world's own 1 cm voxel.
CONTACT_TOLERANCE_M = 0.01
#: How long ``close`` waits for a prebuild still asking the planner.
PREBUILD_JOIN_S = 30.0


def start(session: Any) -> None:
    """Bind this run's timing and start its planner, if its RunSpec names one."""
    session.motion_planner = None
    motion_timing.bind_preflight(session.output_root)
    config = session.config.get("motion_planner")
    if not isinstance(config, Mapping):
        return
    robot = next(iter(session.robots.values()))
    root = session.output_root / "motion-planner"
    client = planner_client(config, python_executable=Path(config["python_executable"]), root=Path(config["root"]),
                            output_root=root, robot_file=Path(robot.curobo_yml_path))
    binding = client.start()
    session.motion_planner = client
    samples = []
    for arm, measured in session._proprioception()["end_effectors"].items():
        joints = [float(value) for value in measured["arm_joint_qpos"]]
        base = pose_matrix(base_pose_xyzw(session, arm))
        pose = pose_matrix(reachability.fk_link6(session, arm, joints))
        samples.append({"joints": joints, "simulator": (np.linalg.inv(base) @ pose).tolist(),
                        "planner": pose_matrix(client.forward_kinematics([joints])[0]).tolist()})
    report = parity(samples, *ROBOT_TOLERANCES[ROBOT])
    (root / "binding.json").write_text(json.dumps({**binding, "fk_parity": report}, sort_keys=True, indent=2) + "\n",
                                       encoding="utf-8")
    if not report["passed"]:
        raise RobodojoError(f"the planner's forward kinematics differs from the simulator's: {report}")


def close(session: Any) -> None:
    join_prebuild(session, PREBUILD_JOIN_S)
    client = getattr(session, "motion_planner", None)
    if client is not None:
        client.close()
    session.motion_planner = None


def prebuild(session: Any) -> None:
    """Have the planner build both arms' scenes of the observation just published, off the control path.

    The snapshots are written here, on the simulator's thread (they read its state); the planner's exchanges run on a
    thread of their own, one arm at a time and the arm that just moved first, so a request that comes before both
    are built waits for at most one; the client serializes them with the requests, which then find their scene kept.
    A newer observation supersedes the arms not yet asked. Nothing is asked of a session without a running planner,
    or for an arm with no current snapshot."""
    client = getattr(session, "motion_planner", None)
    if client is None or getattr(client, "_process", None) is None:
        return
    observation = getattr(session, "last_observation_id", None)
    worlds: dict[str, dict[str, Any]] = {}
    try:
        for arm in session.robots:
            record, joints = world_input(session, arm)
            if record is None:
                return
            worlds[arm] = {"start": {"joint_names": JOINT_NAMES, "joint_positions": joints},
                           "world": {"model": WORLD_MODEL, "input": record}}
    except Exception as error:  # noqa: BLE001 - the observation stands; a request writes its snapshot as before
        _evidence(session, "prebuild", {"observation_id": observation, "arms": [], "status": "error",
                                        "reason": "snapshot_failed", "error": f"{type(error).__name__}: {error}"})
        return
    previous = getattr(session, "_prebuild_joints", None) or {}
    session._prebuild_joints = {arm: item["start"]["joint_positions"] for arm, item in worlds.items()}
    order = sorted(worlds, key=lambda arm: previous.get(arm) == worlds[arm]["start"]["joint_positions"])
    thread = threading.Thread(target=_prepare, args=(session, client, [(arm, worlds[arm]) for arm in order],
                                                     observation), daemon=True, name="planner-prebuild")
    session._planner_prebuilds = [*(item for item in getattr(session, "_planner_prebuilds", ()) if item.is_alive()),
                                  thread]
    thread.start()


def _prepare(session: Any, client: Any, worlds: list[tuple[str, dict[str, Any]]], observation: Any) -> None:
    started, arms, status = time.monotonic(), [], "prepared"
    for arm, world in worlds:
        if getattr(session, "last_observation_id", None) != observation:
            status = "superseded"
            break
        try:
            answer = client.prepare([world])
            arms.append({"arm": arm, "status": answer.get("status"), "reason": answer.get("reason"),
                         "timing_s": answer.get("timing_s")})
            if answer.get("status") != "prepared":
                status = "error"
        except Exception as error:  # noqa: BLE001 - a prebuild is a head start, never an answer
            arms.append({"arm": arm, "status": "error", "reason": "prepare_failed",
                         "error": f"{type(error).__name__}: {error}"})
            status = "error"
            break
    _evidence(session, "prebuild", {"observation_id": observation, "status": status, "arms": arms,
                                    "wall_s": round(time.monotonic() - started, 6)})


def join_prebuild(session: Any, timeout_s: float | None = None) -> None:
    """Wait for the session's prebuilds still asking the planner."""
    for thread in getattr(session, "_planner_prebuilds", ()):
        thread.join(timeout_s)
    session._planner_prebuilds = [item for item in getattr(session, "_planner_prebuilds", ()) if item.is_alive()]


def base_pose_xyzw(session: Any, arm: str) -> list[float]:
    """The arm base in the episode-local world frame, ``[x, y, z, qx, qy, qz, qw]`` (``fk_link6``'s base)."""
    origin = [float(value) for value in session.robots[arm].entity_origin_pose]
    return [*origin[:3], *_unit([*origin[4:], origin[3]])]


def _unit(quaternion: Sequence[float]) -> list[float]:
    return (np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)).tolist()


def _link6(session: Any, arm: str, rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """The simulator's link6 pose of each joint row, with a unit quaternion (its FK is single precision)."""
    poses = [reachability.fk_link6(session, arm, row) for row in rows]
    return [[*pose[:3], *_unit(pose[3:])] for pose in poses]


def _finger_gap_qpos(session: Any, arm: str) -> list[float]:
    manager = session.environment.robot_manager
    fingers = np.asarray(manager.get_end_effector_real_val(session.robots[arm], env_idx_list=[0])[0], dtype=float)
    return [max(float(value), 0.0) for value in fingers.reshape(-1)[:2]]


def _gripper_directive(session: Any, arm: str) -> float:
    """Positive while the jaws are commanded closed (``depth_world``'s convention)."""
    opening = session.gripper_directives.get(arm)
    return 1.0 if opening is not None and float(opening) < 0.5 else -1.0


def world_input(session: Any, arm: str) -> tuple[dict[str, str] | None, list[float]]:
    """This observation's world snapshot for ``arm`` and its measured joints; no snapshot when it is stale."""
    effectors = session._proprioception()["end_effectors"]
    joints = [float(value) for value in effectors[arm]["arm_joint_qpos"]]
    root = getattr(session, "last_observation_root", None)
    if root is None or getattr(session, "last_observation_clock", None) != session._clock():
        return None, joints
    path = session.output_root / "motion-planner/world-inputs" / f"{session.last_observation_id}-{arm}.npz".replace(
        ":", "-")
    cached = getattr(session, "_world_inputs", None) or {}
    if str(path) in cached:
        return cached[str(path)], joints
    cameras = {}
    for camera in sorted(item.name for item in Path(root).iterdir() if (item / "depth-m.npy").is_file()):
        calibration = json.loads((Path(root) / camera / "calibration.json").read_text(encoding="utf-8"))
        cameras[camera] = {"depth_m": np.load(Path(root) / camera / "depth-m.npy", allow_pickle=False),
                           "intrinsic_cv": calibration["intrinsic_cv"], "cam2world_cv": calibration["cam2world_cv"],
                           "depth_range_m": (0.0, 1000.0)}
    other = "right" if arm == "left" else "left"
    record = write_world_inputs(
        path, cameras=cameras, base_pose_world_xyzw=base_pose_xyzw(session, arm), joint_positions=joints,
        gripper_finger_qpos=_finger_gap_qpos(session, arm), gripper_directive=_gripper_directive(session, arm),
        other_arm={"joint_positions": [float(value) for value in effectors[other]["arm_joint_qpos"]],
                   "base_pose_world_xyzw": base_pose_xyzw(session, other)})
    session._world_inputs = {**cached, str(path): record}
    return record, joints


def _unavailable(arm: str, reason: str) -> dict[str, Any]:
    return {"arm": arm, "status": "unavailable", "reason": reason}


def _evidence(session: Any, kind: str, record: Mapping[str, Any]) -> None:
    root = session.output_root / "motion-planner"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _base_pose(session: Any, arm: str, pose_xyz_xyzw: Sequence[float]) -> list[float]:
    from ora.contracts.pose import matrix_quaternion

    local = np.linalg.inv(pose_matrix(base_pose_xyzw(session, arm))) @ pose_matrix(pose_xyz_xyzw)
    return [*local[:3, 3].tolist(), *matrix_quaternion(local[:3, :3].tolist())]


def plan_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's path from its measured joints, in the current public world."""
    arm = request["arm"]
    client = getattr(session, "motion_planner", None)
    if client is None:
        return _unavailable(arm, "no_planner")
    record, joints = world_input(session, arm)
    if record is None:
        return _unavailable(arm, "stale_observation")
    pose, goal_joints = request["goal_pose_xyz_xyzw"], request["goal_joints"]
    plan = {"schema": PLAN_REQUEST_SCHEMA, "robot": ROBOT,
            "start": {"joint_names": JOINT_NAMES, "joint_positions": joints},
            "goal": ({"joint_positions": goal_joints} if pose is None
                     else {"tool_frame": TOOL_FRAME, "pose_base_xyzw": _base_pose(session, arm, pose)}),
            "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}, "constraints": []}
    try:
        answer = client.plan(plan)
    except MotionPlannerError as error:
        _evidence(session, "plans", {"arm": arm, "error": str(error)})
        return _unavailable(arm, "planner_error")
    _evidence(session, "plans", {key: answer.get(key) for key in (
        "status", "reason", "world", "planner", "timing_s", "start_tool_pose_base_xyzw")} | {"arm": arm})
    if answer["status"] == "error":
        return _unavailable(arm, str(answer["reason"]))
    world = answer.get("world") or {}
    summary = world.get("summary") or {}
    why = refusal_message(answer)
    result = {"arm": arm, "status": answer["status"], "reason": answer["reason"], "start_joints": joints,
              "world_sha256": world.get("sha256"), "held_voxels": summary.get("held_voxels"),
              "goal_refusal": summary.get("goal_refusal") or (None if why is None else {"message": why})}
    trajectory = answer.get("trajectory")
    if trajectory is not None:
        result["path_joints"] = trajectory["positions"]
    return result


def solve_reach(session: Any, arm: str, queries: Sequence[Mapping[str, Any]]) -> tuple[dict | None, str | None]:
    """The planner's IK of each query in the world a plan to that pose would see, or why there is no answer."""
    client = getattr(session, "motion_planner", None)
    if client is None:
        return None, "no_planner"
    record, joints = world_input(session, arm)
    if record is None:
        return None, "stale_observation"
    try:
        answer = client.solve_ik({
            "schema": IK_REQUEST_SCHEMA, "robot": ROBOT,
            "start": {"joint_names": JOINT_NAMES, "joint_positions": joints},
            "queries": [{"tool_frame": TOOL_FRAME, "from_index": query["from_index"],
                         "pose_base_xyzw": _base_pose(session, arm, query["pose_xyzw"])} for query in queries],
            "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}})
    except MotionPlannerError as error:
        _evidence(session, "ik", {"arm": arm, "error": str(error)})
        return None, "planner_error"
    _evidence(session, "ik", {"arm": arm, "status": answer["status"], "timing_s": answer.get("timing_s"),
                              "queries": len(queries)})
    if answer["status"] != "solved":
        return None, "planner_error"
    return {**answer, "seed_joints": joints}, None


def pace_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's joint path timed at the native action, wrist-checked, fitted to its budget."""
    arm = request["arm"]
    try:
        paced = motion_pace.pace_path(
            request["joints_rad"], lambda rows: _link6(session, arm, rows), speed=request["speed"], tick_s=NATIVE_ACTION_S, joint_speed_ceiling_rad_s=None,
            wrist_joint_index=WRIST_JOINT_INDEX, wrist_limits_rad=WRIST_LIMITS_RAD,
            budget_ticks=request["budget_ticks"], slow_approach=request["slow_approach"],
            cut_to_budget=request["cut_to_budget"], arm=arm, robot=ROBOT, unwind_check=request["unwind_check"])
    except motion_pace.PaceRefusal as refusal:
        return {"arm": arm, "status": "refused", "reason": refusal.code, "message": str(refusal),
                "constraints": dict(refusal.constraints)}
    report = paced.report()
    _evidence(session, "timing", {"arm": arm, "timing": report.get("timing"), "ticks": report["planned_ticks"]})
    return {"arm": arm, "status": "timed", "reason": None, "path_joints": paced.rows(),
            "fractions": list(paced.knots["fractions"]), "duration_s": report["duration_s"],
            "ticks": report["planned_ticks"], "needed_ticks": int(paced.whole_ticks or report["planned_ticks"]),
            "fraction_of_move": float(paced.fraction), "slow_from_tick": paced.slow_from_knot,
            "speed": dict(report["speed"]), "wrist": dict(report["wrist"])}


def contact_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a deliberate contact's path keeps every link but the fingers out of the world."""
    arm = request["arm"]
    client = getattr(session, "motion_planner", None)
    if client is None:
        return _unavailable(arm, "no_planner")
    record, joints = world_input(session, arm)
    if record is None:
        return _unavailable(arm, "stale_observation")
    try:
        answer = client.check_path({
            "schema": CHECK_REQUEST_SCHEMA, "robot": ROBOT,
            "start": {"joint_names": JOINT_NAMES, "joint_positions": joints}, "path": request["joints_rad"],
            "world": {"model": WORLD_MODEL, "input": record}, "allowed_links": list(FINGER_LINKS),
            "tolerance_m": CONTACT_TOLERANCE_M})
    except MotionPlannerError as error:
        _evidence(session, "contact-paths", {"arm": arm, "error": str(error)})
        return _unavailable(arm, "planner_error")
    _evidence(session, "contact-paths", {key: answer.get(key) for key in (
        "status", "first_blocked_row", "link", "penetration_m", "timing_s")} | {"arm": arm})
    if answer["status"] == "error":
        return _unavailable(arm, str(answer["reason"]))
    return {"arm": arm, "status": answer["status"], "reason": None, "first_blocked_row": answer["first_blocked_row"],
            "limiting_link": answer["link"], "penetration_m": answer["penetration_m"]}


__all__ = ["CONTACT_TOLERANCE_M", "FINGER_LINKS", "PREBUILD_JOIN_S", "ROBOT", "base_pose_xyzw", "close", "contact_path",
           "join_prebuild", "pace_path", "plan_path", "prebuild", "solve_reach", "start", "world_input"]
