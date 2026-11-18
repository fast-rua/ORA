"""RoboLab's place on the shared motion base: planning, the world, IK, timing.

RoboLab supplies what a backend supplies, its robot, frames and executor; the
rest are the services every backend shares:

* **Planner.** One ``MotionPlannerClient`` (``runtime.motion_planner``), a process
  of its own on this worker's card, plans for the Droid
  (``franka_robotiq``, ``planners/curobo_v2/robots.py``). The worker's environment holds no
  planner variables, and ``factory.py`` passes ``runtime`` as it is, so the
  interpreter and source come from the check preflight recorded for this run
  (``binding``), as ``motion_timing.bind_preflight`` does for timing. A direct
  session names them in ``runtime.motion_planner`` itself.
* **World.** ``public_depth_voxel_v1``: the depth of both cameras as the last
  observation published them. A snapshot per observation; an arm that moved since
  has none, and the answer is ``unavailable`` (``stale_observation``).
* **FK parity.** At start the planner's forward kinematics of the measured joints
  must land on the simulator's TCP (``fk_parity``), or the session refuses the planner.
* **Timing.** ``motion_pace.pace_path`` at the 15 Hz native action, wrist-checked on
  ``panda_joint7``. Without a timing worker the documented fallback applies. A pose
  move whose path the wrist check refuses runs its jaw twin when the twin plans and
  paces and the jaws are open (``pace_or_twin``, ``controllers/jaw_flip_retry.py``).

A planner that is absent, or cannot answer, never stops a move: the caller falls
back to the Cartesian servo. A planner that refuses does: nothing moves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.pose import matrix_quaternion
from ora.environment.controllers import jaw_flip_retry, motion_pace, motion_timing, robolab_droid as D
from ora.environment.controllers.kept_plans import KeptPlans
from ora.environment.controllers.depth_world import WORLD_MODEL, pose_matrix, write_world_inputs
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity
from ora.environment.controllers.motion_planner import (
    IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, planner_client, planning_diagnostics,
)
from ora.environment.controllers.motion_worker import MotionPlannerError

from .common import RobolabError, pose_error

ROBOT = "franka_robotiq"
TOOL_FRAME = "panda_hand"
JOINT_NAMES = list(D.ARM_JOINT_NAMES)
#: The arm is the Panda the shared timing table knows.
TIMING_ROBOT = "franka_panda"
WRIST_JOINT_INDEX = 6
#: ``panda_joint7``'s limit in RoboLab's USD (+/-166.003 degrees).
WRIST_LIMITS_RAD = (-2.8973, 2.8973)
#: The arm's root is the frame the cameras' ``cam2world_cv`` and every pose are in.
ROOT_POSE_XYZW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
PREFLIGHT_CHECK = "shared_motion_planner"


def _transform(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> Any:
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = np.asarray(rotation, dtype=np.float64), translation
    return matrix


#: The TCP in ``panda_hand``: flange in hand, then TCP in flange (``robolab_droid``).
HAND_FROM_TCP = _transform(D.HAND_FROM_FLANGE_ROTATION, D.HAND_FROM_FLANGE_TRANSLATION_M) @ _transform(
    np.asarray(D.TCP_AXES_IN_FLANGE, dtype=np.float64).T, D.TCP_OFFSET_FROM_FLANGE_M)
TCP_FROM_HAND = np.linalg.inv(HAND_FROM_TCP)


def _pose(matrix: Any) -> list[float]:
    return [*(float(value) for value in matrix[:3, 3]), *matrix_quaternion(matrix[:3, :3].tolist())]


def hand_from_tcp_pose(tcp_xyzw: Sequence[float]) -> list[float]:
    """The ``panda_hand`` pose in the root frame that puts the TCP at ``tcp_xyzw``."""
    return _pose(pose_matrix([float(value) for value in tcp_xyzw]) @ TCP_FROM_HAND)


def tcp_from_hand_pose(hand_xyzw: Sequence[float]) -> list[float]:
    """The TCP pose in the root frame of a ``panda_hand`` pose."""
    return _pose(pose_matrix([float(value) for value in hand_xyzw]) @ HAND_FROM_TCP)


def binding(config: Mapping[str, Any], output_root: Path) -> dict[str, Any] | None:
    """The run's planner: ``runtime.motion_planner`` with its interpreter and source, or None without one."""
    planner = config.get("motion_planner")
    if not isinstance(planner, Mapping):
        return None
    if "python_executable" in planner:
        return dict(planner)
    record = Path(output_root).parent / "preflight.json"
    checks = json.loads(record.read_text(encoding="utf-8")).get("checks") if record.is_file() else ()
    matches = [item for item in checks or () if isinstance(item, Mapping) and item.get("check") == PREFLIGHT_CHECK]
    detail = matches[0].get("detail") if len(matches) == 1 and matches[0].get("passed") is True else None
    if not isinstance(detail, Mapping):
        raise RobolabError("the shared motion planner's preflight binding is absent")
    return {**planner, "python_executable": str(detail["python_executable"]), "root": str(detail["root"])}


def start(session: Any) -> None:
    """Bind this run's timing and start its planner, if its runtime names one."""
    session.motion_planner = None
    motion_timing.bind_preflight(session.output_root)
    config = binding(session.config, session.output_root)
    if config is None:
        return
    root = session.output_root / "motion-planner"
    client = planner_client({**config, "robot": ROBOT}, python_executable=Path(config["python_executable"]),
                            root=Path(config["root"]), output_root=root)
    started = client.start()
    session.motion_planner = KeptPlans(client)
    joints = session.bridge.arm_qpos()
    measured = pose_matrix(session.bridge.tcp_pose())
    planned = pose_matrix(tcp_from_hand_pose(client.forward_kinematics([joints])[0]))
    report = parity([{"joints": joints, "simulator": measured.tolist(), "planner": planned.tolist()}],
                    *ROBOT_TOLERANCES[ROBOT])
    (root / "binding.json").write_text(json.dumps({**started, "fk_parity": report}, sort_keys=True, indent=2) + "\n",
                                       encoding="utf-8")
    if not report["passed"]:
        raise RobolabError(f"the planner's forward kinematics differs from the simulator's: {report}")


def close(session: Any) -> None:
    client = getattr(session, "motion_planner", None)
    if client is not None:
        client.close()
    session.motion_planner = None
    motion_timing.close()


def _world_input(session: Any) -> dict[str, str] | None:
    """The last observation's world snapshot, or None when the arm moved since."""
    root = session.last_observation_root
    if root is None or session.last_observation_clock != session.bridge.clock():
        return None
    name = str(session.last_observation_id).replace(":", "-")
    if session.world_input_cache is not None and session.world_input_cache[0] == name:
        return session.world_input_cache[1]
    cameras = {}
    for camera in ("agentview", "wrist"):
        calibration = json.loads((Path(root) / camera / "calibration.json").read_text(encoding="utf-8"))
        cameras[camera] = {"depth_m": np.load(Path(root) / camera / "depth-m.npy", allow_pickle=False),
                           "intrinsic_cv": calibration["intrinsic_cv"], "cam2world_cv": calibration["cam2world_cv"],
                           "depth_range_m": (0.0, 1000.0)}
    gap = session.bridge.jaw_separation_m()
    record = write_world_inputs(
        session.output_root / "motion-planner/world-inputs" / f"{name}.npz", cameras=cameras,
        base_pose_world_xyzw=ROOT_POSE_XYZW, joint_positions=session.bridge.arm_qpos(),
        gripper_finger_qpos=[gap / 2.0, gap / 2.0], gripper_directive=1.0 if session.gripper_target == "close" else -1.0)
    session.world_input_cache = (name, record)
    return record


def _evidence(session: Any, kind: str, record: Mapping[str, Any]) -> None:
    root = session.output_root / "motion-planner"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def global_route(interaction: str, current: Sequence[float], target: Sequence[float]) -> bool:
    """Whether a move plans: a contact approach and a short move stay on the servo, as on LIBERO."""
    if interaction == "contact_approach":
        return False
    translation, rotation = pose_error(target, current)
    return translation > D.LOCAL_TRANSLATION_M or rotation > D.LOCAL_ROTATION_RAD


def plan(session: Any, pose: Sequence[float]) -> dict[str, Any]:
    """A path to the TCP pose from the measured joints: ``planned``, ``refused`` or ``unavailable``."""
    client = getattr(session, "motion_planner", None)
    if client is None:
        return {"status": "unavailable", "reason": "no_planner"}
    record = _world_input(session)
    if record is None:
        return {"status": "unavailable", "reason": "stale_observation"}
    joints = session.bridge.arm_qpos()
    request = {"schema": PLAN_REQUEST_SCHEMA, "robot": ROBOT,
               "start": {"joint_names": JOINT_NAMES, "joint_positions": joints},
               "goal": {"tool_frame": TOOL_FRAME, "pose_base_xyzw": hand_from_tcp_pose(pose)},
               "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}, "constraints": []}
    try:
        answer = client.plan(request)
    except MotionPlannerError as error:
        _evidence(session, "plans", {"error": str(error)})
        return {"status": "unavailable", "reason": "planner_error"}
    _evidence(session, "plans", {key: answer.get(key) for key in (
        "status", "reason", "world", "planner", "timing_s", "start_tool_pose_base_xyzw")})
    if answer["status"] == "error":
        return {"status": "unavailable", "reason": str(answer["reason"])}
    if answer["status"] != "planned":
        return {"status": "refused", "reason": answer["reason"], "planning_diagnostics": planning_diagnostics(answer)}
    return {"status": "planned", "reason": None, "start_joints": joints,
            "path_joints": answer["trajectory"]["positions"], "world_sha256": (answer.get("world") or {}).get("sha256")}


def forward_tcp(session: Any, rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """The TCP pose of joint rows, by the planner's forward kinematics."""
    poses = session.motion_planner.forward_kinematics([[float(value) for value in row] for row in rows])
    return [tcp_from_hand_pose(pose) for pose in poses]


def pace(session: Any, planned: Mapping[str, Any], speed: Mapping[str, Any] | None) -> motion_pace.PacedPath:
    """A planned path timed at the native action; raises ``PaceRefusal`` before anything moves."""
    rows = [list(planned["start_joints"]), *planned["path_joints"]]
    left = max(1, session.extended_native_action_limit - session.native_actions)
    paced = motion_pace.pace_path(
        rows, lambda joints: forward_tcp(session, joints), speed=speed, tick_s=D.CONTROL_PERIOD_S,
        joint_speed_ceiling_rad_s=D.JOINT_SPEED_CEILING_RAD_S, joint_accel_ceiling_rad_s2=D.JOINT_ACCEL_CEILING_RAD_S2,
        wrist_joint_index=WRIST_JOINT_INDEX, wrist_limits_rad=WRIST_LIMITS_RAD,
        budget_ticks=min(left, D.PACE_BUDGET_TICKS), robot=TIMING_ROBOT)
    report = paced.report()
    _evidence(session, "timing", {"timing": report.get("timing"), "ticks": report["planned_ticks"]})
    return paced


def pace_or_twin(session: Any, planned: Mapping[str, Any], pose: Sequence[float], speed: Mapping[str, Any] | None,
                 *, twin: bool) -> tuple[motion_pace.PacedPath, list[float], dict[str, Any] | None]:
    """``pace`` of a planned path to the TCP ``pose``, or of the path to its jaw twin when the wrist check refuses it.

    The twin (``controllers/jaw_flip_retry.py``: the Robotiq turned a half-turn about its approach, TCP +z) is tried
    for a pose move (``twin``) with the jaws commanded open, planned and paced as the request was. Returns the paced
    path, the TCP pose it ends at (the request's or the twin's) and the retry's record (None: no twin ran), which the
    path's wrist report then carries; raises the original ``PaceRefusal`` when the twin does not plan and pace either.
    A real call and its dry run both ask this."""
    try:
        return pace(session, planned, speed), list(pose), None
    except motion_pace.PaceRefusal as refusal:
        def admit(goal: list[float]) -> Any:
            again = plan(session, goal)
            return pace(session, again, speed) if again["status"] == "planned" else str(again["reason"])

        retried = None if not twin else jaw_flip_retry.retry(
            pose, refusal, admit, jaws_open=lambda: session.finger_action == D.GRIPPER_OPEN_ACTION)
        if retried is None:
            raise
        goal, paced, record = retried
        return jaw_flip_retry.recorded(paced, record), goal, record


def solve_ik(session: Any, poses: Sequence[Sequence[float]],
             starts: Sequence[int | None]) -> dict[str, Any] | None:
    """One IK answer for TCP poses, chained by ``starts``; None when no planner or world can answer."""
    client = getattr(session, "motion_planner", None)
    record = None if client is None else _world_input(session)
    if record is None:
        return None
    try:
        answer = client.solve_ik({
            "schema": IK_REQUEST_SCHEMA, "robot": ROBOT,
            "start": {"joint_names": JOINT_NAMES, "joint_positions": session.bridge.arm_qpos()},
            "queries": [{"tool_frame": TOOL_FRAME, "pose_base_xyzw": hand_from_tcp_pose(pose), "from_index": start}
                        for pose, start in zip(poses, starts)],
            "world": {"model": WORLD_MODEL, "input": record}, "held_object": {"source": "world"}})
    except MotionPlannerError as error:
        _evidence(session, "ik", {"error": str(error)})
        return None
    return None if answer["status"] != "solved" else answer


def reachable(session: Any, pose: Sequence[float]) -> bool | None:
    """Whether the shared IK solves a pose for the servo's route; None when it cannot answer."""
    answer = solve_ik(session, [pose], [None])
    return None if answer is None else bool(answer["results"][0]["solved"])


__all__ = ["HAND_FROM_TCP", "JOINT_NAMES", "PREFLIGHT_CHECK", "ROBOT", "TCP_FROM_HAND", "TOOL_FRAME",
           "WRIST_JOINT_INDEX", "WRIST_LIMITS_RAD", "binding", "close", "forward_tcp", "global_route",
           "hand_from_tcp_pose", "pace", "pace_or_twin", "plan", "reachable", "solve_ik", "start",
           "tcp_from_hand_pose"]
