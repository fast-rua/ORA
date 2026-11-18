"""RoboCasa's place on the shared motion base: planning, IK and the world.

The PandaOmron's arm is the Panda the shared planner already holds: robosuite's
arm and ``PandaGripper`` on a mount, so it plans as ``franka_panda``. RoboCasa
supplies what a backend supplies: the frames and the executor.

* **Planner.** One ``MotionPlannerClient`` (``runtime.motion_planner``), a process
  of its own on the lane's card, bound by the run's preflight because this
  worker's environment is allowlisted (:func:`planner_binding`).
* **Frame.** The planner's base is ``robot0_link0``, which moves with the mobile
  base and the torso lift, so every request is built from the measured state of
  that moment: after ``move_base`` or a lift the next plan is simply in the new
  frame. At start the planner's forward kinematics must land on the simulator's
  hand (``fk_parity``), and its collision spheres must be the ones this host
  holds for it (``own_spheres_check``: the sphere set's hash, and the host's
  chain on the planner's hand), or the session refuses the planner.
* **World.** ``public_depth_voxel_v1``: the depth of the two public cameras,
  rendered at the plan's own moment exactly as an observation renders them, with
  the arm's and the held load's depth taken out by the planner's spheres. Nothing
  else of the scene enters. What the cameras cannot see is free, the Omron base
  and torso included unless they are in view; a move the world did not foresee
  stops on the executor's measured stops.
* **IK.** The shared solver. A straight line is solved every ``C.LINE_STEP_M``
  against a blank world, so the kinematics are not paid a world per cut, and the
  path is then checked in the real one (:func:`check_line`); a joint-space move
  is the planner's collision-free path.

* **Stands.** A lift or a base move asked about before it runs (the lift's admission, the base-placement search of
  ``controllers/base_placement.py``) is asked in the frame the arm mount would have there (``Stand``), in a world
  input written for that mount: the same public depth, the robot's own body and base taken out of it
  (``base_placement.without_own``), since the robot is not where the cameras saw it then. The robot's own body is the
  planner's own model of it at the measured joints (``own_body``: its collision spheres, ``robot_spheres``, plus the
  world's robot margin, the hand's box, and a held load's spheres, to which ``base_placement.public_points`` adds the
  depth linked to that load), so the scene next to the hand stays. ``move_base`` is admitted on the same depth
  (``base_path_refusal``), its footprint and the arm and load it carries.

Answers carry no planner name: refusals are the neutral reasons mapped to this
backend's public codes, and the full answers stay in ``motion-planner/`` with the
run's other private evidence.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.pose import matrix_quaternion
from ora.environment.benchmarks.robocasa import PUBLIC_CAMERAS
from ora.environment.controllers import base_placement, robocasa_omron as C, robot_spheres
from ora.environment.controllers.base_placement import Stand
from ora.environment.controllers.kept_plans import KeptPlans
from ora.environment.controllers.depth_world import (
    PUBLIC_DEPTH_VOXEL_V1, WORLD_MODEL, DepthWorldError, Scene, pose_matrix, read_world_inputs, write_world_inputs,
)
from ora.environment.controllers.fk_parity import ROBOT_TOLERANCES, parity
from ora.environment.controllers.motion_planner import (
    CHECK_REQUEST_SCHEMA, IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, ROBOTS, planner_client, planning_diagnostics,
)
from ora.environment.controllers.motion_worker import MotionPlannerError

from . import kinematics
from .common import RoboCasaError

ROBOT = "franka_panda"
TOOL_FRAME = ROBOTS[ROBOT]["tool_frame"]
JOINT_NAMES = list(ROBOTS[ROBOT]["joint_names"])
#: A deliberate contact may touch the world with these; the arm's own links may not.
GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object")
#: The planner's neutral refusals as this backend's public codes.
CODES = {
    "goal_unreachable": "target_unreachable_at_base_pose",
    "goal_in_collision": "target_in_collision_at_base_pose",
    "start_in_collision": "path_in_collision_at_base_pose",
    "start_outside_limits": "target_unreachable_at_base_pose",
    "path_blocked": "path_in_collision_at_base_pose",
}
UNAVAILABLE = "motion_planner_unavailable"
#: Joint offsets of the FK parity sample around the measured joints.
PARITY_OFFSETS_RAD = (0.0, 0.3, -0.3, 0.6)
#: What a line's refusal says of its target on its own, asked at the measured base pose and torso height.
ENDPOINT_REACHABLE, ENDPOINT_IN_WORLD, ENDPOINT_UNREACHABLE = (
    "reachable", "reachable_only_through_the_world", "unreachable")
#: The PandaOmron to the base-placement search: its footprint, its base servo, its lift and its arm's reach.
#: The lift is no stand's to choose (``torso_range_m`` None): ``move_base`` does not move it and no public action sets
#: it; ``move_pose`` lifts only a target the arm alone cannot reach. A stand keeps the measured height.
MOBILE_BASE = base_placement.MobileBase(
    footprint=tuple(base_placement.Box(*box, top_lifts=index == C.BASE_FOOTPRINT_LIFTING_BOX)
                    for index, box in enumerate(C.BASE_FOOTPRINT_BOXES_M)),
    servo=base_placement.Servo(
        linear_gain=C.BASE_LINEAR_GAIN, max_linear_m_s=C.BASE_MAX_LINEAR_SPEED_M_S, yaw_gain=C.BASE_YAW_GAIN,
        max_yaw_rad_s=C.BASE_MAX_YAW_RATE_RAD_S, tick_s=C.CONTROL_PERIOD_S,
        position_tolerance_m=C.BASE_POSITION_TOLERANCE_M, yaw_tolerance_rad=C.BASE_YAW_TOLERANCE_RAD,
        max_ticks=C.BASE_MAX_STEPS),
    torso_range_m=None, torso_step_m=C.TORSO_SEARCH_STEP_M, reach_m=C.ARM_MAX_REACH_FROM_SHOULDER_M,
    shoulder_m=C.ARM_SHOULDER_ABOVE_MOUNT_M, floor_m=C.FLOOR_HEIGHT_M)
#: The hand's own box (``OwnBody.boxes``): the planner's hand box (the hand and its open fingers with their buffer and
#: about 2 cm, which holds the real fingers at any opening) ending where its carve box does, about 1 cm past the
#: fingertips, so a surface the fingertips touch stays.
HAND_BOX_M = (*PUBLIC_DEPTH_VOXEL_V1.hand_box_m[:2],
              (PUBLIC_DEPTH_VOXEL_V1.hand_box_m[2][0], PUBLIC_DEPTH_VOXEL_V1.carve_box_tip_m))


def planner_binding(config: Mapping[str, Any], output_root: Path) -> tuple[str, str]:
    """The interpreter and source root of this run's planner: the preflight's, or the named environment."""
    record = Path(output_root).parent / "preflight.json"
    if record.is_file():
        checks = json.loads(record.read_text(encoding="utf-8")).get("checks") or ()
        matches = [item for item in checks if isinstance(item, Mapping) and item.get("check") == C.MOTION_PLANNER_PREFLIGHT_CHECK]
        detail = matches[0].get("detail") if len(matches) == 1 and matches[0].get("passed") is True else None
        if isinstance(detail, Mapping) and isinstance(detail.get("python"), str) and isinstance(detail.get("root"), str):
            return detail["python"], detail["root"]
    python, root = os.environ.get(str(config["python_env"])), os.environ.get(str(config["root_env"]))
    if not python or not root:
        raise RoboCasaError(f"no motion planner binding: {config['python_env']} and {config['root_env']} are unset "
                            "and the run's preflight names none")
    return python, root


def _matrix(pose_xyzw: Sequence[float]) -> Any:
    return pose_matrix(list(pose_xyzw))


def _pose(transform: Any) -> list[float]:
    return [*(float(value) for value in transform[:3, 3]), *matrix_quaternion(transform[:3, :3].tolist())]


def start(session: Any) -> None:
    """Start this session's planner, if its RunSpec names one, and check it against the simulator."""
    session.motion_planner = None
    config = session.planner_config
    if not isinstance(config, Mapping):
        return
    python, root_path = planner_binding(config, session.output_root)
    root = session.output_root / "motion-planner"
    client = planner_client(config, python_executable=Path(python), root=Path(root_path), output_root=root,
                            environment_overrides=C.MOTION_PLANNER_ENVIRONMENT)
    binding = client.start()
    session.motion_planner = KeptPlans(client)
    report = fk_parity(session, client)
    spheres = own_spheres_check(session, client, binding)
    (root / "binding.json").write_text(json.dumps({**binding, "fk_parity": report, "own_spheres": spheres},
                                                  sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RoboCasaError(f"the planner's forward kinematics differs from the simulator's: {report}")
    if not spheres["passed"]:
        raise RoboCasaError(f"the planner's collision spheres are not the ones this host takes for the robot: {spheres}")


def close(session: Any) -> None:
    client = getattr(session, "motion_planner", None)
    if client is not None:
        client.close()
    session.motion_planner = None


def parity_rows(joints: Sequence[float], limits: Sequence[Sequence[float]]) -> list[list[float]]:
    """The measured joints and each one moved by ``PARITY_OFFSETS_RAD``, inside its limits."""
    rows = []
    for index in range(len(joints)):
        for offset in PARITY_OFFSETS_RAD[1 if index else 0:]:
            row = [float(value) for value in joints]
            row[index] = min(max(row[index] + offset, float(limits[index][0])), float(limits[index][1]))
            rows.append(row)
    return rows


def fk_parity(session: Any, client: Any) -> dict[str, Any]:
    """The planner's hand against the simulator's, in the arm's base frame, around the measured joints."""
    io = session.io
    limits = [kinematics.joint_range(io.model, name) for name in C.ARM_JOINTS]
    rows = parity_rows(io.arm_qpos(), limits)
    samples = [{"joints": row, "simulator": simulator.tolist(), "planner": _matrix(planned).tolist()}
               for row, simulator, planned in zip(rows, kinematics.hand_in_base(io.model, io.data, rows),
                                                  client.forward_kinematics(rows))]
    return parity(samples, *ROBOT_TOLERANCES[ROBOT])


def own_spheres_check(session: Any, client: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the host's model of the planner's robot (``robot_spheres``, which takes the robot's own depth out of a
    stand's world and carries the arm along a base move) is the planner's: the sphere set it names in its ready answer
    (``collision_spheres_sha256``), and the host's chain on the planner's own hand at the FK parity rows."""
    io = session.io
    engine = ((binding.get("worker") or {}).get("ready") or {}).get("engine") or {}
    rows = parity_rows(io.arm_qpos(), [kinematics.joint_range(io.model, name) for name in C.ARM_JOINTS])
    # ``parity`` compares its two sides: here the host's chain (its "simulator") and the planner's own hand.
    samples = [{"simulator": robot_spheres.hand(row).tolist(), "planner": _matrix(planned).tolist()}
               for row, planned in zip(rows, client.forward_kinematics(rows))]
    hand = parity(samples, *ROBOT_TOLERANCES[ROBOT])
    named = engine.get("collision_spheres_sha256")
    return {"sha256": robot_spheres.SPHERE_SET_SHA256, "planner_sha256": named, "hand": hand,
            "passed": named == robot_spheres.SPHERE_SET_SHA256 and bool(hand["passed"])}


def _hand_goal(io: Any, grip_pose: Sequence[float], lift_m: float = 0.0,
               mount: Sequence[float] | None = None) -> list[float]:
    """The tool pose, arm base frame, that puts the grip site on ``grip_pose`` with the torso ``lift_m`` higher, the
    arm mount at ``mount`` (world) when given."""
    import numpy as np

    base = _matrix(io.arm_base_pose() if mount is None else mount)
    base[2, 3] += lift_m
    hand_from_grip = np.linalg.inv(_matrix(io.hand_pose())) @ _matrix(io.grip_pose())
    return _pose(np.linalg.inv(base) @ _matrix(grip_pose) @ np.linalg.inv(hand_from_grip))


def _snapshot(session: Any, *, blank: bool = False) -> dict[str, str]:
    """The world input of this moment: the public depth now, or none at all for bare kinematics."""
    import numpy as np

    io = session.io
    key = (session.round_index, io.control_steps, blank)
    cached = session.world_inputs.get(key)
    if cached is not None:
        return cached
    cameras = {public_name: {"depth_m": np.zeros((1, 1)), "intrinsic_cv": np.eye(3), "cam2world_cv": np.eye(4),
                             "depth_range_m": (0.0, 1.0)} for public_name in PUBLIC_CAMERAS} if blank else _cameras(session)
    name = f"round-{session.round_index:04d}-step-{io.control_steps:06d}{'-blank' if blank else ''}.npz"
    record = write_world_inputs(
        session.output_root / "motion-planner/world-inputs" / name, cameras=cameras,
        base_pose_world_xyzw=io.arm_base_pose(), joint_positions=io.arm_qpos(),
        gripper_finger_qpos=io.finger_qpos(), gripper_directive=io.gripper_directive)
    session.world_inputs[key] = record
    return record


def _moment(session: Any) -> dict[str, Any]:
    """What this moment's stands share (the cameras, their public points, the stands' worlds and searches), kept until
    the robot steps."""
    key = (session.round_index, session.io.control_steps)
    moment = getattr(session, "stand_cache", None)
    if not isinstance(moment, dict) or moment.get("key") != key:
        moment = {"key": key, "worlds": {}, "searches": {}}
        session.stand_cache = moment
    return moment


def _cameras(session: Any) -> dict[str, dict[str, Any]]:
    """Both public cameras' depth and calibration at this moment, rendered once, as an observation renders them."""
    moment = _moment(session)
    if "cameras" not in moment:
        cameras: dict[str, dict[str, Any]] = {}
        for public_name, source in PUBLIC_CAMERAS.items():
            _rgb, depth, meta = session._camera(source)
            cameras[public_name] = {"depth_m": depth, "intrinsic_cv": meta["intrinsic"],
                                    "cam2world_cv": meta["cam2world"], "depth_range_m": (meta["near"], meta["far"])}
        moment["cameras"] = cameras
    return moment["cameras"]


def here(session: Any) -> base_placement.Here:
    """The measured state the stands are taken from: base pose, torso height, arm mount and the base's turn centre."""
    io = session.io
    return base_placement.Here(base=tuple(io.base_pose()), torso_m=float(io.torso_height()),
                               mount=tuple(io.arm_base_pose()), turn_centre_m=tuple(io.anchor_offset()))


def body_at(mount_pose: Sequence[float], joints: Sequence[float],
            held_hand: Any = None) -> base_placement.OwnBody:
    """The robot's own body with the arm mount at ``mount_pose`` (world) and the arm at ``joints``: the planner's
    collision spheres there (``robot_spheres``), grown by the world's robot margin as the planner's own world takes the
    arm out (``robot_point_margin_m``), the hand's box (``HAND_BOX_M``), the base link's (``BASE_LINK_BOX_M``) and a
    held load's spheres (``held_hand``, ``[x, y, z, r]`` in the hand frame)."""
    import numpy as np

    mount = pose_matrix(list(mount_pose))
    hand = mount @ robot_spheres.hand(joints)
    held = np.asarray(held_hand if held_hand is not None else np.zeros((0, 4)), dtype=np.float64).reshape(-1, 4)
    held_world = np.hstack([held[:, :3] @ hand[:3, :3].T + hand[:3, 3], held[:, 3:]])
    return base_placement.OwnBody(np.vstack([robot_spheres.spheres(joints, mount), held_world]),
                                  PUBLIC_DEPTH_VOXEL_V1.robot_point_margin_m,
                                  ((hand, HAND_BOX_M), (mount, robot_spheres.BASE_LINK_BOX_M)), len(held))


def held_spheres(session: Any) -> Any:
    """The spheres, hand frame, the planner attaches for a load the jaws hold now (``depth_world.Scene`` of this
    moment's world input, built as the planner builds it, with its own robot spheres), or ``None`` when the jaws hold
    nothing by the world's rule: commanded closed with a gap between ``held_min_gap_m`` and ``held_max_gap_m``. A world
    that cannot be built (recorded in ``base-placement.jsonl``) leaves the load in the scene, where it is seen."""
    io = session.io
    parameters = PUBLIC_DEPTH_VOXEL_V1
    gap = sum(abs(float(value)) for value in io.finger_qpos())
    if not (io.gripper_directive > 0 and parameters.held_min_gap_m < gap < parameters.held_max_gap_m):
        return None
    record = _snapshot(session)
    joints = io.arm_qpos()
    try:
        scene = Scene(read_world_inputs(Path(record["path"]), record["sha256"]),
                      robot_spheres_base=robot_spheres.spheres(joints), start_hand_base=robot_spheres.hand(joints),
                      parameters=parameters)
    except DepthWorldError as error:
        _evidence(session, "base-placement", {"held_load_error": str(error)})
        return None
    return scene.held_spheres_hand


def own_body(session: Any) -> base_placement.OwnBody:
    """The robot's own body at this moment (``body_at``), kept until the robot steps."""
    moment = _moment(session)
    if "body" not in moment:
        io = session.io
        moment["body"] = body_at(io.arm_base_pose(), io.arm_qpos(), held_spheres(session))
    return moment["body"]


def public_points(session: Any) -> base_placement.PublicPoints:
    """This moment's public depth for a footprint and for a world seen from elsewhere (``base_placement``)."""
    moment = _moment(session)
    if "points" not in moment:
        moment["points"] = base_placement.public_points(
            _cameras(session), base=MOBILE_BASE, here=here(session), own=own_body(session),
            max_depth_m=PUBLIC_DEPTH_VOXEL_V1.max_depth_m)
    return moment["points"]


def stand_world(session: Any, stand: Stand) -> dict[str, str]:
    """The world input of ``stand``: this moment's public depth without the robot's own pixels, and the arm mount
    where it would be, its joints as measured."""
    moment = _moment(session)
    key = (round(stand.x_m, 4), round(stand.y_m, 4), round(stand.yaw_rad, 4), round(stand.torso_m, 4))
    cached = moment["worlds"].get(key)
    if cached is not None:
        return cached
    io = session.io
    name = (f"round-{session.round_index:04d}-step-{io.control_steps:06d}-stand-{len(moment['worlds']):03d}.npz")
    record = write_world_inputs(
        session.output_root / "motion-planner/world-inputs" / name,
        cameras=base_placement.without_own(_cameras(session), public_points(session)),
        base_pose_world_xyzw=base_placement.carried(here(session), stand, io.arm_base_pose()),
        joint_positions=io.arm_qpos(), gripper_finger_qpos=io.finger_qpos(), gripper_directive=io.gripper_directive)
    moment["worlds"][key] = record
    return record


def lifted(session: Any, torso_m: float) -> Stand:
    """The base where it stands and the torso at ``torso_m``."""
    return base_placement.lifted(here(session), torso_m)


def _evidence(session: Any, kind: str, record: Mapping[str, Any]) -> None:
    root = session.output_root / "motion-planner"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _start(session: Any) -> dict[str, Any]:
    return {"joint_names": JOINT_NAMES, "joint_positions": session.io.arm_qpos()}


def _world(session: Any, *, blank: bool = False, stand: Stand | None = None) -> dict[str, Any]:
    """The world a request is asked in: this moment's, a blank one, or the one seen from ``stand``."""
    if stand is not None and not blank:
        return {"model": WORLD_MODEL, "input": stand_world(session, stand)}
    return {"model": WORLD_MODEL, "input": _snapshot(session, blank=blank)}


def _mount(session: Any, stand: Stand | None) -> list[float] | None:
    """The arm mount's world pose at ``stand``; ``None`` where it stands now."""
    return None if stand is None else base_placement.carried(here(session), stand, session.io.arm_base_pose())


def _refused(code: str, kind: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    scope = "arm_and_torso_joint_limits_self_collision_public_depth_world"
    return {"refusal": {**detail, "path_kind": kind, "scope": scope}, "code": code,
            "path": None, "poses": None, "torso_height_m": None, "kind": kind}


def _admitted(session: Any, kind: str, path: list[list[float]], stand: Stand | None = None) -> dict[str, Any]:
    io = session.io
    poses = kinematics.grip_poses(io.model, io.data, path)
    if stand is not None:
        poses = [base_placement.carried(here(session), stand, pose) for pose in poses]
    return {"refusal": None, "code": None, "path": path, "kind": kind, "torso_height_m": None, "poses": poses}


def _client(session: Any, *, kept: bool = True) -> Any:
    """The session's planner; ``kept`` False asks past what a dry run keeps (a search's many questions)."""
    client = getattr(session, "motion_planner", None)
    if client is None:
        raise MotionPlannerError("this session has no motion planner")
    return client if kept else getattr(client, "planner", client)


def solve_ik(session: Any, poses: Sequence[Sequence[float]], *, chained: bool = False, blank: bool = False,
             lifts_m: Sequence[float] | None = None, mounts: Sequence[Sequence[float]] | None = None,
             stand: Stand | None = None, kept: bool = True) -> list[dict[str, Any]]:
    """The shared IK of grip-site poses from the measured joints, each cut seeded by the last when ``chained``.

    ``lifts_m`` solves each pose as if the torso stood that much higher, ``mounts`` with the arm mount at each of
    those world poses; ``stand`` asks every pose from there, in the world seen from there unless ``blank``."""
    io = session.io
    lifts = list(lifts_m) if lifts_m is not None else [0.0] * len(poses)
    bases = list(mounts) if mounts is not None else [_mount(session, stand)] * len(poses)
    answer = _client(session, kept=kept).solve_ik({
        "schema": IK_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(session),
        "queries": [{"tool_frame": TOOL_FRAME, "pose_base_xyzw": _hand_goal(io, pose, lift, mount),
                     "from_index": index - 1 if chained and index else None}
                    for index, (pose, lift, mount) in enumerate(zip(poses, lifts, bases))],
        "world": _world(session, blank=blank, stand=stand), "held_object": {"source": "world"}})
    _evidence(session, "ik", {key: answer.get(key) for key in ("status", "reason", "timing_s")}
              | {"queries": len(poses), "solved": sum(bool(row["solved"]) for row in answer.get("results") or ()),
                 "blank": blank, **({"stand": True} if stand is not None or mounts is not None else {})})
    if answer["status"] != "solved":
        raise MotionPlannerError(f"the shared IK failed: {answer.get('reason')}")
    return answer["results"]


def _plan(session: Any, goal: Mapping[str, Any], kind: str, stand: Stand | None = None,
          kept: bool = True) -> dict[str, Any]:
    """A collision-free joint path from the measured joints, or the refusal in this backend's codes."""
    answer = _client(session, kept=kept).plan({
        "schema": PLAN_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(session), "goal": dict(goal),
        "world": _world(session, stand=stand), "held_object": {"source": "world"}, "constraints": []})
    world = answer.get("world") or {}
    _evidence(session, "plans", {key: answer.get(key) for key in ("status", "reason", "timing_s", "planner")}
              | {"world_sha256": world.get("sha256"), "goal_refusal": (world.get("summary") or {}).get("goal_refusal")}
              | ({"stand": True} if stand is not None else {}))
    if answer["status"] == "error":
        raise MotionPlannerError(f"the shared planner failed: {answer.get('reason')}")
    if answer["status"] == "refused":
        return _refused(CODES[answer["reason"]], kind, {"planner_reason": answer["reason"],
                                                        "planning_diagnostics": planning_diagnostics(answer)})
    rows = answer["trajectory"]["positions"]
    return _admitted(session, kind, [[float(value) for value in row] for row in rows[1:]] or rows, stand)


def plan_to_pose(session: Any, pose: Sequence[float], kind: str = "joint", stand: Stand | None = None,
                 kept: bool = True) -> dict[str, Any]:
    """The planned path to a grip-site pose; a target the arm reaches only through the world says so."""
    goal = {"tool_frame": TOOL_FRAME, "pose_base_xyzw": _hand_goal(session.io, pose, mount=_mount(session, stand))}
    plan = _plan(session, goal, kind, stand, kept)
    if plan["code"] == CODES["goal_unreachable"] and solve_ik(session, [pose], blank=True, stand=stand,
                                                               kept=kept)[0]["solved"]:
        return _refused("target_in_collision_at_base_pose", kind, plan["refusal"])
    return plan


def plan_to_joints(session: Any, joints: Sequence[float]) -> dict[str, Any]:
    """The planned path to named arm joints (recovery)."""
    return _plan(session, {"joint_positions": [float(value) for value in joints]}, "joint")


def _check(session: Any, rows: Sequence[Sequence[float]], allowed: Sequence[str],
           stand: Stand | None = None) -> dict[str, Any]:
    answer = _client(session).check_path({
        "schema": CHECK_REQUEST_SCHEMA, "robot": ROBOT, "start": _start(session),
        "path": [list(row) for row in rows], "world": _world(session, stand=stand), "allowed_links": list(allowed),
        "tolerance_m": C.LINE_WORLD_TOLERANCE_M})
    _evidence(session, "line-checks", {key: answer.get(key) for key in (
        "status", "first_blocked_row", "link", "penetration_m", "timing_s")} | ({"stand": True} if stand else {}))
    if answer["status"] == "error":
        raise MotionPlannerError(f"the path check failed: {answer.get('reason')}")
    return answer


def check_line(session: Any, samples: Sequence[Sequence[float]], interaction: str,
               stand: Stand | None = None) -> dict[str, Any] | None:
    """A refusal detail for a line the public world blocks, or ``None``.

    The arm's own links may touch nothing along the whole line. The check's world is the one a plan to its
    last row would use, which clears the robot there, so the arm's pass ends on the start, whose clearing
    changes nothing. A precision line then checks the gripper too, with the goal's own surroundings cleared
    as a plan's are; a contact approach is free to press with it.
    """
    start = session.io.arm_qpos()
    passes = [([*samples, start], GRIPPER_LINKS)]
    if interaction != "contact_approach":
        passes.append((samples, ()))
    for rows, allowed in passes:
        answer = _check(session, rows, allowed, stand)
        row = answer["first_blocked_row"]
        if answer["status"] == "blocked" and row < len(samples):
            return {"path_fraction": round((row + 1) / len(samples), 3), "penetration_m": answer["penetration_m"],
                    "robot_part": "gripper" if answer["link"] in GRIPPER_LINKS else "arm"}
    return None


def endpoint(session: Any, pose: Sequence[float]) -> str:
    """Whether the arm reaches ``pose`` on its own from the measured base pose and torso height: in a blank world,
    then in the public depth world."""
    if not solve_ik(session, [pose], blank=True)[0]["solved"]:
        return ENDPOINT_UNREACHABLE
    return ENDPOINT_REACHABLE if solve_ik(session, [pose])[0]["solved"] else ENDPOINT_IN_WORLD


def _cut_refusal(cut: Sequence[float], result: Mapping[str, Any], fraction: float, jump: Any = None) -> dict[str, Any]:
    """Why the line's IK stopped at a cut: no solution there, or a joint that jumps to another branch."""
    detail: dict[str, Any] = {"path_fraction": fraction, "at_pose": [round(float(value), 4) for value in cut]}
    if jump is None:
        detail["cause"] = "ik_unsolved"
        if all(isinstance(result.get(key), (int, float)) for key in ("position_error_m", "rotation_error")):
            detail["ik_residual"] = {"translation_m": float(result["position_error_m"]),
                                     "orientation_rad": float(result["rotation_error"])}
        return detail
    index, size = jump
    return {**detail, "cause": "joint_branch_jump", "joint": index + 1, "jump_rad": round(size, 4),
            "bound_rad": C.LINE_MAX_JOINT_JUMP_RAD}


def line_path(session: Any, pose: Sequence[float], interaction: str, stand: Stand | None = None, *,
              explain: bool = False, kept: bool = True) -> dict[str, Any]:
    """The straight grip-site line to ``pose``: chained shared IK, then the check in the public world.

    From ``stand`` the line starts where the grip site would be there, the arm joints held. A refusal of the IK names
    the cut, its pose and why (``_cut_refusal``); ``explain`` adds whether the target itself is reachable
    (``endpoint``), which says whether the line or the target is what fails. ``kept`` False asks past what a dry
    run keeps."""
    import numpy as np

    io = session.io
    start = io.arm_qpos()
    origin = io.grip_pose() if stand is None else base_placement.carried(here(session), stand, io.grip_pose())
    cuts = kinematics.line_cuts(origin, pose)
    results = solve_ik(session, cuts, chained=True, blank=True, stand=stand, kept=kept)
    previous = np.asarray(start, dtype=np.float64)
    solutions: list[list[float]] = []
    for index, result in enumerate(results):
        fraction = round((index + 1) / len(results), 3)
        jump = None
        if result["solved"]:
            current = np.asarray(result["joint_positions"], dtype=np.float64)
            steps = np.abs(current - previous)
            if float(np.max(steps)) > C.LINE_MAX_JOINT_JUMP_RAD:
                jump = (int(np.argmax(steps)), float(np.max(steps)))
        if not result["solved"] or jump is not None:
            detail = _cut_refusal(cuts[index], result, fraction, jump)
            if explain:
                detail["endpoint"] = endpoint(session, pose)
            return _refused("target_unreachable_at_base_pose", "line", detail)
        solutions.append(current.tolist())
        previous = current
    samples = [[float(value) for value in row] for row in kinematics.joint_path_samples(start, solutions)]
    blocked = check_line(session, samples, interaction, stand)
    if blocked is not None:
        last = blocked["path_fraction"] >= 1.0
        code = "target_in_collision_at_base_pose" if last else "path_in_collision_at_base_pose"
        return _refused(code, "line", blocked)
    return _admitted(session, "line", samples, stand)


def arm_path(session: Any, pose: Sequence[float], kind: str, interaction: str, stand: Stand | None = None, *,
             explain: bool = False, kept: bool = True) -> dict[str, Any]:
    """One arm move's path at the measured torso height, or from ``stand``; a dead planner refuses, it never moves
    unplanned. ``kept`` False asks past what a dry run keeps (a base search's questions)."""
    try:
        return (line_path(session, pose, interaction, stand, explain=explain, kept=kept) if kind == "line"
                else plan_to_pose(session, pose, kind, stand, kept))
    except MotionPlannerError as error:
        _evidence(session, "plans", {"error": str(error)})
        return _refused(UNAVAILABLE, kind, {})


def joint_path(session: Any, joints: Sequence[float]) -> dict[str, Any]:
    try:
        return plan_to_joints(session, joints)
    except MotionPlannerError as error:
        _evidence(session, "plans", {"error": str(error)})
        return _refused(UNAVAILABLE, "joint", {})


def torso_heights(session: Any, pose: Sequence[float]) -> list[float]:
    """The torso heights, nearest first, at which the arm alone reaches ``pose`` (kinematics only), there and within
    the lift's stop tolerance either side; the measured height is not one of them.

    These are the heights a lift may be admitted at: the session then admits the whole move from each (the path
    from up there in the world seen from there, and its pacing) before it offers one. The lift stops anywhere within
    ``TORSO_TOLERANCE_M`` of its target (``motion.move_torso``), so a height is kept only if the arm reaches the pose
    at both ends of that band: the nearest reachable height is the reach boundary, and a lift that stopped 7-9 mm
    short of it left the target unreachable (RoboCasa smoke 11-05).
    """
    io = session.io
    measured = io.torso_height()
    low, high = C.TORSO_RANGE_M
    count = int(math.floor((high - low) / C.TORSO_SEARCH_STEP_M))
    heights = sorted((low + index * C.TORSO_SEARCH_STEP_M for index in range(count + 1)),
                     key=lambda height: abs(height - measured))
    heights = [height for height in [*heights, high] if abs(height - measured) >= C.TORSO_MIN_CHANGE_M]
    if not heights:
        return []
    band = [min(high, max(low, height + offset)) for height in heights
            for offset in (0.0, -C.TORSO_TOLERANCE_M, C.TORSO_TOLERANCE_M)]
    try:
        seen = solve_ik(session, [pose] * len(band), blank=True, lifts_m=[height - measured for height in band])
    except MotionPlannerError:
        return []
    return [height for index, height in enumerate(heights)
            if all(result["solved"] for result in seen[3 * index:3 * index + 3])]


def base_search(session: Any, pose: Sequence[float], move: tuple[str, str, Any],
                admits: Callable[[Stand], bool]) -> dict[str, Any] | None:
    """Where the base could stand for the refused ``move`` to ``pose`` (``base_placement.search``), asked once per
    moment, pose and move (``(kind, interaction, speed)``).

    ``admits(stand)`` is the move's own admission from a stand (``session._admitted_from``). The search's questions go
    to the planner past what a dry run keeps, and its answer is kept for this moment, so a dry run and the real call
    that follows it answer the same; ``None`` when the planner could not answer."""
    moment = _moment(session)
    kind, interaction, speed = move
    key = (tuple(round(float(value), 5) for value in pose), kind, interaction,
           json.dumps(speed, sort_keys=True, default=str))
    if key in moment["searches"]:
        return moment["searches"][key]
    io, now = session.io, here(session)

    def solve(stands: Sequence[Stand]) -> list[bool]:
        mounts = [base_placement.carried(now, stand, io.arm_base_pose()) for stand in stands]
        return [bool(row["solved"]) for row in solve_ik(session, [pose] * len(stands), blank=True, mounts=mounts,
                                                         kept=False)]

    try:
        result = base_placement.search(pose[:3], now, MOBILE_BASE, public_points(session), solve=solve, admits=admits)
    except MotionPlannerError as error:
        _evidence(session, "base-placement", {"error": str(error)})
        result = None
    else:
        _evidence(session, "base-placement", {"target": list(key[0]), "kind": kind, "interaction": interaction,
                                              "speed": speed, "base": list(now.base), "torso_m": now.torso_m,
                                              **result})
    moment["searches"][key] = result
    return result


def base_path_refusal(session: Any, target: Mapping[str, float]) -> dict[str, Any] | None:
    """``move_base``'s admission: where the base, driven by its servo to ``target``, first meets the public depth seen
    now (``base_placement.sweep``): its footprint (``robot_part`` ``base``), or the arm or its held load carried at its
    joints (``arm``, ``held_object``; the arm has to be retracted first), or ``None``."""
    goal = (float(target["x_m"]), float(target["y_m"]), float(target["yaw_rad"]))
    blocked = base_placement.sweep(public_points(session), MOBILE_BASE, here(session), goal)
    _evidence(session, "base-checks", {"goal": list(goal), "blocked": blocked})
    return None if blocked is None else {"path_kind": "base", **blocked}


def probe(session: Any, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Answer motion.probe queries with the shared IK in the public world (no plan_ref)."""
    results = []
    for query in request["queries"]:
        if query.get("arm") != "right":
            raise RoboCasaError("PandaOmron has only the right arm")
        pose = [*query["pose"]["position_m"], *query["pose"]["orientation_xyzw"]]
        candidates = [("exact", pose)]
        if request.get("variant_policy") == "parallel_jaw_equivalents":
            candidates.append(("jaw_flip_180", kinematics.jaw_equivalent(pose)))
        solved = solve_ik(session, [candidate for _name, candidate in candidates])
        variants = [{
            "variant_id": variant_id, "reachable": bool(result["solved"]), "target_pose_xyzw": candidate,
            "planner": {"kind": "shared_ik", "plan": "none"},
            "endpoint_error": {"translation_m": result["position_error_m"],
                               "orientation_rad": result["rotation_error"]},
            "trajectory_steps": 0, "trajectory_duration_s": 0.0, "plan_ref": None,
        } for (variant_id, candidate), result in zip(candidates, solved)]
        selected = next((variant for variant in variants if variant["reachable"]), None)
        results.append({
            "query_id": query["query_id"], "arm": "right", "reachable": selected is not None,
            "selected_variant_id": selected["variant_id"] if selected else None, "plan_ref": None,
            "selected_target_pose_xyzw": selected["target_pose_xyzw"] if selected else None,
            "variants": variants,
        })
    return results


__all__ = ["CODES", "ENDPOINT_IN_WORLD", "ENDPOINT_REACHABLE", "ENDPOINT_UNREACHABLE", "GRIPPER_LINKS", "HAND_BOX_M",
           "MOBILE_BASE", "ROBOT", "UNAVAILABLE", "arm_path", "base_path_refusal", "base_search", "body_at",
           "check_line", "close", "endpoint", "fk_parity", "held_spheres", "here", "joint_path", "lifted", "line_path",
           "own_body", "own_spheres_check", "parity_rows", "plan_to_joints", "plan_to_pose", "planner_binding",
           "probe", "public_points", "solve_ik", "stand_world", "start", "torso_heights"]
