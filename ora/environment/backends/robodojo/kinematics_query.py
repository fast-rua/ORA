"""RoboDojo answers to the read-only kinematics query (``ora.kinematics-request.v1``).

``ik_chain`` solves one pose after another with the pinned SDK's own CuRobo
IK (``ik.sdk_candidates``), each seeded from the previous solution, so a dense
pose sequence along a path comes back as a continuous joint path. Ranking
``nearest`` takes the admissible candidate nearest the previous solution (no
canonical-first reordering; FK only for the candidates it tries, nearest
first); ``canonical_first`` is move_pose's own selection (``ik.solve_ik``).
A candidate is admissible when it lies inside the live joint limits, its FK
lands within the SDK's IK tolerances, it carries no pitch joint past its
half-turn limit, and its largest joint step from the previous solution stays
within ``max_joint_step_rad`` times the number of poses since that solution.
With a ``screen``, the straight joint path of each step is screened with the
SDK collision spheres (and the other arm at its measured joints). A failed
pose keeps the previous solution as the seed of the next.

``fk`` returns link6 poses of joint vectors (``reachability.fk_link6``), and
``screen_path`` screens joint polylines segment by segment, self-collision per
arm, each moving arm against the SDK planner's own table cuboid (its top face),
and both arms in lockstep (an arm not named holds its retained target). On a
path the shared planner made it is the re-check: the other arm is already in that
planner's world. ``plan_path``, ``pace_path`` and ``contact_path`` are answered by
the shared motion base (``shared_motion.py``).

Nothing here writes a controller or advances physics; the clock is checked.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.contracts.kinematics import RESULT_SCHEMA, validate_request
from ora.environment.controllers.joint_path_clearance import (
    InterArmSphereScreen, JointPathClearanceError, TableSphereScreen,
)

from . import ik, reachability
from .bridge import RobodojoError
from .common import canonical_joints, nearest_equivalent_joints, pitch_joint_wraps, pose_error

IK_MODEL = "robodojo_sdk_ik_seeded_chain_v1"
CLAIM_BOUNDARY = ("SDK IK, FK and collision spheres only (self, the other arm where screened, and the SDK table top "
                  "in screen_path); "
                  "no object or payload collision, no physics, no arrival or task-success claim.")


#: How every IK solve of this query is seeded: CuRobo's own advancing seed
#: sequence (the LM seed solver's Halton buffer, ``sampler_seed`` 451, when
#: ``use_lm_seed`` is on, as in the SDK; else the main solver's), drawn
#: explicitly and passed as ``seed_config``
#: (``ik.sdk_candidates(explicit_seeds=True)``); each row records the
#: generator state before and after its solve, so replay or a fork re-solves
#: it by restoring that state (``ik.restore_generator_state``), never by
#: replaying the solves before it.
SEEDING_RULE = "explicit_solver_seed_sequence_v1"


def _limits_margin(joints: np.ndarray, limits: np.ndarray) -> float:
    return float(np.min(np.minimum(joints - limits[:, 0], limits[:, 1] - joints)))


def _nearest(session: Any, arm: str, pose: Sequence[float], previous: np.ndarray, limits: np.ndarray,
             bound: float, screen: Callable[[Sequence[float], Sequence[float]], Mapping[str, Any]] | None,
             timings: dict[str, float]) -> dict[str, Any]:
    robot = session.robots[arm]
    planner = session.environment.robot_manager.ik_solver[robot.robot_name]
    started = time.monotonic()
    candidates = ik.sdk_candidates(planner, previous.tolist(), list(pose), robot.entity_origin_pose,
                                   explicit_seeds=True)
    timings["ik"] += time.monotonic() - started
    names = list(candidates["joint_names"])
    revolute = list(candidates.get("revolute", [name in ik.ARX_X5_REVOLUTE_JOINT_NAMES for name in names]))
    pitch = list(candidates.get("pitch", [name in ik.ARX_X5_PITCH_JOINT_NAMES for name in names]))
    tolerances = (float(candidates["position_tolerance_m"]), float(candidates["orientation_tolerance_rad"]))
    valid = []
    for index, values in enumerate(np.asarray(candidates["solutions"], dtype=float)):
        if (not np.isfinite(values).all() or not bool(candidates["success"][index])
                or not bool(candidates["feasible"][index])
                or np.any(values < limits[:, 0]) or np.any(values > limits[:, 1])):
            continue
        command, _ = nearest_equivalent_joints(values, previous, limits, revolute)
        command = np.asarray(command, dtype=float)
        valid.append((float(np.max(np.abs(command - previous))), index, command))
    valid.sort(key=lambda item: (item[0], item[1]))
    row: dict[str, Any] = {"status": "ik_failed", "joints_rad": None, "joint_step_rad": None,
                           "position_error_m": None, "orientation_error_rad": None, "limit_margin_rad": None,
                           "canonical": None, "pitch_joint_wrap": False, "path": None,
                           "valid_candidates": len(valid), "seeding": candidates.get("seeding")}
    blocked = None
    for step, _, command in valid:
        if step > bound:
            if row["status"] == "ik_failed":
                row.update(status="discontinuous", joint_step_rad=step)
            break
        started = time.monotonic()
        position, rotation = pose_error(list(pose), reachability.fk_link6(session, arm, command.tolist()))
        timings["fk"] += time.monotonic() - started
        if position > tolerances[0] or rotation > tolerances[1]:
            continue
        if pitch_joint_wraps(command, previous, pitch):
            row["pitch_joint_wrap"] = True
            continue
        if screen is not None:
            started = time.monotonic()
            evidence = dict(screen(previous.tolist(), command.tolist()))
            timings["screen"] += time.monotonic() - started
            if evidence.get("status") != "clear":
                blocked = blocked or {"status": evidence.get("status"), "joint_step_rad": step}
                continue
        return {**row, "status": "ok", "joints_rad": command.tolist(), "joint_step_rad": step,
                "position_error_m": position, "orientation_error_rad": rotation,
                "limit_margin_rad": _limits_margin(command, limits), "canonical": canonical_joints(command, revolute),
                "path": "clear" if screen is not None else None}
    if blocked is not None:
        row.update(status="path_blocked", path=blocked["status"], joint_step_rad=blocked["joint_step_rad"])
    return row


def _canonical_first(session: Any, arm: str, pose: Sequence[float], previous: np.ndarray, limits: np.ndarray,
                     bound: float, other_arm: bool, timings: dict[str, float]) -> dict[str, Any]:
    started = time.monotonic()
    solved = ik.solve_ik(session, arm, list(pose), jump_limit_rad=bound, current=previous.tolist(),
                         static_other_arm=other_arm, explicit_seeds=True)
    timings["ik"] += time.monotonic() - started
    outcome = (solved.get("ik_selection") or {}).get("outcome")
    row: dict[str, Any] = {"status": "ik_failed", "joints_rad": None, "joint_step_rad": solved.get("joint_delta_rad"),
                           "position_error_m": None, "orientation_error_rad": None, "limit_margin_rad": None,
                           "canonical": None, "pitch_joint_wrap": outcome == "pitch_joint_wrap", "path": None,
                           "valid_candidates": (solved.get("ik_selection") or {}).get("valid_endpoint_count"),
                           "seeding": solved.get("seeding")}
    if solved.get("joint_target") is None:
        if outcome in ("self_collision_path", ik.INTER_ARM_OUTCOME):
            row.update(status="path_blocked", path=outcome)
        return row
    command = np.asarray(solved["joint_target"], dtype=float)
    if outcome == "joint_jump":
        return {**row, "status": "discontinuous"}
    position, rotation = pose_error(list(pose), reachability.fk_link6(session, arm, command.tolist()))
    return {**row, "status": "ok", "joints_rad": command.tolist(), "position_error_m": position,
            "orientation_error_rad": rotation, "limit_margin_rad": _limits_margin(command, limits),
            "canonical": bool((solved.get("ik_selection") or {}).get("selected_canonical")), "path": "clear"}


def ik_chain(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    arm = request["arm"]
    limits = np.asarray(ik.articulation_limits(session, arm), dtype=float)
    seed = (np.asarray(ik.measured_arm_joints(session, arm), dtype=float) if request["seed_joints_rad"] is None
            else np.asarray(request["seed_joints_rad"], dtype=float))
    screen_mode = request["screen"]
    screen = None
    if request["ranking"] == "nearest" and screen_mode != "none":
        screen, _ = ik.path_screen(session, arm, static_other_arm=screen_mode == "self_and_measured_other_arm")
    timings = {"ik": 0.0, "fk": 0.0, "screen": 0.0}
    previous, gap, rows, first_failure = seed, 1, [], None
    for index, pose in enumerate(request["poses_xyz_xyzw"]):
        bound = request["max_joint_step_rad"] * gap
        if request["ranking"] == "nearest":
            row = _nearest(session, arm, pose, previous, limits, bound, screen, timings)
        else:
            row = _canonical_first(session, arm, pose, previous, limits, bound,
                                   screen_mode == "self_and_measured_other_arm", timings)
        rows.append({"index": index, **row})
        if row["status"] == "ok":
            previous, gap = np.asarray(row["joints_rad"], dtype=float), 1
        else:
            gap += 1
            first_failure = index if first_failure is None else first_failure
    return {"arm": arm, "seed_joints_rad": seed.tolist(), "rows": rows, "first_failure_index": first_failure,
            "timings_s": {key: round(value, 4) for key, value in timings.items()}}


def fk(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    arm = request["arm"]
    return {"arm": arm, "rows": [{"index": index, "pose_xyz_xyzw": reachability.fk_link6(session, arm, joints)}
                                 for index, joints in enumerate(request["joints_rad"])]}


def table_screen(session: Any) -> TableSphereScreen:
    """Each arm's SDK spheres at its world base against the top of its planner's table cuboid.

    The cuboid is ``CuroboPlanner.scene_model["cuboid"]["table"]`` in the planner
    frame; its top face is placed in the world with the same base transform as
    the inter-arm screen. Missing evidence fails closed.
    """
    cached = getattr(session, "_table_screen", None)
    if cached is not None:
        return cached
    manager = session.environment.robot_manager
    models, bases, tables, buffers = {}, {}, {}, {}
    for arm, robot in session.robots.items():
        planner = manager.ik_solver[robot.robot_name]
        models[arm] = ik.self_collision_model(session, arm, planner)
        try:
            origin = [float(value) for value in robot.entity_origin_pose]
            base = InterArmSphereScreen.base_transform(origin[:3], origin[3:], planner.frame_bias)
            table = planner.scene_model["cuboid"]["table"]
            dims, pose = [float(value) for value in table["dims"]], [float(value) for value in table["pose"]]
            centre = base[:3, :3] @ np.asarray([pose[0], pose[1], pose[2] + dims[2] / 2]) + base[:3, 3]
            kinematics = planner.robot_cfg["robot_cfg"]["kinematics"]
            buffers[arm] = float(kinematics.get("collision_sphere_buffer", 0.0) or 0.0)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
            raise RobodojoError(f"SDK table model is unavailable: {error}") from error
        bases[arm] = base
        tables[arm] = {"top_z_m": float(centre[2]),
                       "xy_extent_m": [[centre[0] - dims[0] / 2, centre[0] + dims[0] / 2],
                                       [centre[1] - dims[1] / 2, centre[1] + dims[1] / 2]]}
    try:
        screen = TableSphereScreen(models, bases, tables, sphere_buffers_m=buffers)
    except JointPathClearanceError as error:
        raise RobodojoError(f"SDK table screen is unavailable: {error}") from error
    session._table_screen = screen
    return screen


def screen_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    polylines = {arm: np.asarray(rows, dtype=float) for arm, rows in request["knots_rad"].items()}
    manager = session.environment.robot_manager
    count = len(next(iter(polylines.values())))
    for arm, rows in polylines.items():
        model = ik.self_collision_model(session, arm, manager.ik_solver[session.robots[arm].robot_name])
        for segment in range(1, count):
            evidence = dict(model.straight_path(rows[segment - 1].tolist(), rows[segment].tolist()))
            if evidence.get("status") != "clear":
                return {"status": "self_collision", "arm": arm, "first_blocked_segment": segment,
                        "segments_screened": count - 1}
    table = table_screen(session)
    for arm, rows in polylines.items():
        for segment in range(1, count):
            evidence = dict(table.straight_path(arm, rows[segment - 1].tolist(), rows[segment].tolist()))
            if evidence.get("status") != "clear":
                return {"status": "table_collision", "arm": arm, "first_blocked_segment": segment,
                        "segments_screened": count - 1, "min_table_clearance_m": evidence.get("min_clearance_m"),
                        "limiting_link": evidence.get("limiting_link")}
    if set(session.robots) != set(ik.OTHER_ARM):
        return {"status": "clear", "arm": None, "first_blocked_segment": None, "segments_screened": count - 1}
    screen = ik.inter_arm_screen(session)
    held = {arm: (ik.measured_arm_joints(session, arm), ik.retained_arm_target(session, arm))
            for arm in ik.OTHER_ARM if arm not in polylines}
    clearance = None
    for segment in range(1, count):
        first = {arm: rows[segment - 1].tolist() for arm, rows in polylines.items()}
        last = {arm: rows[segment].tolist() for arm, rows in polylines.items()}
        for arm, (measured, retained) in held.items():
            first[arm], last[arm] = (list(measured) if segment == 1 else list(retained)), list(retained)
        evidence = dict(screen.straight_paths(first, last))
        if evidence.get("min_clearance_m") is not None:
            clearance = (float(evidence["min_clearance_m"]) if clearance is None
                         else min(clearance, float(evidence["min_clearance_m"])))
        if evidence.get("status") != "clear":
            return {"status": "inter_arm_collision", "arm": None, "first_blocked_segment": segment,
                    "segments_screened": count - 1, "min_inter_arm_clearance_m": clearance,
                    "limiting_links": evidence.get("limiting_links")}
    return {"status": "clear", "arm": None, "first_blocked_segment": None, "segments_screened": count - 1,
            "min_inter_arm_clearance_m": clearance}


def plan_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    from . import shared_motion

    return shared_motion.plan_path(session, request)


def pace_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    from . import shared_motion

    return shared_motion.pace_path(session, request)


def contact_path(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    from . import shared_motion

    return shared_motion.contact_path(session, request)


OPERATIONS = {"ik_chain": ik_chain, "fk": fk, "screen_path": screen_path, "plan_path": plan_path,
              "pace_path": pace_path, "contact_path": contact_path}


def answer(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """One validated kinematics request, answered without advancing physics."""
    normalized = validate_request(request)
    clock = session._clock()
    started = time.monotonic()
    result = OPERATIONS[normalized["operation"]](session, normalized)
    if session._clock() != clock:
        session._fault("kinematics_query_advanced_physics")
        raise RobodojoError("read-only kinematics query advanced physics")
    from .ik_sampler import provenance

    return {"schema": RESULT_SCHEMA, "operation": normalized["operation"], **result, "ik_model": IK_MODEL,
            **provenance(session),
            **({"seeding": SEEDING_RULE} if normalized["operation"] == "ik_chain" else {}),
            "total_s": round(time.monotonic() - started, 4), "claim_boundary": CLAIM_BOUNDARY}


__all__ = ["CLAIM_BOUNDARY", "IK_MODEL", "SEEDING_RULE", "answer", "fk", "ik_chain", "screen_path", "table_screen"]
