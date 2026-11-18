"""The backend's read-only kinematics query, as lib calls it (``ora.kinematics-request.v1``).

Replay: each call is one recorded ``kinematics`` interface call; replay serves
the recorded answer for the exact request and a changed request fails
``NotRecorded`` (no new IK is solved). The environment solves with the SDK's
own CuRobo IK, each pose seeded from the previous solution; lib only builds
requests and reads rows. Requests longer than one call's limit are split, each
part seeded from the last solution of the one before. ``plan_path``, ``pace_path``
and ``contact_path`` are answered by the shared motion base, and recorded the same way.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ...interface.api import DataInterface
from ..values import PublicResult
from .timing import Pace

REQUEST_SCHEMA = "ora.kinematics-request.v1"
MAXIMUM_POSES = 256


class KinematicsUnavailable(RuntimeError):
    """The backend answers no kinematics query (refusal or missing bridge)."""


def _query(api: DataInterface, request: dict) -> dict:
    response = api.planner.kinematics(request=request)
    envelope = response.json()
    if envelope.get("error"):
        raise KinematicsUnavailable(str(envelope["error"]))
    return PublicResult.of(response).data.json()


def ik_chain(api: DataInterface, *, arm: str, poses: Sequence[Sequence[float]], seed: Sequence[float] | None,
             max_joint_step_rad: float, ranking: str = "nearest",
             screen: str = "self_and_measured_other_arm") -> dict:
    """Replay: recorded kinematics answers. One arm's joint solution per pose, each seeded from the previous.

    Returns ``{"rows": [...], "first_failure_index": int | None, "answers": [...]}`` where each row has
    ``status`` (ok, ik_failed, discontinuous, path_blocked) and, when ok, ``joints_rad``.
    Example: ik_chain(api, arm='left', poses=[pose], seed=None, max_joint_step_rad=0.2)."""
    rows, answers, previous = [], [], None if seed is None else [float(v) for v in seed]
    for start in range(0, len(poses), MAXIMUM_POSES):
        part = [[float(v) for v in pose] for pose in poses[start:start + MAXIMUM_POSES]]
        answer = _query(api, {"schema": REQUEST_SCHEMA, "operation": "ik_chain", "arm": arm,
                              "seed_joints_rad": previous, "poses_xyz_xyzw": part, "ranking": ranking,
                              "max_joint_step_rad": float(max_joint_step_rad), "screen": screen})
        answers.append({key: answer.get(key) for key in ("seeding", "ik_model", "timings_s", "total_s",
                                                         "ik_sampler_state", "warning") if key in answer})
        for row in answer["rows"]:
            rows.append({**row, "index": start + int(row["index"])})
            if row["status"] == "ok":
                previous = row["joints_rad"]
    failures = [row["index"] for row in rows if row["status"] != "ok"]
    return {"rows": rows, "first_failure_index": failures[0] if failures else None, "answers": answers}


def plan_path(api: DataInterface, *, arm: str, goal_pose: Sequence[float] | None = None,
              goal_joints: Sequence[float] | None = None) -> dict:
    """Replay: recorded kinematics answers. One arm's collision-free path from its measured joints to a goal.

    Planned in the world of the current public depth, the other arm held where it is. Returns ``status``:
    ``planned`` (``path_joints``), ``refused`` (``reason``, one neutral reason; ``goal_refusal`` when the world
    refused the goal) or ``unavailable`` (``reason``; no planner here, or the observation is stale).
    Example: plan_path(api, arm='left', goal_pose=pose)."""
    return _query(api, {"schema": REQUEST_SCHEMA, "operation": "plan_path", "arm": arm,
                        "goal_pose_xyz_xyzw": None if goal_pose is None else [float(v) for v in goal_pose],
                        "goal_joints": None if goal_joints is None else [float(v) for v in goal_joints]})


def pace_path(api: DataInterface, *, arm: str, joints: Sequence[Sequence[float]], pace: Pace,
              slow_pace: Pace | None = None, budget_ticks: int, cut_to_budget: bool = True,
              unwind_check: bool = True) -> dict:
    """Replay: recorded kinematics answers. One arm's joint path timed, one row per native action.

    Returns ``status`` ``timed`` (``path_joints``, ``fractions``, ``ticks``, ``needed_ticks``,
    ``fraction_of_move``, ``slow_from_tick``, ``speed``, ``wrist``) or ``refused`` (``reason``, ``message``,
    ``constraints``). Example: pace_path(api, arm='left', joints=rows, pace=pace, budget_ticks=50)."""
    speed = {"speed": "explicit", "caps": pace.caps(), "slow_caps": None if slow_pace is None else slow_pace.caps(),
             "slow_approaching": slow_pace is not None}
    return _query(api, {"schema": REQUEST_SCHEMA, "operation": "pace_path", "arm": arm,
                        "joints_rad": [[float(v) for v in row] for row in joints], "speed": speed,
                        "slow_approach": slow_pace is not None, "budget_ticks": int(budget_ticks),
                        "cut_to_budget": bool(cut_to_budget), "unwind_check": bool(unwind_check)})


def contact_path(api: DataInterface, *, arm: str, joints: Sequence[Sequence[float]]) -> dict:
    """Replay: recorded kinematics answers. Whether a contact's path keeps every link but the fingers out of the world.

    Returns ``status`` ``clear``, ``blocked`` (``first_blocked_row``, ``limiting_link``, ``penetration_m``) or
    ``unavailable``. Example: contact_path(api, arm='left', joints=rows)."""
    return _query(api, {"schema": REQUEST_SCHEMA, "operation": "contact_path", "arm": arm,
                        "joints_rad": [[float(v) for v in row] for row in joints]})


def warnings(chain: Mapping) -> list[dict]:
    """Pure computation: the answers of an ``ik_chain`` result that carry a warning (e.g. a restored run whose
    IK generator state was not recorded: ``ik_sampler_state`` ``unrecorded``).

    Example: warnings(ik_chain(api, ...)) == []."""
    return [dict(answer) for answer in chain.get("answers", ())
            if answer.get("warning") or answer.get("ik_sampler_state") == "unrecorded"]


def fk(api: DataInterface, *, arm: str, joints: Sequence[Sequence[float]]) -> list[list[float]]:
    """Replay: recorded kinematics answers. link6 poses [x,y,z,qx,qy,qz,qw] of each joint vector.

    Example: fk(api, arm='left', joints=[q])."""
    poses = []
    for start in range(0, len(joints), MAXIMUM_POSES):
        part = [[float(v) for v in row] for row in joints[start:start + MAXIMUM_POSES]]
        answer = _query(api, {"schema": REQUEST_SCHEMA, "operation": "fk", "arm": arm, "joints_rad": part})
        poses.extend([float(v) for v in row["pose_xyz_xyzw"]] for row in answer["rows"])
    return poses


def screen_path(api: DataInterface, polylines: Mapping[str, Sequence[Sequence[float]]]) -> dict:
    """Replay: recorded kinematics answers. Self and inter-arm sphere screen of joint polylines in lockstep.

    Example: screen_path(api, {'left': [q0, q1, q2]})."""
    count = len(next(iter(polylines.values())))
    if count > MAXIMUM_POSES:
        # Overlapping windows: each starts on the last knot of the one before.
        for start in range(0, count - 1, MAXIMUM_POSES - 1):
            part = {arm: rows[start:start + MAXIMUM_POSES] for arm, rows in polylines.items()}
            result = screen_path(api, part)
            if result.get("status") != "clear":
                return {**result, "first_blocked_segment": start + int(result["first_blocked_segment"])}
        return {"status": "clear", "first_blocked_segment": None, "segments_screened": count - 1}
    return _query(api, {"schema": REQUEST_SCHEMA, "operation": "screen_path",
                        "knots_rad": {arm: [[float(v) for v in row] for row in rows]
                                      for arm, rows in polylines.items()}})


__all__ = ["KinematicsUnavailable", "MAXIMUM_POSES", "REQUEST_SCHEMA", "contact_path", "fk", "ik_chain", "pace_path",
           "plan_path", "screen_path", "warnings"]
