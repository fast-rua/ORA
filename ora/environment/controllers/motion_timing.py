"""Which timing a process's paced moves use, and the joints' hardware jerk limits it certifies them at.

``motion_pace.pace_path`` times a planner's path with the TOPP-RA + Ruckig
timing worker (``ora/environment/planners/timing/``) whenever this process has
one: a backend selects it with ``configure`` (LIBERO does, from
``runtime.motion_planner.timing``), or else ``ORA_TIMING_PYTHON`` names its
interpreter. Without either, ``pace_path`` keeps ``motion_pace``'s own
hand-written timing, the documented fallback, and every receipt says which one
ran (``timing_evidence``). ``configure(None)`` selects the fallback explicitly,
whatever the environment holds (a frozen baseline).

Jerk is the one limit a preset does not set: Ruckig certifies every tick at
the robot's hardware jerk per joint (``JOINT_JERK_LIMITS``). A robot without
a documented value runs at ``DEFAULT_JERK_RAD_S3`` with ``source:
"unverified"``. A robot motion profile's own pace may name its jerk
(``joint_jerk_rad_s3`` of explicit caps, RoboDojo's ARX X5 profile v2
``fast``): ``pace_jerk`` then certifies at that jerk where the robot has no
documented one, and never above a documented one.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.motion_speed import validate_speed

from . import motion_pace as mp
from .motion_retime import RETIME_REQUEST_SCHEMA, TIMING_IMPLEMENTATION, TIMING_PYTHON_ENV, TimingClient
from .motion_worker import MotionPlannerError

#: ``motion_pace``'s own timing, the fallback without a timing interpreter.
FALLBACK_IMPLEMENTATION = "motion_pace"
#: The preflight check that starts the timing worker; its passing detail names the interpreter (path, sha256).
PREFLIGHT_CHECK = "motion_timing_worker"
#: A robot with no documented jerk limit runs at this one per joint, and says so.
DEFAULT_JERK_RAD_S3 = 1000.0
UNVERIFIED = "unverified"
#: Hardware jerk limits per joint, rad/s^3, in the robot's joint order.
JOINT_JERK_LIMITS: dict[str, dict[str, Any]] = {
    "franka_panda": {
        "rad_s3": (7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0),
        "source": "Franka Robotics, Robot and interface specifications, 'Limits for Panda', joint jerk "
                  "(https://support.franka.de/docs/control_parameters.html)",
    },
    # TODO(timing): the ARX X5's and the YAM's joint jerk limits are not published; until a vendor value or a
    # measurement exists both run at the conservative default (docs/reference/motion-speed.md).
    "arx_x5": {"rad_s3": (DEFAULT_JERK_RAD_S3,) * 6, "source": UNVERIFIED},
    "yam": {"rad_s3": (DEFAULT_JERK_RAD_S3,) * 6, "source": UNVERIFIED},
    # TODO(timing): RoboTwin's ALOHA-AgileX arms have no published joint jerk limit either.
    "aloha_agilex": {"rad_s3": (DEFAULT_JERK_RAD_S3,) * 6, "source": UNVERIFIED},
}

_state: dict[str, Any] = {"configured": False, "python": None, "stderr": None, "client": None}


def jerk_limits(robot: str | None, joints: int) -> dict[str, Any]:
    """``{"rad_s3": [per joint], "source": ...}`` for ``robot`` (``None`` or unknown: the unverified default)."""

    known = JOINT_JERK_LIMITS.get(robot or "")
    if known is None:
        return {"rad_s3": [DEFAULT_JERK_RAD_S3] * joints, "source": UNVERIFIED}
    if len(known["rad_s3"]) != joints:
        raise ValueError(f"{robot} has {len(known['rad_s3'])} jerk limits, the path {joints} joints")
    return {"rad_s3": list(known["rad_s3"]), "source": str(known["source"])}


#: ``jerk_source`` of a move certified at its pace's own jerk (a robot without a documented jerk).
PACE_JERK = "pace"


def pace_jerk(robot: str | None, joints: int, pace: mp.Pace) -> dict[str, Any]:
    """The jerk a move of ``pace`` is certified at: the robot's (``jerk_limits``), or the pace's own.

    A pace that names ``joint_jerk_rad_s3`` gets it where the robot's jerk is ``unverified``
    (``source: "pace"``), and at most the documented jerk of each joint where there is one (that source
    kept). Example: pace_jerk("arx_x5", 6, pace)["source"] == "pace" for profile v2's fast."""

    limits = jerk_limits(robot, joints)
    if pace.joint_jerk_rad_s3 is None:
        return limits
    if limits["source"] == UNVERIFIED:
        return {"rad_s3": [float(pace.joint_jerk_rad_s3)] * joints, "source": PACE_JERK}
    return {"rad_s3": [min(float(pace.joint_jerk_rad_s3), value) for value in limits["rad_s3"]],
            "source": limits["source"]}


def configure(python_executable: str | Path | None, *, stderr_path: str | Path | None = None) -> None:
    """Select this process's timing: the worker under ``python_executable``, or ``None`` for the fallback.

    It replaces any earlier choice, ``ORA_TIMING_PYTHON`` included; the worker starts on first use."""

    close()
    _state.update(configured=True, python=None if python_executable is None else Path(python_executable),
                  stderr=None if stderr_path is None else Path(stderr_path))


def reset() -> None:
    """Forget ``configure``: ``ORA_TIMING_PYTHON`` decides again."""

    close()
    _state.update(configured=False, python=None, stderr=None)


def client() -> TimingClient | None:
    """This process's timing client, or ``None`` when the fallback timing applies."""

    if _state["client"] is not None:
        return _state["client"]
    python = _state["python"] if _state["configured"] else os.environ.get(TIMING_PYTHON_ENV) or None
    if python is None:
        return None
    _state["client"] = TimingClient(Path(python), stderr_path=_state["stderr"])
    return _state["client"]


def binding() -> dict[str, Any]:
    """What this process times with, for a session's metadata: the started worker's binding, or the fallback."""

    timing = client()
    if timing is None:
        return {"implementation": FALLBACK_IMPLEMENTATION}
    return timing.start()


def bind_preflight(output_root: str | Path) -> dict[str, Any] | None:
    """Time this worker's moves as its run's preflight bound them; returns the binding, or None without a record.

    For a backend whose worker config cannot carry the Supervisor's binding (RoboTwin's fixed session fields,
    RoboCasa365's): the isolated worker's environment is allowlisted and holds no ``ORA_TIMING_PYTHON``, so it reads
    the same record LIBERO's Supervisor reads, ``preflight.json`` of the run whose ``native/`` is ``output_root``.
    A run without the check (no ``runtime.motion_timing``) takes the fallback explicitly; a changed interpreter is
    refused. No record at all is a direct session or a qualification, and ``ORA_TIMING_PYTHON`` still decides."""

    root = Path(output_root)
    record = root.parent / "preflight.json"
    if not record.is_file():
        return None
    checks = json.loads(record.read_text(encoding="utf-8")).get("checks") or ()
    matches = [item for item in checks if isinstance(item, Mapping) and item.get("check") == PREFLIGHT_CHECK]
    if not matches:
        configure(None)
        return binding()
    detail = matches[0].get("detail") if matches[0].get("passed") is True else None
    if len(matches) != 1 or not isinstance(detail, Mapping) or not isinstance(detail.get("python"), str) \
            or not isinstance(detail.get("python_sha256"), str):
        raise MotionPlannerError("the motion timing preflight binding is absent or invalid")
    configure(detail["python"], stderr_path=root / "timing/worker.stderr.log")
    bound = binding()
    if bound.get("python_sha256") != detail["python_sha256"]:
        raise MotionPlannerError("the motion timing interpreter changed after preflight")
    path = root / "timing/binding.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bound, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return bound


def close() -> None:
    timing, _state["client"] = _state["client"], None
    if timing is not None:
        timing.close()


def timing_evidence(answer: Mapping[str, Any] | None = None, *, fallback: str | None = None) -> dict[str, Any]:
    """``timing`` of a paced move's receipt: which implementation timed it, with what, and how it went."""

    if answer is None:
        return {"implementation": FALLBACK_IMPLEMENTATION,
                "reason": fallback or f"no timing interpreter ({TIMING_PYTHON_ENV})"}
    detail = answer.get("detail") or {}
    return {"implementation": TIMING_IMPLEMENTATION, "versions": dict(answer["planner"].get("versions") or {}),
            "jerk_source": detail.get("jerk_source"), "attempts": detail.get("attempts"),
            "acceleration_scale": detail.get("acceleration_scale"),
            "ruckig_extended_segments": detail.get("ruckig_extended_segments"),
            "max_deviation_m": detail.get("max_deviation_m"), "solve_ms": detail.get("solve_ms"),
            "verification": dict(detail.get("verification") or {})}


def library_pace(timing: Any, path: Sequence[Sequence[float]],
                 fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], *, speed: Mapping[str, Any] | None,
                 pace: mp.Pace, slow_pace: mp.Pace | None, tick_s: float, joint_speed_ceiling_rad_s: float | None,
                 joint_accel_ceiling_rad_s2: float | None, wrist_joint_index: int,
                 wrist_limits_rad: tuple[float, float], budget_ticks: int, slow_approach: bool,
                 cut_to_budget: bool, arm: str, robot: str | None,
                 unwind_check: bool = True) -> mp.PacedPath | str:
    """``motion_pace.pace_path`` by the timing worker: the planner's path on the joint grid, measured with ``fk``, timed by
    ``retime``. Returns the paced path, or the reason the fallback times it (``timing_infeasible``)."""

    measured = mp.joint_path_grid(path, fk, arm=arm, wrist_joint_index=wrist_joint_index)
    nodes = measured.grid.joints[arm]
    joints = int(nodes.shape[1])
    jerk = pace_jerk(robot, joints, pace)
    request = {
        "schema": RETIME_REQUEST_SCHEMA, "joint_names": [f"{arm}_joint{index}" for index in range(joints)],
        "path": {"joint_positions": nodes.tolist(), "tool_poses_xyzw": [list(pose) for pose in measured.poses]},
        "speed": validate_speed(speed) if speed is not None else None, "slow_approach": bool(slow_approach),
        "limits": {"joint_velocity_rad_s": None if joint_speed_ceiling_rad_s is None
                   else [float(joint_speed_ceiling_rad_s)] * joints,
                   "joint_acceleration_rad_s2": None if joint_accel_ceiling_rad_s2 is None
                   else [float(joint_accel_ceiling_rad_s2)] * joints,
                   "joint_jerk_rad_s3": jerk["rad_s3"], "jerk_source": jerk["source"]},
        "tick_s": float(tick_s), "budget_ticks": int(budget_ticks), "cut_to_budget": bool(cut_to_budget),
        "wrist": {"joint_index": int(wrist_joint_index),
                  "limits_rad": [float(wrist_limits_rad[0]), float(wrist_limits_rad[1])],
                  **({} if unwind_check else {"unwind_check": False})},
    }
    answer = timing.retime(request)
    if answer["status"] == "error":
        error = answer["detail"]["error"]
        raise MotionPlannerError(f"the timing worker failed: {error['type']}: {error['message']}")
    if answer["status"] == "refused":
        refusal = answer["detail"]["refusal"]
        if answer["reason"] == "timing_infeasible":
            return f"timing_infeasible: {refusal['message']}"
        raise mp.PaceRefusal(refusal["message"], code=answer["reason"], constraints=refusal["constraints"])
    report = answer["report"]
    rows = [list(map(float, row)) for row in answer["trajectory"]["positions"][1:]]
    grid = measured.grid if report["fraction_of_move"] >= 1.0 else mp.cut(measured.grid, report["part_segments"])
    planned = {"joints": {arm: rows}, "fractions": [float(value) for value in report["fractions"][1:]],
               "duration_s": float(report["duration_s"]), "ticks": int(report["ticks"])}
    zoned = slow_pace if slow_approach else None
    speed_receipt = mp.speed_report(speed, pace, zoned)
    speed_receipt.update(report["peaks"])
    return mp.PacedPath(grid, planned, float(report["fraction_of_move"]), pace, zoned, speed_receipt,
                        dict(report["wrist"]), report["slow_from_tick"] if zoned is not None else None,
                        int(report["needed_ticks"]), {"timing": timing_evidence(answer)})


atexit.register(close)

__all__ = ["DEFAULT_JERK_RAD_S3", "FALLBACK_IMPLEMENTATION", "JOINT_JERK_LIMITS", "PACE_JERK", "PREFLIGHT_CHECK",
           "UNVERIFIED", "pace_jerk",
           "bind_preflight", "binding", "client", "close", "configure", "jerk_limits", "library_pace", "reset",
           "timing_evidence"]
