"""The YAM station's arm paths timed by ORA's shared timing worker (``runtime.motion_timing``, off by default).

The station plans a path itself (``motion.plan_arm_path``: IK samples along its Cartesian line) and, without this,
times it itself too: each sample at the interaction's fixed speeds, interpolated linearly at the control tick, so a
motion starts and stops at full speed. With ``runtime.motion_timing`` the same samples go to the TOPP-RA + Ruckig
worker (``motion_pace.pace_path``, ``ora/environment/planners/timing/``) under the interaction's own caps
(``speed_request``), which adds what the station's timing lacks: an acceleration cap at the controlled point and at
every joint, and a jerk-limited start and stop. Each straight piece of the path (lift, traverse, descend) is timed on
its own, rest to rest, so the arm stops at every corner and its rows pass through the corner itself. The rows are
samples of the same path; ``motion`` runs its per-tick checks on them.

The worker is bound as on RoboTwin: preflight starts it (``motion_timing_worker``) and the session reads the
interpreter from the run's ``preflight.json`` (``motion_timing.bind_preflight``). A session with the switch and no
worker (a direct session without ``ORA_TIMING_PYTHON``) keeps the station's own timing and its receipts say so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.motion_speed import EXPLICIT_SPEED
from ora.environment.controllers import motion_pace, motion_timing, yam_station as Y
from ora.environment.controllers.motion_retime import TIMING_PYTHON_ENV

from .kinematics import pose_xyz_xyzw

#: The jerk table's entry (``motion_timing.JOINT_JERK_LIMITS``): no vendor value is published for the YAM, so the
#: worker certifies its rows at the unverified default and every receipt says ``jerk_source: unverified``.
ROBOT = "yam"
#: The station's own timing, as a receipt's ``motion.timing`` names it.
STATION_IMPLEMENTATION = "yam_fixed_speed"
#: The last arm joint, the one the wrist report reads.
WRIST_JOINT_INDEX = Y.ARM_JOINT_COUNT - 1
#: A path whose largest joint travel is below this does not move: the station's own timing answers its one tick.
STILL_PATH_RAD = 1e-6


def enabled(session: Any) -> bool:
    """Whether this session's runtime selects the shared timing (``runtime.motion_timing``)."""

    return isinstance((getattr(session, "config", None) or {}).get("motion_timing"), Mapping)


def start(session: Any) -> dict[str, Any] | None:
    """Bind the timing worker this run's preflight started; the binding, or None with the switch off.

    A session without a preflight record takes ``ORA_TIMING_PYTHON``; with neither, the station's own timing.
    ``timing/binding.json`` under the session's output names what was bound either way."""

    if not enabled(session):
        return None
    bound = motion_timing.bind_preflight(session.output_root)
    if bound is None or motion_timing.client() is None:
        bound = motion_timing.binding() if motion_timing.client() is not None else station_timing()
        path = Path(session.output_root) / "timing/binding.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bound, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return bound


def close(session: Any) -> None:
    if enabled(session):
        motion_timing.close()


def speed_request(interaction: str) -> dict[str, Any]:
    """An interaction's caps in the shared wire form: the station's speeds, each reached in ``Y.TIMING_RAMP_S``.

    Example: speed_request("precision")["caps"]["linear_m_s"] == 0.05."""

    linear = float(Y.LINEAR_SPEED_M_S[interaction])
    angular = math.degrees(Y.ANGULAR_SPEED_RAD_S[interaction])
    return {"speed": EXPLICIT_SPEED, "slow_approaching": False, "slow_caps": None,
            "caps": {"linear_m_s": linear, "angular_deg_s": angular,
                     "linear_accel_m_s2": linear / Y.TIMING_RAMP_S, "angular_accel_deg_s2": angular / Y.TIMING_RAMP_S,
                     "joint_speed_rad_s": Y.MAX_JOINT_SPEED_RAD_S, "joint_accel_rad_s2": Y.MAX_JOINT_ACCEL_RAD_S2}}


def joint_speed_request(joint_speed_rad_s: float) -> dict[str, Any]:
    """A joint-space move's caps: no cap at the controlled point, every joint at ``joint_speed_rad_s``."""

    return {"speed": EXPLICIT_SPEED, "slow_approaching": False, "slow_caps": None,
            "caps": {"linear_m_s": None, "angular_deg_s": None, "linear_accel_m_s2": None,
                     "angular_accel_deg_s2": None, "joint_speed_rad_s": float(joint_speed_rad_s),
                     "joint_accel_rad_s2": float(joint_speed_rad_s) / Y.TIMING_RAMP_S}}


def station_timing(reason: str | None = None) -> dict[str, Any]:
    """``motion.timing`` of an arm the station timed itself although the switch is on."""

    return {"implementation": STATION_IMPLEMENTATION,
            "reason": reason or f"no timing interpreter ({TIMING_PYTHON_ENV})"}


@dataclass(frozen=True)
class Timed:
    """One arm's path as the shared timing answered it: a row per control tick and its entries in ``motion``."""

    rows: list[np.ndarray]
    speed: dict[str, Any]
    wrist: dict[str, Any]
    timing: dict[str, Any]


def retime(session: Any, arm: str, samples: Sequence[Sequence[float]], speed: Mapping[str, Any],
           stops: Sequence[int] = ()) -> Timed | dict[str, Any] | None:
    """``samples`` (the start first) timed by the shared worker under ``speed``, one row per control tick.

    ``stops`` are the sample indices where the path turns a corner (a waypoint): each piece between them is
    timed rest to rest, so its last row is the corner exactly. None with the switch off: nothing changes. With
    the switch on and no worker, or a path that does not move, the station's own timing applies and the answer
    is its ``motion.timing`` entry. Raises ``motion_pace.PaceRefusal`` (``paced_motion_budget``); the wrist
    turn is reported, never refused, because the path is the station's own Cartesian line, whose turn is the
    requested one by construction."""

    if not enabled(session):
        return None
    points = np.asarray(samples, dtype=float)
    if float(np.max(np.abs(points - points[0]))) <= STILL_PATH_RAD:
        return station_timing("the path does not move")
    if motion_timing.client() is None:
        return station_timing()
    kin = session.kinematics
    limits = kin.arm_limits[arm]
    wrist_limits = (float(limits[WRIST_JOINT_INDEX, 0]), float(limits[WRIST_JOINT_INDEX, 1]))
    caps = speed["caps"]

    def fk(rows: list[list[float]]) -> list[list[float]]:
        return [pose_xyz_xyzw(kin.fk(arm, np.asarray(row, dtype=float))) for row in rows]

    cuts = sorted({0, *(int(index) for index in stops), len(points) - 1})
    rows: list[np.ndarray] = []
    reports, budget = [], Y.TIMING_BUDGET_TICKS
    for first, last in zip(cuts, cuts[1:]):
        piece = points[first:last + 1]
        if float(np.max(np.abs(piece - piece[0]))) <= STILL_PATH_RAD:
            continue
        paced = motion_pace.pace_path(
            piece.tolist(), fk, speed=speed, tick_s=Y.CONTROL_PERIOD_S,
            joint_speed_ceiling_rad_s=caps["joint_speed_rad_s"],
            joint_accel_ceiling_rad_s2=caps["joint_accel_rad_s2"], wrist_joint_index=WRIST_JOINT_INDEX,
            wrist_limits_rad=wrist_limits, budget_ticks=max(1, budget - len(rows)), slow_approach=False, arm=arm,
            robot=ROBOT, unwind_check=False)
        rows.extend(np.asarray(row, dtype=float) for row in paced.rows())
        reports.append(paced.report())
    peaks = ("planned_peak_linear_m_s", "planned_peak_angular_deg_s")
    report = {**reports[0]["speed"],
              **{key: max(item["speed"][key] for item in reports) for key in peaks if key in reports[0]["speed"]}}
    wrist = {**motion_pace.wrist_report(points[0], rows, joint_index=WRIST_JOINT_INDEX, limits_rad=wrist_limits),
             "path_turn_deg": round(sum(item["wrist"]["path_turn_deg"] for item in reports), 2),
             "requested_turn_deg": motion_pace.path_turns(fk([points[0].tolist(), points[-1].tolist()]))[
                 "requested_turn_deg"]}
    implementations = list(dict.fromkeys(item["timing"]["implementation"] for item in reports))
    return Timed(rows, report, wrist, {"implementation": "+".join(implementations),
                                       "segments": [item["timing"] for item in reports]})


def motion_report(timed: Mapping[str, Timed | dict[str, Any]], ticks: int) -> dict[str, Any] | None:
    """A receipt's ``motion`` for the arms the switch covers: speed, wrist and timing per arm, under its name.

    ``speed`` and ``wrist`` hold the arms the worker timed; ``timing`` every arm, the station's own included."""

    if not timed:
        return None
    paced = {arm: value for arm, value in timed.items() if isinstance(value, Timed)}
    return {**({"speed": {arm: dict(value.speed) for arm, value in paced.items()},
                "wrist": {arm: dict(value.wrist) for arm, value in paced.items()}} if paced else {}),
            "planned_ticks": int(ticks), "duration_s": ticks * Y.CONTROL_PERIOD_S,
            "timing": {arm: dict(value.timing if isinstance(value, Timed) else value) for arm, value in timed.items()}}


__all__ = ["ROBOT", "STATION_IMPLEMENTATION", "Timed", "close", "enabled", "joint_speed_request", "motion_report",
           "retime", "speed_request", "start", "station_timing"]
