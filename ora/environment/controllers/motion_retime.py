"""Timing a joint path: the ``retime`` contract and the client of the timing worker.

Timing is a service of its own beside planning. ``retime`` takes a joint path
with the controlled point's pose at every row (the caller's forward kinematics,
so the service knows no robot), a speed request (``ora.contracts.motion_speed``)
and the joints' limits, and answers the path timed at the backend's control tick
(``ora.motion-retime-answer.v1``) or one neutral refusal. The TOPP-RA + Ruckig
worker (``ora/environment/planners/timing/``) answers it under its own small
interpreter (``ORA_TIMING_PYTHON``); ``TimingClient`` starts and talks to it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.motion_speed import validate_speed

from .motion_worker import (
    ERROR_REASONS, JsonLinesWorker, MotionPlannerError, check, file_sha256, number, project_root, unit_pose,
    validate_common, validate_trajectory, vector,
)

RETIME_REQUEST_SCHEMA = "ora.motion-retime-request.v1"
RETIME_ANSWER_SCHEMA = "ora.motion-retime-answer.v1"
TIMING_BINDING_SCHEMA = "ora.motion-timing-binding.v1"
#: Why a path was not timed: the pacing's refusals (``motion_pace``), and ``timing_infeasible``, a path no
#: timing within the limits could follow (TOPP-RA found none, or Ruckig could not certify it at the
#: joints' jerk limits). A plan v2 answer whose timing refused carries the same reason.
TIMING_REFUSAL_REASONS = ("paced_path_wrist_unwind", "paced_motion_budget", "timing_infeasible")
RETIME_STATUSES = ("timed", "refused", "error")
MAXIMUM_RETIME_ROWS = 20000
#: The registered timing implementation and its worker (``ora/environment/planners/timing/``).
TIMING_IMPLEMENTATION = "toppra_ruckig"
TIMING_WORKER_MODULE = "ora.environment.planners.timing.worker"
#: The environment variable naming the timing worker's interpreter.
TIMING_PYTHON_ENV = "ORA_TIMING_PYTHON"


TIMING_FIELDS = {"speed", "slow_approach", "limits", "tick_s", "budget_ticks", "cut_to_budget", "wrist"}
_RETIME_REQUEST_FIELDS = {"schema", "joint_names", "path"} | TIMING_FIELDS
_RETIME_REPORT_FIELDS = {"ticks", "duration_s", "fractions", "slow_from_tick", "fraction_of_move", "needed_ticks",
                         "part_segments", "peaks", "wrist"}


def _positivevector(value: Any, length: int) -> bool:
    return vector(value, length) and all(float(item) > 0 for item in value)


def validate_timing_fields(value: Mapping[str, Any], joint_count: int) -> None:
    """The speed, limits, tick, budget and wrist a path is timed under (a retime request, a plan v2's timing)."""
    if value["speed"] is not None:
        try:
            validate_speed(value["speed"])
        except ValueError as exc:
            raise MotionPlannerError(f"timing speed is malformed: {exc}") from exc
    check(type(value["slow_approach"]) is bool and type(value["cut_to_budget"]) is bool,
          "timing slow_approach and cut_to_budget are booleans")
    limits = value["limits"]
    check(isinstance(limits, Mapping) and set(limits) == {"joint_velocity_rad_s", "joint_acceleration_rad_s2",
                                                          "joint_jerk_rad_s3", "jerk_source"},
          "timing limits fields drifted")
    check(all(limits[key] is None or _positivevector(limits[key], joint_count)
              for key in ("joint_velocity_rad_s", "joint_acceleration_rad_s2"))
          and _positivevector(limits["joint_jerk_rad_s3"], joint_count)
          and isinstance(limits["jerk_source"], str) and bool(limits["jerk_source"]),
          "timing limits are one positive value per joint (velocity and acceleration may be null) and a source")
    check(number(value["tick_s"], positive=True) and float(value["tick_s"]) <= 1, "timing tick is malformed")
    check(type(value["budget_ticks"]) is int and value["budget_ticks"] >= 1, "timing budget is malformed")
    wrist = value["wrist"]
    check(isinstance(wrist, Mapping) and set(wrist) - {"unwind_check"} == {"joint_index", "limits_rad"}
          and wrist.get("unwind_check", True) in (True, False)
          and type(wrist["joint_index"]) is int and 0 <= wrist["joint_index"] < joint_count
          and vector(wrist["limits_rad"], 2) and wrist["limits_rad"][0] < wrist["limits_rad"][1],
          "timing wrist names one joint and its limits")


def validate_retime_request(value: Any) -> dict[str, Any]:
    """The closed retime request: a joint path, the controlled point's pose per row, and how to time it."""
    check(isinstance(value, Mapping) and value.get("schema") == RETIME_REQUEST_SCHEMA
          and set(value) == _RETIME_REQUEST_FIELDS, "retime request fields drifted")
    names = value["joint_names"]
    check(isinstance(names, list) and names and all(isinstance(name, str) and name for name in names)
          and len(set(names)) == len(names), "retime joint names are distinct strings")
    path = value["path"]
    check(isinstance(path, Mapping) and set(path) == {"joint_positions", "tool_poses_xyzw"},
          "retime path fields drifted")
    rows, poses = path["joint_positions"], path["tool_poses_xyzw"]
    check(isinstance(rows, list) and isinstance(poses, list) and 1 <= len(rows) <= MAXIMUM_RETIME_ROWS
          and len(poses) == len(rows), f"a retime path holds 1 to {MAXIMUM_RETIME_ROWS} rows, one pose each")
    check(all(vector(row, len(names)) for row in rows) and all(unit_pose(pose) for pose in poses),
          "retime path rows are finite joints and unit-quaternion poses")
    validate_timing_fields(value, len(names))
    return json.loads(json.dumps(value))


def validate_retime_answer(value: Any, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate one retime answer, against its request when given; returns a JSON copy."""
    fields = {"schema", "status", "reason", "trajectory", "report", "planner", "timing_s", "request", "detail"}
    validate_common(value, RETIME_ANSWER_SCHEMA, request, fields)
    status, reason = value.get("status"), value.get("reason")
    check(status in RETIME_STATUSES, "retime status is unknown")
    check((status == "timed" and reason is None) or (status == "refused" and reason in TIMING_REFUSAL_REASONS)
          or (status == "error" and reason in ERROR_REASONS), "retime status and reason disagree")
    if status == "error":
        check(value.get("trajectory") is None and isinstance(value["detail"].get("error"), Mapping),
              "a timing error carries no trajectory and names its error in detail")
        return json.loads(json.dumps(value, allow_nan=False))
    check(set(value) == fields, "retime answer fields drifted")
    request = validate_retime_request(value["request"])
    if status == "refused":
        refusal = value["detail"].get("refusal")
        check(value["trajectory"] is None and value["report"] is None and isinstance(refusal, Mapping)
              and isinstance(refusal.get("message"), str) and isinstance(refusal.get("constraints"), Mapping),
              "a refused timing carries no trajectory and names its refusal in detail")
    else:
        trajectory = value["trajectory"]
        validate_trajectory(trajectory, request["joint_names"], timed=True)
        check(float(trajectory["dt_s"]) == float(request["tick_s"]), "timed rows are not one per tick")
        report = value["report"]
        rows = len(trajectory["positions"])
        check(isinstance(report, Mapping) and set(report) == _RETIME_REPORT_FIELDS
              and report["ticks"] == rows - 1 >= 1 and vector(report["fractions"], rows)
              and (report["slow_from_tick"] is None
                   or (type(report["slow_from_tick"]) is int and 1 <= report["slow_from_tick"] <= rows - 1))
              and number(report["fraction_of_move"]) and 0 < report["fraction_of_move"] <= 1
              and type(report["needed_ticks"]) is int and report["needed_ticks"] >= report["ticks"]
              and type(report["part_segments"]) is int and report["part_segments"] >= 1
              and isinstance(report["peaks"], Mapping) and isinstance(report["wrist"], Mapping),
              "retime report is malformed")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MotionPlannerError("retime answer is not finite JSON") from exc


def retime_request(timing: Mapping[str, Any], joint_names: list[str], joint_positions: list[list[float]],
                   tool_poses_xyzw: list[list[float]]) -> dict[str, Any]:
    """The retime request of a plan v2's ``timing`` for the path the planner answered."""
    return validate_retime_request({
        "schema": RETIME_REQUEST_SCHEMA, "joint_names": list(joint_names),
        "path": {"joint_positions": joint_positions, "tool_poses_xyzw": tool_poses_xyzw},
        **{key: timing[key] for key in TIMING_FIELDS}})


@dataclass
class TimingClient(JsonLinesWorker):
    """Start the timing worker under its interpreter and retime paths with it.

    The worker is ORA's own module (``TIMING_WORKER_MODULE``), run from this
    checkout under ``python_executable`` (``ORA_TIMING_PYTHON``): a small CPU
    interpreter holding the timing libraries, which no simulator or planner
    interpreter needs. ``stderr_path`` keeps its log (``None``: this process's).
    """

    python_executable: Path
    stderr_path: Path | None = None
    request_timeout_s: float = 60.0
    environment_overrides: Mapping[str, str] | None = None
    _process: subprocess.Popen[str] | None = field(init=False, default=None, repr=False)
    _sequence: int = field(init=False, default=0)
    binding: dict[str, Any] = field(init=False, default_factory=dict)
    _label = "motion timing"

    def __post_init__(self) -> None:
        self.python_executable = Path(self.python_executable).expanduser().absolute()
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise MotionPlannerError(f"motion timing Python is unavailable or not executable ({TIMING_PYTHON_ENV})")
        self.binding = {"schema": TIMING_BINDING_SCHEMA, "implementation": TIMING_IMPLEMENTATION,
                        "python": str(self.python_executable),
                        "python_sha256": file_sha256(self.python_executable)}

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            return dict(self.binding)
        environment = os.environ.copy()
        environment.update(self.environment_overrides or {})
        # toppra imports matplotlib.pyplot at module level; no display is ever needed.
        environment.setdefault("MPLBACKEND", "Agg")
        # One solve at a time: a worker per run must not start a BLAS pool per core.
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        # Only this checkout: a simulator's PYTHONPATH must not shadow the worker's pinned numpy.
        environment["PYTHONPATH"] = str(project_root())
        stderr: Any = None
        if self.stderr_path is not None:
            Path(self.stderr_path).parent.mkdir(parents=True, exist_ok=True)
            stderr = Path(self.stderr_path).open("a", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                [str(self.python_executable), "-m", TIMING_WORKER_MODULE], cwd=project_root(), env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        finally:
            if stderr is not None:
                stderr.close()
        ready = self._exchange("ready").get("ready")
        if not isinstance(ready, Mapping) or ready.get("implementation") != TIMING_IMPLEMENTATION:
            self.close()
            raise MotionPlannerError("motion timing worker readiness identity drift")
        self.binding["worker"] = {"pid": self._process.pid, "ready": dict(ready)}
        return dict(self.binding)

    def retime(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_retime_request(request)
        self.start()
        answer = self._exchange("retime", validated).get("answer")
        try:
            answer = validate_retime_answer(answer, validated)
            if answer["planner"]["implementation"] != TIMING_IMPLEMENTATION:
                raise MotionPlannerError("motion timing answer names another implementation")
        except MotionPlannerError:
            self._invalidate()
            raise
        return answer


__all__ = ["MAXIMUM_RETIME_ROWS", "RETIME_ANSWER_SCHEMA", "RETIME_REQUEST_SCHEMA", "RETIME_STATUSES",
           "TIMING_BINDING_SCHEMA", "TIMING_FIELDS", "TIMING_IMPLEMENTATION", "TIMING_PYTHON_ENV",
           "TIMING_REFUSAL_REASONS", "TIMING_WORKER_MODULE", "TimingClient", "retime_request",
           "validate_retime_answer", "validate_retime_request", "validate_timing_fields"]
