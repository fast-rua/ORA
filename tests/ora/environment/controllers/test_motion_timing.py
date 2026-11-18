"""Which timing a paced move uses: the fallback without a worker, the worker's answer with one, and the jerk table."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from ora.environment.controllers import motion_pace as mp
from ora.environment.controllers import motion_timing
from ora.environment.controllers.motion_planner import MotionPlannerError
from ora.environment.controllers.motion_retime import (
    RETIME_ANSWER_SCHEMA, TIMING_IMPLEMENTATION, TimingClient, validate_retime_answer,
)

TICK = 0.05
LIMITS = (-2.8973, 2.8973)
Q0 = [0.0] * 7
QX = [0.2, 0, 0, 0, 0, 0, 0]


def fk(rows):
    return [[0.4 + 0.3 * q[0], 0.3 * q[1], 0.5, 0.0, 0.0, math.sin(q[6] / 2), math.cos(q[6] / 2)] for q in rows]


def paced(speed=None, **options):
    arguments = dict(tick_s=TICK, joint_speed_ceiling_rad_s=1.0, wrist_joint_index=6, wrist_limits_rad=LIMITS,
                     budget_ticks=2000)
    arguments.update(options)
    return mp.pace_path([Q0, QX], fk, speed=speed, **arguments)


class Fake:
    """A timing client answering from ``respond``; every request is validated as the real one's is."""

    def __init__(self, respond: Any = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.respond = respond or timed

    def retime(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return validate_retime_answer(self.respond(request), request)


def timed(request: dict[str, Any], *, fraction: float = 1.0, segments: int | None = None) -> dict[str, Any]:
    nodes = request["path"]["joint_positions"]
    rows = [nodes[0], nodes[len(nodes) // 2], nodes[-1]]
    zeros = [[0.0] * 7] * 3
    return {"schema": RETIME_ANSWER_SCHEMA, "status": "timed", "reason": None,
            "trajectory": {"joint_names": request["joint_names"], "positions": rows, "velocities": zeros,
                           "accelerations": zeros, "dt_s": request["tick_s"]},
            "report": {"ticks": 2, "duration_s": 0.1, "fractions": [0.0, 0.5, fraction], "slow_from_tick": 2,
                       "fraction_of_move": fraction, "needed_ticks": 5, "part_segments": segments or len(nodes) - 1,
                       "peaks": {"planned_peak_linear_m_s": 0.29, "planned_peak_angular_deg_s": 1.5},
                       "wrist": {"travel_deg": 0.0, "path_turn_deg": 0.0}},
            "planner": {"implementation": TIMING_IMPLEMENTATION, "versions": {"toppra": "0.6.10", "ruckig": "0.19.4"}},
            "timing_s": {"total": 0.01}, "request": request,
            "detail": {"jerk_source": request["limits"]["jerk_source"], "attempts": 1, "acceleration_scale": 1.0,
                       "ruckig_extended_segments": 0, "max_deviation_m": 0.0, "solve_ms": 5.0,
                       "verification": {"max_linear_over_cap": 0.97}}}


def refused(reason: str, message: str, **constraints: Any) -> Any:
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        return {"schema": RETIME_ANSWER_SCHEMA, "status": "refused", "reason": reason, "trajectory": None,
                "report": None, "planner": {"implementation": TIMING_IMPLEMENTATION}, "timing_s": {"total": 0.01},
                "request": request, "detail": {"refusal": {"message": message, "constraints": constraints}}}
    return respond


def use(monkeypatch: pytest.MonkeyPatch, timing: Any) -> Any:
    monkeypatch.setattr(motion_timing, "client", lambda: timing)
    return timing


def test_without_a_worker_the_fallback_times_the_move_and_says_so(monkeypatch):
    monkeypatch.delenv("ORA_TIMING_PYTHON", raising=False)
    assert motion_timing.client() is None
    evidence = paced().report()["timing"]
    assert evidence == {"implementation": "motion_pace", "reason": "no timing interpreter (ORA_TIMING_PYTHON)"}
    assert motion_timing.binding() == {"implementation": "motion_pace"}


def test_the_environment_selects_the_worker_and_configure_overrides_it(monkeypatch, tmp_path):
    monkeypatch.setenv("ORA_TIMING_PYTHON", sys.executable)
    chosen = motion_timing.client()
    assert isinstance(chosen, TimingClient) and chosen.python_executable == Path(sys.executable).absolute()
    assert motion_timing.client() is chosen
    motion_timing.configure(None)
    assert motion_timing.client() is None
    motion_timing.configure(sys.executable, stderr_path=tmp_path / "log")
    assert motion_timing.client().stderr_path == tmp_path / "log"
    motion_timing.reset()
    monkeypatch.delenv("ORA_TIMING_PYTHON")
    assert motion_timing.client() is None


def test_a_broken_configured_interpreter_is_an_error_not_the_fallback(monkeypatch, tmp_path):
    motion_timing.configure(tmp_path / "absent/python")
    with pytest.raises(MotionPlannerError, match="ORA_TIMING_PYTHON"):
        paced()


def test_a_paced_move_sends_the_grid_and_reads_the_rows_the_worker_answers(monkeypatch):
    timing = use(monkeypatch, Fake())
    speed = {"speed": "normal", "slow_approaching": True}
    result = paced(speed, robot="franka_panda", arm="right", joint_accel_ceiling_rad_s2=9.0)
    request = timing.requests[0]
    nodes = np.asarray(request["path"]["joint_positions"])
    # The planner's path on the 0.02 rad joint grid, with the caller's kinematics at every node.
    assert len(nodes) == 11 and np.abs(np.diff(nodes, axis=0)).max() <= 0.02 + 1e-9
    assert request["path"]["tool_poses_xyzw"] == fk(nodes.tolist())
    assert request["joint_names"][0] == "right_joint0" and request["speed"] == speed and request["slow_approach"]
    limits = request["limits"]
    assert limits["joint_velocity_rad_s"] == [1.0] * 7 and limits["joint_acceleration_rad_s2"] == [9.0] * 7
    assert limits["joint_jerk_rad_s3"] == list(motion_timing.JOINT_JERK_LIMITS["franka_panda"]["rad_s3"])
    assert "Franka" in limits["jerk_source"]
    assert request["wrist"] == {"joint_index": 6, "limits_rad": list(LIMITS)}
    # The knots leave out the start, as motion_pace's do.
    assert result.rows() == [nodes[5].tolist(), nodes[-1].tolist()] and result.knots["ticks"] == 2
    assert result.knots["fractions"] == [0.5, 1.0] and result.slow_from_knot == 2 and result.whole_ticks == 5
    assert result.fraction == 1.0 and result.grid.joints["right"].shape == nodes.shape
    report = result.report()
    assert report["speed"]["linear_m_s"] == 0.3 and report["speed"]["planned_peak_linear_m_s"] == 0.29
    assert report["speed"]["slow_approach_m"] == 0.02 and report["planned_ticks"] == 2
    assert report["timing"]["implementation"] == TIMING_IMPLEMENTATION
    assert report["timing"]["versions"] == {"toppra": "0.6.10", "ruckig": "0.19.4"}
    assert report["timing"]["verification"]["max_linear_over_cap"] == 0.97


def test_a_move_without_a_slow_zone_reports_none(monkeypatch):
    use(monkeypatch, Fake())
    assert paced({"speed": "normal", "slow_approaching": True}, slow_approach=False).slow_from_knot is None


def test_a_cut_move_keeps_the_part_of_the_grid(monkeypatch):
    use(monkeypatch, Fake(lambda request: timed(request, fraction=0.5, segments=5)))
    result = paced(cut_to_budget=True)
    assert result.fraction == 0.5 and result.grid.segments == 5
    assert result.report()["budget_limited"] == {"fraction_of_move": 0.5, "needed_ticks": 5}


@pytest.mark.parametrize("code", ["paced_path_wrist_unwind", "paced_motion_budget"])
def test_the_worker_s_refusals_are_the_pacing_s(monkeypatch, code):
    use(monkeypatch, Fake(refused(code, "nothing was executed", tick_budget=2)))
    with pytest.raises(mp.PaceRefusal) as error:
        paced()
    assert error.value.code == code and str(error.value) == "nothing was executed"
    assert error.value.constraints == {"tick_budget": 2}


def test_a_path_the_library_cannot_time_falls_back_and_says_why(monkeypatch):
    use(monkeypatch, Fake(refused("timing_infeasible", "Ruckig cannot follow", attempts=8)))
    result = paced({"speed": "normal", "slow_approaching": False})
    evidence = result.report()["timing"]
    assert evidence == {"implementation": "motion_pace", "reason": "timing_infeasible: Ruckig cannot follow"}
    assert result.knots["ticks"] > 2


def test_a_worker_error_stops_the_move(monkeypatch):
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        return {"schema": RETIME_ANSWER_SCHEMA, "status": "error", "reason": "planner_error", "trajectory": None,
                "report": None, "planner": {"implementation": TIMING_IMPLEMENTATION}, "timing_s": {"total": 0.0},
                "request": request, "detail": {"error": {"type": "ValueError", "message": "broken"}}}

    use(monkeypatch, Fake(respond))
    with pytest.raises(MotionPlannerError, match="ValueError: broken"):
        paced()


def test_the_hardware_jerk_table_cites_its_values_and_names_what_is_unverified():
    panda = motion_timing.jerk_limits("franka_panda", 7)
    assert panda["rad_s3"] == [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]
    assert "support.franka.de" in panda["source"]
    for robot in ("arx_x5", "yam"):
        unverified = motion_timing.jerk_limits(robot, 6)
        assert unverified == {"rad_s3": [motion_timing.DEFAULT_JERK_RAD_S3] * 6, "source": "unverified"}
    assert motion_timing.jerk_limits(None, 3) == {"rad_s3": [1000.0] * 3, "source": "unverified"}
    with pytest.raises(ValueError, match="jerk limits"):
        motion_timing.jerk_limits("franka_panda", 6)


def test_a_pace_s_own_jerk_applies_only_where_the_robot_documents_none():
    fast = mp.Pace(None, None, None, None, 4.5, 60.0, joint_jerk_rad_s3=5000.0)
    preset = mp.Pace(None, None, None, None, 4.0, 40.0)
    assert motion_timing.pace_jerk("arx_x5", 6, fast) == {"rad_s3": [5000.0] * 6, "source": "pace"}
    assert motion_timing.pace_jerk("arx_x5", 6, preset) == motion_timing.jerk_limits("arx_x5", 6)
    # A documented jerk is never exceeded: the pace's is a ceiling there, and the source stays the robot's.
    panda = motion_timing.pace_jerk("franka_panda", 7, fast)
    assert panda["rad_s3"] == [5000.0, 3750.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
    assert "support.franka.de" in panda["source"]


def test_a_paced_move_with_its_own_jerk_sends_it_to_the_worker(monkeypatch):
    timing = use(monkeypatch, Fake())
    caps = {"linear_m_s": None, "angular_deg_s": None, "linear_accel_m_s2": None, "angular_accel_deg_s2": None,
            "joint_speed_rad_s": 4.5, "joint_accel_rad_s2": 60.0, "joint_jerk_rad_s3": 5000.0}
    speed = {"speed": "explicit", "caps": caps, "slow_caps": None, "slow_approaching": False}
    # The fixture path has 7 joints; a robot without a documented jerk (None, like the ARX X5).
    result = paced(speed, robot=None, joint_speed_ceiling_rad_s=None)
    limits = timing.requests[0]["limits"]
    assert limits["joint_jerk_rad_s3"] == [5000.0] * 7 and limits["jerk_source"] == "pace"
    assert result.report()["speed"]["joint_jerk_rad_s3"] == 5000.0
    # Without one the request is what it always was.
    plain = {**speed, "caps": {key: value for key, value in caps.items() if key != "joint_jerk_rad_s3"}}
    plain_result = paced(plain, robot=None, joint_speed_ceiling_rad_s=None)
    assert timing.requests[1]["limits"]["jerk_source"] == "unverified"
    assert "joint_jerk_rad_s3" not in plain_result.report()["speed"]


# ---- a worker whose config cannot carry the Supervisor's binding reads the run's preflight record

def _preflight(run_root, *checks):
    (run_root / "preflight.json").write_text(json.dumps({"checks": list(checks)}), encoding="utf-8")
    return run_root / "native"


def _passed(sha):
    return {"check": motion_timing.PREFLIGHT_CHECK, "passed": True,
            "detail": {"python": "/opt/timing/bin/python", "python_sha256": sha}}


def _configured(monkeypatch, sha):
    seen = {}

    def configure(python, *, stderr_path=None):
        seen.update(python=python, stderr_path=stderr_path)

    monkeypatch.setattr(motion_timing, "configure", configure)
    monkeypatch.setattr(motion_timing, "binding", lambda: {"implementation": "toppra_ruckig", "python_sha256": sha})
    return seen


def test_a_direct_session_has_no_record_and_the_environment_still_decides(tmp_path, monkeypatch):
    seen = _configured(monkeypatch, "a" * 64)
    assert motion_timing.bind_preflight(tmp_path / "native") is None and not seen


def test_the_bound_interpreter_times_the_worker_and_the_binding_is_recorded(tmp_path, monkeypatch):
    seen = _configured(monkeypatch, "a" * 64)
    native = _preflight(tmp_path, _passed("a" * 64))
    assert motion_timing.bind_preflight(native)["implementation"] == "toppra_ruckig"
    assert seen == {"python": "/opt/timing/bin/python", "stderr_path": native / "timing/worker.stderr.log"}
    assert json.loads((native / "timing/binding.json").read_text())["python_sha256"] == "a" * 64


def test_a_changed_interpreter_is_refused(tmp_path, monkeypatch):
    _configured(monkeypatch, "b" * 64)
    with pytest.raises(MotionPlannerError, match="changed after preflight"):
        motion_timing.bind_preflight(_preflight(tmp_path, _passed("a" * 64)))


def test_a_run_without_the_check_takes_the_fallback_explicitly(tmp_path, monkeypatch):
    seen = _configured(monkeypatch, "a" * 64)
    assert motion_timing.bind_preflight(_preflight(tmp_path, {"check": "gpu", "passed": True, "detail": ""}))
    assert seen == {"python": None, "stderr_path": None}


@pytest.mark.parametrize("record", [
    {"check": motion_timing.PREFLIGHT_CHECK, "passed": False, "detail": "ORA_TIMING_PYTHON is missing"},
    {"check": motion_timing.PREFLIGHT_CHECK, "passed": True, "detail": {"python": "/opt/timing/bin/python"}},
])
def test_a_bad_binding_is_refused(tmp_path, monkeypatch, record):
    _configured(monkeypatch, "a" * 64)
    with pytest.raises(MotionPlannerError, match="absent or invalid"):
        motion_timing.bind_preflight(_preflight(tmp_path, record))
