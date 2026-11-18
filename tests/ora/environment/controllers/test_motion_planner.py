"""The planner-neutral contract and the planner client's lifecycle."""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ora.environment.controllers import motion_planner
from ora.environment.controllers.depth_world import WORLD_MODEL
from ora.environment.controllers.motion_planner import (
    FAILURE_REASONS, IK_ANSWER_SCHEMA, IK_REQUEST_SCHEMA, PLAN_ANSWER_SCHEMA, PLAN_ANSWER_SCHEMA_V2,
    PLAN_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA_V2, MotionPlannerClient, MotionPlannerError, planned_retime_request,
    planner_client, planning_diagnostics, timed_plan_answer, validate_ik_answer, validate_ik_request,
    validate_plan_answer, validate_plan_request,
)
from ora.environment.controllers.motion_retime import (
    RETIME_ANSWER_SCHEMA, RETIME_REQUEST_SCHEMA, TIMING_IMPLEMENTATION, TIMING_WORKER_MODULE, TimingClient,
    validate_retime_answer, validate_retime_request,
)

NAMES = [f"panda_joint{index}" for index in range(1, 8)]
JOINTS = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
POSE = [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
WORLD_INPUT = {"path": "/inputs/world.npz", "sha256": "a" * 64}


def _request(world: bool = True, goal: str = "pose") -> dict[str, Any]:
    return {
        "schema": PLAN_REQUEST_SCHEMA, "robot": "franka_panda",
        "start": {"joint_names": NAMES, "joint_positions": JOINTS},
        "goal": ({"tool_frame": "panda_hand", "pose_base_xyzw": POSE} if goal == "pose"
                 else {"joint_positions": [0.1] * 7}),
        "world": {"model": WORLD_MODEL, "input": dict(WORLD_INPUT)} if world else {"model": "none"},
        "held_object": {"source": "world" if world else "none"}, "constraints": [],
    }


def _answer(request: dict[str, Any], status: str = "planned", reason: str | None = None) -> dict[str, Any]:
    world = None
    if request["world"]["model"] != "none":
        world = {"model": WORLD_MODEL, "sha256": "b" * 64, "inputs_sha256": "c" * 64,
                 "summary": {"goal_refusal": None}}
    return {
        "schema": PLAN_ANSWER_SCHEMA, "status": status, "reason": reason,
        "trajectory": {"joint_names": NAMES, "positions": [JOINTS, JOINTS], "velocities": None, "dt_s": 0.02}
        if status == "planned" else None,
        "start_tool_pose_base_xyzw": POSE, "world": world, "planner": {"implementation": "curobo_v2"},
        "timing_s": {"total": 1.5, "world": 0.5}, "request": request, "detail": {},
    }


def test_plan_requests_are_closed_and_world_bound() -> None:
    assert validate_plan_request(_request()) == _request()
    validate_plan_request(_request(world=False, goal="joints"))
    for mutate, match in (
        (lambda r: r["start"].update(joint_names=NAMES[::-1]), "start must name"),
        (lambda r: r["goal"].update(tool_frame="panda_link8"), "pose goal"),
        (lambda r: r["goal"].update(pose_base_xyzw=[0.5, 0, 0.3, 2.0, 0, 0, 0]), "unit quaternion"),
        (lambda r: r["world"]["input"].update(path="relative.npz"), "absolute path"),
        (lambda r: r.update(world={"model": "none"}), "needs a world"),
        (lambda r: r.update(held_object={"source": "spheres", "spheres_tool_frame": [[0, 0, 0.1, -1]]}),
         "held spheres"),
        (lambda r: r.update(constraints=[{"kind": "pose_axes"}]), "no motion constraint"),
        (lambda r: r.update(extra=1), "fields drifted"),
    ):
        request = _request()
        mutate(request)
        with pytest.raises(MotionPlannerError, match=match):
            validate_plan_request(request)


def test_plan_answers_tie_status_reason_trajectory_and_world() -> None:
    request = _request()
    assert validate_plan_answer(_answer(request), request)["status"] == "planned"
    for reason in FAILURE_REASONS[:6]:
        assert validate_plan_answer(_answer(request, "refused", reason))["reason"] == reason
    for mutate, match in (
        (lambda a: a.update(reason="goal_unreachable"), "status and reason"),
        (lambda a: a.update(status="refused", reason="IK_FAIL"), "status and reason"),
        (lambda a: a.update(status="refused", reason="planner_error"), "status and reason"),
        (lambda a: a.update(status="refused", reason="path_blocked"), "carries a trajectory"),
        (lambda a: a["trajectory"].update(joint_names=NAMES[::-1]), "joint order"),
        (lambda a: a.update(world=None), "world names"),
        (lambda a: a.update(planner={}), "implementation"),
        (lambda a: a.update(request={**request, "constraints": []} | {"robot": "x"}), "request evidence"),
    ):
        answer = _answer(request)
        mutate(answer)
        with pytest.raises(MotionPlannerError, match=match):
            validate_plan_answer(answer, request)
    robot_only = _request(world=False)
    with pytest.raises(MotionPlannerError, match="robot-only answer carries a world"):
        validate_plan_answer({**_answer(robot_only), "world": _answer(request)["world"]})
    error = motion_planner.error_answer(PLAN_ANSWER_SCHEMA, None, "planner_error", ValueError("boom"),
                                        implementation="curobo_v2", started=0.0)
    assert validate_plan_answer(error)["detail"]["error"] == {"type": "ValueError", "message": "boom"}


def test_ik_requests_and_answers_chain_queries() -> None:
    request = {
        "schema": IK_REQUEST_SCHEMA, "robot": "franka_panda",
        "start": {"joint_names": NAMES, "joint_positions": JOINTS},
        "queries": [{"tool_frame": "panda_hand", "pose_base_xyzw": POSE, "from_index": None},
                    {"tool_frame": "panda_hand", "pose_base_xyzw": POSE, "from_index": 0}],
        "world": {"model": "none"}, "held_object": {"source": "none"},
    }
    validate_ik_request(request)
    with pytest.raises(MotionPlannerError, match="earlier query"):
        validate_ik_request({**request, "queries": [{**request["queries"][1], "from_index": 1}]})
    blank = {"solved": False, "skipped": False, "refusal": None, "joint_positions": None,
             "tool_pose_base_xyzw": None, "position_error_m": 0.2, "rotation_error": 0.1}
    answer = {"schema": IK_ANSWER_SCHEMA, "status": "solved", "reason": None, "joint_names": NAMES,
              "joint_limits": [[-3.0] * 7, [3.0] * 7], "start_tool_pose_base_xyzw": POSE,
              "results": [blank, {**blank, "skipped": True, "position_error_m": None, "rotation_error": None}],
              "world": None, "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": 0.1},
              "request": request, "detail": {}}
    assert validate_ik_answer(answer, request)["results"][1]["skipped"] is True
    orphan = copy.deepcopy(answer)
    orphan["results"][1] = {**blank, "solved": True, "joint_positions": JOINTS, "tool_pose_base_xyzw": POSE}
    with pytest.raises(MotionPlannerError, match="orphan"):
        validate_ik_answer(orphan, request)


def test_planning_diagnostics_are_library_free() -> None:
    request = _request()
    unreachable = planning_diagnostics(_answer(request, "refused", "goal_unreachable"))
    assert unreachable["phase"] == "ik" and unreachable["outcome"] == "goal_unreachable"
    assert unreachable["recovery"][0].startswith("Try a different target")
    below = _answer(request, "refused", "goal_below_support")
    below["world"]["summary"]["goal_refusal"] = {"message": "raise the target"}
    assert planning_diagnostics(below)["recovery"] == ["Refused before IK: raise the target."]
    assert planning_diagnostics(_answer(request))["outcome"] == "planned"
    from ora.contracts.planning_diagnostics import validate_planning_diagnostics

    validate_planning_diagnostics(unreachable)
    assert unreachable["details_available"] is False and unreachable["candidate_counts"] == {}


def test_planning_diagnostics_count_the_last_captured_ik_attempt() -> None:
    from ora.contracts.planning_diagnostics import validate_planning_diagnostics

    def attempt(accepted: int) -> dict[str, Any]:
        return {"status": "captured", "candidate_counts": {"optimized": 32, "accepted": accepted},
                "rejection_counts": {"solver_infeasible": 32 - accepted, "self_collision": None},
                "unknown_reasons": ["per_term_constraint_costs_not_retained_by_sdk"]}

    answer = _answer(_request(), "refused", "goal_unreachable")
    answer["detail"]["ik_attempts"] = [attempt(3), attempt(0)]
    diagnostics = validate_planning_diagnostics(planning_diagnostics(answer))
    assert diagnostics["details_available"] is True
    assert diagnostics["candidate_counts"] == {"optimized": 32, "accepted": 0}
    assert diagnostics["rejection_counts"] == {"solver_infeasible": 32, "self_collision": None}
    assert diagnostics["evidence_scope"] == "last_recorded_original_ik_attempt"
    answer["detail"]["ik_attempts"] = [attempt(3), {"status": "unavailable", "reason": "ik_metric_capture_failed"}]
    assert planning_diagnostics(answer)["details_available"] is False


#: A CPU worker speaking the protocol: ready, then one canned answer per request.
_WORKER = r'''
import json, sys
argv = sys.argv[1:]
mode = argv[argv.index("--world-model") + 1]
lie = argv[0]
for line in sys.stdin:
    envelope = json.loads(line)
    reply = {"protocol": envelope["protocol"], "request_id": envelope["request_id"], "command": envelope["command"]}
    if envelope["command"] == "close":
        print(json.dumps({**reply, "closed": True}), flush=True)
        break
    if envelope["command"] == "ready":
        reply["ready"] = {"implementation": "curobo_v2", "robot": "franka_panda",
                          "world_model": mode if lie in ("honest", "silent") else lie}
    elif lie == "silent":
        continue
    else:
        request = envelope["request"]
        reply["answer"] = {"schema": "ora.motion-plan-answer.v1", "status": "refused",
                           "reason": "goal_unreachable", "trajectory": None,
                           "start_tool_pose_base_xyzw": [0.5, 0, 0.3, 1, 0, 0, 0], "world": None,
                           "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": 0.1},
                           "request": request, "detail": {}}
    print(json.dumps(reply), flush=True)
'''


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behaviour: str = "honest") -> Any:
    monkeypatch.setattr(motion_planner, "_git", lambda root, *args: "f" * 40 if args[0] == "rev-parse" else "")
    client = MotionPlannerClient("curobo_v2", Path(sys.executable), tmp_path, "f" * 40, tmp_path / "out",
                                 request_timeout_s=5.0)
    launched: list[list[str]] = []
    real_popen = subprocess.Popen

    def launch(arguments: list[str], **kwargs: Any) -> Any:
        launched.append(list(arguments))
        kwargs["stderr"].close()
        return real_popen([sys.executable, "-u", "-c", _WORKER, behaviour, *arguments[3:]],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, bufsize=1)

    monkeypatch.setattr(subprocess, "Popen", launch)
    return client, launched


def test_client_starts_checks_answers_and_closes(tmp_path, monkeypatch) -> None:
    client, launched = _client(tmp_path, monkeypatch)
    binding = client.start()
    assert binding["implementation"] == "curobo_v2" and binding["revision"] == "f" * 40
    assert binding["worker"]["ready"]["world_model"] == "none"
    assert launched[0][1:3] == ["-m", "ora.environment.planners.curobo_v2.worker"]
    answer = client.plan(_request(world=False))
    assert answer["status"] == "refused" and answer["reason"] == "goal_unreachable"
    with pytest.raises(MotionPlannerError, match="plans franka_panda in none"):
        client.plan(_request(world=True))
    client.close()
    assert client._process is None


@pytest.mark.parametrize(("behaviour", "match"), [
    ("public_depth_voxel_v1", "readiness identity"), ("silent", "timed out"),
])
def test_client_retires_a_drifting_or_silent_worker(tmp_path, monkeypatch, behaviour, match) -> None:
    client, _ = _client(tmp_path, monkeypatch, behaviour)
    client.request_timeout_s = 1.0
    with pytest.raises(MotionPlannerError, match=match):
        if behaviour == "silent":
            client.start()
        client.plan(_request(world=False))
    assert client._process is None


def test_client_refuses_an_unknown_implementation_or_unclean_source(tmp_path, monkeypatch) -> None:
    with pytest.raises(MotionPlannerError, match="unknown motion planner"):
        MotionPlannerClient("robokit", Path(sys.executable), tmp_path, "f" * 40, tmp_path)
    monkeypatch.setattr(motion_planner, "_git", lambda root, *args: "e" * 40 if args[0] == "rev-parse" else "")
    with pytest.raises(MotionPlannerError, match="required_revision"):
        MotionPlannerClient("curobo_v2", Path(sys.executable), tmp_path, "f" * 40, tmp_path)


def test_configuration_selects_the_implementation_and_requires_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(motion_planner, "_git", lambda root, *args: "f" * 40 if args[0] == "rev-parse" else "")
    with pytest.raises(MotionPlannerError, match="names no implementation"):
        planner_client({"required_revision": "f" * 40, "world_model": WORLD_MODEL},
                       python_executable=Path(sys.executable), root=tmp_path, output_root=tmp_path)
    neutral = planner_client({"implementation": "curobo_v2", "required_revision": "f" * 40,
                              "world_model": WORLD_MODEL}, python_executable=Path(sys.executable),
                             root=tmp_path, output_root=tmp_path)
    assert isinstance(neutral, MotionPlannerClient) and neutral.world_model == WORLD_MODEL


# -- retime and plan v2 ----------------------------------------------------------------------------

LIMITS = {"joint_velocity_rad_s": [1.0] * 7, "joint_acceleration_rad_s2": None,
          "joint_jerk_rad_s3": [1000.0] * 7, "jerk_source": "unverified"}
TIMING_FIELDS = {"speed": {"speed": "normal", "slow_approaching": True}, "slow_approach": True, "limits": LIMITS,
                 "tick_s": 0.05, "budget_ticks": 200, "cut_to_budget": False,
                 "wrist": {"joint_index": 6, "limits_rad": [-2.8, 2.8]}}
PATH = {"joint_positions": [JOINTS, [value + 0.01 for value in JOINTS]], "tool_poses_xyzw": [POSE, POSE]}


def _retime_request() -> dict[str, Any]:
    return {"schema": RETIME_REQUEST_SCHEMA, "joint_names": NAMES, "path": copy.deepcopy(PATH),
            **copy.deepcopy(TIMING_FIELDS)}


def _retime_answer(request: dict[str, Any], status: str = "timed", reason: str | None = None) -> dict[str, Any]:
    rows = request["path"]["joint_positions"]
    zeros = [[0.0] * 7] * 2
    answer = {"schema": RETIME_ANSWER_SCHEMA, "status": status, "reason": reason, "trajectory": None, "report": None,
              "planner": {"implementation": TIMING_IMPLEMENTATION, "versions": {"toppra": "0.6.10"}},
              "timing_s": {"total": 0.01}, "request": request, "detail": {}}
    if status == "timed":
        answer["trajectory"] = {"joint_names": NAMES, "positions": rows, "velocities": zeros, "accelerations": zeros,
                                "dt_s": 0.05}
        answer["report"] = {"ticks": 1, "duration_s": 0.05, "fractions": [0.0, 1.0], "slow_from_tick": 1,
                            "fraction_of_move": 1.0, "needed_ticks": 1, "part_segments": 1,
                            "peaks": {"planned_peak_linear_m_s": 0.1, "planned_peak_angular_deg_s": 0.0},
                            "wrist": {"min_limit_margin_deg": 10.0}}
    elif status == "refused":
        answer["detail"] = {"refusal": {"message": "no", "constraints": {"tick_budget": 200}}}
    return answer


def test_retime_requests_are_closed() -> None:
    assert validate_retime_request(_retime_request()) == _retime_request()
    for mutate, match in (
        (lambda r: r.update(extra=1), "fields drifted"),
        (lambda r: r["path"]["tool_poses_xyzw"].pop(), "one pose each"),
        (lambda r: r["path"]["tool_poses_xyzw"].__setitem__(0, [0, 0, 0, 2, 0, 0, 0]), "unit-quaternion"),
        (lambda r: r["limits"].update(joint_jerk_rad_s3=[0.0] * 7), "positive value per joint"),
        (lambda r: r["limits"].update(jerk_source=""), "positive value per joint"),
        (lambda r: r.update(speed={"speed": "ludicrous", "slow_approaching": False}), "speed is malformed"),
        (lambda r: r.update(tick_s=0.0), "tick"),
        (lambda r: r.update(budget_ticks=0), "budget"),
        (lambda r: r["wrist"].update(joint_index=7), "wrist"),
    ):
        request = _retime_request()
        mutate(request)
        with pytest.raises(MotionPlannerError, match=match):
            validate_retime_request(request)


def test_retime_answers_tie_status_reason_and_rows() -> None:
    request = _retime_request()
    assert validate_retime_answer(_retime_answer(request), request)["status"] == "timed"
    refused = _retime_answer(request, "refused", "paced_motion_budget")
    assert validate_retime_answer(refused, request)["reason"] == "paced_motion_budget"
    for mutate, match in (
        (lambda a: a.update(reason="goal_unreachable"), "status and reason"),
        (lambda a: a["trajectory"].update(accelerations=None), "accelerations"),
        (lambda a: a["trajectory"].update(dt_s=0.02), "one per tick"),
        (lambda a: a["report"].update(ticks=2), "report is malformed"),
        (lambda a: a["report"].update(fractions=[0.0]), "report is malformed"),
        (lambda a: a["report"].update(slow_from_tick=9), "report is malformed"),
    ):
        answer = _retime_answer(request)
        mutate(answer)
        with pytest.raises(MotionPlannerError, match=match):
            validate_retime_answer(answer, request)
    with pytest.raises(MotionPlannerError, match="names its refusal"):
        validate_retime_answer({**refused, "detail": {}}, request)


def _timed_plan_request() -> dict[str, Any]:
    point = {"tool_frame": "panda_hand", "offset_xyzw": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]}
    return {**_request(world=False), "schema": PLAN_REQUEST_SCHEMA_V2,
            "timing": {**copy.deepcopy(TIMING_FIELDS), "controlled_point": point}}


def _planned_v2(request: dict[str, Any]) -> dict[str, Any]:
    return {**_answer(request), "schema": PLAN_ANSWER_SCHEMA_V2, "timing": None,
            "detail": {"timing_path": copy.deepcopy(PATH)}}


def test_plan_v2_carries_timing_and_v1_stays_closed() -> None:
    request = _timed_plan_request()
    assert validate_plan_request(request) == request
    validate_plan_request({**request, "timing": None})
    with pytest.raises(MotionPlannerError, match="fields drifted"):
        validate_plan_request({**_request(world=False), "timing": None})
    with pytest.raises(MotionPlannerError, match="offset pose"):
        broken = copy.deepcopy(request)
        broken["timing"]["controlled_point"]["tool_frame"] = "panda_link8"
        validate_plan_request(broken)
    with pytest.raises(MotionPlannerError, match="version differs"):
        validate_plan_answer({**_answer(_request(world=False)), "schema": PLAN_ANSWER_SCHEMA_V2, "timing": None})


def test_a_timed_plan_answer_is_the_timing_s_rows_or_its_refusal() -> None:
    request = _timed_plan_request()
    planned = validate_plan_answer(_planned_v2(request), request, untimed=True)
    with pytest.raises(MotionPlannerError):
        validate_plan_answer(planned, request)
    retime = planned_retime_request(planned)
    timed = timed_plan_answer(planned, validate_retime_answer(_retime_answer(retime), retime))
    assert timed["status"] == "planned" and timed["trajectory"] == timed["timing"]["trajectory"]
    assert timed["detail"]["planned_trajectory"]["dt_s"] == 0.02
    refused = timed_plan_answer(planned, _retime_answer(retime, "refused", "paced_path_wrist_unwind"))
    assert refused["status"] == "refused" and refused["reason"] == "paced_path_wrist_unwind"
    assert refused["trajectory"] is None and refused["timing"]["status"] == "refused"
    with pytest.raises(MotionPlannerError, match="trajectory is not its timing"):
        validate_plan_answer({**timed, "trajectory": planned["trajectory"]})
    with pytest.raises(MotionPlannerError, match="keeps its measured path"):
        planned_retime_request({**planned, "detail": {}})


#: A CPU timing worker: ready, then one canned timed answer, or an answer naming another implementation.
_TIMING_WORKER = r'''
import json, sys
behaviour = sys.argv[1]
for line in sys.stdin:
    envelope = json.loads(line)
    reply = {"protocol": envelope["protocol"], "request_id": envelope["request_id"], "command": envelope["command"]}
    if envelope["command"] == "close":
        print(json.dumps({**reply, "closed": True}), flush=True)
        break
    if envelope["command"] == "ready":
        name = "toppra_ruckig" if behaviour != "stranger" else "other"
        reply["ready"] = {"implementation": name, "versions": {"toppra": "0.6.10"}}
    else:
        request = envelope["request"]
        rows = request["path"]["joint_positions"]
        zeros = [[0.0] * 7] * 2
        reply["answer"] = {
            "schema": "ora.motion-retime-answer.v1", "status": "timed", "reason": None,
            "trajectory": {"joint_names": request["joint_names"], "positions": rows, "velocities": zeros,
                           "accelerations": zeros, "dt_s": request["tick_s"]},
            "report": {"ticks": 1, "duration_s": 0.05, "fractions": [0.0, 1.0], "slow_from_tick": None,
                       "fraction_of_move": 1.0, "needed_ticks": 1, "part_segments": 1, "peaks": {}, "wrist": {}},
            "planner": {"implementation": "toppra_ruckig" if behaviour == "honest" else "impostor"},
            "timing_s": {"total": 0.1}, "request": request, "detail": {}}
    print(json.dumps(reply), flush=True)
'''


def _timing_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behaviour: str = "honest") -> Any:
    launched: list[dict[str, Any]] = []
    real_popen = subprocess.Popen

    def launch(arguments: list[str], **kwargs: Any) -> Any:
        launched.append({"arguments": list(arguments), "env": kwargs["env"]})
        return real_popen([sys.executable, "-u", "-c", _TIMING_WORKER, behaviour], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    monkeypatch.setattr(subprocess, "Popen", launch)
    return TimingClient(Path(sys.executable), stderr_path=tmp_path / "timing/worker.stderr.log",
                        request_timeout_s=5.0), launched


def test_timing_client_starts_its_own_worker_and_answers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/a/simulator/site-packages")
    client, launched = _timing_client(tmp_path, monkeypatch)
    binding = client.start()
    assert binding["implementation"] == TIMING_IMPLEMENTATION and binding["worker"]["ready"]["versions"]
    assert launched[0]["arguments"][1:] == ["-m", TIMING_WORKER_MODULE]
    # Only this checkout: a simulator's PYTHONPATH never reaches the worker's pinned numpy.
    assert launched[0]["env"]["PYTHONPATH"] == str(Path(motion_planner.__file__).resolve().parents[3])
    assert launched[0]["env"]["MPLBACKEND"] == "Agg" and launched[0]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert client.retime(_retime_request())["status"] == "timed"
    client.close()
    assert client._process is None


@pytest.mark.parametrize(("behaviour", "match"), [("stranger", "readiness identity"), ("impostor", "another")])
def test_timing_client_retires_a_drifting_worker(tmp_path, monkeypatch, behaviour, match) -> None:
    client, _ = _timing_client(tmp_path, monkeypatch, behaviour)
    with pytest.raises(MotionPlannerError, match=match):
        client.retime(_retime_request())
    assert client._process is None


def test_timing_client_refuses_a_missing_interpreter(tmp_path) -> None:
    with pytest.raises(MotionPlannerError, match="ORA_TIMING_PYTHON"):
        TimingClient(tmp_path / "absent/python")


class _Timing:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def start(self) -> dict[str, Any]:
        return {"implementation": TIMING_IMPLEMENTATION}

    def retime(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return validate_retime_answer(_retime_answer(request), request)

    def close(self) -> None:
        self.requests.append({"closed": True})


def test_a_planner_client_times_a_planned_path_in_one_call(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    timing = _Timing()
    client.timing = timing
    client._exchange = lambda command, request=None: {"answer": _planned_v2(request)}  # type: ignore[method-assign]
    client._process = object()  # type: ignore[assignment]
    answer = client.plan(_timed_plan_request())
    client._process = None
    assert answer["schema"] == PLAN_ANSWER_SCHEMA_V2 and answer["timing"]["status"] == "timed"
    assert timing.requests[0]["path"] == PATH and timing.requests[0]["joint_names"] == NAMES
    assert answer["trajectory"]["dt_s"] == 0.05 and answer["trajectory"]["accelerations"]


def test_a_timed_plan_needs_a_timing_client_and_an_untimed_one_never_starts_it(tmp_path, monkeypatch) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    with pytest.raises(MotionPlannerError, match="needs a timing client"):
        client.plan(_timed_plan_request())
    timing = _Timing()
    client.timing = timing
    refused = {**_answer(_timed_plan_request(), "refused", "path_blocked"), "schema": PLAN_ANSWER_SCHEMA_V2,
               "timing": None}
    client._exchange = lambda command, request=None: {"answer": refused}  # type: ignore[method-assign]
    client._process = object()  # type: ignore[assignment]
    assert client.plan(_timed_plan_request())["reason"] == "path_blocked"
    client._process = None
    assert timing.requests == []


def _prepared_world(path: str = "/inputs/world.npz") -> dict[str, Any]:
    return {"start": {"joint_names": NAMES, "joint_positions": JOINTS},
            "world": {"model": WORLD_MODEL, "input": {"path": path, "sha256": "a" * 64}}}


def test_a_prepare_request_names_one_world_per_arm_at_most() -> None:
    from ora.environment.controllers.motion_planner import (
        MAXIMUM_PREPARED_WORLDS, PREPARE_REQUEST_SCHEMA, validate_prepare_request,
    )

    request = {"schema": PREPARE_REQUEST_SCHEMA, "robot": "franka_panda", "worlds": [_prepared_world()] * 2}
    assert validate_prepare_request(request) == request and MAXIMUM_PREPARED_WORLDS == 2
    for worlds in ([], [_prepared_world()] * 3, [{**_prepared_world(), "world": {"model": "none"}}]):
        with pytest.raises(MotionPlannerError):
            validate_prepare_request({**request, "worlds": worlds})


def test_prepare_never_starts_a_worker_and_exchanges_one_at_a_time(tmp_path, monkeypatch) -> None:
    import threading
    import time as clock

    from ora.environment.controllers.motion_planner import PREPARE_ANSWER_SCHEMA
    from ora.environment.controllers.motion_worker import JsonLinesWorker

    monkeypatch.setattr(motion_planner, "_git", lambda root, *args: "f" * 40 if args[0] == "rev-parse" else "")
    client = MotionPlannerClient("curobo_v2", Path(sys.executable), tmp_path, "f" * 40, tmp_path / "out",
                                 world_model=WORLD_MODEL)
    with pytest.raises(MotionPlannerError, match="not running"):
        client.prepare([_prepared_world()])
    inside, overlaps = [], []

    def exchange(self, command, request=None):
        overlaps.append(bool(inside))
        inside.append(command)
        clock.sleep(0.05)
        inside.pop()
        return {"answer": {"schema": PREPARE_ANSWER_SCHEMA, "status": "prepared", "prepared": 1}}

    monkeypatch.setattr(JsonLinesWorker, "_exchange", exchange)
    client._process = object()
    threads = [threading.Thread(target=client.prepare, args=([_prepared_world()],)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(overlaps) == 4 and not any(overlaps), "the client serializes its exchanges"
    assert client.prepare([_prepared_world()])["status"] == "prepared"
