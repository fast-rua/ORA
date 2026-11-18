"""Planner-neutral motion planning: the request, the answer and the planner worker's lifecycle.

One motion system serves every backend, and the planner behind it is
swappable. A backend asks in robot terms -- start joints, a goal tool pose in
the robot base frame or goal joints, a world reference with its hash, a held
object, constraints -- and gets back a timed joint trajectory, a status and,
for anything not planned, one neutral reason from ``FAILURE_REASONS``. A
planner library's own status names stay inside an answer's ``detail``, which
is evidence and never reaches the policy.

The world is ORA's own (``depth_world.py``, numpy only): the worker builds it
from the request's world input and converts it to its planner's layout with
one small export function. Every answer names the world it was planned in by
the world's sha256, which does not depend on the planner.

An implementation is a worker module started as ``python -m <module>`` under
its own interpreter; it reads one JSON request per line (``ready``, ``plan``,
``ik``, ``close``) and writes one JSON answer per line. ``IMPLEMENTATIONS``
registers each one; ``MotionPlannerClient`` starts, checks, talks to and stops
it. ``prepare`` (``ora.motion-prepare-request.v1``) asks a worker to build the
goal-independent scenes of up to ``MAXIMUM_PREPARED_WORLDS`` world inputs before
any request needs them; it answers no question and changes no answer. The
client serializes its exchanges, so a backend may send ``prepare`` from a
thread of its own while the policy decides.

Timing is a service of its own (``motion_retime.py``). A
``ora.motion-plan-request.v2`` may carry ``timing``: its client then has the path
the planner answered timed before it returns, one call for the backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .depth_world import WORLD_MODEL
from .motion_retime import (
    RETIME_ANSWER_SCHEMA, TIMING_FIELDS, TIMING_REFUSAL_REASONS, TimingClient, retime_request,
    validate_retime_answer, validate_timing_fields,
)
from .motion_worker import (
    ERROR_REASONS, PROTOCOL, JsonLinesWorker, MotionPlannerError, check, file_sha256, number, project_root,
    sha256, unit_pose, validate_common, validate_trajectory, vector,
)

PLAN_REQUEST_SCHEMA = "ora.motion-plan-request.v1"
PLAN_ANSWER_SCHEMA = "ora.motion-plan-answer.v1"
#: v2: the v1 request plus ``timing`` (``None``, or how to time the planned path), and its answer plus
#: ``timing`` (``None``, or the retime answer whose rows the answer's trajectory then is).
PLAN_REQUEST_SCHEMA_V2 = "ora.motion-plan-request.v2"
PLAN_ANSWER_SCHEMA_V2 = "ora.motion-plan-answer.v2"
IK_REQUEST_SCHEMA = "ora.motion-ik-request.v1"
IK_ANSWER_SCHEMA = "ora.motion-ik-answer.v1"
CHECK_REQUEST_SCHEMA = "ora.motion-check-request.v1"
CHECK_ANSWER_SCHEMA = "ora.motion-check-answer.v1"
#: Build the scenes of world inputs now (one per arm of a two-arm robot), so the requests that follow find them.
PREPARE_REQUEST_SCHEMA = "ora.motion-prepare-request.v1"
PREPARE_ANSWER_SCHEMA = "ora.motion-prepare-answer.v1"
MAXIMUM_PREPARED_WORLDS = 2
BINDING_SCHEMA = "ora.motion-planner-binding.v1"

#: Why a goal was not planned. A refusal is an answer about the request; an
#: error is the planner failing (or running out of time) and says nothing
#: about the goal.
REFUSAL_REASONS = (
    "goal_unreachable",      # no joint configuration reaches the goal pose in the world
    "goal_in_collision",     # the goal joints collide with the world or the robot itself
    "goal_below_support",    # the world refused the goal before any solver ran
    "start_in_collision",    # the start joints collide with the world or the robot itself
    "start_outside_limits",  # the start joints are outside the joint limits
    "path_blocked",          # the goal is valid but no collision-free trajectory was found
)
FAILURE_REASONS = REFUSAL_REASONS + ERROR_REASONS
PLAN_STATUSES = ("planned", "refused", "error")
#: ``none`` plans against the robot alone; the other in the public-depth world.
WORLD_MODELS = ("none", WORLD_MODEL)
HELD_OBJECT_SOURCES = ("none", "world", "spheres")
MAXIMUM_IK_QUERIES = 256
MAXIMUM_HELD_SPHERES = 64
MAXIMUM_CHECK_ROWS = 2000

#: Robots an implementation may be asked to plan for: the joint order of every
#: request and trajectory, and the tool frame a pose goal names.
ROBOTS = {
    "franka_panda": {
        "joint_names": tuple(f"panda_joint{index}" for index in range(1, 8)),
        "tool_frame": "panda_hand",
    },
    "arx_x5": {"joint_names": tuple(f"joint{index}" for index in range(1, 7)), "tool_frame": "link6"},
    # Either ALOHA arm: the description's two chains are the same joint for joint, so one model plans both.
    "aloha_agilex": {"joint_names": tuple(f"fl_joint{index}" for index in range(1, 7)), "tool_frame": "fl_link6"},
    # RoboLab's Droid: V2's Franka arm with the Robotiq's spheres on the Franka hand's links.
    "franka_robotiq": {
        "joint_names": tuple(f"panda_joint{index}" for index in range(1, 8)),
        "tool_frame": "panda_hand",
    },
    # Either YAM station arm (``planners/curobo_v2/yam.py``): one arm of i2rt's station URDF, in its own base frame.
    "yam_linear_4310": {"joint_names": tuple(f"left_joint{index}" for index in range(1, 7)), "tool_frame": "left_tcp"},
}
#: The robots whose description is not bundled with the implementation: the client names the file.
ROBOT_FILE_ROBOTS = ("arx_x5", "aloha_agilex", "yam_linear_4310")


@dataclass(frozen=True)
class Implementation:
    """One registered planner: its worker module and how its source is put on the path."""

    worker_module: str
    #: Source directories relative to the implementation root that go on ``PYTHONPATH``.
    source_paths: tuple[str, ...]
    world_models: tuple[str, ...]
    robots: tuple[str, ...]


IMPLEMENTATIONS = {
    "curobo_v2": Implementation(
        worker_module="ora.environment.planners.curobo_v2.worker", source_paths=(".",),
        world_models=WORLD_MODELS, robots=("franka_panda", "arx_x5", "aloha_agilex", "franka_robotiq", "yam_linear_4310"),
    ),
}


def _robot(value: Any) -> dict[str, Any]:
    check(value in ROBOTS, f"unknown robot {value!r}")
    return ROBOTS[value]


def _validate_start(robot: Mapping[str, Any], start: Any) -> None:
    check(isinstance(start, Mapping) and set(start) == {"joint_names", "joint_positions"}
          and start["joint_names"] == list(robot["joint_names"])
          and vector(start["joint_positions"], len(robot["joint_names"])),
          "start must name the robot's joints in order with finite positions")


def _validate_world(world: Any) -> str:
    check(isinstance(world, Mapping) and world.get("model") in WORLD_MODELS, "world model is unknown")
    if world["model"] == "none":
        check(set(world) == {"model"}, "a robot-only request names no world input")
    else:
        source = world.get("input")
        check(set(world) == {"model", "input"} and isinstance(source, Mapping)
              and set(source) == {"path", "sha256"} and isinstance(source["path"], str)
              and Path(source["path"]).is_absolute() and sha256(source["sha256"]),
              "a world input is an absolute path and its sha256")
    return str(world["model"])


def _validate_held(held: Any, world_model: str) -> None:
    check(isinstance(held, Mapping) and held.get("source") in HELD_OBJECT_SOURCES, "held object source is unknown")
    if held["source"] == "spheres":
        spheres = held.get("spheres_tool_frame")
        check(set(held) == {"source", "spheres_tool_frame"} and isinstance(spheres, list)
              and 1 <= len(spheres) <= MAXIMUM_HELD_SPHERES
              and all(vector(sphere, 4) and sphere[3] > 0 for sphere in spheres),
              f"held spheres are 1 to {MAXIMUM_HELD_SPHERES} [x, y, z, r] in the tool frame")
    else:
        check(set(held) == {"source"}, "held object fields drifted")
    check(held["source"] != "world" or world_model != "none", "a held object from the world needs a world")


def _validate_plan_timing(timing: Any, robot: Mapping[str, Any]) -> None:
    check(isinstance(timing, Mapping) and set(timing) == TIMING_FIELDS | {"controlled_point"},
          "plan timing fields drifted")
    point = timing["controlled_point"]
    check(isinstance(point, Mapping) and set(point) == {"tool_frame", "offset_xyzw"}
          and point["tool_frame"] == robot["tool_frame"] and unit_pose(point["offset_xyzw"]),
          "plan timing names the controlled point as an offset pose in the robot's tool frame")
    validate_timing_fields(timing, len(robot["joint_names"]))


_PLAN_REQUEST_FIELDS = {"schema", "robot", "start", "goal", "world", "held_object", "constraints"}


def validate_plan_request(value: Any) -> dict[str, Any]:
    """The closed plan request (v1, or v2 with ``timing``); returns a JSON copy."""
    schema = value.get("schema") if isinstance(value, Mapping) else None
    check(isinstance(value, Mapping) and (
       (schema == PLAN_REQUEST_SCHEMA and set(value) == _PLAN_REQUEST_FIELDS)
       or (schema == PLAN_REQUEST_SCHEMA_V2 and set(value) == _PLAN_REQUEST_FIELDS | {"timing"})),
       "plan request fields drifted")
    robot = _robot(value["robot"])
    if schema == PLAN_REQUEST_SCHEMA_V2 and value["timing"] is not None:
        _validate_plan_timing(value["timing"], robot)
    _validate_start(robot, value["start"])
    goal = value["goal"]
    if isinstance(goal, Mapping) and set(goal) == {"tool_frame", "pose_base_xyzw"}:
        check(goal["tool_frame"] == robot["tool_frame"] and unit_pose(goal["pose_base_xyzw"]),
              "a pose goal names the robot's tool frame and a finite pose with a unit quaternion")
    else:
        check(isinstance(goal, Mapping) and set(goal) == {"joint_positions"}
              and vector(goal["joint_positions"], len(robot["joint_names"])),
              "a goal is a tool pose or the robot's joint positions")
    world_model = _validate_world(value["world"])
    _validate_held(value["held_object"], world_model)
    # Reserved for partial-pose and path constraints; no implementation takes one yet.
    check(value["constraints"] == [], "no motion constraint is supported yet")
    return json.loads(json.dumps(value))


def validate_ik_request(value: Any) -> dict[str, Any]:
    """The closed IK request: start joints and 1..MAXIMUM_IK_QUERIES chained tool poses."""
    check(isinstance(value, Mapping) and value.get("schema") == IK_REQUEST_SCHEMA
          and set(value) == {"schema", "robot", "start", "queries", "world", "held_object"},
          "IK request fields drifted")
    robot = _robot(value["robot"])
    _validate_start(robot, value["start"])
    queries = value["queries"]
    check(isinstance(queries, list) and 1 <= len(queries) <= MAXIMUM_IK_QUERIES,
          f"an IK request holds 1 to {MAXIMUM_IK_QUERIES} queries")
    for index, query in enumerate(queries):
        check(isinstance(query, Mapping) and set(query) == {"tool_frame", "pose_base_xyzw", "from_index"}
              and query["tool_frame"] == robot["tool_frame"] and unit_pose(query["pose_base_xyzw"]),
              "an IK query names the tool frame and a finite pose with a unit quaternion")
        start = query["from_index"]
        check(start is None or (type(start) is int and 0 <= start < index),
              "an IK query's from_index names an earlier query")
    world_model = _validate_world(value["world"])
    _validate_held(value["held_object"], world_model)
    return json.loads(json.dumps(value))


def validate_prepare_request(value: Any) -> dict[str, Any]:
    """The closed prepare request: 1..MAXIMUM_PREPARED_WORLDS starts, each with the world input built there."""
    check(isinstance(value, Mapping) and value.get("schema") == PREPARE_REQUEST_SCHEMA
          and set(value) == {"schema", "robot", "worlds"}, "prepare request fields drifted")
    robot = _robot(value["robot"])
    worlds = value["worlds"]
    check(isinstance(worlds, list) and 1 <= len(worlds) <= MAXIMUM_PREPARED_WORLDS,
          f"a prepare request names 1 to {MAXIMUM_PREPARED_WORLDS} worlds")
    for item in worlds:
        check(isinstance(item, Mapping) and set(item) == {"start", "world"}, "a prepared world names its start")
        _validate_start(robot, item["start"])
        check(_validate_world(item["world"]) != "none", "a robot-only world has no scene to prepare")
    return json.loads(json.dumps(value))


def validate_check_request(value: Any) -> dict[str, Any]:
    """The closed path check: the world a joint path runs in, and the links allowed to touch it.

    ``allowed_links`` touch freely; any other link may lie at most ``tolerance_m`` inside the world
    (the world's own depth noise). The world is the one a plan to the path's last row would see."""
    check(isinstance(value, Mapping) and value.get("schema") == CHECK_REQUEST_SCHEMA
          and set(value) == {"schema", "robot", "start", "path", "world", "allowed_links", "tolerance_m"},
          "path check fields drifted")
    robot = _robot(value["robot"])
    _validate_start(robot, value["start"])
    rows = value["path"]
    check(isinstance(rows, list) and 1 <= len(rows) <= MAXIMUM_CHECK_ROWS
          and all(vector(row, len(robot["joint_names"])) for row in rows),
          f"a checked path holds 1 to {MAXIMUM_CHECK_ROWS} joint rows")
    check(_validate_world(value["world"]) != "none", "a path check needs a world")
    links = value["allowed_links"]
    check(isinstance(links, list) and all(isinstance(link, str) and link for link in links),
          "allowed_links are link names")
    check(number(value["tolerance_m"]) and 0 <= float(value["tolerance_m"]) <= 0.05, "tolerance_m is malformed")
    return json.loads(json.dumps(value))


def validate_check_answer(value: Any, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One path check's answer: ``clear``, or ``blocked`` at a row by a link, or an error."""
    fields = {"schema", "status", "reason", "first_blocked_row", "link", "penetration_m", "world", "planner",
              "timing_s", "request", "detail"}
    validate_common(value, CHECK_ANSWER_SCHEMA, request, fields)
    status = value.get("status")
    check(status in ("clear", "blocked", "error"), "path check status is unknown")
    if status == "error":
        check(value.get("reason") in ERROR_REASONS and isinstance(value["detail"].get("error"), Mapping),
              "a path check error names its reason and error")
        return json.loads(json.dumps(value, allow_nan=False))
    check(set(value) == fields and value["reason"] is None and (status == "blocked") == (
        type(value["first_blocked_row"]) is int and isinstance(value["link"], str)), "path check fields drifted")
    return json.loads(json.dumps(value, allow_nan=False))


def _validate_world_summary(world: Any, request: Mapping[str, Any]) -> None:
    model = request["world"]["model"]
    check(isinstance(world, Mapping) and world.get("model") == model and sha256(world.get("sha256"))
          and sha256(world.get("inputs_sha256")) and isinstance(world.get("summary"), Mapping),
          "an answer's world names its model, its sha256, its inputs' sha256 and a summary")


def validate_plan_answer(value: Any, request: Mapping[str, Any] | None = None, *,
                         untimed: bool = False) -> dict[str, Any]:
    """Validate one plan answer (v1, or v2 to a v2 request), against its request when given; returns a JSON copy.

    A v2 answer to a request with ``timing`` carries the retime answer in ``timing``: planned, its
    trajectory is the timed rows; refused by the timing, its reason is the timing's. ``untimed`` reads a
    planner worker's own answer, before its client has the path timed (``timing`` is still ``None``)."""
    fields = {"schema", "status", "reason", "trajectory", "start_tool_pose_base_xyzw", "world",
              "planner", "timing_s", "request", "detail"}
    version_two = isinstance(value, Mapping) and value.get("schema") == PLAN_ANSWER_SCHEMA_V2
    if version_two:
        fields = fields | {"timing"}
    validate_common(value, PLAN_ANSWER_SCHEMA_V2 if version_two else PLAN_ANSWER_SCHEMA, request, fields)
    asked = value.get("request") if isinstance(value.get("request"), Mapping) else request
    timed = version_two and not untimed and isinstance(asked, Mapping) and asked.get("timing") is not None
    check(asked is None or version_two == (asked.get("schema") == PLAN_REQUEST_SCHEMA_V2),
          "a plan answer's version differs from its request's")
    status, reason = value.get("status"), value.get("reason")
    check(status in PLAN_STATUSES, "plan status is unknown")
    check((status == "planned" and reason is None)
          or (status == "refused" and (reason in REFUSAL_REASONS or (timed and reason in TIMING_REFUSAL_REASONS)))
          or (status == "error" and reason in ERROR_REASONS), "plan status and reason disagree")
    if status == "error":
        check(value.get("trajectory") is None and isinstance(value["detail"].get("error"), Mapping),
              "a planner error carries no trajectory and names its error in detail")
        return json.loads(json.dumps(value, allow_nan=False))
    check(set(value) == fields, "plan answer fields drifted")
    request = value["request"]
    validate_plan_request(request)
    check(unit_pose(value["start_tool_pose_base_xyzw"]), "start tool pose is malformed")
    if version_two:
        timing = value["timing"]
        if not timed or (status == "refused" and reason in REFUSAL_REASONS):
            check(timing is None, "a plan answer carries timing it was not asked for or never reached")
        else:
            timing = validate_retime_answer(timing)
            check(timing["status"] == ("timed" if status == "planned" else "refused")
                  and timing["reason"] == reason
                  and timing["request"]["joint_names"] == request["start"]["joint_names"],
                  "a plan answer's timing disagrees with its status")
            check(status != "planned" or value["trajectory"] == timing["trajectory"],
                  "a timed plan's trajectory is not its timing's rows")
    if status == "planned":
        validate_trajectory(value["trajectory"], request["start"]["joint_names"], timed=timed)
    else:
        check(value["trajectory"] is None, "an unplanned answer carries a trajectory")
    if request["world"]["model"] == "none":
        check(value["world"] is None, "a robot-only answer carries a world")
    else:
        _validate_world_summary(value["world"], request)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MotionPlannerError("plan answer is not finite JSON") from exc


def validate_ik_answer(value: Any, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate one IK answer query by query; chained queries without a seed are skipped."""
    fields = {"schema", "status", "reason", "joint_names", "joint_limits", "start_tool_pose_base_xyzw",
              "results", "world", "planner", "timing_s", "request", "detail"}
    validate_common(value, IK_ANSWER_SCHEMA, request, fields)
    if value.get("status") == "error":
        check(value.get("reason") in ERROR_REASONS and isinstance(value["detail"].get("error"), Mapping),
              "an IK error names its reason and error")
        return json.loads(json.dumps(value, allow_nan=False))
    check(value.get("status") == "solved" and value.get("reason") is None and set(value) == fields,
          "IK answer fields drifted")
    request = value["request"]
    validate_ik_request(request)
    names = request["start"]["joint_names"]
    limits = value["joint_limits"]
    check(value["joint_names"] == names and isinstance(limits, list) and len(limits) == 2
          and all(vector(row, len(names)) for row in limits)
          and all(low < high for low, high in zip(*limits)), "IK joint limits are malformed")
    check(unit_pose(value["start_tool_pose_base_xyzw"]), "IK start tool pose is malformed")
    results, queries = value["results"], request["queries"]
    check(isinstance(results, list) and len(results) == len(queries), "IK answered another number of queries")
    keys = {"solved", "skipped", "refusal", "joint_positions", "tool_pose_base_xyzw",
            "position_error_m", "rotation_error"}
    for query, result in zip(queries, results):
        check(isinstance(result, Mapping) and set(result) == keys
              and type(result["solved"]) is bool and type(result["skipped"]) is bool, "IK result fields drifted")
        start = query["from_index"]
        seeded = start is None or results[start]["solved"] is True
        check(result["skipped"] is not seeded and not (result["solved"] and result["skipped"]),
              "IK skipped a seeded query or solved an orphan")
        check(result["refusal"] in (None, "goal_below_support")
              and not (result["refusal"] and (result["solved"] or result["skipped"])), "IK refusal is malformed")
        if result["solved"]:
            check(vector(result["joint_positions"], len(names)) and unit_pose(result["tool_pose_base_xyzw"]),
                  "IK solution is malformed")
        else:
            check(result["joint_positions"] is None and result["tool_pose_base_xyzw"] is None,
                  "an unsolved IK query carries a solution")
        for key in ("position_error_m", "rotation_error"):
            check(result[key] is None or (number(result[key]) and not result["skipped"]),
                  f"IK {key} is malformed")
    world = value["world"]
    if request["world"]["model"] == "none":
        check(world is None, "a robot-only IK answer carries a world")
    else:
        worlds = world.get("queries") if isinstance(world, Mapping) else None
        check(isinstance(world, Mapping) and set(world) == {"model", "inputs_sha256", "queries"}
              and world["model"] == request["world"]["model"] and sha256(world["inputs_sha256"])
              and isinstance(worlds, list) and len(worlds) == len(results)
              and all((summary is None) == result["skipped"] for summary, result in zip(worlds, results)),
              "IK world evidence does not match its queries")
        for summary in worlds:
            if summary is not None:
                check(isinstance(summary, Mapping) and sha256(summary.get("sha256"))
                      and summary.get("inputs_sha256") == world["inputs_sha256"],
                      "IK queries were solved in different scenes")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MotionPlannerError("IK answer is not finite JSON") from exc


def error_answer(schema: str, request: Any, reason: str, error: BaseException, *, implementation: str,
                 started: float, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The answer of a request the planner could not handle; ``request`` is ``None`` if it never validated."""
    answer = {"schema": schema, "status": "error", "reason": reason, "request": request,
              "planner": {"implementation": implementation}, "timing_s": {"total": time.time() - started},
              "detail": {**dict(detail or {}), "error": {"type": type(error).__name__,
                                                         "message": str(error) or type(error).__name__}}}
    if schema in (PLAN_ANSWER_SCHEMA, PLAN_ANSWER_SCHEMA_V2, RETIME_ANSWER_SCHEMA):
        answer["trajectory"] = None
    if schema == RETIME_ANSWER_SCHEMA:
        answer["report"] = None
    return answer


def refusal_message(answer: Mapping[str, Any]) -> str | None:
    """The sentence that says why a plan answer refused its goal, or ``None``: a pose goal's refusal has one."""
    refusal = ((answer.get("world") or {}).get("summary") or {}).get("goal_refusal")
    diagnosis = (answer.get("detail") or {}).get("goal_diagnosis")
    source = refusal if isinstance(refusal, Mapping) else diagnosis if isinstance(diagnosis, Mapping) else {}
    return source.get("message")


def planning_diagnostics(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded, library-free planning feedback (``ora.planning-diagnostics.v1``) of one plan answer."""
    request = answer.get("request") or {}
    joint_goal = "joint_positions" in (request.get("goal") or {})
    reason = answer.get("reason")
    phase = {"goal_unreachable": "ik", "goal_below_support": "ik", "goal_in_collision": "joint_space",
             "start_in_collision": "start_state", "start_outside_limits": "start_state"}.get(
        reason, "unknown" if answer.get("status") == "error" else "joint_space" if joint_goal
        else "trajectory_planning")
    recovery: list[str] = []
    # An implementation may keep each IK attempt's seed population in
    # ``detail.ik_attempts``; the last captured one gives the counts.
    attempts = (answer.get("detail") or {}).get("ik_attempts")
    last = attempts[-1] if isinstance(attempts, list) and attempts else None
    captured = isinstance(last, Mapping) and last.get("status") == "captured"
    unknown = (list(last.get("unknown_reasons") or []) if captured
               else ["no_target_ik_metrics_retained_for_this_request"])
    if reason == "goal_unreachable":
        recovery = ["Try a different target position or orientation, or an intermediate pose.",
                    "This finite IK search does not prove that no valid solution exists."]
        why = refusal_message(answer)
        if why is not None:
            recovery.insert(0, f"{why[0].upper()}{why[1:]}.")
    refusal = ((answer.get("world") or {}).get("summary") or {}).get("goal_refusal")
    if reason == "goal_below_support" and isinstance(refusal, Mapping):
        recovery, unknown = [f"Refused before IK: {refusal['message']}."], ["no_solver_ran_for_a_refused_goal"]
    return {"phase": phase, "outcome": reason or "planned",
            "candidate_counts": dict(last["candidate_counts"]) if captured else {},
            "rejection_counts": dict(last["rejection_counts"]) if captured else {},
            "details_available": captured, "recovery": recovery, "unknown_reasons": unknown,
            "evidence_scope": "last_recorded_original_ik_attempt" if captured else "no_target_ik_details"}


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True,
                               timeout=30, check=False)
    if completed.returncode:
        raise MotionPlannerError(completed.stderr.strip() or completed.stdout.strip() or "git probe failed")
    return completed.stdout.strip()


@dataclass
class MotionPlannerClient(JsonLinesWorker):
    """Start one registered planner worker, check its identity, and exchange requests with it.

    The worker runs under ``python_executable`` with the implementation's
    source (``root``, a clean checkout at ``required_revision``) first on
    ``PYTHONPATH``. A request that times out, or an answer that breaks the
    contract, retires the worker: a late answer is never read as another's.
    A plan v2 request with ``timing`` has the planned path timed by ``timing``
    (a ``TimingClient``) before the answer returns.
    """

    implementation: str
    python_executable: Path
    root: Path
    required_revision: str
    output_root: Path
    robot: str = "franka_panda"
    world_model: str = "none"
    device: str = "cuda:0"
    request_timeout_s: float = 180.0
    environment_overrides: Mapping[str, str] | None = None
    timing: TimingClient | None = None
    #: The robot's description, for a robot of ``ROBOT_FILE_ROBOTS``.
    robot_file: Path | None = None
    _process: subprocess.Popen[str] | None = field(init=False, default=None, repr=False)
    _sequence: int = field(init=False, default=0)
    binding: dict[str, Any] = field(init=False, default_factory=dict)
    #: One exchange at a time: ``prepare`` may come from a thread of its own.
    _lock: Any = field(init=False, default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        spec = IMPLEMENTATIONS.get(self.implementation)
        if spec is None:
            raise MotionPlannerError(f"unknown motion planner implementation {self.implementation!r}")
        if self.world_model not in spec.world_models or self.robot not in spec.robots:
            raise MotionPlannerError(f"{self.implementation} does not plan {self.robot} in {self.world_model}")
        if (self.robot in ROBOT_FILE_ROBOTS) != (self.robot_file is not None):
            raise MotionPlannerError(f"{self.robot} takes {'a' if self.robot in ROBOT_FILE_ROBOTS else 'no'} robot file")
        # absolute(), not resolve(): a venv's bin/python is a SYMLINK to the base
        # interpreter, and Python decides it is in a venv from the pyvenv.cfg next
        # to the path it was invoked as. Resolving the symlink lands on the shared
        # base interpreter, which has no pyvenv.cfg, so the venv's site-packages
        # silently disappear and the worker dies on its first import.
        self.python_executable = self.python_executable.expanduser().absolute()
        self.root = self.root.expanduser().resolve()
        self.output_root = self.output_root.expanduser().resolve()
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise MotionPlannerError("motion planner Python is unavailable or not executable")
        revision = _git(self.root, "rev-parse", "HEAD")
        if revision != self.required_revision or _git(self.root, "status", "--porcelain", "--untracked-files=no"):
            raise MotionPlannerError("motion planner source must be clean and match required_revision")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.binding = {
            "schema": BINDING_SCHEMA, "implementation": self.implementation, "revision": revision,
            "python": str(self.python_executable), "python_sha256": file_sha256(self.python_executable),
            "root": str(self.root), "device": self.device, "robot": self.robot, "world_model": self.world_model,
            **({"robot_file": str(self.robot_file), "robot_file_sha256": file_sha256(Path(self.robot_file))}
               if self.robot_file is not None else {}),
        }

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            return dict(self.binding)
        spec = IMPLEMENTATIONS[self.implementation]
        environment = os.environ.copy()
        environment.update(self.environment_overrides or {})
        checkout = project_root()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(checkout), *(str((self.root / path).resolve()) for path in spec.source_paths),
            environment.get("PYTHONPATH"),
        )))
        self._process = subprocess.Popen(
            [str(self.python_executable), "-m", spec.worker_module, "--device", self.device,
             "--robot", self.robot, "--world-model", self.world_model,
             *(("--robot-file", str(self.robot_file)) if self.robot_file is not None else ())],
            cwd=checkout, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(self.output_root / "worker.stderr.log").open("a", encoding="utf-8"), text=True, bufsize=1,
        )
        ready = self._exchange("ready").get("ready")
        if (not isinstance(ready, Mapping) or ready.get("implementation") != self.implementation
                or ready.get("robot") != self.robot or ready.get("world_model") != self.world_model):
            self.close()
            raise MotionPlannerError("motion planner worker readiness identity drift")
        self.binding["worker"] = {"pid": self._process.pid, "ready": dict(ready)}
        if self.timing is not None:
            self.binding["timing"] = self.timing.start()
        return dict(self.binding)

    def _exchange(self, command: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            return super()._exchange(command, request)

    def prepare(self, worlds: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Have a running worker build the scenes of ``worlds`` (``{start, world}`` each) now; its answer.

        Never starts a worker: a worker that is not running has nothing to keep them in."""
        validated = validate_prepare_request({"schema": PREPARE_REQUEST_SCHEMA, "robot": self.robot,
                                              "worlds": [dict(item) for item in worlds]})
        check(all(item["world"]["model"] == self.world_model for item in validated["worlds"]),
              f"this planner prepares {self.world_model} worlds")
        if self._process is None:
            raise MotionPlannerError(f"{self._label} worker is not running")
        answer = self._exchange("prepare", validated).get("answer")
        check(isinstance(answer, Mapping) and answer.get("schema") == PREPARE_ANSWER_SCHEMA
              and answer.get("status") in {"prepared", "error"}, "motion planner prepare answer drifted")
        return dict(answer)

    def _answer(self, command: str, request: Mapping[str, Any], validate: Any) -> dict[str, Any]:
        self.start()
        answer = self._exchange(command, request).get("answer")
        try:
            answer = validate(answer, request)
            if answer["planner"]["implementation"] != self.implementation or (
                    answer["status"] != "error" and request["world"]["model"] != self.world_model):
                raise MotionPlannerError("motion planner answer names another planner or world model")
        except MotionPlannerError:
            self._invalidate()
            raise
        return answer

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_plan_request(request)
        check(validated["robot"] == self.robot and validated["world"]["model"] == self.world_model,
              f"this planner plans {self.robot} in {self.world_model}")
        timing = validated.get("timing")
        if timing is None:
            return self._answer("plan", validated, validate_plan_answer)
        if self.timing is None:
            raise MotionPlannerError("a plan request with timing needs a timing client")
        answer = self._answer("plan", validated,
                              lambda value, asked: validate_plan_answer(value, asked, untimed=True))
        if answer["status"] != "planned":
            return answer
        return timed_plan_answer(answer, self.timing.retime(planned_retime_request(answer)))

    def solve_ik(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_ik_request(request)
        check(validated["robot"] == self.robot and validated["world"]["model"] == self.world_model,
              f"this planner solves {self.robot} in {self.world_model}")
        return self._answer("ik", validated, validate_ik_answer)

    def check_path(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_check_request(request)
        check(validated["robot"] == self.robot and validated["world"]["model"] == self.world_model,
              f"this planner checks {self.robot} in {self.world_model}")
        return self._answer("check", validated, validate_check_answer)

    def forward_kinematics(self, rows: list[list[float]]) -> list[list[float]]:
        """The planner's own tool poses of joint rows (``ROBOTS``' order), base frame."""
        self.start()
        answer = self._exchange("fk", {"joint_positions": rows}).get("answer")
        poses = answer.get("tool_poses_base_xyzw") if isinstance(answer, Mapping) else None
        if not isinstance(poses, list) or len(poses) != len(rows) or not all(unit_pose(pose) for pose in poses):
            self._invalidate()
            raise MotionPlannerError("motion planner forward kinematics drifted")
        return poses

    def close(self) -> None:
        if self.timing is not None:
            self.timing.close()
        super().close()


def planned_retime_request(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The retime request of a planned v2 answer: its ``timing`` over the path the planner measured.

    A worker answering a plan v2 with ``timing`` keeps, in ``detail.timing_path``, its trajectory as
    a grid at most ``motion_pace.JOINT_GRID_SPACING_RAD`` apart with the controlled point's pose at every
    row, from its own kinematics and the request's ``controlled_point`` offset."""
    path = (answer.get("detail") or {}).get("timing_path")
    check(isinstance(path, Mapping) and set(path) == {"joint_positions", "tool_poses_xyzw"},
          "a planned answer to a timed request keeps its measured path in detail.timing_path")
    request = answer["request"]
    return retime_request(request["timing"], request["start"]["joint_names"], path["joint_positions"],
                          path["tool_poses_xyzw"])


def timed_plan_answer(answer: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    """A planned v2 answer with its path timed: the timed rows, or the timing's refusal or error.

    The planner's own trajectory stays in ``detail.planned_trajectory``."""
    result = {**dict(answer), "timing": dict(timing), "detail": {**dict(answer["detail"]),
                                                                  "planned_trajectory": answer["trajectory"]}}
    status = timing["status"]
    result.update(status="planned" if status == "timed" else status, reason=timing["reason"],
                  trajectory=timing["trajectory"])
    if status == "error":
        result["detail"]["error"] = dict(timing["detail"]["error"])
    return validate_plan_answer(result)


def planner_client(config: Mapping[str, Any], *, python_executable: Path, root: Path, output_root: Path,
                   environment_overrides: Mapping[str, str] | None = None, timing: TimingClient | None = None,
                   robot_file: Path | None = None) -> Any:
    """The client a ``runtime.motion_planner`` configuration selects.

    ``implementation`` names an entry of ``IMPLEMENTATIONS``; a configuration
    without one is refused. ``timing`` serves plan v2 requests that ask for timing.
    """
    if "implementation" not in config:
        raise MotionPlannerError(f"runtime.motion_planner names no implementation (one of {sorted(IMPLEMENTATIONS)})")
    return MotionPlannerClient(implementation=str(config["implementation"]), python_executable=python_executable,
                               root=root, required_revision=str(config["required_revision"]),
                               robot=str(config.get("robot", "franka_panda")), timing=timing,
                               robot_file=robot_file, output_root=output_root,
                               device=str(config.get("device", "cuda:0")),
                               request_timeout_s=float(config.get("request_timeout_s", 180.0)),
                               environment_overrides=environment_overrides,
                               world_model=str(config.get("world_model", "none")))


__all__ = [
    "BINDING_SCHEMA", "CHECK_ANSWER_SCHEMA", "CHECK_REQUEST_SCHEMA", "ERROR_REASONS", "FAILURE_REASONS", "IK_ANSWER_SCHEMA", "IK_REQUEST_SCHEMA",
    "IMPLEMENTATIONS", "Implementation", "MAXIMUM_PREPARED_WORLDS", "MotionPlannerClient", "MotionPlannerError",
    "PLAN_ANSWER_SCHEMA", "PLAN_ANSWER_SCHEMA_V2", "PLAN_REQUEST_SCHEMA", "PLAN_REQUEST_SCHEMA_V2", "PLAN_STATUSES",
    "PROTOCOL", "REFUSAL_REASONS", "ROBOTS", "ROBOT_FILE_ROBOTS", "WORLD_MODELS", "error_answer", "planned_retime_request",
    "planner_client", "planning_diagnostics", "timed_plan_answer", "validate_check_answer", "validate_check_request",
    "validate_ik_answer", "validate_ik_request", "validate_plan_answer", "validate_plan_request",
    "PREPARE_ANSWER_SCHEMA", "PREPARE_REQUEST_SCHEMA", "validate_prepare_request",
]
