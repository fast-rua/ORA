"""What a script may ask with ``robot.query`` (``library: extended``): the queries, their arguments and their results.

One source for the three places that name them: the host that answers (``query.py``), the sandbox that sends
(``robot.py``) and the documentation a script looks up (``docs.py``: the index line and the entry of each query, the
same parts a function has). Nothing here imports ORA, so the documentation can be generated without the toolkit.

Every query is a read of the environment's robot model or of the measured state, served through the episode's
``DataInterface`` (the toolkit lib: ``motion_plan.kinematics``, ``motion``, ``observation``). It moves nothing, is no
tool call and counts against no call limit; it counts against the script's query allowance (``contract.QUERY_LIMITS``).
Lengths are metres, rotations are xyzw quaternions in the ``world`` frame, and every angle is in degrees, as every angle
a policy sees (``composition/surface/units.py``): the argument and result names carry ``_deg``.
"""

from __future__ import annotations

import copy
from typing import Any

#: The queries, in the order the index lists them.
NAMES = ("ik", "fk", "screen_path", "reach_check", "probe_motion", "align_plan", "robot_state")

#: The documentation entry of a query is named ``query.<name>`` (``query_<name>`` in a script, as a tool's dots are).
PREFIX = "query"

#: How many poses or knots one query takes: one call of the environment's bridge each.
KINEMATICS_ITEMS = 256
REACH_ITEMS = 16
PROBE_ITEMS = 12
ALIGN_ITEMS = 12
#: ``max_joint_step_deg`` when a script gives none: a little over the 14.3 degrees the motion tools admit between two
#: grid poses of a straight line.
DEFAULT_STEP_DEG = 15.0

_POSE = {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7,
         "description": "[x, y, z, qx, qy, qz, qw]: metres and an xyzw quaternion in the world frame, the pose of the "
                        "arm's controlled frame (tcp:<arm>). An ora_lib Pose in world is accepted."}
_JOINTS = {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6,
           "description": "The arm's six joint positions, in degrees."}
_ARM = {"type": "string", "description": "The arm: left or right (the environment names the arms it has)."}
_COUNT = {"type": "integer", "minimum": 0}


def _poses(maximum: int, what: str) -> dict[str, Any]:
    return {"type": "array", "minItems": 1, "maxItems": maximum, "items": _POSE,
            "description": f"The poses {what}, in order. One pose may be given alone."}


_NOTE = ("It is read-only: it moves nothing, it is no tool call and no call limit counts it, and it counts against the "
         "script's query allowance (`allowance_left.queries`: a count and seconds). A query that is refused for its "
         "arguments is not counted; one the environment cannot answer says `query_unavailable` and why.")

#: name -> (description, arguments, required, result, example)
_QUERIES: dict[str, dict[str, Any]] = {
    "ik": {
        "description": (
            "Inverse kinematics of one arm for a sequence of world poses, each solution seeded from the previous "
            "one, read from the environment's robot model.\n"
            "`poses` are the arm's controlled frame in world. A row is `ok` when the environment found a joint "
            "solution inside the limits that lands on the pose, within `max_joint_step_deg` of the previous "
            "solution (per pose since the last good one) and whose straight joint path from it `screen` finds "
            "clear; otherwise `ik_failed` (no solution), `discontinuous` (only a jump) or `path_blocked`. A failed "
            "pose keeps the previous solution as the next seed. It is the solver the motion tools plan with; it "
            "plans nothing, and `screen` knows the robot's own links and the other arm, not objects or the table. "
            + _NOTE),
        "arguments": {
            "arm": _ARM,
            "poses": _poses(KINEMATICS_ITEMS, "to solve"),
            "seed_joints_deg": {**_JOINTS, "description": "The joints the first solution is nearest to, in degrees. "
                                "Default: the arm's measured joints."},
            "max_joint_step_deg": {"type": "number", "exclusiveMinimum": 0, "maximum": 360,
                                   "description": "The largest joint change admitted between two consecutive "
                                   f"poses, in degrees. Default {DEFAULT_STEP_DEG:g}; 180 admits any branch."},
            "ranking": {"enum": ["nearest", "canonical_first"], "description": "nearest: the candidate nearest the "
                        "previous solution (default). canonical_first: move_pose's own selection of an endpoint."},
            "screen": {"enum": ["none", "self", "self_and_measured_other_arm"],
                       "description": "What each step's straight joint path is screened against (default "
                       "self_and_measured_other_arm: the arm's own links and the other arm at its measured joints)."},
        },
        "required": ["arm", "poses"],
        "result": {"type": "object", "properties": {
            "rows": {"type": "array", "description": "One row per pose, in order.", "items": {
                "type": "object", "properties": {
                    "index": {"type": "integer"},
                    "status": {"enum": ["ok", "ik_failed", "discontinuous", "path_blocked"]},
                    "joints_deg": {"type": ["array", "null"], "items": {"type": "number"},
                                   "description": "The solution, in degrees; null unless ok."},
                    "joint_step_deg": {"type": ["number", "null"], "description": "The largest joint change from the "
                                       "previous solution, in degrees."},
                    "position_error_m": {"type": ["number", "null"], "description": "FK of the solution against the "
                                         "pose, metres."},
                    "orientation_error_deg": {"type": ["number", "null"], "description": "The same, degrees."},
                    "limit_margin_deg": {"type": ["number", "null"], "description": "The nearest joint's distance to "
                                         "its limit, degrees."},
                    "path": {"type": ["string", "null"], "description": "clear when the screened path from the "
                             "previous solution is clear; the screen's finding when not."}}}},
            "first_failure_index": {"type": ["integer", "null"], "description": "The first row that is not ok."},
            "answers": {"type": "array", "description": "One per call to the environment: its seeding rule, ik_model, "
                        "timings_s, total_s and any warning it carries."}}},
        "example": ('from ora_lib import Direction, current\n'
                    'pose = current().pose("right")                 # the measured controlled frame, in world\n'
                    'higher = pose + Direction([0, 0, 0.05], frame="world")\n'
                    'answer = robot.query("ik", arm="right", poses=[higher])\n'
                    'row = answer["rows"][0]\n'
                    'print(row["status"], row["joints_deg"], row["position_error_m"])'),
    },
    "fk": {
        "description": (
            "Forward kinematics of one arm: the world pose of the arm's controlled frame at each joint vector, read "
            "from the environment's robot model.\n"
            "`joints_deg` is one vector of the arm's six joints or a list of them, in degrees. " + _NOTE),
        "arguments": {
            "arm": _ARM,
            "joints_deg": {"type": "array", "minItems": 1, "maxItems": KINEMATICS_ITEMS, "items": _JOINTS,
                           "description": "The joint vectors to evaluate, in degrees. One vector may be given alone."},
        },
        "required": ["arm", "joints_deg"],
        "result": {"type": "object", "properties": {
            "poses": {"type": "array", "items": _POSE, "description": "One pose per joint vector, in order."}}},
        "example": ('state = robot.query("robot_state")\n'
                    'joints = state["arms"]["right"]["joint_positions_deg"]\n'
                    'print(robot.query("fk", arm="right", joints_deg=joints)["poses"][0])'),
    },
    "screen_path": {
        "description": (
            "Screen joint paths for collisions: the arms' joint polylines, one segment after the other, against "
            "themselves, the table top and each other, read from the environment's robot model.\n"
            "`knots_deg` maps `left` and/or `right` to knots (six joints each, degrees); the arms move in lockstep, "
            "so they need the same number of knots, and an arm left out holds its retained target. `status` is "
            "`clear`, `self_collision`, `table_collision` or `inter_arm_collision`, with the first blocked segment. "
            "It models the robot's links and the table top, not objects, so clear is no promise about the scene. "
            + _NOTE),
        "arguments": {
            "knots_deg": {"type": "object", "minProperties": 1, "description": "Each arm's knots, in degrees.",
                          "additionalProperties": {"type": "array", "minItems": 1, "maxItems": KINEMATICS_ITEMS,
                                                   "items": _JOINTS}},
        },
        "required": ["knots_deg"],
        "result": {"type": "object", "properties": {
            "status": {"enum": ["clear", "self_collision", "table_collision", "inter_arm_collision"]},
            "first_blocked_segment": {"type": ["integer", "null"], "description": "The index of the knot that ends "
                                      "the first blocked segment."},
            "segments_screened": _COUNT,
            "arm": {"type": ["string", "null"], "description": "The arm whose path is blocked, where one is."},
            "min_inter_arm_clearance_m": {"type": ["number", "null"], "description": "Metres."},
            "min_table_clearance_m": {"type": ["number", "null"], "description": "Metres."},
            "limiting_link": {"type": ["string", "null"]},
            "limiting_links": {"type": "object"},
            "claim_boundary": {"type": "string", "description": "What the screen does and does not cover."}}},
        "example": ('here = robot.query("robot_state")["arms"]["right"]["joint_positions_deg"]\n'
                    'there = [0.0, 20.0, 0.0, 0.0, 0.0, 0.0]\n'
                    'print(robot.query("screen_path", knots_deg={"right": [here, there]})["status"])'),
    },
    "reach_check": {
        "description": (
            "Whether one arm can hold each of up to 16 world poses, by the environment's read-only inverse "
            "kinematics, without planning a motion.\n"
            "Per pose: `reachable`, the `reason` when not, the endpoint errors and the joint-limit margin. "
            "`interaction` is the move_pose interaction the pose is reached with (one for all poses, or a list); "
            "`from_index` chains a pose onto the configuration the pose at that index reached, as a pregrasp "
            "followed by its approach would execute. The answer is for the current observation. Where "
            "`matches_move_pose_admission` is false it is a reference, not the move's own verdict: `probe_motion` is. "
            + _NOTE),
        "arguments": {
            "arm": _ARM,
            "poses": _poses(REACH_ITEMS, "to check"),
            "interaction": {"description": "Default free_space.", "oneOf": [
                {"enum": ["free_space", "precision", "contact_approach"]},
                {"type": "array", "items": {"enum": ["free_space", "precision", "contact_approach"]}}]},
            "from_index": {"type": "array", "items": {"type": ["integer", "null"]},
                           "description": "Per pose, the index of an earlier pose to start from, or null "
                           "(the measured joints). Default: all null."},
        },
        "required": ["arm", "poses"],
        "result": {"type": "object", "properties": {
            "arm": {"type": "string"}, "ik_model": {"type": "string"},
            "matches_move_pose_admission": {"type": "boolean"},
            "claim_boundary": {"type": "string", "description": "What the answer does and does not claim."},
            "results": {"type": "array", "description": "One row per pose, in order.", "items": {
                "type": "object", "properties": {
                    "reachable": {"type": "boolean"}, "reason": {"type": ["string", "null"]},
                    "route": {"type": ["string", "null"]},
                    "position_error_m": {"type": ["number", "null"]},
                    "orientation_error_deg": {"type": ["number", "null"]},
                    "joint_limit_margin_deg": {"type": ["number", "null"]},
                    "joint_delta_deg": {"type": ["number", "null"]}}}}}},
        "example": ('from ora_lib import Direction, current\n'
                    'pose = current().pose("right")\n'
                    'goals = [pose + Direction([0, 0, 0.05 * step], frame="world") for step in (1, 2)]\n'
                    'print([row["reachable"] for row in robot.query("reach_check", arm="right", poses=goals)["results"]])'),
    },
    "probe_motion": {
        "description": (
            "Plan up to 12 world poses with the environment's motion planner, without executing them.\n"
            "Per pose it says whether a trajectory exists, how long it is in steps and seconds, how far the "
            "joints travel and, when there is none, why: the verdict a fast free-space move_pose to that pose gets, "
            "where the motion tools plan their moves, with the same refusal code. It calls the planner, so it takes "
            "seconds, and its answer is for the current observation: any action makes it stale. "
            "`variant_policy` is `exact` (the pose as given) or `parallel_jaw_equivalents` (also the pose turned "
            "half a turn about the approach axis, which grips the same way). " + _NOTE),
        "arguments": {
            "arm": _ARM,
            "poses": _poses(PROBE_ITEMS, "to plan"),
            "variant_policy": {"enum": ["exact", "parallel_jaw_equivalents"], "description": "Default exact."},
        },
        "required": ["arm", "poses"],
        "result": {"type": "object", "properties": {
            "results": {"type": "array", "description": "One entry per pose, in order.", "items": {
                "type": "object", "properties": {
                    "query_id": {"type": "string"}, "arm": {"type": "string"}, "reachable": {"type": "boolean"},
                    "selected_variant_id": {"type": ["string", "null"]},
                    "variants": {"type": "array", "description": "The poses tried.", "items": {
                        "type": "object", "properties": {
                            "variant_id": {"type": "string"}, "reachable": {"type": "boolean"},
                            "target_pose_xyzw": _POSE, "trajectory_steps": _COUNT,
                            "trajectory_duration_s": {"type": ["number", "null"]},
                            "joint_delta_deg": {"description": "Joint travel to the pose, degrees."},
                            "route": {"type": "string", "description": "The route it would take, where the motion "
                                      "tools plan their moves (as in a dry run)."},
                            "failure": {"type": "object", "description": "code and message, when there is no "
                                        "trajectory."},
                            "endpoint_error": {"type": "object", "description": "translation_m and "
                                               "orientation_deg of the planned end against the pose."},
                            "planner": {"type": "object", "description": "The planner's public diagnostics."}}}}}}},
            "world_state_changed": {"type": "boolean"}}},
        "example": ('from ora_lib import Direction, current\n'
                    'goal = current().pose("right") + Direction([0, 0, 0.05], frame="world")\n'
                    'plan = robot.query("probe_motion", arm="right", poses=[goal])["results"][0]\n'
                    'print(plan["reachable"], [v["trajectory_duration_s"] for v in plan["variants"]])'),
    },
    "align_plan": {
        "description": (
            "Rank up to 12 world poses by the environment's inverse kinematics, as the align_plan tool does for the "
            "approach poses it builds, but for poses you give.\n"
            "Per pose and arm: `reachable`, the position and orientation errors of the best solution, the "
            "iterations and the rejection reason. `arm` is left, right or both; `approach_side` labels the batch "
            "and is echoed, it does not change the answer. No motion planner runs. " + _NOTE),
        "arguments": {
            "arm": {"type": "string", "description": "left, right or both (the environment names the arms it has)."},
            "poses": _poses(ALIGN_ITEMS, "to rank"),
            "approach_side": {"enum": ["top", "left", "right", "front", "back"], "description": "Default top."},
        },
        "required": ["arm", "poses"],
        "result": {"type": "object", "properties": {
            "requested_arm": {"type": "string"}, "approach_side": {"type": "string"},
            "motion_planner_called": {"type": "boolean"},
            "candidates": {"type": "array", "description": "One per pose and arm.", "items": {
                "type": "object", "properties": {
                    "source_index": {"type": "integer", "description": "The pose it answers."},
                    "arm": {"type": "string"}, "reachable": {"type": "boolean"},
                    "position_error_m": {"type": ["number", "null"]},
                    "orientation_error_deg": {"type": ["number", "null"]},
                    "iterations": {"type": ["integer", "null"]},
                    "rejection_reason": {"type": ["string", "null"]},
                    "claim_boundary": {"type": "string"}}}}}},
        "example": ('from ora_lib import Direction, current\n'
                    'pose = current().pose("right")\n'
                    'ranked = robot.query("align_plan", arm="right", poses=[pose + Direction([0, 0, 0.03], frame="world")])\n'
                    'print([(c["source_index"], c["reachable"]) for c in ranked["candidates"]])'),
    },
    "robot_state": {
        "description": (
            "The robot's measured state in the current observation, typed.\n"
            "Per arm: the pose of its end-effector frame as published (with the frame it is given in), its joint "
            "positions in degrees and its normalized gripper opening (1 open, 0 closed; not metres). `public` is "
            "every published proprioception field, angles in degrees. " + _NOTE),
        "arguments": {},
        "required": [],
        "result": {"type": "object", "properties": {
            "arms": {"type": "object", "description": "By arm name.", "additionalProperties": {
                "type": "object", "properties": {
                    "pose": {"type": ["object", "null"], "description": "frame, xyz (metres) and xyzw as published; "
                             "the frame is the published one, not necessarily world."},
                    "joint_positions_deg": {"type": "array", "items": {"type": "number"}},
                    "gripper_normalized": {"type": ["number", "null"]}}}},
            "public": {"type": "object", "description": "The whole published proprioception, angles in degrees."}}},
        "example": ('state = robot.query("robot_state")\n'
                    'print({arm: round(item["gripper_normalized"], 2) for arm, item in state["arms"].items()})'),
    },
}


def entry_name(name: str) -> str:
    """The documentation entry's name of query ``name`` (``query.ik``; a script also writes ``query_ik``)."""

    return f"{PREFIX}.{name}"


def purpose(name: str) -> str:
    """The first line of a query's description: what it is for."""

    return _QUERIES[name]["description"].split("\n", 1)[0]


def argument_names(name: str) -> tuple[str, ...]:
    """The arguments ``robot.query(name, ...)`` takes."""

    return tuple(_QUERIES[name]["arguments"])


def required(name: str) -> tuple[str, ...]:
    return tuple(_QUERIES[name]["required"])


def signature(name: str) -> str:
    """``robot.query("ik", arm, poses, seed_joints_deg?, ...)``: the call with its arguments, required ones first."""

    spec = _QUERIES[name]
    names = [*spec["required"], *(f"{item}?" for item in spec["arguments"] if item not in spec["required"])]
    return f'robot.query("{name}"' + "".join(f", {item}" for item in names) + ")"


def definition(name: str) -> dict[str, Any]:
    """The query as a function definition, in the form ``docs.py`` generates an entry from (``signature`` and
    ``lookup`` are what name its call and its entry)."""

    spec = _QUERIES[name]
    schema: dict[str, Any] = {"type": "object", "additionalProperties": False,
                              "properties": copy.deepcopy(spec["arguments"])}
    if spec["required"]:
        schema["required"] = list(spec["required"])
    return {"name": entry_name(name), "description": spec["description"], "inputSchema": schema,
            "signature": signature(name), "lookup": entry_name(name).replace(".", "_")}


def result_schema(name: str) -> dict[str, Any]:
    return copy.deepcopy(_QUERIES[name]["result"])


def example(name: str) -> str:
    return str(_QUERIES[name]["example"])


__all__ = ["ALIGN_ITEMS", "DEFAULT_STEP_DEG", "KINEMATICS_ITEMS", "NAMES", "PREFIX", "PROBE_ITEMS", "REACH_ITEMS",
           "argument_names", "definition", "entry_name", "example", "purpose", "required", "result_schema",
           "signature"]
