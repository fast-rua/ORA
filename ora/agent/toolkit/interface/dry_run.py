"""``dry_run``: an action that moves the robot plans itself and stops before it acts.

Every action that moves the robot takes it, on every backend, in tool-call and code mode, with one result: a
``RESULT_SCHEMA`` record of what the call would do, with its preview (``lib/action_preview.py``). Reads and ``wait_ticks``
have none: nothing moves. The tool-call argument and a script's ``dry_run=True`` both end at ``dry``: the flag reaches
the tool out of band, the tool compiles its request, runs the planner the real call runs and returns the record instead
of acting. Outside a dry run ``requested()`` is False and costs a context-variable read. A dry run predicts the robot's
own motion (its model, the table height, public depth), never the world's response, and runs no physics.

Where the environment plans a move when it executes it, the tool still compiles and validates the action, and at the
point it would hand it over raises ``Proposed`` with it; the runtime asks the environment the read-only question
(``ora/agent/tools/runtime/action_preview.py``). The same plan then runs when the same action is proposed.

Nothing in a dry run can act: the tools return before their trajectory is executed, the data interface a tool gets
inside ``dry`` has ``NoActions`` in place of its action capability, and a tool of a surface whose environment plans
stops at ``Proposed`` (``ora/agent/tools/runtime/toolkit_dispatch.py``, ``robot_tools.py``).
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from ora.contracts.action_plan import DRY_RUN_ARGUMENT, MOVING_TOOLS, body_points, segment

#: The tools a dry run can be asked of: every tool that moves the robot.
TOOLS = MOVING_TOOLS
#: The refusal of a dry run no plan can answer, with ``constraints.reason`` saying why.
UNAVAILABLE = "dry_run_unavailable"
#: The name of the argument; the runtime strips it before the tool sees the call.
ARGUMENT = DRY_RUN_ARGUMENT

_flag: ContextVar[bool] = ContextVar("toolkit_dry_run", default=False)


def requested() -> bool:
    """Whether the call in progress is a dry run (code mode asked for one with ``dry``)."""

    return _flag.get()


@contextlib.contextmanager
def dry() -> Iterator[None]:
    """The tool calls made inside the block are dry runs."""

    token = _flag.set(True)
    try:
        yield
    finally:
        _flag.reset(token)


def asks(name: str, arguments: Any) -> bool:
    """Whether a call asks for a dry run: ``dry_run: true`` on a tool that moves the robot."""

    return (isinstance(arguments, Mapping) and arguments.get(ARGUMENT) is True
            and str(name).rsplit(".", 1)[-1] in TOOLS)


def split(name: str, arguments: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """The call's arguments as the tool takes them, and whether it asks for a dry run."""

    if not isinstance(arguments, Mapping) or ARGUMENT not in arguments or str(name).rsplit(".", 1)[-1] not in TOOLS:
        return arguments, False
    return {key: value for key, value in arguments.items() if key != ARGUMENT}, asks(name, arguments)


class DryRunAction(RuntimeError):
    """A dry run reached for the environment's action capability; it has none."""


class Proposed(DryRunAction):
    """A dry run reached the point where the action goes to the environment that plans it. ``commitment`` is what a
    compound action committed to, which only the tool knows."""

    def __init__(self, action: Mapping[str, Any], commitment: Mapping[str, Any] | None = None) -> None:
        super().__init__("a dry run never submits an action")
        self.action, self.commitment = dict(action), commitment


@dataclass(frozen=True, slots=True)
class NoActions:
    """The action capability of a tool inside a dry run: it refuses every action.

    ``admit`` is the trusted admission of a proposal where the environment plans: the environment keeps its plan for the
    action it was asked about, so that is the action the real call would admit."""

    admit: Callable[[dict], dict] | None = None

    def execute_action(self, action: dict, budget: int):
        raise DryRunAction("a dry run never executes an action")

    def submit_action(self, *, action: dict, preparation: dict | None = None):
        raise Proposed(action if self.admit is None else self.admit(action))


_POSE = {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7}
_JOINTS = {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6}
_AXIS = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
#: What a dry run returns (the tool's own result schema is not extended: tool-call mode advertises it). Poses are world
#: ``[x, y, z, qx, qy, qz, qw]`` in metres, angles in degrees.
RESULT_SCHEMA: dict = {
    "type": "object",
    "required": ["dry_run", "action", "status", "feasible"],
    "properties": {
        "dry_run": {"const": True},
        "action": {"type": "string", "description": "The tool that was dry-run."},
        "status": {"enum": ["planned", "infeasible"]},
        "feasible": {"type": "boolean", "description": "The call would run: it compiled, solved, was screened clear and fit "
                                                          "its budget. A move cut to its budget is feasible with "
                                                          "`motion.fraction_of_move` below 1."},
        "arms": {"type": "object", "description": "Per arm planned.", "additionalProperties": {
            "type": "object", "properties": {
                "target_pose": {**_POSE, "description": "The compiled target (world)."},
                "predicted_pose": {**_POSE, "description": "Where the plan's last knot puts the controlled point (world); "
                                                           "the cut point when the move is cut to its budget."},
                "predicted_approach_world": {**_AXIS, "description": "Code mode adds it where the robot's controlled "
                                             "frame is published: predicted_pose's approach axis in world, a unit "
                                             "vector, as move_pose's approach_world takes it."},
                "predicted_jaw_line_world": {**_AXIS, "description": "Code mode adds it beside "
                                             "predicted_approach_world: predicted_pose's jaw-closing axis in world, a "
                                             "unit vector, as move_pose's jaw_line_world takes it."},
                "end_joints_deg": {**_JOINTS, "description": "The IK solution the plan ends at, joints 1 to 6."},
                "native_actions": {"type": ["integer", "null"], "description": "Planned up to the first contact; null "
                                                                               "where only a servo moves this arm."},
                "servo_not_counted": {"type": "boolean", "description": "Part of the move is a servo, whose native actions "
                                                                        "only its run says."},
                "hand_travel_m": {"type": "number"}, "hand_turn_deg": {"type": "number", "description": "All the turning "
                                                                                                      "the hand does."},
                "rise_above_ends_m": {"type": ["number", "null"], "description": "How far the path's highest point, up "
                                      "to the first contact, is above the higher of its two ends (0 where an end is "
                                      "highest): the jaw centre's, or the controlled point's where the jaw geometry is "
                                      "not published."},
                "rise_at_fraction": {"type": ["number", "null"], "description": "Where that highest point is: 0 at the "
                                                                                "start, 1 at the end."},
                "held_object": {"type": "object", "description": "Where the jaws hold something, from the measured "
                                "state (last command close, jaws stopped apart), whatever closed them: target_ref (the "
                                "target named when they closed, or null), extent (localized: that target's box; "
                                "between_jaws: only the gap the jaws measure, what reaches past them is not measured; "
                                "unknown), box_size_m and jaw_gap_m. Its box is swept in clearance and drawn.",
                                "properties": {"target_ref": {"type": ["string", "null"]},
                                               "extent": {"enum": ["localized", "between_jaws", "unknown"]},
                                               "box_size_m": {"type": ["array", "null"], "items": {"type": "number"}},
                                               "jaw_gap_m": {"type": ["number", "null"]}}},
                "clearance": {"type": "object", "description": "The least distance from the jaws along the path to what "
                                                               "the depth images show; claim_boundary says what it covers.",
                              "properties": {"status": {"enum": ["measured", "beyond", "ends_only", "unknown"]},
                                             "enters_cleared_space": {
                                                 "type": "object", "description": "Within the last 5 cm of an end, where "
                                                 "the jaws enter what a planner clears there on purpose.",
                                                 "additionalProperties": True},
                                             "clearance_m": {"type": "number"},
                                             "inside_m": {"type": "number", "description": "Where the jaws' boxes "
                                                          "reach into what the depth images show (clearance_m 0): how "
                                                          "far the deepest point seen lies inside them; at_fraction and "
                                                          "position_m name it."},
                                             "at_fraction": {"type": "number", "description": "0 at the start, 1 at the end."},
                                             "position_m": {"type": "array", "items": {"type": "number"}},
                                             "part": {"type": "string"}, "camera": {"type": "string"},
                                             "clearance_at_least_m": {"type": "number"}, "reason": {"type": "string"}}},
                "opening_from": {"type": ["number", "null"]}, "opening_to": {"type": ["number", "null"]},
                "jaw_width_from_m": {"type": ["number", "null"]}, "jaw_width_to_m": {"type": ["number", "null"]},
                "joint6": {"type": "object", "description": "The wrist roll along the plan, in degrees.", "properties": {
                    "start_deg": {"type": "number"}, "end_deg": {"type": "number"}, "min_deg": {"type": "number"},
                    "max_deg": {"type": "number"}, "travel_deg": {"type": "number"},
                    "min_limit_margin_deg": {"type": "number", "description": "The least margin to the wrist roll's limit."}}},
                "method": {"type": "string", "description": "A recovery's path: measured_retrace or joint_interpolation."},
                "unwind_order": {"type": "string", "description": "A return_home's joint order: straight_joint_line, "
                                                                   "fold_then_yaw or yaw_then_fold."},
                "lift": {"type": "string", "description": "A return_home's lift before the unwind: planned or skipped."},
                "undo": {"type": "object", "description": "A recover_pose's: the call it would undo (`undone_call`), "
                                                           "whether the retrace completes and the calls left to undo.",
                         "properties": {"undone_call": {"type": "object", "additionalProperties": True},
                                        "complete": {"type": "boolean"}, "path_left_fraction": {"type": "number"},
                                        "undoable_calls": {"type": "array", "items": {"type": "object"}}}},
            }}},
        "route": {"type": "string", "description": "line, joint, global or staged (move_pose and its primitives); "
                                                    "measured_retrace or retract_then_unwind (recovery)."},
        "screening": {"type": "object", "properties": {
            "status": {"type": "string"}, "segments_screened": {"type": "integer"},
            "candidates_screened": {"type": "integer", "description": "A recovery's: the candidate paths screened."},
            "min_inter_arm_clearance_m": {"type": ["number", "null"]}, "against": {"type": "string"}}},
        "motion": {"type": "object", "description": "The plan: native actions, duration, fraction of the move it covers.",
                   "properties": {
                       "native_actions": {"type": "integer", "description": "The trajectory's, without the settle."},
                       "whole_native_actions": {"type": "integer", "description": "What the whole move needs."},
                       "settle_native_actions": {"type": "integer"}, "duration_s": {"type": "number"},
                       "fraction_of_move": {"type": "number", "description": "1 when the move fits its budget, less when "
                                                                            "it is cut to it."},
                       "native_action_budget": {"type": "integer", "description": "What the call may spend."},
                       "expect_contact": {"type": "boolean", "description": "Whether the call set it, as the real "
                                                                            "call's `motion.expect_contact` will say."},
                       "remaining_m": {"type": "object", "description": "Per arm, how far the cut point is from the target."},
                       "joint_line_bridges": {"type": "array", "items": {"type": "object"}}, "ik_warning": {"type": "string"}}},
        "segments": {"type": "array", "description": "What would run, in order: each stretch's kind (arm, base, torso, "
                                                         "gripper), arm, label, planned (false: a servo's commanded line, "
                                                         "not a path) and not_predicted (true: not predicted).",
                     "items": {"type": "object", "additionalProperties": True}},
        "base": {"type": "object", "description": "The base's numbers, where the action moves it.",
                 "additionalProperties": True},
        "summary": {"type": "string", "description": "One line per arm: what its jaws hold, cost, travel, turn, rise "
                                                       "above the ends, least clearance or how far the jaws reach into "
                                                       "what the depth shows."},
        "image": {"type": "object", "description": "The picture: the cameras it is drawn on, or why there is none.",
                  "additionalProperties": True},
        "speed": {"type": "object"},
        "compiled_pose": {"type": "object", "additionalProperties": True},
        "failure_code": {"type": ["string", "null"]},
        "failure_stage": {"type": ["string", "null"], "description": "request, route, ik, screening, budget or plan."},
        "refusal": {"type": "object", "description": "The refusal the real call would give, when infeasible.", "properties": {
            "code": {"type": "string"}, "message": {"type": "string"}, "field": {"type": "string"},
            "constraints": {"type": "object", "additionalProperties": True}}},
        "claim_boundary": {"type": "string"},
    },
}


__all__ = ["ARGUMENT", "DryRunAction", "NoActions", "Proposed", "RESULT_SCHEMA", "TOOLS", "UNAVAILABLE", "asks",
           "body_points", "dry", "requested", "segment", "split"]
