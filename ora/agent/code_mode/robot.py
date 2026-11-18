"""Inside the sandbox: the ``robot`` library a ``bash`` command's programs import (standard library only).

``shell.run_command`` binds this file at ``/ora/robot.py`` and puts ``/ora`` on
``PYTHONPATH`` (with ``library: extended``, at ``/ora/lib/robot.py`` and ``/ora/lib``),
with the host's socket inherited as ``ORA_CODE_MODE_FD``. Every
Python process the command starts may ``from robot import tools, handoff``:

- ``tools.<name>(**arguments)`` (or ``tools.call(name, arguments)``) sends
  ``{"op": "call", "name": ..., "arguments": {...}}`` and waits for the host's
  answer (``ora.code-mode-rpc.v1``); a refused call raises ``ToolRefused``;
- ``handoff(reason, **data)`` sends ``{"op": "handoff", "reason": ...,
  "data": ...}``; the host then stops the whole command;
- ``docs(name=None, part=None)`` sends ``{"op": "docs", "name": ..., "part": ...}``
  and returns the function's documentation (``library: extended``; with no
  name, one line per function), which ``help(tools.<name>)`` shows too;
- ``explain(call)`` sends ``{"op": "explain", "call_id": ...}`` and returns what
  that call did in causal order (``library: extended``; ``call`` is a ``Result``,
  the ``ToolRefused`` of a refused call, or a call id);
- ``query(name, **arguments)`` sends ``{"op": "query", "name": ..., "arguments": ...}`` and returns the
  answer's ``result``: a read of the environment's robot model or of the measured state (``library: extended``;
  ``query_spec.py`` lists them), no tool call, counted against the script's query allowance only; a refused
  query raises ``ToolRefused``.

Processes of one command may run at once; a lock file in ``/tmp`` makes each
call and its answer one exchange on the shared socket.

With ``library: extended`` (``ORA_CODE_MODE_LIBRARY=extended`` here) every call returns a
``Result``: the answer as before, a dict, with its outcome read into fields
(``ok``, ``status``, ``failure_code``, ``failure_stage``, ``pose_after``,
``summary``, ``target_error_m``, ``target_error_deg``, ``cut_short``, ``contact`` and the public gripper
signals), the call's id (``call_id``, which ``explain`` takes), the explanation of a failed call
(``explain``) and the whole answer under ``.raw``; a refused call's ``ToolRefused``
carries one as ``.result``. After an action ``pose_after`` is each arm's measured pose in the
observation the action published (the mounted ``/ora/observation``), where the answer states none.
``reused`` is True for an answer the policy session kept from an earlier identical call on the
same observation. ``move_pose`` and ``move_to`` also take typed values (``target``) and fill in
the tool's mode parameters. ``tools.<name>`` also takes positional arguments, in the order of the
function's signature, required ones first (``tools.wait_ticks(15)``). Without it a call takes keywords and
returns the answer unchanged.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import sys
import time
from collections.abc import Mapping
from typing import Any

FD_ENV = "ORA_CODE_MODE_FD"
LOCK_PATH = "/tmp/.ora-robot.lock"
#: ``extended`` when the contract's ``library`` is (``shell.command_bwrap``).
LIBRARY_ENV = "ORA_CODE_MODE_LIBRARY"
#: Where the extended library mounts the current published observation (``library.py``).
OBSERVATION_MOUNT = "/ora/observation"
#: How long ``handoff`` waits with its output closed, so a program's pipeline can print before the command stops.
HANDOFF_PIPE_S = 0.3


class ToolRefused(Exception):
    """A tool call the Supervisor refused or that failed; ``answer`` is what it said (with ``library: extended``,
    ``result`` is its ``Result``)."""

    result: Result | None = None

    def __init__(self, name: str, answer: dict[str, Any]) -> None:
        error = answer.get("error") if isinstance(answer.get("error"), dict) else {}
        self.name = name
        self.code = error.get("code")
        self.answer = answer
        super().__init__(f"{name}: {error.get('message') or error.get('code') or 'refused'}")


def _json_safe(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else repr(value)
    if depth < 32 and isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if depth < 32 and isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):  # NumPy arrays and scalars
        try:
            return _json_safe(value.tolist(), depth + 1)
        except Exception:  # noqa: BLE001 - fall back to the repr
            pass
    return repr(value)


# --- library: extended -- typed results ---------------------------------------------------------------------------
#
# Where a call's outcome sits in the answer the host sends (``agent/pi/code_mode.py``, ``_code_mode_answer``): an
# action's effect, the public motion effect (``status``, ``failure_stage``, ``failure_code``, ``physical_cause``,
# ``detail``), is ``receipt`` beside ``result`` for a tool whose action the script's parent executes (LIBERO,
# RoboTwin, RoboCasa365, RoboLab), and ``result.receipt`` for a synchronous toolkit tool (RoboDojo); a multi-stage
# synchronous tool (``grasp_execute``) adds ``result.stages``, each with its receipt, and ``result.verification``.
# A receipt that still wraps its record (``receipt.receipt``) is read inside. A read-only tool has no effect: its
# ``result`` is all there is (``observe`` adds ``public_robot_facts``).

#: An action's ``status`` when it did what it was asked (``completed``: a skill sequence's).
SUCCEEDED = frozenset({"reached", "executed", "completed"})
#: A ``failure_code`` of a move that still reached: the pose is inside the endpoint acceptance, only the joints had not
#: settled (``backends/robodojo/trajectory.py``). The prompt's rule holds for it: ``reached`` is ``ok``.
REACHED_ANYWAY = frozenset({"arm_not_settled"})
#: How a multi-stage tool's stage may end without failing it (as the skill-sequence projection reads it).
STAGE_PASSED = frozenset({"executed", "skipped", "not_requested", "terminated"})
#: ``physical_cause`` values that are a public contact estimate.
CONTACT_CAUSES = frozenset({"blocked_by_contact", "pushed_by_contact"})
#: Stops a measured contact made (``detail.endpoint.stop_reason`` of an arm, ``detail.base.stop_reason`` of a mobile
#: base; ``session_core/stop_detail.py``): the gripper contact stop and the base's bumper contact.
CONTACT_STOPS = {"endpoint": "contact_detected", "base": "base_contact"}
ARMS = ("left", "right")
#: A result's typed fields, in the order its repr shows them.
FIELDS = ("ok", "status", "failure_code", "failure_stage", "summary", "pose_after", "target_error_m", "target_error_deg",
          "cut_short", "contact", "measured_aperture_mm", "measured_opening", "commanded_opening", "gripper_command", "feasible",
          "predicted_pose", "route")
#: The fields each arm has (``Result.arms``); the result's own field is the targeted arm's.
ARM_FIELDS = ("pose_after", "target_error_m", "target_error_deg", "cut_short", "measured_aperture_mm",
              "measured_opening", "commanded_opening", "gripper_command")
#: What only a run of a call has (a ``Plan`` raises ``PlanNotRun`` on one), and a plan's fields as its repr shows them.
RUN_FIELDS = ("ok", "status", "pose_after", "target_error_m", "target_error_deg", "cut_short", "contact",
              "measured_aperture_mm", "measured_opening", "commanded_opening", "gripper_command", "arms")
PLAN_FIELDS = ("feasible", "failure_code", "failure_stage", "summary", "predicted_pose", "route")


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _pose(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 7 or any(_number(item) is None for item in value):
        return None
    return [float(item) for item in value]


def _measured_poses() -> dict[str, list[float]]:
    """Each arm's measured pose (world ``[x, y, z, qx, qy, qz, qw]``) in the mounted current observation; nothing where
    none is mounted (a closed observation too) or it cannot be read."""

    try:
        with open(os.path.join(OBSERVATION_MOUNT, "current.json"), encoding="utf-8") as stream:
            index = json.load(stream)
        if index.get("error") or not isinstance(index.get("proprioception"), str):
            return {}
        with open(os.path.join(OBSERVATION_MOUNT, index["proprioception"]), encoding="utf-8") as stream:
            facts = _get(json.load(stream), "end_effectors")
    except (OSError, ValueError):
        return {}
    poses = {arm: _pose(_get(facts, arm, "pose_xyz_xyzw")) for arm in ARMS}
    return {arm: pose for arm, pose in poses.items() if pose is not None}


def _unwrapped(receipt: Any) -> Mapping[str, Any] | None:
    """A receipt, or the record it wraps (``receipt.receipt``)."""

    if not isinstance(receipt, Mapping):
        return None
    nested = receipt.get("receipt")
    return nested if isinstance(nested, Mapping) and "status" not in receipt else receipt


class _Fields(dict):
    """One arm's typed fields: a dict whose keys also read as attributes."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(name)


class Result(dict):
    """A tool call's answer (dict access unchanged, the whole answer also under ``.raw``) with its outcome as fields.

    - ``ok``: the call was answered and its outcome succeeded: an action's ``status`` is ``reached``, ``executed`` or
      ``completed`` with no ``failure_code`` (a move ``reached`` whose joints had not settled, ``arm_not_settled``, counts); a read-only call returned no error, no ``success: false`` and no
      ``status: failed``. (``self["ok"]`` stays the answer's own flag: the call was not refused.)
    - ``status``, ``failure_code``, ``failure_stage``: the action's (its public motion effect, or for a multi-stage
      tool its own status and its first stage that did not pass); a read-only call's ``status`` if it has one; a
      refused call's ``status`` is ``refused`` and its ``failure_code`` the error's code.
    - ``pose_after`` (world ``[x, y, z, qx, qy, qz, qw]``), ``target_error_m``, ``target_error_deg``: where the answer
      carries them; a receipt in degrees (``orientation_deg``) is converted.
    - ``cut_short``: True when the budget stopped the move on its path, short of its target; ``target_error_m`` is
      then the distance left to the target (None for ``return_home`` and ``recover_pose``, whose ``summary`` gives the
      joint travel left) and ``target_error_deg`` None. None for any other outcome.
    - ``summary``: what happened in one line: an action's own summary, or why the call was refused (a planner's
      refusal says what stopped the goal and what near it would work). None for a call that says neither.
    - ``contact``: True when a public contact estimate says so (``physical_cause`` ``blocked_by_contact`` or
      ``pushed_by_contact``, ``contact_motion.contact_estimated`` or ``force_threshold_exceeded``) or a measured contact
      ended the move (``endpoint.stop_reason`` ``contact_detected``, ``base.stop_reason`` ``base_contact``), False when
      the estimate says no contact, None without one.
    - ``measured_aperture_mm``, ``measured_opening`` (0 closed to 1 open), ``commanded_opening`` (0 to 1) and
      ``gripper_command`` (``open`` or ``close``): the public gripper signals. Whether a grasp holds is not one of
      them: judge it from these, the views and a re-measurement.
    - ``arms``: each arm's ``pose_after``, ``target_error_*``, ``cut_short`` and gripper signals. A result's own field is the value of
      its one targeted arm (the larger error when two arms were targeted); None where arms differ or nothing says.
    - ``reused``: True when the policy session kept this answer from an earlier identical call on the same
      observation (a perception call: ``vision_ground``, ``vision_segment``, ``geometry_query``); else False.

    After an action (the host numbers it: ``answer["action"]``) the answer carries no pose, and ``pose_after`` is each
    arm's measured pose in the observation that action published, read from the mount (``/ora/observation``), the
    same measurement ``observe`` reports; a receipt's own ``after`` pose wins where it has one.

    - ``call_id``: the call's id in this episode's record (what ``explain`` takes), ``explain``: what the call did, in
      causal order, for a failed call of a tool that has an explanation (``robot.explain(result)`` asks for any call's).

    A dry run (``tools.move_to(..., dry_run=True)``: the plan the call would run, nothing executed) answers a ``Plan``,
    which reads the plan and has none of a run's fields.

    A field that does not apply, or that the backend's answer does not carry, is None. A key the answer does not have
    raises ``KeyError`` saying where it is (``r["result"]["candidates"]``) or which keys there are; ``in`` and
    ``.get`` answer as a dict's.
    """

    def __init__(self, tool: str, answer: Mapping[str, Any]) -> None:
        super().__init__(answer)
        self.tool, self.raw = tool, answer
        self.call_id, self.explain = answer.get("call_id"), answer.get("explain")
        result = answer.get("result") if isinstance(answer.get("result"), Mapping) else {}
        error = answer.get("error") if isinstance(answer.get("error"), Mapping) else result.get("error")
        error = error if isinstance(error, Mapping) else None
        effect = _unwrapped(answer.get("receipt"))  # an action the script's parent executed
        stages: list[Mapping[str, Any]] = []
        if effect is None:  # a synchronous tool: its own receipt, and a multi-stage tool's stages
            effect = _unwrapped(result.get("receipt"))
            stages = [stage for stage in result.get("stages") or [] if isinstance(stage, Mapping)] \
                if isinstance(result.get("stages"), list) else []
        #: Every receipt of the call in order, the last the call's own: each signal is the latest one says.
        effects = [item for item in (*(_unwrapped(stage.get("receipt")) for stage in stages), effect) if item]
        self.status = self.failure_code = self.failure_stage = None
        if stages:  # a multi-stage tool: its own status, its first stage that did not pass
            self.status = result.get("status")
            failed = next((stage for stage in stages if stage.get("status") not in STAGE_PASSED), None)
            if failed is not None:
                self.failure_stage = failed.get("stage")
                self.failure_code = failed.get("reason") or _get(failed, "receipt", "failure_code")
            elif effect is not None:
                self.failure_code, self.failure_stage = effect.get("failure_code"), effect.get("failure_stage")
        elif effect is not None:
            self.status = effect.get("status")
            self.failure_code, self.failure_stage = effect.get("failure_code"), effect.get("failure_stage")
        else:
            self.status = result.get("status")
            if result.get("dry_run") is True:  # a plan: why it would not run
                self.failure_code, self.failure_stage = result.get("failure_code"), result.get("failure_stage")
        self.status = self.status if isinstance(self.status, str) else None
        if error is not None and self.failure_code is None:
            self.failure_code = error.get("code")
        if not answer.get("ok") and self.status is None:
            self.status = "refused"
        self.ok = bool(answer.get("ok")) and error is None and (self.failure_code is None or (
            self.failure_code in REACHED_ANYWAY and self.status == "reached")) and (
            self.status in SUCCEEDED if effects or stages
            else result.get("success") is not False and self.status != "failed")
        self.summary = _summary(effects, error) or (result.get("summary") if isinstance(result.get("summary"), str)
                                                    and result.get("dry_run") is True else None)
        self.contact = _contact(effects)
        self.arms = {arm: _Fields(fields) for arm, fields in _arm_signals(result, effects).items()}
        targeted = _targeted(effects, self.arms)
        self.reused = answer.get("reused") is True
        self.dry_run = answer.get("dry_run") is True
        plan = result if self.dry_run and result.get("dry_run") is True else {}
        planned = plan.get("arms") if isinstance(plan.get("arms"), Mapping) else {}
        self.feasible = plan.get("feasible") if isinstance(plan.get("feasible"), bool) else None
        self.predicted_poses = {arm: _pose(_get(fields, "predicted_pose")) for arm, fields in planned.items()}
        self.predicted_pose = list(self.predicted_poses.values())[0] if len(self.predicted_poses) == 1 else None
        solutions = [_get(fields, "end_joints_deg") for fields in planned.values()]
        self.end_joints_deg = solutions[0] if len(solutions) == 1 else None
        self.route = plan.get("route") if isinstance(plan.get("route"), str) else None
        self.screening = plan.get("screening") if isinstance(plan.get("screening"), Mapping) else None
        if answer.get("action") is not None:  # an action: where each arm is in the observation it published
            measured = _measured_poses()
            for arm, pose in measured.items():
                fields = self.arms.setdefault(arm, _Fields(dict.fromkeys(ARM_FIELDS)))
                fields["pose_after"] = fields["pose_after"] or pose
            targeted = targeted or _targeted(effects, self.arms)  # an arm the receipt names, else every arm measured
        loose = _armless_signals(result, effects)
        for name in ARM_FIELDS:
            values = [self.arms[arm][name] for arm in targeted if self.arms[arm][name] is not None]
            if name in ("target_error_m", "target_error_deg", "cut_short") and values:
                setattr(self, name, max(values))
            else:
                setattr(self, name, values[0] if len(values) == 1 else None if values else loose.get(name))

    def __repr__(self) -> str:
        shown = ", ".join(f"{name}={getattr(self, name)!r}" for name in FIELDS if getattr(self, name) is not None)
        return f"Result({self.tool}: {shown}{', reused=True' if self.reused else ''})"

    def __missing__(self, key: Any) -> Any:
        result = self.get("result")
        if isinstance(result, Mapping) and key in result:
            raise KeyError(f'{key} is under result: r["result"]["{key}"]')
        raise KeyError(f"no {key} in this answer; its keys are {', '.join(map(str, self))}")


class PlanNotRun(RuntimeError):
    """A run's field (``ok``, ``status``, a receipt's) read on a ``Plan``: nothing has run. The call without
    ``dry_run=True`` runs it."""


class Plan(Result):
    """A call's plan, not its run (``dry_run`` True, ``explain`` always): what any ``dry_run=True`` call returns, a
    gripper command's included. Nothing moved and nothing was spent.

    ``feasible`` says whether the call would run, ``failure_code`` and ``failure_stage`` (request, route, ik,
    screening, budget, plan) why not; ``predicted_pose`` (world ``[x, y, z, qx, qy, qz, qw]``) is where the plan ends
    the one arm it moves (the cut point when the move is cut to its budget), ``predicted_poses`` every arm's,
    ``end_joints_deg`` the IK solution it ends at, ``route`` the path kind, ``screening`` the self, table and inter-arm
    screen and ``summary`` one line per arm (native actions, hand travel and turn, rise above the path's ends, least
    clearance); the picture of the path comes with the script's result, printed or not. The same call without
    ``dry_run=True`` runs it and returns the run's ``Result``. A run's fields (``ok``, ``status``, a receipt's) raise
    ``PlanNotRun``: ``if r.ok`` on a plan is not whether it would run (``feasible`` is)."""

    def __getattribute__(self, name: str) -> Any:
        if name in RUN_FIELDS and "_planned" in object.__getattribute__(self, "__dict__"):
            tool = object.__getattribute__(self, "tool")
            raise PlanNotRun(f"{tool} returned a plan and nothing has run: .{name} is a run's; tools.{tool}(...) "
                             "without dry_run=True runs it and returns a Result that has it (.feasible says whether "
                             "it would run)")
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        shown = ", ".join(f"{name}={getattr(self, name)!r}" for name in PLAN_FIELDS if getattr(self, name) is not None)
        return f"Plan({self.tool}: {shown})"


def _summary(effects: list[Mapping[str, Any]], error: Mapping[str, Any] | None) -> str | None:
    """The last receipt's summary line, else the refusal's own words."""
    for effect in reversed(effects):
        if isinstance(effect.get("action_summary"), str):
            return effect["action_summary"]
    return error.get("message") if error is not None and isinstance(error.get("message"), str) else None


def _contact(effects: list[Mapping[str, Any]]) -> bool | None:
    flags: list[Any] = []
    for effect in effects:
        motion = _get(effect, "detail", "contact_motion")
        flags += [True if effect.get("physical_cause") in CONTACT_CAUSES else None,
                  _get(motion, "contact_estimated"), _get(motion, "force_threshold_exceeded"),
                  *(True if _get(effect, "detail", block, "stop_reason") == stop else None
                    for block, stop in CONTACT_STOPS.items())]
    flags = [flag for flag in flags if isinstance(flag, bool)]
    return any(flags) if flags else None


def _targeted(effects: list[Mapping[str, Any]], arms: Mapping[str, Any]) -> list[str]:
    """The arms the call's action targeted (``detail.target_arms``, or ``targeted``), else every arm with a signal."""

    for effect in reversed(effects):
        named = _get(effect, "detail", "target_arms")
        if isinstance(named, list) and named:
            return [arm for arm in named if arm in arms]
        flagged = [arm for arm in arms if _get(effect, "detail", "arms", arm, "targeted") is True]
        if flagged:
            return flagged
    return list(arms)


def _gripper(fields: dict[str, Any], outcome: Any) -> None:
    """A ``gripper_outcome`` (or ``gripper_feedback``, or a public robot facts ``gripper``) into gripper signals."""

    if not isinstance(outcome, Mapping):
        return
    for name, key in (("measured_aperture_mm", "measured_aperture_mm"),
                      ("measured_opening", "measured_opening_normalized"),
                      ("commanded_opening", "commanded_opening_normalized")):
        if _number(outcome.get(key)) is not None:
            fields[name] = float(outcome[key])
    if outcome.get("command_target") in ("open", "close"):
        fields["gripper_command"] = outcome["command_target"]


def _effect_signals(effect: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """One arm's signals in one receipt."""

    fields: dict[str, Any] = {}
    detail = _get(effect, "detail")
    record = _get(detail, "arms", arm)
    error = _get(record, "pose_error")
    left = _number(_get(record, "motion_progress", "remaining_to_goal_m"))
    if _get(record, "motion_progress", "outcome") == "cut_short" and left is not None:
        fields.update(target_error_m=left, cut_short=True)  # its pose_error is the cut point's
    elif _get(record, "recovery_progress", "outcome") == "cut_short":
        fields["cut_short"] = True  # a recovery has no goal pose: its distance left is in joints
    elif isinstance(error, Mapping):
        degrees, radians = _number(error.get("orientation_deg")), _number(error.get("orientation_rad"))
        fields.update(target_error_m=_number(error.get("translation_m")), target_error_deg=degrees if degrees is not
                      None else math.degrees(radians) if radians is not None else None)
    target = _number(_get(detail, "gripper_targets_effective", arm))
    if target is not None:
        fields["commanded_opening"] = target
    targets = _get(detail, "gripper_targets_effective")
    if isinstance(targets, Mapping) and list(targets) == [arm]:  # a single-arm gripper action's own outcome (LIBERO)
        _gripper(fields, _get(detail, "gripper_outcome"))
    _gripper(fields, _get(detail, "gripper_feedback", arm))
    _gripper(fields, _get(record, "gripper_outcome"))
    command = _get(record, "command_preservation", "gripper_command_target_after")
    if command in ("open", "close"):
        fields["gripper_command"] = command
    pose = _pose(_get(effect, "after", arm, "pose_xyz_xyzw"))
    if pose is not None:
        fields["pose_after"] = pose
    return {name: value for name, value in fields.items() if value is not None}


def _arm_signals(result: Mapping[str, Any], effects: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each arm's pose after, target error and gripper signals: the latest receipt's, then the result's own facts."""

    arms: dict[str, dict[str, Any]] = {}
    facts = _get(result, "public_robot_facts", "end_effectors")
    for arm in ARMS:
        fields: dict[str, Any] = {}
        for effect in effects:
            signals = _effect_signals(effect, arm)
            fields.update(signals)
            if "cut_short" in signals:
                fields.pop("target_error_deg", None)
        if isinstance(_get(facts, arm), Mapping):  # a result that carries the public robot facts (observe)
            measured = {"pose_after": _pose(_get(facts, arm, "pose_xyz_xyzw"))}
            _gripper(measured, _get(facts, arm, "gripper"))
            fields.update({name: value for name, value in measured.items() if value is not None})
        if fields:
            arms[arm] = {name: fields.get(name) for name in ARM_FIELDS}
    return arms


def _armless_signals(result: Mapping[str, Any], effects: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Signals no arm is named for: a single-arm gripper action's ``detail.gripper_outcome`` and a grasp's measured
    jaw opening (``verification.gripper_opening_m``)."""

    fields: dict[str, Any] = {}
    for effect in effects:
        _gripper(fields, _get(effect, "detail", "gripper_outcome"))
    for verification in (*(_get(effect, "detail", "verification") for effect in effects), result.get("verification")):
        opening = _number(_get(verification, "gripper_opening_m"))
        if opening is not None:
            fields["measured_aperture_mm"] = opening * 1000.0
    return fields


def typed(name: str, answer: Mapping[str, Any]) -> Result:
    """``name``'s answer as a ``Result``; a dry run's plan as a ``Plan``."""

    result = answer.get("result")
    if answer.get("ok") and answer.get("dry_run") is True and isinstance(result, Mapping) and result.get("dry_run") is True:
        plan = Plan(str(name), answer)
        plan._planned = True  # from here on a run's field raises
        return plan
    return Result(str(name), answer)


# --- library: extended -- the library fills tool parameters -------------------------------------------------------
#
# A typed value is an object with ``frame`` and ``xyz`` (a point), ``xyz`` and ``xyzw`` (a pose) or ``v`` (a
# direction) -- ``ora_lib``'s ``Point``, ``Pose`` and ``Direction`` -- its ``to_json()``, or such a dict. Frames:
# ``world``, ``tcp:<arm>`` (the arm's controlled frame), ``camera:<name>``, ``pixel:<name>``.

#: The keys that make a ``move_pose`` or ``move_to`` arm target one written in the library's form. ``interaction`` is
#: a key of both forms, the library's and the tool's: it never says which form a target is in.
TARGET_KEYS = frozenset({"position", "approach", "jaw_line", "orientation"})


def _typed_value(value: Any) -> tuple[str, dict[str, Any]] | None:
    """``(frame, {xyz, xyzw, v})`` of a typed value, or None for anything else."""

    if callable(getattr(value, "to_json", None)) and not isinstance(value, Mapping):
        value = value.to_json()
    if isinstance(value, Mapping):
        if not isinstance(value.get("frame"), str):
            return None
        return value["frame"], {key: value[key] for key in ("xyz", "xyzw", "v") if key in value}
    frame = getattr(value, "frame", None)
    if not isinstance(frame, str):
        return None
    return frame, {key: getattr(value, key) for key in ("xyz", "xyzw", "v") if hasattr(value, key)}


def _vector(value: Any, size: int, what: str) -> list[float]:
    """One finite ``size``-vector (a batch is refused)."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != size or any(_number(item) is None for item in value):
        raise ValueError(f"{what} must be one {size}-vector of finite numbers, not {value!r}")
    return [float(item) for item in value]


def _framed(value: Any, what: str) -> tuple[str, dict[str, Any]]:
    typed_value = _typed_value(value)
    if typed_value is None:
        raise TypeError(f"{what} needs a frame: a typed value (ora_lib's Point, Pose or Direction) or a dict such as "
                        f"{{'frame': 'world', 'xyz': [x, y, z]}}, not {value!r}")
    return typed_value


def _world_direction(value: Any, what: str) -> list[float]:
    frame, fields = _framed(value, what)
    if "v" not in fields:
        raise TypeError(f"{what} must be a direction (v), not a point or pose")
    if frame != "world":
        raise ValueError(f"{what} is in frame {frame!r}; move_pose takes world directions: convert it with "
                         f"{what}.to('world')")
    return _vector(fields["v"], 3, what)


def _identity(xyzw: list[float]) -> bool:
    return max(abs(xyzw[0]), abs(xyzw[1]), abs(xyzw[2])) < 1e-9 and abs(abs(xyzw[3]) - 1.0) < 1e-9


def target(position: Any = None, *, approach: Any = None, jaw_line: Any = None, orientation: Any = None,
           interaction: str | None = None, arm: str | None = None) -> dict[str, Any]:
    """One arm's ``move_pose`` target from typed values: under its arm in ``targets`` where ``move_pose``'s line shows
    ``targets``, else spread into the call (``tools.move_pose(**target(...))``).

    - ``position``: a point or pose in ``world`` -> ``position_mode: absolute`` (``position_m``); in ``tcp:<arm>``
      -> ``position_mode: delta``, ``position_frame: end_effector`` (``translation_m``); None holds the position.
    - ``approach`` and ``jaw_line``, two ``world`` directions -> ``orientation_mode: world_axes``
      (``approach_world``, ``jaw_line_world``); ``orientation``, a ``world`` pose -> ``absolute_quaternion``; a
      ``world`` pose given as ``position`` brings its orientation too; else the orientation is held.
    - ``interaction``: passed on (``free_space``, ``precision``, ``contact_approach``).

    A value in another frame (a camera's, a pixel's) is refused with the conversion to make; ``arm`` refuses a
    ``tcp`` frame of the other arm.
    """

    result: dict[str, Any] = {"position_mode": "hold_current", "orientation_mode": "hold_current"}
    rotation: list[float] | None = None
    if position is not None:
        frame, fields = _framed(position, "position")
        if "xyz" not in fields:
            raise TypeError("position must be a point or pose (xyz), not a direction")
        xyz = _vector(fields["xyz"], 3, "position")
        quaternion = _vector(fields["xyzw"], 4, "position orientation") if "xyzw" in fields else None
        if frame == "world":
            result.update(position_mode="absolute", position_m=xyz)
            rotation = quaternion
        elif frame.startswith("tcp:"):
            if arm is not None and frame != f"tcp:{arm}":
                raise ValueError(f"position is in {frame!r}, but it moves the {arm} arm: convert it with "
                                 f"position.to('tcp:{arm}') or position.to('world')")
            if quaternion is not None and not _identity(quaternion):
                raise ValueError("a rotation in a tcp frame has no move_pose form here: convert the pose with "
                                 "position.to('world')")
            result.update(position_mode="delta", translation_m=xyz, position_frame="end_effector")
        else:
            raise ValueError(f"position is in frame {frame!r}; move_pose takes world or tcp:<arm> positions: convert "
                             "it with position.to('world')")
    if (approach is None) != (jaw_line is None):
        raise ValueError("approach and jaw_line go together (orientation_mode world_axes)")
    given = [name for name, value in (("approach and jaw_line", approach), ("orientation", orientation),
                                      ("the position's orientation", rotation)) if value is not None]
    if len(given) > 1:
        raise ValueError(f"give the orientation once: {' and '.join(given)} both set it")
    if approach is not None:
        result.update(orientation_mode="world_axes", approach_world=_world_direction(approach, "approach"),
                      jaw_line_world=_world_direction(jaw_line, "jaw_line"))
    elif orientation is not None:
        frame, fields = _framed(orientation, "orientation")
        if "xyzw" not in fields:
            raise TypeError("orientation must be a pose (xyzw); for axes give approach and jaw_line")
        if frame != "world":
            raise ValueError(f"orientation is in frame {frame!r}: convert it with orientation.to('world')")
        result.update(orientation_mode="absolute_quaternion", orientation_xyzw=_vector(fields["xyzw"], 4,
                                                                                       "orientation"))
    elif rotation is not None:
        result.update(orientation_mode="absolute_quaternion", orientation_xyzw=rotation)
    if interaction is not None:
        result["interaction"] = interaction
    return result


def _move_to_target(value: Any, arm: str | None) -> dict[str, Any]:
    """One arm's ``move_to`` target from a typed point; ``arm`` None for a single-arm surface's ``position_m``."""

    what = f"move_to's {arm} position" if arm is not None else "move_to's position_m"
    if isinstance(value, Mapping) and _typed_value(value) is None:
        extra = set(value) - {"position"}
        if extra:
            raise ValueError(f"move_to's {arm} target takes only position, not {', '.join(sorted(extra))}")
        value = value.get("position")
    frame, fields = _framed(value, what)
    if "xyz" not in fields:
        raise TypeError(f"{what} must be a point (xyz)")
    if frame != "world":
        raise ValueError(f"move_to takes a world position, not one in {frame!r}: convert it with .to('world'), or "
                         "for an offset in tcp:<arm> use move_pose")
    if "xyzw" in fields and not _identity(_vector(fields["xyzw"], 4, "orientation")):
        raise ValueError("move_to holds the orientation; to reach a pose use move_pose")
    return {"position_m": _vector(fields["xyz"], 3, what)}


def _fill_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """``move_pose``'s and ``move_to``'s arguments with every arm target given by typed values written in the tool's
    parameters (a single-arm ``move_to``'s typed ``position_m`` too); a target already in the tool's form, and every
    other tool's arguments, are left as they are."""

    arguments = dict(arguments)
    targets = arguments.get("targets")
    if name == "move_to" and targets is None and _typed_value(arguments.get("position_m")) is not None:
        arguments["position_m"] = _move_to_target(arguments["position_m"], None)["position_m"]
        return arguments
    if name not in ("move_pose", "move_to") or not isinstance(targets, Mapping):
        return arguments
    filled: dict[str, Any] = {}
    for arm, value in targets.items():
        if value is None:
            filled[arm] = value
        elif name == "move_to":
            plain = isinstance(value, Mapping) and _typed_value(value) is None and "position" not in value
            filled[arm] = value if plain else _move_to_target(value, str(arm))
        elif _typed_value(value) is not None:
            filled[arm] = target(value, interaction=value.get("interaction") if isinstance(value, Mapping) else None,
                                 arm=str(arm))
        elif isinstance(value, Mapping) and set(value) & TARGET_KEYS:
            mixed = set(value) - TARGET_KEYS - {"interaction"}
            if mixed:
                raise ValueError(f"move_pose's {arm} target mixes the library's keys ({', '.join(sorted(TARGET_KEYS))})"
                                 f" with the tool's ({', '.join(sorted(mixed))}): use one form")
            filled[arm] = target(**dict(value), arm=str(arm))
        else:
            filled[arm] = value
    arguments["targets"] = filled
    return arguments


class _Connection:
    def __init__(self) -> None:
        self.sock = socket.socket(fileno=os.dup(int(os.environ[FD_ENV])))

    def exchange(self, message: dict[str, Any], *, answer: bool = True) -> dict[str, Any]:
        with open(LOCK_PATH, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.sock.sendall((json.dumps(_json_safe(message), ensure_ascii=False) + "\n").encode("utf-8"))
            if not answer:
                return {}
            data = b""
            while not data.endswith(b"\n"):
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    raise SystemExit(70)  # the host went away
                data += chunk
            return json.loads(data)


_connection: _Connection | None = None


def _channel() -> _Connection:
    global _connection
    if _connection is None:
        _connection = _Connection()
    return _connection


_DOCS: dict[tuple[str | None, str | None], str] = {}


def docs(name: str | None = None, part: str | None = None) -> str:
    """A function's documentation, or one ``part`` of it (``"arguments"``); with no name, one line per function."""

    key = (None if name is None else str(name), None if part is None else str(part))
    if key not in _DOCS:
        answer = _channel().exchange({"op": "docs", "name": key[0], "part": key[1]})
        if not answer.get("ok"):
            raise ToolRefused("docs", answer)
        _DOCS[key] = str(answer.get("text") or "")
    return _DOCS[key]


_EXPLAIN: dict[str, str] = {}


def _call_id(call: Any) -> str:
    """The id of the call a ``Result``, a ``ToolRefused`` (its ``.result``), an answer dict or an id string names."""

    found = call
    if isinstance(call, BaseException):
        found = getattr(call, "result", None) or getattr(call, "answer", None)
    ident = found if isinstance(found, str) else getattr(found, "call_id", None) or (
        found.get("call_id") if isinstance(found, Mapping) else None)
    if not isinstance(ident, str) or not ident:
        raise ValueError("explain needs the result of a call (what tools.<name>(...) returned, or the .result of its "
                         "ToolRefused) or a call id")
    return ident


def explain(call: Any) -> str:
    """What one call did, in causal order: each stage with its criterion, the measured value and the verdict, then the
    conclusion and, for a failed call, what can be done next. ``call`` is a ``Result``, a ``ToolRefused`` or a call id."""

    ident = _call_id(call)
    if ident not in _EXPLAIN:
        answer = _channel().exchange({"op": "explain", "call_id": ident})
        if not answer.get("ok"):
            raise ToolRefused("explain", answer)
        _EXPLAIN[ident] = str(answer.get("text") or "")
    return _EXPLAIN[ident]


def _pose_list(value: Any, what: str) -> Any:
    """A typed pose in ``world`` as ``[x, y, z, qx, qy, qz, qw]``; any other value is left for the host to check."""

    typed_value = _typed_value(value)
    if typed_value is None:
        return value
    frame, fields = typed_value
    if "xyz" not in fields or "xyzw" not in fields:
        raise TypeError(f"{what} must be a pose (xyz and xyzw), not a point or a direction")
    if frame != "world":
        raise ValueError(f"{what} is in frame {frame!r}; a query takes world poses: convert it with {what}.to('world')")
    return _vector(fields["xyz"], 3, what) + _vector(fields["xyzw"], 4, what)


def query_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A query's arguments with the typed poses of ``poses`` (a pose alone, or a list of them) written as the lists
    the host takes."""

    arguments = dict(arguments)
    poses = arguments.get("poses")
    if hasattr(poses, "tolist"):
        poses = poses.tolist()
    if _typed_value(poses) is not None:
        arguments["poses"] = [_pose_list(poses, "poses")]
    elif isinstance(poses, (list, tuple)):
        arguments["poses"] = [_pose_list(item, f"poses[{index}]") for index, item in enumerate(poses)]
    return arguments


def query(name: str, **arguments: Any) -> Any:
    """A read of the environment's robot model or of the measured state: ``ik``, ``fk``, ``screen_path``,
    ``reach_check``, ``probe_motion``, ``align_plan`` or ``robot_state``; ``robot.docs("query_<name>")`` has the
    arguments (angles in degrees) and the result. It is no tool call; it counts against the script's query allowance."""

    if os.environ.get(LIBRARY_ENV) != "extended":
        raise RuntimeError("robot.query needs the extended library (policy.code_mode.library: extended)")
    answer = _channel().exchange({"op": "query", "name": str(name), "arguments": query_arguments(arguments)})
    if not answer.get("ok"):
        raise ToolRefused("query", answer)
    return answer.get("result")


def _bind_positional(documentation: str, positional: tuple[Any, ...], keywords: Mapping[str, Any]) -> dict[str, Any]:
    """``tools.<name>(*positional, **keywords)``'s arguments: each positional one takes the next argument of the
    signature, the first line of the function's ``documentation`` (``docs(name, "description")``), required ones first.
    Too many, or a name given twice, is a ``TypeError`` that shows the signature."""

    signature = documentation.split("\n", 1)[0].strip()
    names = [name.rstrip("?") for name in signature.partition("(")[2].rpartition(")")[0].split(", ") if name]
    if len(positional) > len(names):
        raise TypeError(f"{signature} takes {len(names)} argument(s), not {len(positional)}")
    bound = dict(zip(names, positional))
    twice = sorted(set(bound) & set(keywords))
    if twice:
        raise TypeError(f"{signature} got {', '.join(twice)} twice, by position and by name")
    return {**bound, **keywords}


class ToolFunction:
    # ``tools.<name>``: a call of it calls the tool; its documentation (``__doc__``, what ``help`` shows) is
    # ``docs(name)``, looked up on first use.

    def __init__(self, tools: "Tools", name: str) -> None:
        self._tools, self.__name__ = tools, name

    def __call__(self, **arguments: Any) -> Any:
        return self._tools.call(self.__name__, arguments)

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        try:
            return docs(self.__name__)
        except ToolRefused:
            return None

    def __repr__(self) -> str:
        return f"<function tools.{self.__name__}>"


class ExtendedFunction(ToolFunction):
    # ``tools.<name>`` with ``library: extended``: positional arguments too. A ``docs`` lookup gives their names; it is
    # no tool call and counts against nothing.

    __doc__ = ToolFunction.__doc__  # a class's own ``__doc__`` (None) would hide the property

    def __call__(self, *positional: Any, **arguments: Any) -> Any:
        if positional:
            arguments = _bind_positional(docs(self.__name__, "description"), positional, arguments)
        return self._tools.call(self.__name__, arguments)


class Tools:
    """``tools.<name>(**arguments)``: a dotted tool name is written with underscores."""

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        extended = os.environ.get(LIBRARY_ENV) == "extended"
        arguments = _fill_arguments(str(name), arguments or {}) if extended else dict(arguments or {})
        answer = _channel().exchange({"op": "call", "name": str(name), "arguments": arguments})
        if not answer.get("ok"):
            refused = ToolRefused(str(name), answer)
            if extended:
                refused.result = typed(str(name), answer)
            raise refused
        return typed(str(name), answer) if extended else answer

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return (ExtendedFunction if os.environ.get(LIBRARY_ENV) == "extended" else ToolFunction)(self, name)


tools = Tools()


def handoff(reason: str, **data: Any) -> None:
    """Stop the command here and hand control back to the model with ``reason`` and ``data``."""

    sys.stdout.flush()
    sys.stderr.flush()
    os.close(1)  # the host stops the whole command on this message: `python x.py | tail` prints on these closing
    os.close(2)
    time.sleep(HANDOFF_PIPE_S)
    _channel().exchange({"op": "handoff", "reason": str(reason), "data": data}, answer=False)
    os._exit(75)


__all__ = ["Plan", "PlanNotRun", "Result", "ToolRefused", "docs", "explain", "handoff", "query", "query_arguments",
           "target", "tools", "typed"]
