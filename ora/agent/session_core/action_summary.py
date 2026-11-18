"""One plain-language line at the head of the last-action result.

``last_action_summary_v1``: the policy reads, before the detail, a line such as
``left arm 104 mm / 120° from target, plan rejected: inter arm collision, not
moved``. It restates only what the projected effect already carries:

* distance from target per targeted arm -- the arm's ``pose_error``, else the
  last endpoint acceptance residual, else LIBERO's judged endpoint residual;
  for a closed skill, the last stage that measured an endpoint residual. A move
  its budget cut short states the measured distance left to its goal instead,
  and a recovery the joint distance left: its ``pose_error`` is the cut point's;
* for a gripper command, measured opening and effective target;
* the outcome -- ``status`` and, when it adds something, ``failure_code``
  (a closed skill also names its ``failure_stage``); a move whose
  ``motion_progress`` says the native-action budget ended it names that, not the
  public ``controller_timeout`` it maps to;
* whether the arm moved -- ``synchronization.executed``, the native-action
  window, or the measured TCP translation, else ``motion unknown``.

A move that stopped under way adds one clause, the same on every backend:
how far along it stopped (``stop_evidence.path_fraction``, else the distance
moved against the distance left), which way it was moving, the joint that fell
behind where the receipt can tell, and, when nothing measured a contact, that
it probably touched something, worded as the inference it is.

Where the receipt names the stop that ended a move (``detail.endpoint.stop_reason``
for an arm, ``detail.base.stop_reason`` for a mobile base), the outcome is
followed by that stop in words when the outcome does not already say it: a
contact stop, which is ``reached`` wherever it stopped, and each stop that left a
move short (joint lag, joints at rest short of the path's end, a settle that
ran out, base contact). A stop a measured contact made (the gripper contact
stop, base contact) is said as measured, never as the inferred contact, and
with no force value.

A move the planner refused, or that its plan's validation refused
(``failure_stage`` ``plan_validation``, RoboCasa's admission), ends with the
planner's own words: the first ``recovery`` sentence of its planning
diagnostic, which says why and what near the goal plans, unless it is one of
the fixed hints a planner gives when it found no cause. That sentence is
quoted, not composed here. A refusal whose effect carries ``admission_refusal``
then restates it (``toolkit/tools/_refusal_words.py``): where a line's IK
stopped and why, whether the target itself is reachable from here, which robot
part enters the depth and how far along, a target's distance from the arm mount
and the reach, the base's footprint meeting the depth, and the nearest base pose
the base-placement search found. A move that lifted the torso first says so,
from its ``torso`` block.

The line adds no other hint or cause, and no next step. Angles are in degrees
whatever unit the detail uses. The shared contact detail measures the right
arm's TCP only, so its translation is cited only when the right arm is the one
target, or when the effect names no target arm (LIBERO's single-arm hybrid
receipts). A surface declaring ``per_arm_realized_translation_v1`` carries each
targeted arm's own translation in ``detail.arms``, and those are cited instead.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..toolkit.tools._refusal_words import admission_words, first_recovery, torso_words

CAPABILITY = "last_action_summary_v1"
SUMMARY_KEY = "action_summary"
_ENDPOINT_STAGES = ("final_endpoint", "controller_terminal_endpoint")
#: ``motion_progress.outcome`` of a move cut to fit its budget (``environment.backends.robodojo.trajectory``).
_CUT_SHORT = "cut_short"
#: The outcomes of a move the native-action budget ended: cut to fit it, or out of it before the last knot.
_BUDGET_OUTCOMES = (_CUT_SHORT, "budget")
_BUDGET_CODE = "native_action_budget_exhausted"
#: The stops under way: the arm could not follow, a measured force ended it, or it left its target.
_COULD_NOT_FOLLOW = ("controller_tracking_error", "controller_stalled", "blocked")
_STOPPED_UNDER_WAY = (*_COULD_NOT_FOLLOW, "contact_stop", "tracking_diverged")
#: The inference a stop may add, only where no contact was measured.
_INFERRED_CONTACT = "probably touched something (inferred, no contact sensor)"
#: A stop the receipt names, in words: an arm's ``detail.endpoint.stop_reason`` and a mobile base's
#: ``detail.base.stop_reason``. ``failure_code`` ``blocked`` covers the first three arm stops and base contact.
_ARM_STOPS = {
    "joint_lag": "ended by joint lag",
    "settle_stalled": "ended with the joints at rest short of the path's end",
    "settle_timeout": "ended still settling after the last target",
    "contact_detected": "ended by the gripper contact stop",
}
_BASE_STOPS = {"base_contact": "ended by base contact"}
#: The stops a measured contact made: said as measured, never as the inferred contact.
_MEASURED_CONTACT_STOPS = frozenset({"contact_detected", "base_contact"})
#: Under this travel a direction would be noise.
_MIN_TRAVEL_M = 0.002
#: The outcome of a move the planner refused (``receipt_projection`` folds every planner refusal into it).
_PLANNER_REFUSED = "planner_failed"
#: The stage of a move its plan's validation refused before it moved (RoboCasa's admission, on the shared planner).
_PLAN_VALIDATION = "plan_validation"


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _millimetres(metres: float) -> str:
    value = metres * 1000.0
    return f"{value:.0f} mm" if abs(value) >= 10 else f"{value:.1f} mm"


def _degrees(radians: float) -> str:
    value = math.degrees(radians)
    return f"{value:.0f}°" if abs(value) >= 10 else f"{value:.1f}°"


def _words(code: Any) -> str:
    return str(code).replace("_", " ")


def _from_target(arm: Any, position: Any, orientation: Any) -> str | None:
    position, orientation = _finite(position), _finite(orientation)
    parts = [*([_millimetres(position)] if position is not None else []),
             *([_degrees(orientation)] if orientation is not None else [])]
    if not parts:
        return None
    return f"{f'{arm} arm ' if arm else ''}{' / '.join(parts)} from target"


def _left_to_goal(state: Any) -> float | None:
    """The measured distance left to the goal of an arm whose move was cut short, else ``None``."""
    progress = state.get("motion_progress") if isinstance(state, Mapping) else None
    if isinstance(progress, Mapping) and progress.get("outcome") == _CUT_SHORT:
        return _finite(progress.get("remaining_to_goal_m"))
    return None


def _recovery_left(state: Any) -> float | None:
    """The joint distance left to an arm whose recovery its budget cut short, else ``None``: it has no goal pose."""
    progress = state.get("recovery_progress") if isinstance(state, Mapping) else None
    if isinstance(progress, Mapping) and progress.get("outcome") == _CUT_SHORT:
        return _finite(progress.get("joint_distance_end_rad"))
    return None


def _pose_residuals(detail: Mapping[str, Any]) -> list[str]:
    rows: list[str] = []
    arms, targets = detail.get("arms"), detail.get("target_arms")
    if isinstance(arms, Mapping) and isinstance(targets, list):
        for arm in targets:
            state = arms.get(arm)
            turn = _recovery_left(state)
            if turn is not None:
                rows.append(f"{arm} arm {_degrees(turn)} of joint travel left")
                continue
            left = _left_to_goal(state)
            error = state.get("pose_error") if isinstance(state, Mapping) else None
            if left is not None:
                error = {"translation_m": left}
            if isinstance(error, Mapping):
                row = _from_target(arm, error.get("translation_m", error.get("position_m")),
                                   error.get("orientation_rad"))
                rows.extend([row] if row else [])
    if rows:
        return rows
    for check in reversed(detail.get("acceptance_checks") or []):
        if isinstance(check, Mapping) and check.get("stage") in _ENDPOINT_STAGES \
                and isinstance(check.get("arms"), Mapping):
            for arm, value in sorted(check["arms"].items()):
                residual = value.get("residual") if isinstance(value, Mapping) else None
                if isinstance(residual, Mapping):
                    row = _from_target(arm, residual.get("position_m"), residual.get("orientation_rad"))
                    rows.extend([row] if row else [])
            if rows:
                return rows
    contract = detail.get("endpoint_contract")
    judged = contract.get("judged_residual") if isinstance(contract, Mapping) else None
    if isinstance(judged, Mapping):
        row = _from_target(None, judged.get("position_m"), judged.get("orientation_rad"))
        rows.extend([row] if row else [])
    return rows


def _gripper(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> list[str]:
    checks = [check for check in detail.get("acceptance_checks") or [] if isinstance(check, Mapping)]
    gripper_checks = [check for check in checks if check.get("stage") == "gripper_endpoint"]
    if effect.get("action") != "set_gripper" and not gripper_checks:
        return []
    feedback = detail.get("gripper_feedback")
    targets = detail.get("target_arms") if isinstance(detail.get("target_arms"), list) else []
    rows = []
    for arm in targets:
        state = feedback.get(arm) if isinstance(feedback, Mapping) else None
        measured = _finite(state.get("measured_opening_normalized")) if isinstance(state, Mapping) else None
        if measured is None:
            continue
        target = _finite(state.get("effective_target_opening_normalized"))
        rows.append(f"{arm} gripper opening {measured:.2f}"
                    + (f" / target {target:.2f}" if target is not None else ""))
    if rows:
        return rows
    for check in gripper_checks:
        for arm, value in sorted((check.get("arms") or {}).items()):
            residual = value.get("residual") if isinstance(value, Mapping) else None
            error = _finite(residual.get("opening_normalized")) if isinstance(residual, Mapping) else None
            if error is not None:
                rows.append(f"{arm} gripper {error:.2f} from target opening")
    return rows


def _budget_ended(detail: Any) -> bool:
    arms = detail.get("arms") if isinstance(detail, Mapping) else None
    states = arms.values() if isinstance(arms, Mapping) else ()
    return any(isinstance(state, Mapping) and state.get("targeted")
               and (isinstance(state.get("motion_progress"), Mapping)
                    and state["motion_progress"].get("outcome") in _BUDGET_OUTCOMES
                    or _recovery_left(state) is not None) for state in states)


def _outcome(effect: Mapping[str, Any], *, name_stage: bool = False) -> str:
    status = effect.get("status")
    text = _words(status)
    stage = effect.get("failure_stage")
    if name_stage and isinstance(stage, str) and stage:
        text += f" at {_words(stage)}"
    code = _BUDGET_CODE if _budget_ended(effect.get("detail")) else effect.get("failure_code")
    if isinstance(code, str) and code and code != status:
        text += f": {_words(code)}"
    cause = effect.get("physical_cause")
    if isinstance(cause, str) and cause and cause != "unknown":
        text += f" ({_words(cause)})"
    return text


def _measured_translation(detail: Mapping[str, Any]) -> float | None:
    # The contact detail reads the right arm (LIBERO's only arm) whatever was
    # targeted; LIBERO's hybrid receipts name no target arms at all.
    if detail.get("target_arms", ["right"]) != ["right"]:
        return None
    contact = detail.get("contact_motion")
    vector = contact.get("realized_translation_world_m") if isinstance(contact, Mapping) else None
    if not isinstance(vector, list) or len(vector) != 3 or any(_finite(item) is None for item in vector):
        return None
    return math.dist(vector, (0.0, 0.0, 0.0))


def _arm_translations(detail: Mapping[str, Any]) -> list[tuple[str, float]]:
    arms, targets = detail.get("arms"), detail.get("target_arms")
    if not isinstance(arms, Mapping) or not isinstance(targets, list):
        return []
    rows = []
    for arm in targets:
        state = arms.get(arm)
        vector = state.get("realized_translation_world_m") if isinstance(state, Mapping) else None
        if isinstance(vector, list) and len(vector) == 3 and all(_finite(item) is not None
                                                                  for item in vector):
            rows.append((str(arm), math.dist(vector, (0.0, 0.0, 0.0))))
    return rows


def _motion(detail: Mapping[str, Any]) -> str:
    sync = detail.get("synchronization")
    executed = sync.get("executed") if isinstance(sync, Mapping) else None
    window = detail.get("execution_window")
    used = window.get("native_actions_used") if isinstance(window, Mapping) else None
    budget = window.get("native_action_budget") if isinstance(window, Mapping) else None
    used = used if isinstance(used, int) and not isinstance(used, bool) else None
    if executed is False or used == 0:
        return "not moved"
    moved = _arm_translations(detail)
    if len(moved) == 1:
        return f"TCP moved {_millimetres(moved[0][1])}"
    if moved:
        return ", ".join(f"{arm} TCP moved {_millimetres(value)}" for arm, value in moved)
    translation = _measured_translation(detail)
    if translation is not None:
        return f"TCP moved {_millimetres(translation)}"
    if used is not None:
        return f"ran {used} of {budget} native actions" if isinstance(budget, int) else (
            f"ran {used} native actions")
    return "executed" if executed is True else "motion unknown"


def _travel(detail: Mapping[str, Any], arm: Any) -> list[float] | None:
    """``arm``'s measured TCP translation; the contact detail's only for what it measures, as ``_measured_translation``."""
    arms = detail.get("arms")
    state = arms.get(arm) if isinstance(arms, Mapping) else None
    contact = detail.get("contact_motion")
    vector = state.get("realized_translation_world_m") if isinstance(state, Mapping) else None
    if vector is None and isinstance(contact, Mapping) and detail.get("target_arms", ["right"]) == ["right"]:
        vector = contact.get("realized_translation_world_m")
    if not isinstance(vector, list) or len(vector) != 3 or any(_finite(item) is None for item in vector):
        return None
    return [float(item) for item in vector]


def _heading(vector: list[float]) -> str | None:
    """The way a world-frame translation went, in words: up, down, or its larger horizontal axis."""
    length = math.dist(vector, (0.0, 0.0, 0.0))
    if length < _MIN_TRAVEL_M:
        return None
    if abs(vector[2]) >= 0.7 * length:
        return "moving down" if vector[2] < 0 else "moving up"
    axis = 0 if abs(vector[0]) >= abs(vector[1]) else 1
    return f"moving {'-' if vector[axis] < 0 else '+'}{'xy'[axis]}"


def _stop_clause(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> str | None:
    """How far a move that stopped under way got, which way, what fell behind and what that suggests."""
    if effect.get("failure_code") not in _STOPPED_UNDER_WAY:
        return None
    arms = detail.get("arms") if isinstance(detail.get("arms"), Mapping) else {}
    targets = detail.get("target_arms") if isinstance(detail.get("target_arms"), list) else []
    stopped = next((arm for arm, state in arms.items() if isinstance(state, Mapping) and "stop_evidence" in state),
                   targets[0] if len(targets) == 1 else None)
    state = arms.get(stopped) if isinstance(arms.get(stopped), Mapping) else {}
    evidence = state.get("stop_evidence") if isinstance(state.get("stop_evidence"), Mapping) else {}
    vector = _travel(detail, stopped)
    fraction = _finite(evidence.get("path_fraction"))
    error = state.get("pose_error") if isinstance(state.get("pose_error"), Mapping) else {}
    left = _finite(error.get("translation_m", error.get("position_m")))
    if fraction is None and vector is not None and left is not None:
        moved = math.dist(vector, (0.0, 0.0, 0.0))
        fraction = moved / (moved + left) if moved + left > 0 else None
    parts = [*([f"stopped {fraction:.0%} along its path"] if fraction is not None and fraction >= 0.01 else []),
             *([_heading(vector)] if vector is not None and _heading(vector) else [])]
    clauses = [" ".join(parts)] if parts else []
    lag = _finite(evidence.get("lag_rad"))
    if isinstance(evidence.get("lagging_joint"), int) and lag is not None:
        clauses.append(f"joint {evidence['lagging_joint']} fell {_degrees(lag)} behind")
    if (effect.get("failure_code") in _COULD_NOT_FOLLOW and effect.get("physical_cause") in (None, "unknown")
            and not _MEASURED_CONTACT_STOPS.intersection(_stop_reasons(detail))):
        clauses.append(_INFERRED_CONTACT)
    return ", ".join(clauses) or None


def _stop_reasons(detail: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The arm's ``endpoint.stop_reason`` and the base's ``base.stop_reason``, each ``None`` when not named."""
    def named(block: Any) -> str | None:
        reason = block.get("stop_reason") if isinstance(block, Mapping) else None
        return reason if isinstance(reason, str) else None
    return named(detail.get("endpoint")), named(detail.get("base"))


def _stop_named(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> str | None:
    """The stop that ended a move, in words: a contact stop always, an arm's other stops when it did not reach."""
    arm, base = _stop_reasons(detail)
    if arm in _ARM_STOPS and (arm in _MEASURED_CONTACT_STOPS or effect.get("status") != "reached"):
        return _ARM_STOPS[arm]
    return _BASE_STOPS.get(base) if base else None


def _planner_said(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> list[str]:
    """The first ``recovery`` sentence of each planning diagnostic of a move the planner, or its plan's validation,
    refused.

    LIBERO and RoboCasa keep one diagnostic for the move (``detail.planning_diagnostics``); RoboDojo and RoboTwin keep
    one per targeted arm, at ``arms.<arm>.planning_diagnostics`` or under the arm's ``planner``.
    """
    if (_PLANNER_REFUSED not in (effect.get("status"), effect.get("failure_code"))
            and effect.get("failure_stage") != _PLAN_VALIDATION):
        return []
    found: list[tuple[str | None, str]] = []
    own = first_recovery(detail.get("planning_diagnostics"))
    if own:
        found.append((None, own))
    arms = detail.get("arms") if isinstance(detail.get("arms"), Mapping) else {}
    targets = detail.get("target_arms") if isinstance(detail.get("target_arms"), list) else list(arms)
    for arm in targets:
        state = arms.get(arm) if isinstance(arms.get(arm), Mapping) else {}
        planner = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
        text = first_recovery(state.get("planning_diagnostics")) or first_recovery(planner.get("planning_diagnostics"))
        if text and all(text != seen for _, seen in found):
            found.append((str(arm), text))
    named = len(found) > 1 or len(targets) > 1  # say whose words they are when two arms moved
    return [f"{arm} arm: {text}" if arm and named else text for arm, text in found]


def _skill_summary(effect: Mapping[str, Any], detail: Mapping[str, Any]) -> list[str]:
    stages = [stage for stage in detail["stages"] if isinstance(stage, Mapping)]
    rows: list[str] = []
    for stage in reversed(stages):
        for measurement in stage.get("measurements") or []:
            value = measurement.get("value") if isinstance(measurement, Mapping) else None
            errors = value.get("errors_by_arm") if isinstance(value, Mapping) else None
            if measurement.get("quantity") == "endpoint_residual" and isinstance(errors, Mapping):
                for arm, error in sorted(errors.items()):
                    if isinstance(error, Mapping):
                        row = _from_target(arm, error.get("translation_m", error.get("position_m")),
                                           error.get("orientation_rad"))
                        rows.extend([row] if row else [])
        if rows:
            break
    parts = [*rows, _outcome(effect, name_stage=True)]
    committed = detail.get("committed_sequence")
    if isinstance(committed, list) and committed:
        parts.append(f"{len(stages)} of {len(committed)} stages attempted")
    return parts


def action_summary(effect: Any) -> str | None:
    """The summary line for one projected effect, or ``None`` for an unknown shape."""
    if not isinstance(effect, Mapping) or not isinstance(effect.get("status"), str):
        return None
    detail = effect.get("detail") if isinstance(effect.get("detail"), Mapping) else {}
    if isinstance(detail.get("stages"), list):
        parts = _skill_summary(effect, detail)
    else:
        clause, named = _stop_clause(effect, detail), _stop_named(effect, detail)
        lifted = torso_words(detail.get("torso"))
        parts = [*_pose_residuals(detail), *_gripper(effect, detail), _outcome(effect), *([named] if named else []),
                 _motion(detail), *([lifted] if lifted else []), *([clause] if clause else [])]
    if effect.get("round_terminated") is True:
        parts.append("round ended")
    line = ", ".join(parts)
    said = [*_planner_said(effect, detail), *admission_words(detail.get("admission_refusal"))]
    return " ".join([f"{line}.", *said]) if said else line


__all__ = ["CAPABILITY", "SUMMARY_KEY", "action_summary"]
