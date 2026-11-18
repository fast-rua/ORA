"""recover_pose and return_home as one smooth joint trajectory planned in lib (``lib.motion_plan.recovery``).

Where joint trajectories run (``move_pose._smooth.available``):

* recover_pose (``arms``) undoes each arm's latest not-yet-undone arm-moving
  tool call: it retraces that call's measured path (``exec.recovery.<arm>.undo``,
  the last 5 calls) back to where the call started, at move_pose's normal
  speed; a completed retrace pops the entry, so the next call undoes the one
  before. The jaws are held (an undone grasp does not reopen them). Without a
  call to undo it refuses before anything moves and names return_home and
  move_pose instead.
* return_home (``arms``) lifts link6 up to 0.10 m and unwinds the joints to the
  episode reset joints in a screened order, at move_pose's fast speed.

The host checks the restated stack tops or home anchors; grippers are held.
This module only maps the calls onto the lib plans.
"""

import math
from collections.abc import Mapping

from ...interface import dry_run
from ...interface.types import NotRecorded
from ...lib.motion_plan import checks, kept, kinematics, recovery
from ...lib.observation import observe
from .. import _dry_run
from ..budgets import native_action_budget
from ..errors import ToolRefusal
from .._receipts import model_receipt
from ..move_pose import _smooth
from ..move_pose._smooth import FULL, NORMAL
from ..move_pose.tool import measured_states

#: Paces per method: the retrace at move_pose's normal speed, joint interpolation at its fast speed (without a
#: motion profile; ``paces`` is a call's, joint interpolation at its profile's fast).
PACES = {recovery.RETRACE: NORMAL, recovery.JOINT: FULL}


def paces(config):
    """Pure computation: ``PACES`` of this call, joint interpolation at the fast of its context's motion profile."""
    return {**PACES, recovery.JOINT: _smooth.speeds(config)[_smooth.DEFAULT_SPEED][0]}
#: Hold the end until every joint is this close and this slow (counts toward the budget).
SETTLE = {"native_actions": 2, "joint_tolerance_rad": 0.003, "joint_speed_rad_s": 0.05}
INSTEAD = ("Use return_home to go straight back to the episode start pose, or move_pose to move to a pose "
           "of your choice.")
#: Why recover_pose has no move to retrace (failure_code -> what the message says).
NOTHING_TO_RETRACE = {
    "retrace_not_recorded": "this run records no undo paths (it predates them)",
    "retrace_nothing_to_undo": ("nothing is left to undo: only the most recent 5 arm-moving tool calls per arm "
                                "can be undone, one per call, and none is left in this round (none made yet, or "
                                "all already undone); older moves cannot be retraced"),
    "retrace_arm_off_path": ("the arm no longer stands on the path of its latest undoable tool call (it moved "
                             "since by another route)"),
}


def arms_argument(arguments, name):
    """The validated ``arms`` of a call (sorted), or ``invalid_motion_arguments``."""
    if not isinstance(arguments, Mapping) or set(arguments) != {"arms"}:
        raise ToolRefusal(f"{name} takes exactly arms", code="invalid_motion_arguments",
                          constraints={"allowed": ["arms"], "example": {"arms": ["left"]}})
    arms = arguments["arms"]
    if (not isinstance(arms, list) or not arms or len(set(arms)) != len(arms)
            or any(arm not in ("left", "right") for arm in arms)):
        raise ToolRefusal("arms must list left, right or both, each once", code="invalid_motion_arguments",
                          field="arms", received=arms, constraints={"allowed": ["left", "right"]})
    return sorted(arms)


def _available(config, name):
    if not _smooth.available(config):
        if dry_run.requested():
            raise _dry_run.unavailable("this surface's recoveries are not joint trajectories planned on the policy "
                                       "side: the environment plans them when they execute")
        raise ToolRefusal(f"{name} needs synchronous execution on a backend that executes joint trajectories",
                          code="tool_unavailable", retryable=False)


def _finite(values):
    return isinstance(values, list) and all(type(v) in (int, float) and math.isfinite(v) for v in values)


def _kept(api, obs, config, name, start):
    """The plan a dry run kept for this call on this observation, its key and its warnings; a dry run always plans."""
    identity = obs.identity.json()
    key = kept.key(identity, tool=name, start_joints=start,
                   budget=_smooth.allowance(native_action_budget(name), config)[0])
    record = None if dry_run.requested() else kept.recall(api, identity, key)
    return (None if record is None else kept.recovery_plan(record)), key, (record or {}).get("warnings", ())


def _plan(api, config, name, options, start, goals, paces):
    # The call's budget, at most what the episode and, inside a code-mode script, the script has left.
    allowance, _ = _smooth.allowance(native_action_budget(name), config)
    try:
        return recovery.plan(api, options=options, start_joints=start, goals=goals, paces=paces, settle=SETTLE,
                             native_action_budget=allowance)
    except checks.PlanRefusal as error:
        raise ToolRefusal(str(error), code="plan_rejected",
                          constraints={"failure_code": error.code, **error.constraints}) from error
    except kinematics.KinematicsUnavailable as error:
        raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error


def _undo_report(stack, planned, arm, end, completed):
    """Which call this retrace undid, whether completely or how much of its path is left, what remains undoable."""
    top = stack[-1]
    done = completed and planned.fraction >= 1.0
    left = 0.0 if done else recovery.path_position(
        top["measured_path_rad"], end if end is not None else planned.knots["joints"][arm][-1])
    remaining = stack[:-1] if done else stack
    return {"undone_call": _call(top), "complete": done,
            **({} if done else {"path_left_fraction": round(left, 3)}),
            "undoable_calls": [_call(entry) for entry in reversed(remaining)]}


def _result(api, config, obs, name, planned, start, recovery_argument, route, warnings=(), undo=None, reused=False):
    result = recovery.run(api, planned, action_id=f"action-{config['execution']['generation']}",
                          observation=obs.identity.json(), recovery=recovery_argument)
    receipt = _smooth._previewed(model_receipt(result["receipt"]), reused)
    after = receipt.get("after") or {}
    arms = receipt.setdefault("arms", {})
    for arm in planned.options:
        # The measured joints: the receipt's after-state when it carries one, else the trajectory evidence.
        end = (after.get(arm) or {}).get("arm_joint_qpos") or recovery.measured_end(planned, arm, result["summary"])
        progress = {**recovery.outcome(planned, arm, start[arm], end, result["summary"]), "route": route}
        if undo is not None:
            # The environment pops the entry when the retrace ran every knot of its whole path.
            progress["undo"] = _undo_report(undo[arm], planned, arm, end,
                                            result["summary"].get("status") == "completed")
        if warnings:
            progress["ik_warning"] = ("This run predates IK-generator recording, so the lift was planned with IK "
                                      "answers recomputed from the generator's initial state.")
        arms[arm] = {**dict(arms.get(arm) or {}), "recovery_progress": progress}
    compiled = {"semantic_primitive": name, "canonical_action": "follow_joint_trajectory",
                "targets": {arm: {"strategy": "retrace" if name == "recover_pose" else "home",
                                  "method": planned.options[arm].method} for arm in planned.options}}
    return {"status": "executed", "action": name, "compiled_pose": compiled, "receipt": receipt,
            "remaining_budget": result["remaining_budget"]}


def _nothing(arm, code, **evidence):
    return ToolRefusal(f"recover_pose cannot retrace the {arm} arm: {NOTHING_TO_RETRACE[code]}. {INSTEAD}",
                       code="plan_rejected", retryable=False,
                       constraints={"failure_code": code, "arm": arm, "native_actions": 0,
                                    "use_instead": ["return_home", "move_pose"], **evidence})


def _call(entry):
    return {key: entry.get(key) for key in ("tool_call_id", "tool", "ordinal")}


def run_retrace(api, config, arms):
    """Replay: recorded observation, undo stacks, kinematics answers and the one recorded trajectory primitive.

    Each arm retraces the top entry of ``exec.recovery.<arm>.undo`` (its latest undoable tool call) from where
    it stands on that call's measured path back to its start; a refusal before anything moves names the reason
    and return_home / move_pose. The result's ``recovery_progress.undo`` says which call was undone, whether
    the undo is complete or how much of its path remains, and which calls can still be undone."""
    _available(config, "recover_pose")
    obs = observe(api)
    manifest = obs.manifest.json()
    measured_states(manifest)
    effectors = manifest["proprioception"]["end_effectors"]
    start, options, goals, stated, stacks = {}, {}, {}, {}, {}
    for arm in arms:
        here = effectors[arm].get("arm_joint_qpos")
        if not _finite(here):
            raise _nothing(arm, "retrace_arm_off_path", measured="arm joints unavailable")
        start[arm] = [float(v) for v in here]
        try:
            moves = api.state.read(f"exec.recovery.{arm}.undo")["moves"]
        except (NotRecorded, KeyError, TypeError) as error:
            raise _nothing(arm, "retrace_not_recorded") from error
        if not moves:
            raise _nothing(arm, "retrace_nothing_to_undo")
        top = moves[-1]
        nodes, info = recovery.retrace_nodes(top["arm_joint_qpos"], top["measured_path_rad"], start[arm])
        if nodes is None:
            raise _nothing(arm, "retrace_arm_off_path", latest_call=_call(top),
                           joint_distance_before_rad=info.get("joint_distance_before_rad"))
        options[arm] = [recovery.Option(recovery.RETRACE, (nodes,), {"retrace": info})]
        goals[arm] = top["pose_xyz_xyzw"]
        stated[arm] = {"moves": len(moves), "arm_joint_qpos": list(top["arm_joint_qpos"])}
        stacks[arm] = moves
    planned, key, _ = _kept(api, obs, config, "recover_pose", start)
    reused = planned is not None
    planned = planned or _plan(api, config, "recover_pose", options, start, goals, paces(config))
    if dry_run.requested():  # the plan the call would run, not its trajectory
        kept.keep(api, obs.identity.json(), key, kept.recovery_record(planned))
        return _dry_run.recovered("recover_pose", planned, route=recovery.RETRACE, undo={
            arm: _undo_report(stacks[arm], planned, arm, None, True) for arm in arms},
            view=_dry_run.view(api, obs, config))
    return _result(api, config, obs, "recover_pose", planned, start,
                   {"targets": {arm: "retrace" for arm in arms}, "retrace": stated}, recovery.RETRACE,
                   undo=stacks, reused=reused)


def _home_options(api, config, arm, anchor, effector, bounds, warnings):
    begin = [float(v) for v in effector["arm_joint_qpos"]]
    unwind = recovery.unwind_target(anchor["arm_joint_qpos"], begin)
    compact = {key: unwind[key] for key in ("status", "equivalent_turns", "pitch_vetoed_turns") if key in unwind}
    planned = (recovery.global_option(api, arm=arm, target=unwind["joint_target"], detail={"unwind_target": compact})
               if config["motion"].get("global_planner") else None)
    lift, report, warned = recovery.plan_lift(api, arm=arm, pose=effector["pose_xyz_xyzw"], joints=begin,
                                              bounds=bounds)
    warnings.extend(warned)
    found = []
    if lift is not None:
        found += recovery.joint_options(begin, unwind["joint_target"], before=(lift,),
                                        detail={"retract": report, "unwind_target": compact})
    skipped = report if lift is None else {**report, "status": "skipped", "reason": "path_blocked"}
    return ([planned] if planned is not None else []) + found + recovery.joint_options(
        begin, unwind["joint_target"], before=([begin, begin],), detail={"retract": skipped, "unwind_target": compact})


def run_home(api, config, arms):
    """Replay: recorded observation, home anchors, kinematics answers and the one recorded trajectory primitive.

    Each arm lifts link6 up to 0.10 m straight up, then unwinds to ``exec.recovery.<arm>.home`` in the first
    clear order; missing anchors and blocked paths refuse before anything moves."""
    _available(config, "return_home")
    obs = observe(api)
    manifest = obs.manifest.json()
    measured_states(manifest)
    effectors = manifest["proprioception"]["end_effectors"]
    anchors = {}
    for arm in arms:
        try:
            anchor = api.state.read(f"exec.recovery.{arm}.home")
            anchors[arm] = {"strategy": "home", "arm_joint_qpos": anchor["arm_joint_qpos"],
                            "pose_xyz_xyzw": anchor["pose_xyz_xyzw"]}
        except (NotRecorded, KeyError, TypeError):
            anchors[arm] = None
        if anchors[arm] is None or not (_finite(anchors[arm]["arm_joint_qpos"])
                                        and _finite(effectors[arm].get("arm_joint_qpos"))):
            raise ToolRefusal("the episode reset joints or the measured joints are unavailable",
                              code="plan_rejected", constraints={"failure_code": "recovery_state_unavailable",
                                                                 "arm": arm, "native_actions": 0})
    start = {arm: [float(v) for v in effectors[arm]["arm_joint_qpos"]] for arm in arms}
    planned, key, warnings = _kept(api, obs, config, "return_home", start)
    reused = planned is not None
    if not reused:
        warnings, options = [], {}
        try:
            for arm in arms:
                options[arm] = _home_options(api, config, arm, anchors[arm], effectors[arm],
                                             config["motion"]["workspace_bounds_m"], warnings)
        except kinematics.KinematicsUnavailable as error:
            raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error
        planned = _plan(api, config, "return_home", options, start,
                        {arm: anchors[arm]["pose_xyz_xyzw"] for arm in arms}, paces(config))
    if dry_run.requested():
        kept.keep(api, obs.identity.json(), key, kept.recovery_record(planned, warnings))
        return _dry_run.recovered("return_home", planned, route="retract_then_unwind",
                                  view=_dry_run.view(api, obs, config))
    return _result(api, config, obs, "return_home", planned, start,
                   {"targets": {arm: "home" for arm in arms}, "recovery_targets": anchors}, "retract_then_unwind",
                   warnings=warnings, reused=reused)


__all__ = ["INSTEAD", "NOTHING_TO_RETRACE", "PACES", "SETTLE", "arms_argument", "paces", "run_home", "run_retrace"]
