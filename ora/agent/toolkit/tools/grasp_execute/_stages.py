"""Execute prepared geometry stage by stage; verification remains read-only.

Where joint trajectories run (``move_pose._smooth.available``), each arm stage (pregrasp, approach,
lift) is one trajectory at move_pose's fast speed with the stage's interaction picking the route
(``_smooth.run_action``), and open/close are set_gripper's rate-limited jaw commands
(``set_gripper.tool.rate_limited_targets``); elsewhere, the released primitives. An arm stage may spend
what the sequence's budget has left, as a pregrasp at the fast speed needs up to 30 native actions; the
released primitive's 20 cut a third of the recorded pregrasps short of their goal, so the grasp stopped
before it closed. The stage sequence, candidates, verification and held-state bookkeeping are the same
either way.
"""
from ...lib.motion_plan import checks, kinematics
from ...lib.observation import observe
from . import _lift_evidence, _verify
from .._receipts import model_receipt, model_stages
from ..budgets import native_action_budget
from ..move_pose import _smooth
from ..set_gripper.tool import rate_limited_targets
from .. import _held


def _refused(action, error):
    """The stage record of a path refused before anything moved (no action was sent)."""
    code = getattr(error, "code", "planner_failed")
    return {"action_id": action["action_id"], "status": "refused_before_motion", "reason": code,
            "native_actions": 0, "constraints": dict(getattr(error, "constraints", {}) or {}),
            "stage_evidence": {"execution_status": "plan_rejected", "failure_code": code, "consumed": False}}


def stage_result(stage, arm, receipt):
    """Keep the released sparse-motion endpoint and applied-close predicates."""
    evidence = receipt.get("stage_evidence", {})
    detail = receipt.get("detail", {})
    arms = detail.get("arms") or {}
    outcomes = {name: value["gripper_outcome"] for name, value in arms.items()
                if isinstance(value, dict) and isinstance(value.get("gripper_outcome"), dict)}
    measurements = ([{"quantity": "gripper_outcome", "value": outcomes}] if outcomes else [])
    reached = evidence.get("execution_status") == "reached" and evidence.get("consumed") is True
    reason = "executed"
    if stage in {"open", "close"}:
        applied_close = (stage == "close" and evidence.get("failure_code") in {
            "gripper_not_reached", "gripper_stalled_before_target"}
            and evidence.get("execution_status") == "not_reached"
            and receipt.get("native_actions", 0) > 0
            and evidence.get("gripper_directives", {}).get(arm) == 0.0)
        reached = (evidence.get("consumed") is True
                   and evidence.get("command_invariant_verified") is True
                   and (evidence.get("execution_status") == "reached" or applied_close))
        if applied_close and evidence.get("failure_code") == "gripper_stalled_before_target":
            reason = "gripper_stalled_before_target"
    if not reached:
        reason = evidence.get("failure_code") or receipt.get("reason") or ("gripper_command_unavailable" if stage in {"open", "close"} else "endpoint_not_reached")
    return {"stage": stage, "status": "executed" if reached else "not_reached",
            "reason": reason, "action_id": receipt.get("action_id"),
            "measurements": measurements, "receipt": receipt}


def run(api, config, commitment, sequence):
    """Live: publish after each primitive. Replay: no online action fallback."""
    stages = []
    remaining = 0
    arm = commitment["arm"]
    original_intent = api.state.read(f"tool.held.{arm}")
    spent = 0
    committed_intent = _held.intent("grasp_execute", commitment["source_ref"],
                                    [*commitment["poses"]["approach"]["position_m"],
                                     *commitment["poses"]["approach"]["orientation_xyzw"]])
    final = observe(api)
    smooth = _smooth.available(config)
    left = native_action_budget("grasp_execute")
    for index, stage in enumerate(sequence):
        if stage == "verify":
            measured = _verify.measure(stage, final.manifest.json(), commitment, stages)
            if "lift" in sequence:
                measured["verification"]["lift_probe"] = _lift_evidence.measure(
                    api, commitment=commitment, observation=final,
                    workspace_bounds_m=config["execution"]["workspace_bounds_m"])
            if measured["verification"].get("jaw_hold", {}).get("held") == "unlikely":
                _held.stage(api, {arm: None})
            return {"status": "executed", "skill": "grasp_execute",
                    "candidate_id": commitment["candidate_id"], "stages": model_stages(stages),
                    "verification": measured["verification"], "remaining_budget": remaining,
                    "target_ref": commitment["source_ref"],
                    "claim_boundary": "measured_jaws_and_visible_surface;not_attachment_or_task_success",
                    "_runtime": {"skill_feedback": {
                        "committed_sequence": list(sequence), "arm": arm,
                        "verify": measured,
                    }}}
        if stage == "lift":
            commitment = {**commitment, "lift_baseline": final}
        identity = final.identity.json()
        pose = commitment["poses"].get(stage)
        interaction = "contact_approach" if stage == "approach" else (
            "precision" if stage == "lift" else "free_space")
        action = {"schema": "ora-motion-action.v3", "action": "move_pose" if pose is not None else "set_gripper",
                  "action_id": f"action-{config['execution']['generation']}:{index:02d}-{stage}",
                  "observation_id": identity["observation_id"],
                  "bound_observation_sha256": identity["observation_sha256"],
                  "arguments": {"targets": {arm: pose if pose is not None else stage},
                                **({"interactions": {arm: interaction}} if pose is not None else {})}}
        updates = {}
        if stage == "open":
            updates[arm] = None
        elif stage == "close":
            updates[arm] = committed_intent
        previous = _held.stage(api, updates)
        if smooth and pose is not None:
            try:
                receipt, remaining_now, _ = _smooth.run_action(api, action, config, budget=left)
                value = {"receipt": receipt, "remaining_budget": remaining_now}
            except (checks.PlanRefusal, kinematics.KinematicsUnavailable) as error:
                _held.stage(api, previous)
                value = {"receipt": _refused(action, error), "remaining_budget": left}
        else:
            if smooth:
                effectors = final.manifest.json()["proprioception"]["end_effectors"]
                action["arguments"]["targets"] = rate_limited_targets({arm: stage}, effectors, config)
            value = _held.execute(api, action, previous,
                                  native_action_budget("set_gripper") if smooth else 20)
        spent += value["receipt"].get("native_actions", 0)
        remaining = left = value["remaining_budget"]
        record = stage_result(stage, arm, value["receipt"])
        stages.append(record)
        final = observe(api)
        if record["status"] != "executed":
            # The released after_skill records the commitment even when a
            # begun grasp stops before close. Keep that annotation for re-close,
            # but never change it when this entire parent spent zero actions.
            _held.stage(api, {arm: committed_intent if spent > 0 else original_intent})
            return {"status": "stopped", "skill": "grasp_execute",
                    "candidate_id": commitment["candidate_id"], "stages": model_stages(stages),
                    "receipt": model_receipt(value["receipt"]), "remaining_budget": remaining,
                    "_runtime": {"skill_feedback": {
                        "committed_sequence": list(sequence), "arm": arm,
                    }}}
    raise ValueError("prepared grasp sequence lacks verification")
