"""The tool-runtime half of a closed skill: commit, drive, verify, release.

Where this sits
---------------

``skill_ledger`` is the state; this module is the driver that pumps it, and
``orchestrator/action_executor.execute_skill_sequence`` is the loop that calls
the driver. The split is:

* the **tool runtime** (here) owns the geometry, because it holds the candidate
  artifacts and their observation binding, and it owns the ledger, because the
  ledger lives beside its other control state;
* the **trusted executor** owns the stage loop and the dispatch boundary; and
* the **coordinator** owns the turn, so it is the only thing that can execute
  several Supervisor actions without opening one.

It cannot be only the coordinator: the coordinator is Agent-neutral and has no
candidate geometry, and driving stages from there would write actions the
AgentSession never proposed -- ``record_environment_receipt`` refuses a receipt
whose ``action_id`` is not the session's ``last_action_id``, so every driven
stage would end the episode. It cannot be only the session loop either: the
agent core's turn loop returns as soon as a mutation submits, so a driver there would
own manipulation semantics inside the agent core (``ora-pi/``), which owns the provider
loop and not the tool contract, and it would not run for a managed runtime
sharing the same surface.

Why nothing resumes
-------------------

The stages between the decision and the receipt all run inside one coordinator
execution with no Agent turn between them, so no committed sequence crosses a
turn boundary and none is resumed: every submitted action carries
``stop_before_commit`` false and ``resume_from`` zero.

Not wedging
-----------

``build_skill_receipt`` releases the ledger on terminal executor paths,
including before a raised backend fault propagates. That covers everything the
executor can see; it does not cover a sequence that died with the executor --
a crashed worker, a killed run -- which would otherwise leave a ``running``
ledger no later call could clear. So the ledger is also reconciled by whatever
call comes next, in the holding gate's pending/reconcile shape: a ledger whose
committed observation is no longer the current one, with no action still
pending, has already been decided somewhere this process cannot see and is
released as abandoned. A skill that can wedge is worse than three explicit
calls, so neither half of that is optional.
"""

from __future__ import annotations


import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.contracts import TARGET_STATE_ACTION_SCHEMA, MOTION_ACTION_V3_SCHEMA

from . import (
    jaw_hold,
    ledger as skill_ledger,
    receipt as skill_receipt,
    lift_evidence as skill_retention,
)
from ..perception import localization, target_identity
from .candidates import _grasp_candidate

from ora.agent.error_classes import ToolError
from ora.agent.toolkit.tools.grasp_execute.tool import CLAIM_BOUNDARY


#: The ToolSurface capability a profile must declare to advertise these tools.
#: It names what the driver actually needs: one public end effector, the
#: selected motion envelope for each backend, and the two planners
#: whose candidates the skills consume.
CAPABILITY = "closed_manipulation_skills_v1"

LEDGER_FILENAME = "manipulation-skill-ledger.json"

#: The committed stage targets, written once at the moment the policy decided
#: and replayed for every stage afterwards. They are stored rather than
#: recomputed because the stages run while the action is still pending, when
#: `_current()` correctly refuses to serve a policy read, and because replaying
#: what was committed is what "the policy is asked once" means.
COMMITMENT_FILENAME = "manipulation-skill-commitment.json"

#: Jaw directives in the v4 envelope: 0 is closed, 1 is the open profile. These
#: are the same two values `_atomic_gripper` submits.
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0


# --------------------------------------------------------------------------
# Ledger storage and reconciliation
# --------------------------------------------------------------------------

def ledger_path(host: Any) -> Path:
    return Path(host.control_root) / LEDGER_FILENAME


def load_ledger(host: Any) -> dict[str, Any] | None:
    path = ledger_path(host)
    if not path.is_file():
        return None
    return skill_ledger.validate_state(
        json.loads(path.read_text(encoding="utf-8"))
    )


def store_ledger(host: Any, state: Mapping[str, Any] | None) -> None:
    path = ledger_path(host)
    if state is None:
        path.unlink(missing_ok=True)
        return
    _write_json(path, state)


def load_commitment(host: Any) -> dict[str, Any] | None:
    path = Path(host.control_root) / COMMITMENT_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_commitment(host: Any, value: Mapping[str, Any]) -> None:
    _write_json(Path(host.control_root) / COMMITMENT_FILENAME, value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reconcile(host: Any) -> dict[str, Any] | None:
    """Release a ledger whose sequence can no longer report.

    Called before committing a skill, which is where a stuck ledger
    would otherwise block the policy. A live sequence is one whose committed
    observation is still the current one, or one whose action is still pending
    execution. Anything else has already been decided somewhere this process
    cannot see, and holding it would block every later call with a decision the
    policy cannot take back.
    """

    state = load_ledger(host)
    if state is None or state["phase"] in {"completed", "aborted", "terminated"}:
        return state
    control_path = Path(host.control_root) / "current-observation.json"
    if not control_path.is_file():
        return state
    control = json.loads(control_path.read_text(encoding="utf-8"))
    pending = (Path(host.control_root) / "submitted-action.json").is_file()
    if pending or control.get("observation_id") == state["observation_id"]:
        return state
    released, _transition = skill_ledger.abandon(
        state, "sequence_did_not_report_before_the_next_observation",
    )
    store_ledger(host, released)
    return released


# --------------------------------------------------------------------------
# Committing one skill call
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Driving one stage
# --------------------------------------------------------------------------

def drive_skill_stage(
    host: Any, action: Mapping[str, Any], *, stage: str, stage_index: int,
    observation: Mapping[str, Any], observation_root: Path,
    completed: Sequence[Mapping[str, Any]],
    capture_observation: Callable[[], tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    """Produce one stage's action or measurement from the committed candidate."""

    del observation, observation_root
    arguments = action["arguments"]
    state = load_ledger(host)
    commitment = load_commitment(host)
    if (
        state is None or commitment is None
        or state["skill"] != arguments["skill"]
        or commitment.get("candidate_id") != arguments["candidate_id"]
    ):
        return {
            "kind": "refusal", "status": "refused",
            "reason": "committed_ledger_is_missing_or_belongs_to_another_skill",
        }
    if (arguments["sequence"] != state["sequence"]
            or not 0 <= stage_index < len(state["sequence"])
            or state["sequence"][stage_index] != stage):
        return {"kind": "refusal", "status": "refused", "reason": "committed_stage_sequence_mismatch"}
    try:
        _confirm_candidate_still_bound(host, state)
    except ToolError as refusal:
        return {"kind": "refusal", "status": "refused", "reason": refusal.code}
    if stage in skill_ledger.READ_ONLY_STAGES:
        fresh, _root = capture_observation()
        measured = _measure(host, stage, fresh, commitment, completed)
        if state["skill"] == "grasp_execute" and "lift" in state["sequence"]:
            probe = skill_retention.measure(
                host, commitment=commitment, observation=fresh, observation_root=_root,
            )
            measured["verification"]["lift_probe"] = probe
        return measured
    if stage == "lift":
        if not completed or completed[-1].get("stage") != "close" or completed[-1].get("status") != "executed":
            return {"kind": "refusal", "status": "refused", "reason": "lift_requires_executed_close"}
        fresh, root = capture_observation()
        commitment = {**commitment, "lift_baseline": skill_retention.capture_baseline(fresh, root)}
        store_commitment(host, commitment)
    pose = commitment["poses"].get(stage)
    gripper = commitment["grippers"].get(stage)
    if pose is None and gripper is None:
        return {
            "kind": "refusal", "status": "refused",
            "reason": f"stage_{stage}_has_no_committed_target",
        }
    sparse = (
        host.surface.supports("dual_arm_sparse_targets_v1")
        or host.surface.supports("motion_intent_router_v1")
    )
    if sparse:
        # Candidate coordinates are public TCP coordinates. They must enter the
        # same sparse adapter as manual motion, where TCP-to-planner conversion,
        # controller validation and measured jaw evidence are owned. Every
        # surface declaring the sparse topology also declares the router.
        if pose is not None and gripper is not None:
            raise ValueError("a sparse skill stage must separate pose and gripper")
        arm = commitment["arm"]
        interaction = "contact_approach" if stage in {"approach", "descend"} else (
            "precision" if stage in {"settle", "lift"} else "free_space"
        )
        return {
            "kind": "mutation",
            "action": {
                "schema": MOTION_ACTION_V3_SCHEMA,
                "observation_id": action["observation_id"],
                "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
                "action": "move_pose" if pose is not None else "set_gripper",
                "arguments": {
                    "targets": {arm: pose if pose is not None else (
                        "close" if gripper == GRIPPER_CLOSED else "open"
                    )},
                    **({"interactions": {arm: interaction}} if pose is not None else {}),
                },
                "bound_observation_sha256": action["bound_observation_sha256"],
            },
        }
    return {
        "kind": "mutation",
        "action": {
            "schema": TARGET_STATE_ACTION_SCHEMA,
            "observation_id": action["observation_id"],
            "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
            "action": "target_state",
            "arguments": {
                "arm": commitment["arm"],
                **({"pose": pose} if pose is not None else {}),
                **({"gripper": gripper} if gripper is not None else {}),
            },
            "bound_observation_sha256": action["bound_observation_sha256"],
        },
    }


def _confirm_candidate_still_bound(host: Any, state: Mapping[str, Any]) -> None:
    """Refuse by name if the committed candidate is no longer what it was.

    The candidate artifacts are immutable and content-addressed, so this is not
    a re-derivation: it is the check that the evidence the commitment rests on
    is still present and still names the observation it was bound to. A stage
    that cannot confirm that must not execute against what is left.
    """

    _grasp_candidate(host, str(state["candidate_id"]), str(state["observation_id"]))


def _measure(
    host: Any, stage: str, observation: Mapping[str, Any],
    commitment: Mapping[str, Any], completed: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Measure a read-only stage against the freshly captured observation."""

    hold_signal = getattr(getattr(host, "surface", None), "supports", lambda _name: False)(jaw_hold.CAPABILITY)
    del host
    effectors = observation.get("proprioception", {}).get("end_effectors", {})
    record = effectors.get(commitment["arm"]) if isinstance(effectors, Mapping) else None
    evidence = localization.closure_evidence(
        gripper_opening_m=localization.measured_jaw_separation_m(record),
        localization=commitment.get("localization"),
    )
    if hold_signal and evidence["strength"] == "unavailable":
        evidence["jaw_hold"] = jaw_hold.verify_stage_evidence(completed, commitment["arm"], record)
    semantics = commitment.get("semantics")
    semantics = (
        semantics if isinstance(semantics, Mapping)
        else target_identity.unnamed_target()
    )
    return {
        "kind": "measurement", "status": "executed",
        "reason": f"verified_against_{observation.get('observation_id')}",
        "measurements": [
            {"quantity": "closure_strength", "value": evidence["strength"]},
            *([{"quantity": "jaw_hold", "value": evidence["jaw_hold"]["held"]}] if "jaw_hold" in evidence else []),
            {"quantity": "gripper_opening_m", "value": evidence["gripper_opening_m"]},
            {
                "quantity": "target_largest_extent_m",
                "value": evidence["target_largest_extent_m"],
            },
            # What the policy named, beside what the jaws measured. Neither is
            # a verdict on the other, and the pair is the whole evidence for a
            # question the geometry cannot reach.
            {"quantity": "committed_target_phrase", "value": semantics["phrase"]},
            {"quantity": "committed_target_score", "value": semantics["score"]},
            {
                "quantity": "committed_target_observation_id",
                "value": semantics["observation_id"],
            },
            {"quantity": "committed_target_selection", "value": semantics["selection"]},
        ],
        "verification": evidence,
    }


# --------------------------------------------------------------------------
# Reporting and releasing
# --------------------------------------------------------------------------

def build_skill_receipt(
    host: Any, *, action: Mapping[str, Any], status: str,
    stages: Sequence[Mapping[str, Any]], observation: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    """Advance the ledger to its terminal phase and assemble one receipt.

    Called on every exit path the executor has, which is what makes the ledger
    safe to hold across an action at all: whatever happened, the sequence
    reports and the ledger stops being a decision the policy cannot take back.
    """

    state = load_ledger(host)
    verification: Mapping[str, Any] | None = None
    if state is not None:
        for item in stages:
            if skill_ledger.pending_stage(state) != item["stage"]:
                continue
            state, _transition = skill_ledger.record(
                state, stage=str(item["stage"]), status=str(item["status"]),
                reason=str(item["reason"]), action_id=item.get("action_id"),
                measurements=item.get("measurements", ()),
            )
        if status == "terminated":
            # A pre-stage terminal check records a skipped stage. The skill
            # still ends by lifecycle, independent of that stage's outcome.
            state["phase"] = "terminated"
        if state["phase"] not in {"completed", "aborted", "terminated"}:
            state, _transition = skill_ledger.abandon(state, stop_reason)
        store_ledger(host, state)
    for item in stages:
        if isinstance(item.get("verification"), Mapping):
            verification = item["verification"]
    receipt = skill_receipt.assemble(
        skill=str(action["arguments"]["skill"]),
        status=status,
        rationale=str(action["arguments"]["rationale"]),
        candidate_id=str(action["arguments"]["candidate_id"]),
        observation_id=str(action["observation_id"]),
        observation_sha256=str(action["bound_observation_sha256"]),
        stages=[
            skill_receipt.stage(
                str(item["stage"]), str(item["status"]), str(item["reason"]),
                action_id=item.get("action_id"),
                measurements=item.get("measurements", ()),
            )
            for item in stages
        ],
        verification=verification,
        # Current images use the ordinary observation path. Prepared grasps
        # separately bind a historical post-close baseline below.
        images=(),
        claim_boundary=CLAIM_BOUNDARY,
    )
    receipt["stop_reason"] = stop_reason
    receipt["fresh_observation_id"] = str(observation.get("observation_id") or "")
    receipt["cost"] = {
        "robot_decisions": 1,
        "dispatched_stage_actions": sum(
            1 for item in stages if item.get("action_id") is not None
        ),
        "semantics": (
            "the Harness robot-decision budget counts this whole skill as one "
            "decision; the simulator advanced once per dispatched stage action"
        ),
    }
    receipt["ledger_released"] = (
        state is None or state["phase"] in {"completed", "aborted", "terminated"}
    )
    commitment = load_commitment(host)
    if isinstance(commitment, Mapping) and "precommit_motion_check" in commitment:
        receipt["precommit_motion_check"] = commitment["precommit_motion_check"]
    baseline = commitment.get("lift_baseline") if isinstance(commitment, Mapping) else None
    if isinstance(baseline, Mapping) and action["arguments"]["skill"] == "grasp_execute":
        receipt["observation_evidence"] = [{
            "arm": commitment["arm"],
            **{key: baseline[key] for key in (
                "role", "observation_root", "observation_id", "observation_sha256",
            )},
        }]
    # `assemble` measured the receipt it returned; the four fields above made it
    # a different object. Re-measuring is the whole point of that field -- a
    # recorded size that describes an earlier version of the body is exactly the
    # size claim about a different object the receipt module warns against.
    for _ in range(4):
        measured = skill_receipt.body_bytes(receipt)
        if receipt["delivery"]["body_bytes"] == measured:
            break
        receipt["delivery"]["body_bytes"] = max(
            measured, receipt["delivery"]["body_bytes"],
        )
    return receipt
