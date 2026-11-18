"""Trusted control-file boundary for portable grasp tools, never a simulator executor."""

import json
from pathlib import Path
from ora.contracts import validate_runtime_action
from ora.agent.toolkit.interface.types import (
    AccessDenied,
    PublicData,
    PublicToolFailure,
)
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.skills import driver, ledger
from ora.agent.toolkit.host.skills.lift_evidence import CAPABILITY, LIFT_M
from ora.agent.toolkit.host.reach.probe_feedback import (
    probe_request_fields,
    IK_CANDIDATE_FEEDBACK_CAPABILITY,
)
from ora.agent.toolkit.host.motion_rules.clearance_runtime import note_grasp_execute


def configuration(host):
    """Freeze public preparation state; this function performs no motion or reconciliation."""
    current, _, _ = host._current()
    failure = None
    try:
        host._ensure_mutation_available("grasp_execute")
    except ToolError as error:
        failure = error.as_dict()
    return dict(
        generation=current["generation"],
        ledger=driver.load_ledger(host),
        action_pending=(host.control_root / "submitted-action.json").is_file(),
        mutation_refusal=failure,
        prepared_grasp=host.surface.supports(CAPABILITY),
        lift_m=LIFT_M,
        precommit=host.surface.supports("grasp_precommit_motion_check_v1"),
        depth_inliers=host.surface.supports("public_target_geometry_depth_inliers_v2"),
        workspace_bounds_m=dict(host.workspace_bounds_m),
        probe_request_fields=probe_request_fields(host.surface),
        ik_candidate_feedback=host.surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY),
    )


def submit_adapter(host):
    """Validate the original action plus prepared public files before publishing control."""

    def submit(parameters, inputs, store):
        try:
            host._ensure_mutation_available("grasp_execute")
            current, _, _ = host._current()
            action = validate_runtime_action(parameters["action"])
            if (
                action["action"] != "execute_skill_sequence"
                or action["arguments"]["skill"] != "grasp_execute"
                or action["observation_id"] != current["observation_id"]
                or action["bound_observation_sha256"] != current["observation_sha256"]
                or set(inputs) != {"skill_ledger", "skill_commitment"}
            ):
                raise AccessDenied("prepared action boundary mismatch")
            state = ledger.validate_state(
                json.loads(store.read(inputs["skill_ledger"]))
            )
            commitment = json.loads(store.read(inputs["skill_commitment"]))
            args = action["arguments"]
            if (
                any(
                    state[k] != args[k]
                    for k in (
                        "skill",
                        "candidate_id",
                        "rationale",
                        "stop_before_commit",
                        "sequence",
                    )
                )
                or state["phase"] != "running"
                or state["completed"]
                or args["resume_from"] != 0
                or any(
                    state[k] != current[k]
                    for k in ("observation_id", "observation_sha256")
                )
                or any(
                    commitment[k] != state[k]
                    for k in ("skill", "candidate_id", "observation_id")
                )
            ):
                raise AccessDenied("prepared skill files differ from action")
            # Same control filenames/JSON encoding the protected executor consumes.
            # The action is the final publication; partial preparation never starts motion.
            driver.store_ledger(host, state)
            driver.store_commitment(host, commitment)
            result = host._persist_action(
                host.control_root / "submitted-action.json", dict(current), action
            )
            note_grasp_execute(host)
            return PublicData.from_json(
                dict(accepted=True, action_id=action["action_id"])
            )
        except ToolError as error:
            raise PublicToolFailure(error.as_dict()) from error

    return submit


def reconcile_before_grasp(host, args):
    """Preserve control-ledger housekeeping after the original argument gates.

    No action or observation changes here. The frozen input table contains the
    resulting public ledger, so replay has no filesystem reconciliation hook.
    """
    from ora.agent.toolkit.host.skills import receipt

    host._ensure_mutation_available("grasp_execute")
    if set(args) != {"candidate_id", "rationale"}:
        raise ToolError(
            "grasp_execute requires candidate_id and rationale",
            code="invalid_arguments",
        )
    receipt.validate_rationale(args["rationale"])
    driver.reconcile(host)
