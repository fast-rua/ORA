"""Derived grasp observation feedback, never an environment execution receipt.

Stage/verify interpretations belong to the tool. Action identities, counts and
publication roots belong to the execution owner. This join only feeds the old
skill projector; it cannot authorize actions or change the execution ledger.
"""

from copy import deepcopy

from ora.agent.toolkit.tools._receipts import model_receipt


def assemble(result, feedback, execution, publications, *, terminal=False):
    """Bind semantic stage feedback to this parent's immutable action evidence."""
    sequence = feedback["committed_sequence"]
    if sequence not in (["open", "pregrasp", "approach", "close", "lift", "verify"],
                        ["approach", "close", "verify"]):
        raise ValueError("unsupported grasp feedback sequence")
    actions, stages = execution["actions"], result["stages"]
    arm = feedback["arm"]
    if (arm not in {"left", "right"} or len(stages) != len(actions)
            or [item["stage"] for item in stages] != sequence[:len(actions)]):
        raise ValueError("grasp feedback differs from executed stage prefix")
    projected = []
    for stage, action in zip(stages, actions, strict=True):
        name = stage["stage"]
        primitive = action["action"]
        expected = "set_gripper" if name in {"open", "close"} else "move_pose"
        if (stage.get("action_id") != action["action_id"]
                or primitive["action"] != expected
                or set(primitive["arguments"].get("targets", {})) != {arm}
                or stage.get("receipt") != model_receipt(action["receipt"])):
            raise ValueError("grasp feedback differs from execution evidence")
        projected.append({key: deepcopy(stage[key]) for key in (
            "stage", "status", "reason", "action_id", "measurements")})
    verify = feedback.get("verify")
    if verify is not None:
        if len(actions) != len(sequence) - 1 or result.get("status") != "executed":
            raise ValueError("verification precedes completion of grasp stages")
        projected.append({"stage": "verify", **{key: deepcopy(verify[key])
                          for key in ("status", "reason", "measurements", "verification")}})
    status = "completed" if verify is not None else "aborted"
    stop_reason = "sequence_completed" if verify is not None else f"{stages[-1]['stage']}_not_reached"
    if terminal:
        status, stop_reason = "terminated", "environment_terminal"
    evidence = []
    names = [item["stage"] for item in stages]
    if "lift" in names:
        close = actions[names.index("close")]
        lift = actions[names.index("lift")]
        if close["after_observation"] != lift["before_observation"]:
            raise ValueError("grasp baseline is not the lift input publication")
        pointer = publications[close["action_id"]]
        if pointer["observation_id"] != close["after_observation"]["observation_id"]:
            raise ValueError("grasp baseline identity differs from publication")
        evidence.append({"role": "post_close_before_lift", "arm": arm,
                         "observation_root": pointer["source_root"],
                         "observation_id": pointer["observation_id"],
                         "observation_sha256": pointer["observation_sha256"]})
    return {
        # Only a compatibility input to the existing public projector, not a
        # record_environment_receipt call or a replacement for a worker receipt.
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": actions[-1]["action_id"], "skill": "grasp_execute",
        "candidate_id": result["candidate_id"], "committed_sequence": list(sequence),
        "resumed_from": 0, "stage_receipts": projected,
        "status": status, "stop_reason": stop_reason,
        "terminal_after_action": terminal,
        "skill_receipt": {"observation_evidence": evidence},
    }
