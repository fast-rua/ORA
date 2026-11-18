"""Released confirmations, with immediate zero-spend feedback (MV-04).

The ledger controls turns. Visible submitted confirmations never dispatch again.
"""

NO_MOTION_MESSAGE = (
    "This action used 0 native steps; no motion was executed. "
    "Continue in this turn using the receipt below; no new observation turn will follow this result."
)


def submitted(name: str, result: dict, metadata: dict, *, execution=None) -> dict:
    """Preserve confirmations except accepted zero-spend, which needs immediate feedback."""
    if name == "grasp_execute":
        visible = dict(metadata["legacy_submission"]) if "legacy_submission" in metadata else result
    elif result.get("status") == "executed":
        visible = {"status": "submitted", "action": result["action"],
                   **{key: result[key] for key in ("compiled_pose", "intent") if key in result}}
    else:
        return result
    if (execution and execution["native_actions_used"] == 0
            and any(item["refusal"] is None for item in execution["actions"])):
        from ora.agent.toolkit.tools._receipts import model_receipt
        visible = {**visible, "no_motion": {
            "native_actions": 0, "message": NO_MOTION_MESSAGE,
            "receipts": [model_receipt(item["receipt"]) for item in execution["actions"]],
        }}
    return visible
