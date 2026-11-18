"""Apply receipt accounting for proposal and synchronous parent execution paths."""
import time
from ora.agent import PolicyRuntimeEvent
from ..evaluation.decision_accounting import is_zero_write_refusal
from .records import _append, _status


def record_receipt(loop, action, receipt, execute_duration_s, *, inline_consumed=None, inline_used=None):
    # A certified zero-write refusal spends no robot decision.
    consumed_decision = (not is_zero_write_refusal(receipt)
                         if inline_consumed is None else inline_consumed)
    refused = not consumed_decision if inline_used is None else inline_used == 0
    loop.decisions += consumed_decision
    loop.round_decisions += consumed_decision
    loop.refused_proposals += refused
    loop.round_refused_proposals += refused
    loop.native_allowance.record(receipt, action)
    owner = getattr(loop, "execution_state_owner", None)
    if owner is not None and (inline_consumed is None or inline_consumed):
        owner.finish_tool_call(receipt)
    loop.previous_public_action = action
    loop.pending_execution = {"action": action, "receipt": receipt}
    _append(loop.events, {
        "type": "motion_receipt", "record": receipt, "time": time.time(),
        "duration_s": execute_duration_s, "duration_clock": "monotonic",
        "consumed_decision": consumed_decision,
    })
    receipt_id = receipt.get("receipt_id") or receipt.get("id")
    loop.interaction.record_execution(
        receipt_id=str(receipt_id) if receipt_id is not None else None,
        consumed_decision=consumed_decision,
        **({"world_changed": inline_used > 0, "refused": refused}
           if inline_used is not None else {}),
    )
    loop.deliver_policy_event(
        PolicyRuntimeEvent(
            kind="environment_receipt", payload=receipt,
            observation_id=loop.interaction.observation_id,
            observation_step=loop.interaction.observation_step,
        ),
        stage="environment_receipt",
    )
    _status(loop.status_path, run_id=loop.run_id, state="running", decisions=loop.decisions, refused_proposals=loop.refused_proposals, model_calls=loop.model_calls, official_success=None)
