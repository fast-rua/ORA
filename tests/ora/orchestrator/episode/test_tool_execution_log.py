"""Recovery returns completed evidence, never permission to retry a tool body."""

import pytest
from ora.orchestrator.episode.tool_execution_log import ToolExecutionLog, WorldStateUnknown


def observation(step):
    return {"observation_id": f"obs-{step}", "step": step}


def complete(log, index, used=2):
    log.begin(action={"action_id": f"action-{index}"}, admitted=3, observation=observation(index))
    log.complete(used=used, observation=observation(index + 1), receipt={"native_actions": used, "action_id": f"action-{index}"},
                 receipt_event_id=f"receipt-{index}", receipt_ref={"sha256": str(index) * 64})


def test_all_completed_receipts_survive_restart_before_and_after_parent_result(tmp_path):
    log = ToolExecutionLog(tmp_path, "tool-call", "model")
    log.create(requested=20, admitted=6, observation=observation(0))
    complete(log, 0)
    complete(log, 1)
    resumed = ToolExecutionLog(tmp_path, "tool-call", "model")
    before_result = resumed.recover()
    assert before_result["native_actions_used"] == 4
    assert before_result["native_actions_remaining"] == 2
    assert before_result["final_observation"] == observation(2)
    assert [r["receipt_event_id"] for r in before_result["actions"]] == ["receipt-0", "receipt-1"]
    with pytest.raises(RuntimeError, match="already has execution"):
        resumed.create(requested=20, admitted=6, observation=observation(0))
    resumed.finish()
    assert ToolExecutionLog(tmp_path, "tool-call", "model").recover() == before_result


def test_unclosed_intent_refuses_recovery_and_preserves_completed_receipts(tmp_path):
    log = ToolExecutionLog(tmp_path, "tool-call", "model")
    log.create(requested=20, admitted=6, observation=observation(0))
    complete(log, 0)
    log.begin(action={"action_id": "action-1"}, admitted=3, observation=observation(1))
    resumed = ToolExecutionLog(tmp_path, "tool-call", "model")
    with pytest.raises(WorldStateUnknown):
        resumed.recover()
    assert resumed.value["actions"][0]["receipt_event_id"] == "receipt-0"
    with pytest.raises(WorldStateUnknown):
        resumed.finish()


def test_zero_spend_refusal_keeps_actual_publication_identity(tmp_path):
    log = ToolExecutionLog(tmp_path, "tool-call", "model")
    log.create(requested=20, admitted=0, observation=observation(4))
    log.begin(action={"action_id": "refused"}, admitted=0, observation=observation(4))
    log.complete(used=0, observation=observation(4), receipt={"status": "refused", "action_id": "refused"},
                 receipt_event_id="receipt-zero", receipt_ref={"sha256": "a" * 64},
                 refusal="native_action_budget_exhausted")
    result = log.finish()
    assert result["initial_observation"] == result["final_observation"]
    assert result["native_actions_used"] == 0
    assert len(result["actions"]) == 1


@pytest.mark.parametrize("damage", ["budget", "chain", "partial"])
def test_corrupt_execution_evidence_never_grants_more_physics(tmp_path, damage):
    import json
    log = ToolExecutionLog(tmp_path, "tool-call", "model")
    log.create(requested=20, admitted=6, observation=observation(0))
    complete(log, 0)
    value = json.loads(log.path.read_text())
    if damage == "budget":
        value["native_actions_used"] = 1
    if damage == "chain":
        value["actions"][0]["before_observation"] = observation(9)
    log.path.write_text("{" if damage == "partial" else json.dumps(value))
    with pytest.raises(WorldStateUnknown):
        ToolExecutionLog(tmp_path, "tool-call", "model").recover()


def refuse(log, action_id, step):
    log.begin(action={"action_id": action_id}, admitted=3, observation=observation(step))
    log.complete(used=0, observation=observation(step), receipt={"status": "refused", "action_id": action_id},
                 receipt_event_id=f"receipt-refused-{step}", receipt_ref={"sha256": "a" * 64},
                 refusal="stale_or_mismatched_recovery_target")


def test_a_refused_identity_is_carried_by_the_next_action_on_the_same_observation(tmp_path):
    # A code-mode script's parent: the refused action published nothing, so the script's next action is
    # bound to the same observation and carries the same identity; an executed identity never repeats.
    log = ToolExecutionLog(tmp_path, "script-call", "model")
    log.create(requested=20, admitted=6, observation=observation(0))
    refuse(log, "action-0", 0)
    complete(log, 0)
    with pytest.raises(ValueError, match="already present"):
        log.begin(action={"action_id": "action-0"}, admitted=3, observation=observation(1))
    result = log.finish()
    assert [(item["action_id"], item["refusal"]) for item in result["actions"]] == [
        ("action-0", "stale_or_mismatched_recovery_target"), ("action-0", None)]
    assert result["native_actions_used"] == 2
    assert ToolExecutionLog(tmp_path, "script-call", "model").recover() == result


@pytest.mark.parametrize("damage", ["executed_twice", "refusal_spent"])
def test_a_repeated_identity_is_evidence_only_after_refusals_that_spent_nothing(tmp_path, damage):
    import json
    log = ToolExecutionLog(tmp_path, "script-call", "model")
    log.create(requested=20, admitted=6, observation=observation(0))
    refuse(log, "action-0", 0)
    complete(log, 0)
    value = json.loads(log.path.read_text())
    first = value["actions"][0]
    if damage == "executed_twice":
        first["refusal"] = None
    else:  # balanced, but a refusal that consumed physics is no refusal
        first["native_actions_used"] = 1
        value["native_actions_used"] += 1
        value["native_actions_remaining"] -= 1
    log.path.write_text(json.dumps(value))
    with pytest.raises(WorldStateUnknown):
        ToolExecutionLog(tmp_path, "script-call", "model").recover()
