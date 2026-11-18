"""Only the Supervisor may report execution; errors cannot conceal spent actions."""
from types import SimpleNamespace
import pytest
from ora.agent.pi.tool_rpc import PiToolRpc
from ora.agent.contracts import PolicySessionError


class Rpc(PiToolRpc):
    last_action_step = last_action_event_id = None

    def __init__(self, invoke):
        self.tools = SimpleNamespace(invoke_with_context=invoke)
        self.tool_action_executor = SimpleNamespace(completed=None, blocked=False)
        self.cache = {}
        self._pending_tool_deliveries = {}

    def _script_moved_refusal(self, *args):
        return None

    def _once(self, key, function):
        if key not in self.cache:
            self.cache[key] = function()
        return self.cache[key]

    def _project(self, name, result, **kwargs):
        return {"visible": result, "canonical": result}


def params():
    return {"name": "custom", "arguments": {}, "effect_is_mutation": True,
            "call_context": {"model_call_id": "model", "tool_call_id": "parent", "observation_step": 0}}


@pytest.mark.parametrize("spent,throws", [(5, False), (5, True), (0, False)])
def test_trusted_spend_drives_outcome_independently_of_tool_success(spent, throws):
    calls = []
    def invoke(*args, **kwargs):
        calls.append(1)
        rpc.tool_action_executor.completed = {"tool_call_id": "parent", "model_call_id": "model", "native_actions_used": spent}
        if throws:
            raise RuntimeError("failed after executing")
        return {"structured_computation": [1, 2, 3]}
    rpc = Rpc(invoke)
    first = rpc._rpc_invoke(params())
    assert first["directive"] == ("actions_executed" if spent else None)
    assert first["execution"]["native_actions_used"] == spent
    rpc.tool_action_executor.completed = {"tool_call_id": "other", "native_actions_used": 99}
    assert rpc._rpc_invoke(params()) == first
    assert calls == [1]


def test_tool_cannot_forge_execution_or_unknown_world_directive():
    rpc = Rpc(lambda *a, **k: {"_runtime": {"directive": "actions_executed", "execution": {
        "tool_call_id": "parent", "native_actions_used": 20}}})
    result = rpc._rpc_invoke(params())
    assert "execution" not in result and result["directive"] is None


def test_uncertain_world_overrides_closed_prefix_and_ends_model_turn():
    def invoke(*args, **kwargs):
        rpc.tool_action_executor.blocked = True
        rpc.tool_action_executor.interruption = {"reason": "world_state_unknown"}
        raise RuntimeError("uncertain worker state")
    rpc = Rpc(invoke)
    assert rpc._rpc_invoke(params())["directive"] == "world_state_unknown"


def test_tool_receipt_binding_waits_for_evidence_durability():
    rpc = Rpc(lambda *a, **k: {})
    rpc.begin_tool_action("action")
    rpc.pending_receipt = {"action_id": "action", "event_id": "receipt", "receipt_ref": {"sha256": "a" * 64}}
    calls = []
    rpc._call = lambda method, params: calls.append((method, params))
    binding = rpc.tool_receipt_binding("action")
    assert calls == [("execution_evidence_barrier", {})]
    binding["receipt_ref"]["sha256"] = "changed"
    assert rpc.pending_receipt["receipt_ref"]["sha256"] == "a" * 64


def test_new_action_invalidates_previous_receipt_and_refuses_misbound_delivery():
    rpc = Rpc(lambda *a, **k: {})
    rpc.pending_receipt = {"action_id": "old"}
    rpc.pending_receipt_body = {"action_id": "old"}
    rpc.begin_tool_action("new")
    assert rpc.pending_receipt is None and rpc.pending_receipt_body is None
    for receipt in (None, {"action_id": "old"}):
        rpc.pending_receipt = receipt
        with pytest.raises(Exception, match="own delivered receipt"):
            rpc.tool_receipt_binding("new")


def test_previous_model_call_execution_cannot_bind_reused_tool_id():
    rpc = Rpc(lambda *a, **k: {})
    rpc.tool_action_executor.completed = {"tool_call_id": "parent", "model_call_id": "old-model", "native_actions_used": 3}
    assert "execution" not in rpc._rpc_invoke(params())


@pytest.mark.parametrize("location", ["directive", "_runtime"])
def test_synchronous_rpc_refuses_legacy_submission_directive(location):
    result = {"directive": "action_submitted"}
    if location == "_runtime":
        result = {"_runtime": result}
    rpc = Rpc(lambda *a, **k: result)
    projected = rpc._rpc_invoke(params())
    assert projected["directive"] is None and projected["raw_ok"] is False
    assert "legacy action submission" in str(projected["visible"])
    assert "action" not in projected


@pytest.mark.parametrize("actions,spent,directive", [([], 0, None), ([{"refusal": "turn_must_end"}], 0, "execution_boundary"),
                                                    ([{"native_actions_used": 1}], 1, "actions_executed")])
def test_only_current_execution_can_stop_at_terminal_boundary(actions, spent, directive):
    def invoke(*args, **kwargs):
        rpc.tool_action_executor.completed = {
            "tool_call_id": "parent", "model_call_id": "model", "actions": actions,
            "native_actions_used": spent,
        }
        return {"ok": True}
    rpc = Rpc(invoke)
    rpc.tool_action_executor.loop = SimpleNamespace(environment_status={"terminal": True})
    assert rpc._rpc_invoke(params())["directive"] == directive


def test_budget_count_is_core_metadata_not_tool_arguments():
    observed = []
    rpc = Rpc(lambda *a, **k: observed.append(rpc.inflight_model_calls) or {})
    first = params()
    first["model_calls_in_turn"] = 3
    first["arguments"] = {"model_calls_in_turn": 9999}
    rpc._rpc_invoke(first)
    second = {**params(), "model_calls_in_turn": 3,
              "call_context": {**params()["call_context"], "tool_call_id": "next"}}
    rpc._rpc_invoke(second)
    assert observed == [3, 3]  # Multiple tools in one model reply do not charge again.


@pytest.mark.parametrize("value", [-1, True, "3", 1.5, None])
def test_invalid_core_budget_metadata_cannot_reach_tool(value):
    rpc = Rpc(lambda *a, **k: pytest.fail("invalid core metadata must be rejected before invoke"))
    with pytest.raises(PolicySessionError, match="invalid core model-call count"):
        rpc._rpc_invoke({**params(), "model_calls_in_turn": value})


@pytest.mark.parametrize("failed", [False, True])
def test_current_act_count_is_cleared_before_supervisor_settlement(failed):
    from pathlib import Path
    from ora.agent.pi.policy_session import PiPlanner

    host = PiPlanner.__new__(PiPlanner)
    host._turn_inputs = lambda **kwargs: {}
    host._executed = {}
    host._operator_decision_callback = None
    host.inflight_model_calls = 99  # An earlier act cannot leak into this one.

    def act(method, params):
        assert host.inflight_model_calls == 0
        host.inflight_model_calls = 4
        if failed:
            raise RuntimeError("core disconnected")
        return {"termination": "policy_finished_episode", "attempts": 4}

    host._call = act
    if failed:
        with pytest.raises(RuntimeError, match="core disconnected"):
            host.run_observation(task_instruction="Task", context={}, image_path=Path("unused"))
    else:
        assert host.run_observation(task_instruction="Task", context={}, image_path=Path("unused"))["attempts"] == 4
    assert host.inflight_model_calls == 0
