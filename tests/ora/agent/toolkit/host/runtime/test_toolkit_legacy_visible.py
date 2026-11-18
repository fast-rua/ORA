"""Actual dispatch/result validation and fixed parent budgets on the legacy-visible surface (v5, tool_output).

The legacy-visible v3 and its observation delivery (released results byte-equal to the closed skills', the
accepted-zero-spend notice) were retired on 2026-10-31."""
import pytest

from ora.agent.error_classes import ToolError
from ora.agent.toolkit.tools.budgets import native_action_budget
from tests.support.agent.toolkit_legacy import host
from tests.support.agent.toolkit_recording import proposal


@pytest.mark.parametrize("name", ["move_to", "grasp_execute", "compute"])
def test_model_cannot_supply_any_budget(tmp_path, monkeypatch, name):
    runtime, client = host(tmp_path, monkeypatch)
    try:
        with pytest.raises(ToolError):
            runtime.invoke_with_context(name, {"native_action_budget": 0}, call_context=proposal(runtime, name, {}))
        assert runtime.toolkit_executor.budgets == []
        assert native_action_budget("grasp_execute") == 810
    finally:
        client.close()


def test_explicit_tool_output_passes_runtime_result_validation(tmp_path, monkeypatch):
    runtime, client = host(tmp_path, monkeypatch, mode="tool_output")
    try:
        args = {"ticks": 1}
        result = runtime.invoke_with_context("wait_ticks", args, call_context=proposal(runtime, "wait_ticks", args))
        assert result["status"] == "executed"
        assert result["remaining_budget"] == 17
        assert result["receipt"]["native_actions"] == 3
    finally:
        client.close()


def test_grasp_early_refusal_keeps_original_error_instead_of_requiring_confirmation():
    from ora.agent.toolkit.host.legacy_results import submitted
    result = {"error": {"code": "invalid_grasp_candidate"}}
    assert submitted("grasp_execute", result, {}) is result


def test_refused_zero_spend_has_no_accepted_motion_notice():
    from ora.agent.toolkit.host.legacy_results import submitted
    original = {"status": "refused", "reason": "budget"}
    evidence = {"native_actions_used": 0, "actions": [{"refusal": "budget", "receipt": {}}]}
    assert submitted("move_to", original, {}, execution=evidence) == original
