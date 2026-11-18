"""Synchronous-parent delivery loss terminates; ended historical calls do not.

The actual Pi subprocess is killed at its built-in crash points. A CPU-only
Synthetic runtime supplies read-only data; the production owner owns the ledger.
"""
import json

import pytest

from tests.support.agent_core import episodes

pytestmark = pytest.mark.usefixtures("pi_core")


@pytest.mark.parametrize("crash", ["in_invoke:1", "after_invoke:1", "after_tool_result:1", "after_finish:1"])
def test_current_read_only_parent_process_loss_is_terminal(tmp_path, monkeypatch, crash):
    _bind_owner(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", crash)
    responses = [episodes.tool_reply(("read", "observe", {"views": ["agentview"]}))]
    if crash == "after_finish:1":
        # Terminal lifecycle also has no physical action; delivery still matters.
        responses = [episodes.tool_reply(("read", "finish_episode", {
            "outcome": "failure", "reason": "done", "attempts": ["read"]}))]
    run = episodes.run_core("pi", tmp_path, responses=responses, actions=2)
    result = run["result"]
    assert result["termination"] == "tool_execution_interrupted"
    assert result["behavioral_sample_valid"] is False
    evidence = result["tool_execution_interruption"]
    assert evidence["tool_call_id"] == "read"
    assert evidence["model_call_id"] == "model-call-000000"
    assert evidence["last_action_id"] is None
    assert json.loads((run["root"] / "status.json").read_text())["tool_execution_interruption"] == evidence
    assert not [event for event in run["events"] if event["type"] == "agent_core_resumed"]


def test_loss_in_second_read_only_call_names_current_parent(tmp_path, monkeypatch):
    _bind_owner(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", "after_invoke:2")
    responses = [episodes.tool_reply(
        ("first", "observe", {"views": ["agentview"]}),
        ("current", "observe", {"views": ["wrist"]}),
    )]
    run = episodes.run_core("pi", tmp_path, responses=responses, actions=2)
    evidence = run["result"]["tool_execution_interruption"]
    assert evidence["reason"] == "tool_execution_interrupted"
    assert evidence["tool_call_id"] == "current"
    assert evidence["last_action_id"] is None


def test_completed_read_only_reply_allows_later_provider_restart(tmp_path, monkeypatch):
    _bind_owner(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", "after_request:2")
    responses = [episodes.tool_reply(("read", "observe", {"views": ["agentview"]})),
                 episodes.tool_reply(("finish", "finish_episode", {
                     "outcome": "failure", "reason": "done", "attempts": ["read"]}))]
    run = episodes.run_core("pi", tmp_path, responses=responses, actions=2)
    assert run["result"]["termination"] == "policy_finished_episode"
    assert "tool_execution_interruption" not in run["result"]
    assert [event for event in run["events"] if event["type"] == "agent_core_resumed"]
    assert len([event for event in run["events"] if event["type"] == "tool_result"
                and event.get("tool_call_id") == "read"]) == 1


def _bind_owner(monkeypatch):
    from ora.orchestrator.episode import loop as loop_module
    from ora.orchestrator.episode.tool_execution import ToolActionExecutor

    def bind(loop):
        executor = ToolActionExecutor(loop)
        loop.toolkit_executor = executor
        loop.planner.bind_tool_action_executor(executor, observation_receipts=False)
        tools = loop.tools
        original = tools.invoke_with_context

        def invoke(name, arguments, *, call_context):
            from unittest.mock import patch
            with patch.object(loop.native_allowance, "remaining", lambda: 100), executor.invocation(
                    name, dict(call_context), 0):
                return original(name, arguments, call_context=call_context)

        monkeypatch.setattr(tools, "invoke_with_context", invoke)

    monkeypatch.setattr(loop_module, "bind_script_actions", bind)
