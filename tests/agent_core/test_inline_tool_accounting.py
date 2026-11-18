"""Real Pi RPCs, the synchronous owner and published snapshots on a CPU backend."""

import json

import pytest

from ora.agent.evidence.policy_state import verify_policy_state
from ora.orchestrator.episode import loop as loop_module
from ora.orchestrator.episode import observation as observation_module
from ora.orchestrator.episode.tool_execution import ToolActionExecutor
from ora.orchestrator.episode.execution_state import ExecutionStateOwner
from tests.support.agent_core import episodes
from tests.support.orchestrator import multi_round

pytestmark = pytest.mark.usefixtures("pi_core")


class CountedSession(multi_round.SuccessSyntheticSession):
    def execute(self, payload):
        result = super().execute(payload)
        return {**result, "receipt": {**result["receipt"], "native_actions": 1}}


def run_inline(tmp_path, monkeypatch, responses, *, terminal_after, compound=False):
    owners = []

    def bind(loop):
        loop.execution_state_owner = ExecutionStateOwner()
        executor = ToolActionExecutor(loop)
        owners.append(executor)
        loop.toolkit_executor = executor
        loop.planner.bind_tool_action_executor(executor, observation_receipts=False)
        original = loop.tools.invoke_with_context

        def invoke(name, arguments, *, call_context):
            mutation = name in {"close_gripper", "open_gripper"}
            with executor.invocation(name, dict(call_context), 20 if mutation else 0):
                if not mutation:
                    return original(name, arguments, call_context=call_context)
                # A deterministic test tool; all accounting, execution, publication,
                # receipt delivery and Pi turn management use production code.
                for index in range(2 if compound and name == "close_gripper" else 1):
                    pointer = json.loads((loop.control / "current-observation.json").read_text())
                    action = {
                        "schema": "ora-action.v3", "action": "set_gripper",
                        "arguments": {"arm": "right", "gripper": float(name == "open_gripper")},
                        "action_id": f"{call_context['tool_call_id']}-{index}",
                        "observation_id": pointer["observation_id"],
                        "bound_observation_sha256": pointer["observation_sha256"],
                    }
                    result = executor.execute_action(action, 20)
                return {"ok": True, **result}

        monkeypatch.setattr(loop.tools, "invoke_with_context", invoke)

    monkeypatch.setattr(loop_module, "bind_script_actions", bind)
    # The typed RoboDojo state store does not define Synthetic camera names.
    # Keep the real policy-state publisher and hash chain, skip only that export.
    monkeypatch.setattr(observation_module, "publish_state", lambda *args: None)
    plan = episodes.synthetic_plan(actions=10)
    plan["runtime"]["native_action_limit"] = 100
    session = CountedSession(tmp_path / "native", plan["task"], 0, max_actions=terminal_after,
                             main_camera="agentview", camera_names=("agentview", "wrist"))
    result = multi_round.run(tmp_path, plan, responses,
                            environment_session_factory=lambda **kwargs: session)
    return result, owners[0], session


def test_terminal_check_then_finish_reaches_provider_and_preserves_success(tmp_path, monkeypatch):
    responses = [episodes.tool_reply(("act", "close_gripper", {})),
                 episodes.tool_reply(("check", "check_task", {})),
                 episodes.tool_reply(("finish", "finish_episode", {"outcome": "success"}))]
    result, owner, session = run_inline(tmp_path, monkeypatch, responses, terminal_after=1)
    assert result["termination"] == "trusted_runtime_boundary", result
    assert result["official_success"] is True
    assert len(multi_round.provider_requests(tmp_path)) == 3
    events = multi_round.agent_events(tmp_path)
    assert [event["tool_call_id"] for event in events if event["type"] == "tool_result"] == [
        "act", "check", "finish"]
    assert session.action_sequence == 1
    assert owner.loop.round_model_calls == 3


def test_terminal_physical_request_is_refused_without_another_worker_action(tmp_path, monkeypatch):
    responses = [episodes.tool_reply(("act", "close_gripper", {})),
                 episodes.tool_reply(("refuse", "open_gripper", {})),
                 episodes.tool_reply(("finish", "finish_episode", {"outcome": "success"}))]
    result, owner, session = run_inline(tmp_path, monkeypatch, responses, terminal_after=1)
    assert session.action_sequence == 1
    assert len(multi_round.provider_requests(tmp_path)) == 3
    events = multi_round.agent_events(tmp_path)
    refusal = next(event for event in events if event["type"] == "tool_result"
                   and event["tool_call_id"] == "refuse")
    body = json.loads((tmp_path / refusal["artifact_ref"]["path"]).read_text())
    assert body["ok"] is False
    assert "terminal" in json.dumps(body)
    assert owner.completed["actions"] == []  # finish only, no new execution
    assert result["official_success"] is True


def test_publication_counts_all_current_requests_once_across_primitives_and_turns(tmp_path, monkeypatch):
    responses = [episodes.tool_reply(("read1", "check_task", {})),
                 episodes.tool_reply(("read2", "check_task", {})),
                 episodes.tool_reply(("compound", "close_gripper", {})),
                 episodes.tool_reply(("read3", "check_task", {})),
                 episodes.tool_reply(("single", "open_gripper", {})),
                 episodes.tool_reply(("finish", "finish_episode", {
                     "outcome": "failure", "reason": "test finished", "attempts": ["moved"]}))]
    result, owner, session = run_inline(tmp_path, monkeypatch, responses, terminal_after=10, compound=True)
    assert result["termination"] == "policy_finished_episode", result
    steps = verify_policy_state(tmp_path)["steps"]
    limit = owner.loop.plan["budgets"]["max_model_calls"]
    assert [step["remaining_budget"]["model_calls"] for step in steps] == [
        limit, limit - 3, limit - 3, limit - 5]
    # Reading the sealed chain also verifies that no published snapshot was edited.
    assert owner.loop.round_model_calls == 6
    assert owner.loop.planner.inflight_model_calls == 0
    assert session.action_sequence == 3
    assert len(multi_round.provider_requests(tmp_path)) == 6
