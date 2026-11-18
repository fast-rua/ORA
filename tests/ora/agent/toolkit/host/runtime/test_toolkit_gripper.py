"""Synchronous gripper migration preserves commands and intent semantics."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests.support.agent.toolkit_tools import install
from ora.agent.error_classes import ToolError
from tests.support.agent.toolkit_tools import make_host, use_toolkit_v5
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal


class Executor:
    def __init__(self, host, used):
        self.host, self.used = host, used
        self.requests, self.snapshots, self.budgets = [], [], []
        # The plan v5 delivers receipts by: the tool's own result.
        self.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": "tool_output"})

    @contextmanager
    def invocation(self, name, context, requested):
        self.budgets.append(requested)
        yield SimpleNamespace(recovered=False)

    def execute_action(self, action, budget):
        self.requests.append(action)
        self.snapshots.append(self.host.toolkit_state_access.values())
        return {"receipt": {"status": "reached" if self.used else "not_executed",
                            "action_id": action["action_id"], "native_actions": self.used,
                            "stage_evidence": {"consumed": bool(self.used)}},
                "remaining_budget": self.budgets[-1] - self.used}


def setup_host(tmp_path, monkeypatch, used=3):
    host, client, _ = make_host(tmp_path, monkeypatch)
    complete_cameras(host)
    use_toolkit_v5(host)
    host.toolkit_executor = Executor(host, used)
    install(host)
    return host, client


@pytest.mark.parametrize("targets,reference", [
    ({"left": "open"}, None), ({"right": "close"}, "target-intended"),
    ({"left": "close", "right": "close"}, "ambiguous"),
    ({"left": "open", "right": "close"}, "target-intended"),
])
def test_gripper_dispatch_preserves_old_command_and_annotation(tmp_path, monkeypatch, targets, reference):
    host, client = setup_host(tmp_path, monkeypatch)
    try:
        args = {"targets": targets, "target_ref": reference}
        result = host.invoke_with_context("set_gripper", args, call_context=proposal(host, "set_gripper", args))
        executor = host.toolkit_executor
        # v5's fixed set_gripper budget (``toolkit/tools/budgets.py``), and its rate-limited command per arm.
        assert executor.budgets == [30]
        sent = executor.requests[0]["arguments"]["targets"]
        assert {arm: target["opening"] for arm, target in sent.items()} == targets
        assert executor.requests[0]["action"] == "set_gripper"
        assert result["remaining_budget"] == 27
        assert "stage_evidence" not in result["receipt"]
        assert not (host.control_root / "submitted-action.json").exists()
        if "close" in targets.values():
            assert result["intent"] == {"target_ref": reference, "role": "agent_annotation_only"}
        else:
            assert "intent" not in result
        if list(targets.values()).count("close") == 1 and reference:
            arm = next(arm for arm, command in targets.items() if command == "close")
            held = executor.snapshots[0][f"tool.held.{arm}"]
            assert held["target_ref"] == reference
            assert held["role"] == "dispatch_intent_not_grasp_evidence"
            assert len(held["attach_pose_xyz_xyzw"]) == 7
        else:
            assert executor.snapshots[0]["tool.held.left"] is None
            assert executor.snapshots[0]["tool.held.right"] is None
    finally:
        client.close()


@pytest.mark.parametrize("command,reference", [("open", None), ("close", "new")])
def test_zero_spend_restores_old_intent_but_positive_spend_keeps_dispatch_intent(tmp_path, monkeypatch, command, reference):
    host, client = setup_host(tmp_path, monkeypatch, used=0)
    old = {"source": "set_gripper", "target_ref": "old", "attach_pose_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1]}
    try:
        host.toolkit_state_access.write("tool.held.left", old)
        args = {"targets": {"left": command}, "target_ref": reference}
        host.invoke_with_context("set_gripper", args, call_context=proposal(host, "set_gripper", args))
        assert host.toolkit_state_access.read("tool.held.left") == old
        assert host.toolkit_executor.snapshots[0]["tool.held.left"] != old
    finally:
        client.close()


@pytest.mark.parametrize("arguments,message", [
    ({"targets": {"left": "hold"}}, "input enum"),
    ({"targets": {"left": "close"}, "target_ref": " "}, "non-empty"),
    ({"targets": {}}, "minProperties"),
])
def test_invalid_gripper_request_never_dispatches(tmp_path, monkeypatch, arguments, message):
    host, client = setup_host(tmp_path, monkeypatch)
    try:
        with pytest.raises(ToolError, match=message):
            host.invoke_with_context("set_gripper", arguments, call_context=proposal(host, "set_gripper", arguments))
        assert host.toolkit_executor.requests == []
    finally:
        client.close()


def test_round_reset_clears_holding_intents(tmp_path, monkeypatch):
    host, client = setup_host(tmp_path, monkeypatch)
    try:
        host.toolkit_state_access.write("tool.held.left", {"target_ref": "old-round"})
        host.reset_round(1)
        assert host.toolkit_state_access.read("tool.held.left") is None
        assert host.toolkit_state_access.read("tool.held.right") is None
    finally:
        client.close()
