"""The remaining robot tools on toolkit v5 keep the released tools' public errors and measurements."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from tests.support.agent.toolkit_tools import install
from ora.agent.error_classes import ToolError
from ora.contracts.action import validate_runtime_action
from tests.support.agent.kinematics_bridge import Bridge
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_atomic import timeline
from tests.support.agent.toolkit_recording import proposal


class Executor:
    def __init__(self):
        self.requests = []
        self.budgets = []
        # The plan v5 delivers receipts by: the tool's own result.
        self.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": "tool_output"})

    @contextmanager
    def invocation(self, name, context, requested):
        self.budget = requested
        yield SimpleNamespace(recovered=False)

    def execute_action(self, action, budget):
        # v5 fixes each tool's budget offline (``toolkit/tools/budgets.py``); the call never supplies one.
        self.budgets.append(budget)
        # The real execution owner validates/canonicalizes before dispatch.
        self.requests.append(validate_runtime_action(action))
        return {"receipt": {"action_id": action["action_id"], "status": "reached", "native_actions": 1},
                "remaining_budget": self.budget - 1}


#: The grants toolkit v5 requires (its surface.json ``required_grants``).
V5_GRANTS = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}


def setup(root, monkeypatch, *, migrated):
    host, client, _ = make_host(root, monkeypatch)
    timeline(host)
    if migrated:
        # v5 plans each move through the environment's read-only IK bridge; the toy answers it by hand.
        host.align_plan_callback = Bridge()
        host.toolkit_grants = V5_GRANTS
        host.surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=V5_GRANTS)
        host.allowed_tools = tuple(d["name"] for d in host.surface.definitions)
        host.toolkit_executor = Executor()
        install(host)
    return host, client


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("reference", ["red block", "candidate-0"])
@pytest.mark.parametrize("name", ["move_to", "move_pose", "rotate_wrist", "rotate_pitch"])
def test_free_text_close_does_not_poison_later_motion(tmp_path, monkeypatch, arm, reference, name):
    host, client = setup(tmp_path, monkeypatch, migrated=True)
    try:
        close = {"targets": {arm: "close"}, "target_ref": reference}
        host.invoke_with_context("set_gripper", close, call_context=proposal(host, "set_gripper", close))
        pose = host._current()[2]["proprioception"]["end_effectors"][arm]["pose_xyz_xyzw"]
        pose[2] += .01
        target = (15 if name.startswith("rotate_") else
                  {"position_m": pose[:3], **({"position_mode": "absolute", "orientation_mode": "absolute_quaternion",
                                                 "orientation_xyzw": pose[3:]} if name == "move_pose" else {})})
        args = {"targets": {arm: target}}
        result = host.invoke_with_context(name, args, call_context=proposal(host, name, args, index=1))
        assert result["status"] == "executed"
        assert len(host.toolkit_executor.requests) == 2
        assert host.toolkit_state_access.read(f"tool.held.{arm}")["target_ref"] == reference
    finally:
        client.close()


def test_missing_recovery_anchor_is_public_plan_refusal(tmp_path, monkeypatch):
    """With no arm-moving call to undo, recover_pose refuses as a plan, names why and moves nothing."""
    host, client = setup(tmp_path, monkeypatch, migrated=True)
    try:
        args = {"arms": ["left"]}
        with pytest.raises(ToolError) as caught:
            host.invoke_with_context("recover_pose", args, call_context=proposal(host, "recover_pose", args))
        error = caught.value.as_dict()
        assert error["code"] == "plan_rejected"
        assert error["constraints"]["failure_code"] in {"retrace_nothing_to_undo", "retrace_not_recorded"}
        assert not host.toolkit_executor.requests
    finally:
        client.close()


@pytest.mark.parametrize("args,kind", [({"message": "fixture issue", "kind": "confusion"}, "confusion"),
                                       ({"message": "fixture issue", "kind": "defect"}, "defect"),
                                       ({"message": "fixture issue", "kind": "question"}, None),
                                       ({"message": "x", "extra": True}, None)])
def test_report_issue_takes_message_and_kind_only(tmp_path, monkeypatch, args, kind):
    """v5 declares message and kind (defect or confusion); any other field or kind is refused, and nothing runs."""
    host, client = setup(tmp_path, monkeypatch, migrated=True)
    try:
        if kind is None:
            with pytest.raises(ToolError) as caught:
                host.invoke_with_context("report_issue", args, call_context=proposal(host, "report_issue", args))
            assert caught.value.as_dict()["code"] == "invalid_arguments"
            assert not (host.control_root / "openeta-reported-issues.jsonl").exists()
        else:
            host.invoke_with_context("report_issue", args, call_context=proposal(host, "report_issue", args))
            record = json.loads((host.control_root / "openeta-reported-issues.jsonl").read_bytes())
            assert (record["message"], record["kind"]) == ("fixture issue", kind)
        assert not host.toolkit_executor.requests
    finally:
        client.close()


def test_report_issue_records_its_kind(tmp_path):
    """The file-backed surfaces declare kind (defect, the default, or confusion); the record carries it."""
    from ora.agent.toolkit.host.lifecycle import report_adapter

    report = report_adapter(SimpleNamespace(control_root=tmp_path))
    report({"issue": {"message": " which face is the working face? ", "kind": "confusion"}}, None, None)
    report({"issue": {"message": "malformed result"}}, None, None)
    rows = [json.loads(line) for line in (tmp_path / "openeta-reported-issues.jsonl").read_text().splitlines()]
    assert [(r["message"], r["kind"]) for r in rows] == [
        ("which face is the working face?", "confusion"), ("malformed result", "defect")]
    with pytest.raises(ToolError):
        report({"issue": {"message": "x", "kind": "question"}}, None, None)


def test_rotation_and_wait_help_describe_synchronous_receipt_and_budget():
    surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=V5_GRANTS)
    definitions = {item["name"]: item for item in surface.definitions}
    for name in ("rotate_wrist", "rotate_pitch", "wait_ticks"):
        description = definitions[name]["description"]
        assert "remaining_budget" in description
        assert "Native action budgets are fixed offline and cannot be supplied as tool arguments." in description
        assert "status=submitted" not in description
        if name != "wait_ticks":
            assert "previous applied gripper drive target" in description
