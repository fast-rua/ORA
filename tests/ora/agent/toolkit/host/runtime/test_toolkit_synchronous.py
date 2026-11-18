"""Real portable dispatch uses the injected executor, never the legacy proposal path."""
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from tests.support.agent.toolkit_tools import install
from ora.agent.error_classes import ToolError
from tests.support.agent.kinematics_bridge import Bridge
from tests.support.agent.toolkit_tools import make_host, use_toolkit_v5
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal


class Executor:
    def __init__(self):
        self.requests = []
        self.budgets = []
        # The plan v5 delivers receipts by: the tool's own result.
        self.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": "tool_output"})

    @contextmanager
    def invocation(self, name, context, requested):
        self.budgets.append(requested)
        yield SimpleNamespace(recovered=False)

    def execute_action(self, action, budget):
        self.requests.append((action, budget))
        return {"receipt": {"status": "reached", "action_id": action["action_id"],
                            "native_actions": 3, "stage_evidence": {"consumed": True}}, "remaining_budget": self.budgets[-1] - 3}


def setup_host(tmp_path, monkeypatch):
    host, client, _ = make_host(tmp_path, monkeypatch)
    complete_cameras(host)
    use_toolkit_v5(host)
    # v5 plans each move through the environment's read-only IK bridge; the toy answers it by hand.
    host.align_plan_callback = Bridge()
    host.toolkit_executor = Executor()
    install(host)
    return host, client


def test_move_to_executes_and_returns_public_receipt(tmp_path, monkeypatch):
    host, client = setup_host(tmp_path, monkeypatch)
    try:
        _, _, manifest = host._current()
        position = manifest["proprioception"]["end_effectors"]["left"]["pose_xyz_xyzw"][:3]
        position[2] += .01
        args = {"targets": {"left": {"position_m": position}}}
        result = host.invoke_with_context("move_to", args, call_context=proposal(host, "move_to", args))
        assert result["status"] == "executed"
        assert "stage_evidence" not in result["receipt"]
        # move_to's fixed budget is 250 (``toolkit/tools/budgets.py``); the executor spent 3.
        assert host.toolkit_executor.budgets == [250]
        assert result["receipt"]["native_actions"] == 3 and result["remaining_budget"] == 247
        assert len(host.toolkit_executor.requests) == 1
        assert not (host.control_root / "submitted-action.json").exists()
    finally:
        client.close()


def test_a_budget_argument_is_refused_before_any_tool_body(tmp_path, monkeypatch):
    """Budgets are offline configuration: a call that names one is refused, whatever the value, and a read-only
    tool runs on 0."""
    host, client = setup_host(tmp_path, monkeypatch)
    try:
        for index, value in enumerate((1, 0)):
            args = {"expressions": "total = 1 + 2", "native_action_budget": value}
            with pytest.raises(ToolError, match="native_action_budget"):
                host.invoke_with_context("compute", args, call_context=proposal(host, "compute", args, index=index))
        assert host.toolkit_executor.budgets == []
        args = {"expressions": "total = 1 + 2"}
        result = host.invoke_with_context("compute", args, call_context=proposal(host, "compute", args, index=2))
        assert result["outputs"][0]["value"] == 3
        assert host.toolkit_executor.budgets == [0]
    finally:
        client.close()


def test_toolkit_capability_has_no_legacy_submission_method_or_adapter(tmp_path, monkeypatch):
    from ora.agent.toolkit.host.runtime.dispatch import TOOLS
    host, client = setup_host(tmp_path, monkeypatch)
    original = TOOLS["compute"]
    def probe(api, args):
        assert not hasattr(api.act, "submit_action")
        assert "submit_action" not in api.act._backend.adapters
        with pytest.raises(AttributeError):
            api.act.submit_action(action={"action_id": "bypass"})
        return original(api, args)
    monkeypatch.setitem(TOOLS, "compute", probe)
    try:
        args = {"expressions": "total = 1 + 2"}
        result = host.invoke_with_context("compute", args, call_context=proposal(host, "compute", args))
        assert result["outputs"][0]["value"] == 3
        assert host.toolkit_executor.requests == []
        assert not (host.control_root / "submitted-action.json").exists()
    finally:
        client.close()


def test_a_toolset_tool_under_a_new_name_runs_and_only_a_declared_mutation_tool_gets_a_budget(tmp_path, monkeypatch):
    """I-0005: a toolset adds tools under new names; the surface's mutation_tools decides which may act."""
    import dataclasses
    from ora.agent.toolkit.host.runtime import dispatch as toolkit_dispatch
    from ora.agent.toolkit.tools import budgets
    host, client, _ = make_host(tmp_path, monkeypatch)
    try:
        complete_cameras(host)
        use_toolkit_v5(host)
        host.toolkit_executor = Executor()
        seen = []
        for name in ("slow_slide_block", "measure_blocks"):
            monkeypatch.setitem(toolkit_dispatch.TOOLS, name, lambda api, args, name=name: seen.append(name) or {"status": "ok"})
        extra = [{"name": name, "description": name, "inputSchema": {"type": "object", "properties": {}}}
                 for name in ("slow_slide_block", "measure_blocks")]
        host.surface = dataclasses.replace(host.surface, definitions=[*host.surface.definitions, *extra],
                                           mutation_tools=(*host.surface.mutation_tools, "slow_slide_block"))
        host.allowed_tools = (*host.allowed_tools, "slow_slide_block", "measure_blocks")
        install(host)
        # The toolset's revision gives both a budget offline; only the declared mutation tool may spend one.
        original = budgets.native_action_budget
        monkeypatch.setattr(budgets, "native_action_budget",
                            lambda name: 12 if name in ("slow_slide_block", "measure_blocks") else original(name))
        assert host.invoke_with_context("slow_slide_block", {},
                                        call_context=proposal(host, "slow_slide_block", {}))["status"] == "ok"
        assert host.toolkit_executor.budgets == [12]
        with pytest.raises(ToolError, match="native_action_budget"):
            host.invoke_with_context("measure_blocks", {}, call_context=proposal(host, "measure_blocks", {}, index=1))
        assert seen == ["slow_slide_block"]
    finally:
        client.close()


def test_the_shipped_synchronous_surface_executes_exactly_the_builtin_tools():
    """Execution tools come from mutation_tools; on the shipped synchronous surface (v5) that is the built-in set."""
    from ora.agent.composition.schemas.toolkit_execution import EXECUTION_TOOLS
    GRANTS = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=GRANTS)
    # v5 adds return_home (2026-10-28), an action tool the toolset declares in mutation_tools.
    builtin = EXECUTION_TOOLS | {"return_home"}
    assert set(surface.mutation_tools) == builtin & {d["name"] for d in surface.definitions}


def test_recovery_tools_get_their_surface_budget(tmp_path, monkeypatch):
    """v5 gives recover_pose and return_home what one trajectory may use (the legacy-visible v3, which kept main's
    20 per primitive, was retired on 2026-10-31)."""
    from ora.agent.toolkit.host.runtime import dispatch as toolkit_dispatch
    from ora.agent.toolkit.tools.budgets import native_action_budget
    profile, budgets = "robodojo_placement_toolkit_v5", {"recover_pose": 250, "return_home": 250}
    host, client, _ = make_host(tmp_path, monkeypatch)
    try:
        complete_cameras(host)
        grants = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
        host.toolkit_grants = grants
        host.surface = resolve_tool_surface(profile, toolkit_grants=grants, toolkit_receipt_delivery="tool_output")
        host.allowed_tools = tuple(d["name"] for d in host.surface.definitions)
        host.toolkit_executor = Executor()
        # tool_output delivery: the result is the tool's own, remaining_budget as the executor reports it.
        host.toolkit_executor.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": "tool_output"})

        def tool(api, args, name):
            from ora.agent.toolkit.lib.values import PublicResult
            executed = PublicResult.of(api.act.execute_action({"action_id": "action-0"}, 20)).data.json()
            return {"status": "executed", "action": name, "compiled_pose": {}, **executed}

        for name in budgets:
            monkeypatch.setitem(toolkit_dispatch.TOOLS, name, lambda api, args, name=name: tool(api, args, name))
        install(host)
        for index, (name, budget) in enumerate(budgets.items()):
            args = {"arms": ["left"]}
            result = host.invoke_with_context(name, args, call_context=proposal(host, name, args, index=index))
            assert host.toolkit_executor.budgets[-1] == budget == native_action_budget(name)
            assert result["remaining_budget"] == budget - 3
    finally:
        client.close()
