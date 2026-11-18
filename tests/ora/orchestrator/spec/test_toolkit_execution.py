"""Versioned synchronous surface and independent task-check authority.

On toolkit v5, the synchronous surface left (the layered v2, which ran without the checker and took each budget from
the call, was retired on 2026-11-01)."""
from dataclasses import replace

import pytest
from ora.agent.code_mode.contract import resolve_contract
from ora.agent.composition.surface.tool_surface import resolve_tool_surface, ToolSurfaceError
from ora.agent.toolkit.tools.budgets import NATIVE_ACTION_BUDGETS
from ora.orchestrator.spec.native_policy import _resolve_code_mode
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.agent.surface_catalog import catalog_grants
from tests.support.orchestrator.run_specs import robodojo_native_spec

PROFILE = "robodojo_placement_toolkit_v5"
GRANTS = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
#: Each surface that is to move to the toolkit, with the backend its code mode runs on.
ENVIRONMENT_PLANNED = {"libero_placement_direct_overhead_v1": "libero_pro",
                       "robotwin_toolchain_b2_closed_noservo_v2": "robotwin",
                       "robocasa_mobile_direct_v1": "robocasa", "robolab_direct_v1": "robolab", "yam_direct_v1": "yam"}


def test_synchronous_profile_defaults_and_required_grants():
    spec = robodojo_native_spec("general_pickup")
    spec["budgets"].pop("max_robot_decisions", None)
    plan = resolve_run_spec(spec)
    assert "use_toolkit_tools" not in plan
    assert plan["policy"]["baseline"]["id"] == "robodojo_toolkit_v7"
    assert plan["toolkit_observation_receipts"] is False
    assert plan["toolkit_receipt_delivery"] == "tool_output"
    assert plan["budgets"]["max_robot_decisions"] == 100
    assert plan["toolkit_grants"] == {"tools": [], "state_keys": []}
    # The surface's tools, which the default (code mode) gives the model as functions of the robot library.
    names = set(plan["policy"]["tool_allowlist"])
    assert "check_task" not in names and len(names) == 19
    functions = next(layer for layer in plan["policy_composition"]["tool_surface"]["derived_layers"]
                     if layer["kind"] == "code_mode_v1")["script_functions"]
    assert set(functions) <= names and "check_task" not in functions


def test_a_named_decision_budget_is_honoured_above_the_default():
    spec = robodojo_native_spec("general_pickup")
    spec["budgets"]["max_robot_decisions"] = 200
    assert resolve_run_spec(spec)["budgets"]["max_robot_decisions"] == 200


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "200", None])
def test_an_invalid_decision_budget_is_refused_not_defaulted(value):
    spec = robodojo_native_spec("general_pickup")
    spec["budgets"]["max_robot_decisions"] = value
    with pytest.raises(RunSpecError, match="budgets.max_robot_decisions must be a positive integer"):
        resolve_run_spec(spec)


@pytest.mark.parametrize("grants", [None, {"tools": ["check_task"]}, {"state_keys": ["privileged.task_check"]}])
def test_the_checker_is_granted_whole_or_not_at_all(grants):
    with pytest.raises(ToolSurfaceError, match="both"):
        resolve_tool_surface(PROFILE, toolkit_grants=grants)


def test_no_robot_tool_takes_a_budget_argument():
    """Each budget is offline configuration (``toolkit/tools/budgets.py``), never a schema field."""
    surface = resolve_tool_surface(PROFILE, toolkit_grants=GRANTS)
    for definition in surface.definitions:
        assert "native_action_budget" not in definition["inputSchema"].get("properties", {})
    assert set(NATIVE_ACTION_BUDGETS) <= set(surface.mutation_tools)


def code_mode(surface, kind, **asked):
    policy = {"agent_core": "pi", "code_mode": resolve_contract({"library": "extended", **asked})}
    _resolve_code_mode(policy, surface, {"kind": kind})
    return policy["code_mode"]


@pytest.mark.parametrize("profile,kind", ENVIRONMENT_PLANNED.items())
def test_moving_a_surface_to_the_toolkit_does_not_change_code_mode_s_card(profile, kind):
    """V10: a resolved contract is what the card, the prompt layers and robot.docs read."""
    surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
    moved = replace(surface, capabilities=(*surface.capabilities, "toolkit_only_v1"))
    for asked in ({}, {"prompt": "v3", "source": True}):
        assert code_mode(moved, kind, **asked) == code_mode(surface, kind, **asked)
        assert not {"query", "source"} & set(code_mode(moved, kind, **asked).get("script_reads", ()))


@pytest.mark.parametrize("profile", ["robodojo_placement_toolkit_v5", "robodojo_placement_toolkit_v6"])
def test_robodojo_keeps_its_query_and_its_source(profile):
    surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
    assert code_mode(surface, "robodojo", prompt="v3", source=True)["script_reads"][-2:] == ["query", "source"]
