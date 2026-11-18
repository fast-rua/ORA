"""Manifest and legacy audit reconstruction preserve explicit tool authority."""
from copy import deepcopy

import pytest

from ora.agent.composition.surface.tool_surface import ToolSurfaceError
from ora.orchestrator.evaluation.checks import _expected_context_manifest_tools
from ora.orchestrator.evaluation.context_manifest import _advertised_tools
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator.run_specs import robodojo_native_spec


@pytest.fixture
def legacy_plan():
    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["baseline"] = "robodojo_toolkit_v6"  # the legacy-visible surface left (v5, tool_output)
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    return resolve_run_spec(spec)


@pytest.mark.parametrize("delivery", ["tool_output"])
def test_manifest_and_legacy_audit_reconstruction_use_plan_authority(legacy_plan, delivery):
    plan = legacy_plan
    plan["toolkit_receipt_delivery"] = delivery
    planned = plan["policy_composition"]["tool_surface"].pop("definitions")
    advertised = _advertised_tools(plan, plan["policy"], {})
    definitions = [{k: v for k, v in d.items() if k != "state_effect"} for d in advertised.definitions]
    assert {d["name"] for d in definitions} <= {d["name"] for d in planned}
    assert "check_task" in [d["name"] for d in definitions]
    assert _expected_context_manifest_tools(plan) == [d["name"] for d in definitions]
    result_schema = advertised.resolved_surface.result_schemas["move_to"]["properties"]
    assert ("receipt" in result_schema) == (delivery == "tool_output")


@pytest.mark.parametrize("grants", [None, {}, {"tools": ["check_task"]}, {"state_keys": ["privileged.task_check"]}])
def test_reconstruction_never_supplies_missing_grants(legacy_plan, grants):
    plan = deepcopy(legacy_plan)
    plan["toolkit_grants"] = grants
    plan["policy_composition"]["tool_surface"].pop("definitions")
    with pytest.raises(ToolSurfaceError, match="requires both"):
        _advertised_tools(plan, plan["policy"], {})
    with pytest.raises(ToolSurfaceError, match="requires both"):
        _expected_context_manifest_tools(plan)

