"""Default surfaces reach both provider wires with complete help and closed schemas.

The positive H-admission checks, bound to the exp08h package code, moved to
``test_exp08h_placement_bundle.py`` with relocation U7 (2026-09-28). The
correctness and unified surfaces these checks were written against are retired;
the same projection, grounding and acknowledgement mechanisms are pinned on the
maintained defaults.
"""

from __future__ import annotations

import copy
import json

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.tool_surface import validate_tool_help_contract
from ora.agent.session_core.gateway import openai_tool_definitions
from ora.agent.providers.anthropic_request import translate_anthropic_request


DEFAULTS = {
    "libero": "libero_placement_direct_overhead_v1",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo": "robodojo_placement_toolkit_v5",
}


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
@pytest.mark.parametrize("candidate_count", [0, 1, 2])
def test_unified_grounding_composes_preview_and_perception_results_through_real_dispatch(
    tmp_path, monkeypatch, backend, candidate_count,
):
    from types import SimpleNamespace
    from tests.support.agent.grounding_grasp_runtime import _runtime, _select_correctness_surface
    from ora.agent.session_core.domain import PolicyDomainMixin

    from tests.support.agent.toolkit_run import recameraed

    runtime, _ = _runtime(tmp_path, toolkit=True)
    _select_correctness_surface(runtime, DEFAULTS[backend], serve=True)
    camera = "agentview"
    if backend == "robotwin":
        recameraed(runtime, {"agentview": "head_camera"})
        camera = "head_camera"
    ground = runtime.grounding_backend.ground

    def bounded_ground(**arguments):
        candidates, diagnostics = ground(**arguments)
        candidates = candidates[:candidate_count]
        return candidates, {**diagnostics, "returned_candidate_count": len(candidates)}

    monkeypatch.setattr(runtime.grounding_backend, "ground", bounded_ground)
    result = runtime.invoke("vision_ground", {"camera": camera, "text": "red object"})
    assert len(result["candidates"]) == candidate_count
    assert len(result["candidate_previews"]) == candidate_count
    assert len(result["images_base64"]) == candidate_count + 1
    assert result["perception_evidence"]["semantic_identity_verified"] is False
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = SimpleNamespace(surface=runtime.surface)
    visible = planner._causal_tool_result("vision_ground", result)
    assert visible["candidate_previews"] == [
        {key: value for key, value in item.items() if key not in {"sha256", "path"}}
        for item in result["candidate_previews"]
    ]
    assert visible["perception_evidence"]["semantic_identity_verified"] is False
    # diagnostics is a producer-built block the result schema declares for the
    # policy (grasp.plan keeps its candidate ledger there), so a declaration
    # now makes it promised evidence. backend stays provenance and stays out.
    assert "backend" not in visible


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
@pytest.mark.parametrize("tool", ["move_to", "move_pose", "rotate_wrist", "rotate_pitch"])
def test_unified_motion_ack_is_not_a_consumed_execution_receipt(tmp_path, monkeypatch, backend, tool):
    from tests.support.agent.stack_parity import on_toolkit, stand_in
    from tests.support.agent.toolkit_recording import proposal
    from ora.agent.session_core.domain import PolicyDomainMixin

    # A stand-in host of the backend's default surface on the toolkit (``tests/support/agent/stack_parity.py``).
    runtime = on_toolkit(stand_in(tmp_path, backend, monkeypatch))
    arm = "right" if backend == "libero" else "left"
    pose = runtime._current()[2]["proprioception"]["end_effectors"][arm]["pose_xyz_xyzw"]
    target = ({"angle_deg": 2} if tool.startswith("rotate")
              else {"position_m": [pose[0], pose[1], pose[2] + 0.005]})
    if tool == "move_pose":
        target.update(position_mode="absolute", orientation_mode="absolute_quaternion", orientation_xyzw=pose[3:])
    if backend == "libero":
        args = target
    else:
        args = {"targets": {arm: 2 if tool.startswith("rotate") else target}}
    result = runtime.invoke_with_context(tool, args, call_context=proposal(runtime, tool, args))
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = runtime
    visible = planner._causal_tool_result(tool, result)
    assert visible["status"] == "submitted"
    assert visible["compiled_pose"]
    assert "action" in visible
    assert not {"execution_status", "motion_status", "measured_endpoint", "consumed"} & visible.keys()


@pytest.mark.parametrize("profile", sorted(DEFAULTS.values()))
def test_default_complete_help_and_closed_schema_survive_real_provider_projection(profile):
    surface = surface_of(profile)
    original = copy.deepcopy(surface.definitions)
    validate_tool_help_contract(surface.definitions, surface.result_schemas)
    names = [item["name"] for item in surface.definitions]
    assert len(names) == len(set(names))
    assert set(surface.mutation_tools) < set(names)
    assert surface.lifecycle_tools == ("finish_episode",)
    assert set(surface.result_schemas) == set(names)

    projected = openai_tool_definitions(names, definitions=surface.definitions)
    assert [item["function"]["name"] for item in projected] == names
    body = translate_anthropic_request({"model": "cpu-fixture", "max_tokens": 64,
        "messages": [{"role": "user", "content": "Schema projection only."}],
        "tools": projected})
    assert len(body["tools"]) == len(names)
    assert len({item["name"] for item in body["tools"]}) == len(names)
    aliases = {}
    for block in body.get("system", []):
        text = block.get("text", "")
        if text.startswith("<wire_tool_aliases>\n"):
            aliases = json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])
    for original_tool, native, wire in zip(surface.definitions, projected, body["tools"], strict=True):
        name = original_tool["name"]
        assert wire["name"] == aliases.get(name, name)
        assert wire["description"] == native["function"]["description"] == original_tool["description"]
        schema = native["function"]["parameters"]
        expected_wire_schema = ({"type": "object", **schema}
                                if "type" not in schema and "oneOf" in schema else schema)
        assert wire["input_schema"] == expected_wire_schema
        assert wire["input_schema"]["type"] == "object"
        branches = schema.get("oneOf", [schema])
        assert all(branch["type"] == "object" and branch["additionalProperties"] is False
                   for branch in branches)
    assert surface.definitions == original
