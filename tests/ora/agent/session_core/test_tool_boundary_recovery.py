"""Correctable inputs and public evidence survive the native_api provider boundary.

A malformed tool call inside an episode (bad JSON, no function name) is checked
on the agent core in ``tests/agent_core/test_provider_faults.py``.
"""

from __future__ import annotations

from tests.support.agent.surface_catalog import catalog_grants, handlers_only
import copy
import json
from types import SimpleNamespace

import pytest
pytestmark = pytest.mark.usefixtures("ready_model_services")


from ora.agent.composition import resolve_tool_surface
from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.surface.tool_surface import (
    ToolSurfaceError, validate_schema_instance,
)
from ora.agent.context.causal_value import _causal_projection_violations
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.read_tool import ARTIFACT_READ_TOOL
from ora.agent.memory.catalog import ContextArtifactCatalog
from ora.agent.memory.tool_runtime import RecoverableArtifactToolRuntime
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.session_core.gateway import _openai_parameters, openai_tool_definitions
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.providers.anthropic_request import translate_anthropic_request
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.error_classes import ToolError


@pytest.mark.parametrize("keyword,bound,accepted,rejected", [
    ("minimum", -2, [-2, 0, 2], [-3]),
    ("maximum", 2, [-2, 0, 2], [3]),
    ("exclusiveMinimum", -2, [0, 2], [-3, -2]),
    ("exclusiveMaximum", 2, [-2, 0], [2, 3]),
])
def test_declared_numeric_bounds_accept_exact_closed_edges_and_reject_outside(
    keyword, bound, accepted, rejected,
):
    schema = {"type": "number", keyword: bound}
    for value in accepted:
        validate_schema_instance(value, schema)
    for value in rejected:
        with pytest.raises(ToolSurfaceError, match=rf"\$result.*{keyword} {bound}"):
            validate_schema_instance(value, schema)


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -float("inf")])
def test_numeric_schema_never_accepts_boolean_or_nonfinite_values(bad):
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(bad, {"type": "number", "minimum": -1, "maximum": 1})


def test_numeric_bound_error_names_nested_array_reference_and_rejected_branches():
    schema = {
        "$defs": {"count": {"oneOf": [
            {"type": "integer", "minimum": 0, "maximum": 3}, {"type": "null"},
        ]}},
        "type": "object", "additionalProperties": False,
        "properties": {"counts": {"type": "array", "items": {"$ref": "#/$defs/count"}}},
        "required": ["counts"],
    }
    validate_schema_instance({"counts": [0, 3, None]}, schema, root_path="$input")
    for bad, keyword in [(-1, "minimum"), (4, "maximum")]:
        with pytest.raises(ToolSurfaceError) as caught:
            validate_schema_instance({"counts": [0, bad]}, schema, root_path="$input")
        message = str(caught.value)
        assert "$input.counts[1]" in message
        assert "oneOf[0]" in message and "oneOf[1]" in message and keyword in message


def test_nullable_branch_cannot_skip_a_sibling_numeric_bound():
    schema = {"oneOf": [{"type": "number"}, {"type": "null"}], "minimum": 0}
    validate_schema_instance(None, schema)
    validate_schema_instance(0, schema)
    with pytest.raises(ToolSurfaceError, match="minimum"):
        validate_schema_instance(-1, schema)


#: The kept default surfaces.
LIBERO = "libero_placement_direct_overhead_v1"
ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"


def _planner(surface):
    """The shared result projection (``_causal_tool_result``) of a session on ``surface``."""
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = SimpleNamespace(surface=surface)
    return planner


def _wire(protocol, messages, tools):
    """The provider wire request for these messages and tools: OpenAI sends the execution
    request as assembled, Anthropic its translation."""
    request = {"model": "offline-test", "max_tokens": 128, "stream": False, "tool_choice": "auto",
               "messages": messages, "tools": tools}
    return translate_anthropic_request(request) if protocol == "anthropic" else request


@pytest.fixture
def artifact_reader(tmp_path):
    surface = surface_of(LIBERO)
    runtime = RecoverableArtifactToolRuntime(
        base=SimpleNamespace(surface=surface, definitions=lambda: []), run_root=tmp_path,
        event_path=tmp_path / "events.jsonl",
    )
    catalog = ContextArtifactCatalog(tmp_path, "recovery-test", ArtifactStore(tmp_path))
    reference = catalog.put_bytes(
        b'{"json_value":[{"reachable":false}],"status":"ok"}',
        kind="context_tool_result", suffix=".json", media_type="application/json",
        round_index=0, action_step=1,
    )
    runtime.bind_context_artifacts(catalog)
    return runtime, reference["ref"]


@pytest.mark.parametrize("pointer,extra", [
    ("/json_value", {"limit": 8192}),
    ("/results/0/reachable", {}),
    ("bad-pointer", {}),
])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_artifact_input_error_reaches_wire_with_recovery_fields(
    artifact_reader, pointer, extra, protocol,
):
    runtime, ref = artifact_reader
    with pytest.raises(ToolError) as caught:
        runtime._read_ref({"ref": ref, "json_pointer": pointer, **extra})
    result = _tool_error(caught.value)
    visible = _planner(runtime.surface)._causal_tool_result("artifact_read", result)
    error = visible["error"]
    assert error["code"] == "invalid_arguments" and error["retryable"] is True
    assert error["field"] == "json_pointer" and error["received"] is not None
    assert error["constraints"]["allowed"]
    if not extra:
        assert error["constraints"]["top_level_keys"] == ["json_value", "status"]
    wire = json.dumps(_wire(protocol, [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "read-1", "type": "function", "function": {
                "name": "artifact.read", "arguments": json.dumps({"ref": ref}),
            },
        }]},
        {"role": "tool", "name": "artifact.read", "tool_call_id": "read-1",
         "content": json.dumps(visible)},
    ], openai_tool_definitions(["artifact.read"], definitions=[ARTIFACT_READ_TOOL])))
    assert "invalid_arguments" in wire and "constraints" in wire and "allowed" in wire
    assert not _causal_projection_violations(visible)
    assert runtime._read_ref({"ref": ref, "json_pointer": "/json_value/0/reachable"})["json_value"] is False


def test_artifact_schema_declares_pointer_and_range_modes(artifact_reader):
    _runtime, ref = artifact_reader
    schema = ARTIFACT_READ_TOOL["inputSchema"]
    for arguments in ({"ref": ref}, {"ref": ref, "offset": 2, "limit": 10},
                      {"ref": ref, "json_pointer": "/json_value"}):
        validate_schema_instance(arguments, schema, root_path="$input")
    with pytest.raises(ToolSurfaceError, match=r"\$input.*oneOf\[0\].*oneOf\[1\]"):
        validate_schema_instance(
            {"ref": ref, "json_pointer": "/json_value", "limit": 8192},
            schema, root_path="$input",
        )


def test_unexpected_key_error_is_still_a_nonretryable_internal_fault():
    result = _tool_error(KeyError("worker implementation defect"))
    assert result["error"]["code"] == "internal_tool_error"
    assert result["error"]["retryable"] is False


def test_missing_orientation_names_input_and_the_missing_field():
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = surface_of(ROBOTWIN)
    with pytest.raises(ToolError) as caught:
        runtime._validate_public_arguments("move_pose", {
            "targets": {"left": {"position_mode": "absolute", "position_m": [0.0, 0.0, 0.9],
                                  "interaction": "free_space"}},
        })
    message = str(caught.value)
    assert "$input" in message and "$result" not in message
    assert "orientation_mode" in message
    assert caught.value.code == "invalid_arguments" and caught.value.retryable is True


@pytest.mark.parametrize("profile_id", sorted(PROFILE_CATALOG))
def test_registered_profile_wire_resolves_every_local_schema_reference(profile_id):
    surface = resolve_tool_surface(profile_id, toolkit_grants=catalog_grants(profile_id))

    def visit(node, root):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                target = root
                for token in ref[2:].split("/"):
                    target = target[token.replace("~1", "/").replace("~0", "~")]
            for value in node.values():
                visit(value, root)
        elif isinstance(node, list):
            for value in node:
                visit(value, root)

    for definition in surface.definitions:
        original = copy.deepcopy(definition["inputSchema"])
        projected = _openai_parameters(original)
        visit(projected, projected)
        assert original == definition["inputSchema"]


def test_observation_and_point_follow_their_declared_public_schema():
    planner = _planner(surface_of(ROBOTWIN))
    facts = {"observation_id": "actual-3",
             "end_effectors": {"left": {"pose_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1],
                                        "gripper_command_target": "open"}}}
    observed = {
        "success": True, "observation_id": "actual-3", "status": "actual",
        "public_robot_facts": facts, "views": ["head_camera"], "object_gt": "must remain private",
    }
    visible = planner._causal_tool_result("observe", observed)
    assert visible["success"] is True and visible["observation_id"] == "actual-3"
    assert visible["public_robot_facts"] == facts
    assert "object_gt" not in visible
    point = {"success": True, "status": "solved", "observation_id": "actual-3", "view": "head_camera",
             "mark_count": 1, "solved_count": 1, "failed_count": 0,
             "marks": [{"point_id": "point-1", "status": "solved", "xyz_m": [0.1, 0.2, 0.9],
                        "action_workspace": {"admissible": False}}]}
    assert planner._causal_tool_result("mark_point", point) == point
    assert not _causal_projection_violations(visible)


def test_unregistered_refusal_retains_planner_recovery_without_pose_pair():
    planner = _planner(surface_of(LIBERO))
    effect = planner._minimal_action_effect({
        "schema": "future-generic-motion-receipt", "action": "recover_motion",
        "motion_status": "refused", "feedback": {
            "planning_failure_reason": "INVALID_START_STATE_JOINT_LIMITS",
        },
    })
    assert (effect["status"], effect["failure_stage"], effect["failure_code"]) == (
        "plan_rejected", "planner", "invalid_start_state",
    )


def test_declared_diagnostics_survives_the_operational_filter():
    """A schema-declared content property is promised evidence.

    ``grasp_plan`` builds its candidate ledger into ``diagnostics`` deliberately
    for the policy and the result schema declares the property, but the shared
    operational filter removed the whole block. A policy therefore could not see
    how many candidates the backend returned, how many survived compilation, or
    why the rest were rejected — and so could not cite an untried candidate.

    ``backend`` is the other half of that filter and is deliberately excluded:
    model id, version and config hash are provenance the policy cannot act on,
    and sixty schema sites declare it as an artefact of how those schemas were
    written rather than as a promise. Three older tests assert it stays out.
    When the v4 generation removes those declarations, this exclusion and the
    split in ``projection.py`` both go away.
    """

    planner = _planner(surface_of(ROBOTWIN))
    result = {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-7",
        "target_ref": "target-" + "a" * 24,
        "camera": "head_camera",
        "candidates": [],
        "backend": {"id": "contact_graspnet_pytorch", "version": "53e9966d"},
        "diagnostics": {
            "raw_candidate_limit": 32,
            "returned_candidate_count": 29,
            "compiled_candidate_count": 3,
            "rejected": [{"rank": 4, "reasons": ["unreachable"]}],
        },
    }
    visible = planner._causal_tool_result("grasp_plan", copy.deepcopy(result))
    assert visible["diagnostics"] == result["diagnostics"]
    assert "backend" not in visible
    assert not _causal_projection_violations([{"role": "tool", "content": json.dumps(visible)}])


def test_operational_keys_stay_stripped_below_the_declared_top_level():
    """``declared`` is not propagated: a nested ``backend`` is still provenance."""

    planner = _planner(surface_of(ROBOTWIN))
    visible = planner._causal_tool_result("grasp_plan", {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-8",
        "candidates": [{"candidate_id": "grasp-" + "b" * 24,
                        "backend": {"id": "leaked"},
                        "diagnostics": {"internal": 1}}],
        "diagnostics": {"raw_candidate_limit": 32},
    })
    assert visible["diagnostics"] == {"raw_candidate_limit": 32}
    candidate = visible["candidates"][0]
    assert "backend" not in candidate and "diagnostics" not in candidate


def test_privilege_filter_still_applies_to_declared_properties():
    """Relaxing the operational filter must not relax the privilege boundary."""

    planner = _planner(surface_of(ROBOTWIN))
    visible = planner._causal_tool_result("grasp_plan", {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-9",
        "candidates": [],
        "diagnostics": {"raw_candidate_limit": 32, "path": "/runs/x", "sha256": "d" * 64},
    })
    assert visible["diagnostics"] == {"raw_candidate_limit": 32}


def test_every_stale_target_refusal_names_the_ref_it_refused():
    """A refusal that does not say which ref went stale cannot be acted on.

    Three tools refuse a target bound to an earlier observation. Two of them
    carried the value and one of those also names the way back; the third --
    the one `align_plan` reaches, and the most-called of the three -- passed
    only `field="target_ref"`, so thirteen refusals in one cohort told the
    policy that something was stale without saying what. The ref is the whole
    content of the refusal.
    """

    import ast
    import inspect
    import sys

    from ora.agent.toolkit.host.runtime import core as runtime_core

    # The runtime is runtime/core.py plus its tool-family mixin modules.
    modules = sorted({
        sys.modules[klass.__module__] for klass in runtime_core.ManipulationToolRuntime.__mro__
        if klass.__module__.startswith("ora.agent.toolkit.host.runtime.")
    }, key=lambda module: module.__name__)
    nodes = [
        node for module in modules
        for node in ast.walk(ast.parse(inspect.getsource(module)))
    ]
    missing = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = getattr(function, "id", None) or getattr(function, "attr", None)
        if name != "ToolError":
            continue
        keywords = {kw.arg for kw in node.keywords}
        code = next(
            (kw.value.value for kw in node.keywords
             if kw.arg == "code" and isinstance(kw.value, ast.Constant)),
            None,
        )
        # A refusal that names no input field is not about something the policy
        # passed -- "observation changed before action persistence" has no ref
        # to report. One that does name a field is refusing a value, and the
        # value is what makes the refusal actionable.
        if (
            code in {"stale_observation", "target_ref_unavailable",
                     "target_geometry_unavailable"}
            and "field" in keywords
            and "received" not in keywords
        ):
            missing.append((code, node.lineno))
    assert not missing, (
        "these target refusals name a field but not the value they refused: "
        f"{missing}"
    )


@pytest.mark.parametrize('profile,reader_name', [
    (LIBERO, 'artifact_read'),
    (ROBOTWIN, 'artifact_read'),
    ('robodojo_placement_toolkit_v5', 'artifact_read'),
])
@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_delivered_reader_matches_pointer_and_recovers_exact_evidence(tmp_path, profile, reader_name, protocol):
    from ora.agent.session_core.gateway import NativeToolGateway
    from ora.agent.memory.markers import (
        recoverable_artifact_envelope, recoverable_artifact_pointer,
        encode_recoverable_artifact_marker, decode_recoverable_artifact_marker,
    )
    surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
    from pathlib import Path
    base = handlers_only(ManipulationToolRuntime(control_root=tmp_path / 'control',
        manual_path=Path(__file__).resolve().parents[4] / 'ora/policy/embodiments/aloha-agilex/manual-v1.md',
        event_path=tmp_path / 'base-events.jsonl', workspace_bounds_m={
            'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]}, tool_surface_profile=profile, toolkit_grants=catalog_grants(profile),
        segmentation_backend=SimpleNamespace(), grounding_backend=SimpleNamespace(),
        grasp_backend=SimpleNamespace()))
    runtime = RecoverableArtifactToolRuntime(base, tmp_path, tmp_path / 'events.jsonl')
    catalog = ContextArtifactCatalog(tmp_path, 'wire-reader', ArtifactStore(tmp_path))
    runtime.bind_context_artifacts(catalog, event_sink=lambda *args, **kwargs: None)
    gateway = NativeToolGateway(runtime)
    definitions = gateway.definitions()  # Actual allowlist filter that dropped the live reader.
    delivered = [item['function']['name'] for item in definitions]
    assert delivered == [item['name'] for item in surface.definitions] + [reader_name]
    raw = b'{"result":{"reachable":false},"note":"exact original evidence"}'
    reference = catalog.put_bytes(raw, kind='context_tool_result', suffix='.json',
        media_type='application/json', round_index=0, action_step=1)
    pointer = recoverable_artifact_pointer(reference)
    assert pointer['open'] == reader_name
    assert decode_recoverable_artifact_marker(encode_recoverable_artifact_marker(
        recoverable_artifact_envelope(reference))) == pointer
    result = gateway.execute(pointer['open'], {'ref': pointer['ref'], 'json_pointer': '/result/reachable'}, call_context={})
    assert result['json_value'] is False
    chunks, offset = [], 0
    while True:
        page = gateway.execute(pointer['open'], {'ref': pointer['ref'], 'offset': offset, 'limit': 11}, call_context={})
        chunks.append(page['content'])
        offset = page['next_offset']
        if page['eof']:
            break
    assert ''.join(chunks).encode() == raw
    with pytest.raises(ToolError) as caught:
        gateway.execute('observe', {'ref': pointer['ref']}, call_context={})
    error = _planner(runtime.surface)._causal_tool_result('observe', _tool_error(caught.value))
    assert error['error']['code'] == 'invalid_arguments'
    assert error['error']['constraints']['open']['tool'] == reader_name
    visible = _planner(runtime.surface)._causal_tool_result(reader_name, result)
    wire = _wire(protocol, [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'read-1', 'type': 'function',
         'function': {'name': reader_name, 'arguments': json.dumps({'ref': pointer['ref'], 'json_pointer': '/result/reachable'})}}]},
        {'role': 'tool', 'name': reader_name, 'tool_call_id': 'read-1', 'content': json.dumps(visible)},
    ], definitions)
    assert '"json_value": false' in json.dumps(wire).replace('\\"', '"')
    assert reference['ref'] in json.dumps(wire)
