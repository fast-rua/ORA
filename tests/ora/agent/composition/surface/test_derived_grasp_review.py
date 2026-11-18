"""Derived review metadata survives result admission and the next provider request."""
import json
from pathlib import Path

import pytest

from ora.agent.composition.surface import tool_names

from ora.agent.composition.surface.tool_surface import resolve_tool_surface, validate_schema_instance, ToolSurfaceError
from ora.agent.session_core.constants import NATIVE_SLOTS_V3
from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.anthropic_request import translate_anthropic_request
from tests.support.agent.observation_delivery import Delivery, host

ROOT = Path(__file__).resolve().parents[5]
SURFACES = {'libero': 'libero_placement_direct_overhead_v1',
            'robotwin': 'robotwin_toolchain_b2_closed_noservo_v2'}


def assert_next_request(tmp_path, runtime, result):
    # The tool message and image message the Pi core records for this result.
    planner = host(tmp_path / 'wire', tools=runtime, context_composition=NATIVE_SLOTS_V3)
    delivered = Delivery(planner).tool_result('grasp_plan', result, call_id='call1')
    messages = [{'role': 'user', 'content': 'Inspect candidates.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
            'function': {'name': 'grasp_plan', 'arguments': '{}'}}]},
        delivered['message'], delivered['image_message']]
    request = {'model': 'offline', 'max_tokens': 64, 'messages': messages}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        text = json.dumps(body)
        assert 'global_public_scene_candidates' in text
        assert result['candidates'][0]['candidate_id'] in text
        assert 'internal_tool_error' not in text
        for encoded in result['images_base64']:
            assert encoded in text


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_complete_grasp_result_reaches_next_provider_request(tmp_path, backend):
    from tests.support.agent.grounding_grasp_runtime import _runtime, _select_correctness_surface
    from tests.support.agent.toolkit_run import recameraed
    if backend == 'libero':
        runtime, _ = _runtime(tmp_path, toolkit=True)
        _select_correctness_surface(runtime, SURFACES['libero'], serve=True)
        target = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'vision_ground'), {'camera': 'agentview', 'text': 'red object'})['candidates'][0]['target_ref']
        args = {'target_ref': target}
    else:
        runtime, _ = _runtime(tmp_path, toolkit=True)
        _select_correctness_surface(runtime, SURFACES['robotwin'], serve=True)
        recameraed(runtime, {'agentview': 'head_camera'})
        target = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'vision_ground'), {'camera': 'head_camera', 'text': 'red object'})['candidates'][0]['target_ref']
        args = {'arm': 'right', 'target_ref': target}
    result = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'grasp_plan'), args)
    assert result.get('scene_review', {}).get('status') == 'available', result
    assert result['candidates']
    assert [m['camera'] for m in result['image_mapping']] == ['', '']
    validate_schema_instance({k: v for k, v in result.items() if k != 'images_base64'}, runtime.surface.result_schemas[
        tool_names.as_advertised(runtime.surface.result_schemas, 'grasp_plan')])
    assert_next_request(tmp_path, runtime, result)


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_derived_metadata_does_not_open_physical_camera_boundary(backend):
    surface = resolve_tool_surface(SURFACES[backend])
    schema = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'grasp_plan')]['properties']['image_mapping']
    item = {'image_index': 0, 'role': 'local_contact_candidates', 'camera': '',
            'candidate_id': '', 'target_ref': '', 'observation_step': None}
    validate_schema_instance([item], schema)
    for change in ({'role': 'saved_public_view'}, {'camera': 'observer_camera'}, {'camera': 'invented_camera'}):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance([{**item, **change}], schema)
    observe = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'observe')]['properties']['image_mapping']
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance([{**item, 'role': 'saved_public_view'}], observe)


def test_the_named_pointcloud_remains_usable():
    # LIBERO's default observe names its point cloud. The derived-review tools
    # (grasp_verify_retention, place_plan) left with the last surface that
    # declared them.
    surface = resolve_tool_surface(SURFACES['libero'])
    item = {'image_index': 0, 'role': 'saved_public_view', 'camera': 'pointcloud_top',
            'candidate_id': '', 'target_ref': '', 'observation_step': 0}
    validate_schema_instance([item], surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'observe')]['properties']['image_mapping'])


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_mark_point_binds_the_batch_panel_mapping_instead_of_the_derived_review_one(backend):
    """These surfaces publish the batched mark_point, which names its own panel.

    The derived-review entry mark_point used to carry was the label
    `perception_feedback.enrich` applies when a result has images and no mapping
    of its own. A batched call always builds one, and it says what the image
    actually is -- one composite panel, bound to a `source_index` rather than to
    its position. On LIBERO that also ends `pointcloud_top` appearing here,
    which is the same loss as the batch refusing a point-cloud view to mark on:
    recorded in `docs/reference/public-interfaces.md`, not silent.
    """
    surface = resolve_tool_surface(SURFACES[backend])
    schema = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'mark_point')]['properties']['image_mapping']
    panel = {'image_index': 0, 'role': 'mark_batch_panel', 'camera': 'agentview',
             'source_index': 0, 'candidate_id': None, 'target_ref': None,
             'observation_step': 0}
    validate_schema_instance([panel], schema)
    validate_schema_instance([{**panel, 'source_index': None}], schema)
    for change in ({'role': 'public_geometry_review'}, {'role': 'saved_public_view'}):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance([{**panel, **change}], schema)
