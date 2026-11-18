"""H6 source-safe decision feedback and actual protocol delivery."""
import base64
import json

import pytest

from ora.agent.composition import validate_schema_instance
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.session_core.gateway import NativeToolGateway
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.providers.anthropic_request import translate_anthropic_request



def test_batch_result_images_match_candidates_and_current_or_historical_view(tmp_path):
    from tests.support.robotwin.public_run import RoboTwinPublicRun
    from tests.support.agent.batch_segmentation import BatchBackend
    run = RoboTwinPublicRun(tmp_path, toolkit=True)
    # No schema is patched in, so this also checks the runtime honours the
    # batch result contract the resolved surface declares.
    run.runtime.segmentation_backend = BatchBackend()
    result = run.runtime.invoke('vision_segment', {'camera': 'head_camera', 'points_uv': [[20, 20], [60, 60]]})
    assert len(result['images_base64']) == len(result['candidates']) + 1
    for candidate, mapping in zip(result['candidates'], result['image_mapping'][1:]):
        assert mapping['target_ref'] == candidate['target_ref']
        assert mapping['candidate_id'] == candidate['candidate_id']
    run.publish({'left': 'close'})
    old = run.runtime.invoke('observe', {'step': 0, 'views': ['head_camera']})
    session = object.__new__(PolicyDomainMixin)
    session.tools = run.runtime
    projected = session._causal_tool_result('observe', old)
    assert projected['image_mapping'][0]['observation_step'] == 0
    assert projected['measurement_contract'] == old['measurement_contract']
    assert projected['status'] == 'actual_historical_observation'


def test_grasp_global_review_and_two_provider_wire(tmp_path):
    from tests.support.agent.grounding_grasp_runtime import _runtime, _select_correctness_surface
    from ora.agent.session_core.constants import NATIVE_SLOTS_V3
    from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
    from tests.support.agent.observation_delivery import Delivery, host

    runtime, _ = _runtime(tmp_path, toolkit=True)
    _select_correctness_surface(runtime, 'libero_placement_direct_overhead_v1', serve=True)
    target = runtime.invoke('vision_ground', {'camera': 'agentview', 'text': 'red object'})['candidates'][0]['target_ref']
    result = runtime.invoke('grasp_plan', {'target_ref': target})
    assert result['scene_review']['status'] == 'available'
    assert [m['role'] for m in result['image_mapping']] == ['local_contact_candidates', 'global_public_scene_candidates']
    assert len(result['images_base64']) == 2
    # The manifest says what this surface does with the review: it delivers it.
    manifest = json.loads(next(runtime.artifact_root.glob('grasp-queries/*/manifest.json')).read_text())
    assert manifest['review_visualization']['policy_visible'] is True
    assert ('observed_extent' in manifest['review_visualization']) is runtime.surface.supports(
        'observed_extent_scene_review_v1')
    # The tool message and image message the Pi core records for this result.
    planner = host(tmp_path / 'wire', tools=runtime, context_composition=NATIVE_SLOTS_V3)
    delivered = Delivery(planner).tool_result('grasp_plan', result, call_id='call1')
    image_message = delivered['image_message']
    assert image_message is not None
    messages = [{'role': 'user', 'content': 'Inspect the grasp candidates.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
            'function': {'name': 'grasp_plan', 'arguments': json.dumps({'target_ref': target})}}]},
        delivered['message'], image_message]
    request = {'model': 'offline', 'max_tokens': 64, 'messages': messages}
    openai = OpenAICompatibleProvider.prepare_request(request)
    anthropic = translate_anthropic_request(request)
    for body in (openai, anthropic):
        text = json.dumps(body)
        assert 'scene_review' in text and 'global_public_scene_candidates' in text
        assert result['candidates'][0]['candidate_id'] in text
        for encoded in result['images_base64']:
            assert encoded in text


def test_the_live_dojo_review_delivers_the_observed_extent_cell(tmp_path):
    """observed_extent_scene_review_v1 through the real dispatcher and delivery.

    The live RoboDojo surface's image 1 is the 1168x768 review whose top-left
    cell is the observed extent.
    """
    import base64
    import io

    from PIL import Image

    import tests.support.robodojo.closed_skills_default as parity
    from tests.support.agent.reach_check import _stub_bridge
    from tests.support.agent.toolkit_run import on_surface, served

    # The live surface's own grasp client (it fuses the X5 top-down family) against the in-process fake service.
    runtime = on_surface(parity.tool_runtime(tmp_path / 'live', serve=False), 'robodojo_placement_toolkit_v5')
    runtime.align_plan_callback = _stub_bridge([])
    served(runtime)
    target = runtime.invoke('vision_ground', {'camera': 'head', 'text': 'red mug', 'max_candidates': 1})
    result = runtime.invoke('grasp_plan', {'target_ref': target['candidates'][0]['target_ref'], 'arm': 'left'})
    assert [m['role'] for m in result['image_mapping']] == [
        'local_contact_candidates', 'global_public_scene_candidates']
    with Image.open(io.BytesIO(base64.b64decode(result['images_base64'][1]))) as image:
        assert image.size == (1168, 768)
    manifest = json.loads(next(runtime.artifact_root.glob('grasp-queries/*/manifest.json')).read_text())
    review = manifest['review_visualization']
    assert review['policy_visible'] is True
    assert 'review_renderer' in manifest['identity']
    assert review['observed_extent']['status'] == 'available'
    assert review['observed_extent']['camera_order'] == ['head', 'left_wrist']


def test_every_tool_that_passes_the_contract_gate_has_a_contract():
    """The gate and the table were two literals for one fact, and they drifted.

    Canonicalising `enrich`'s input moved the gate to `motion_probe` and left
    the table keyed `motion.probe`, so a `motion.probe` call passed the gate and
    raised `KeyError` on the next line -- reaching a policy as a non-retryable
    `internal_tool_error` on the 47 registered profiles that still declare it,
    and doing it after the tool had already done its work.

    The gate is the table's key set now. This asserts they have not become two
    things again, and that every name in it is one the dispatch can reach.
    """

    import inspect
    import re

    from ora.agent.toolkit.host.perception import feedback as perception_feedback
    from ora.agent.composition.surface import tool_names

    source = inspect.getsource(perception_feedback.enrich)
    table = re.search(r"contracts = \{(.*?)\n        \}", source, re.S)
    assert table, "the measurement-contract table moved; this test lost its subject"
    keys = set(re.findall(r'\n *"([a-z_.]+)": "', table.group(1)))

    assert keys == set(perception_feedback.MEASUREMENT_CONTRACTS)
    for name in keys:
        assert tool_names.canonical(name) == name, (
            f"{name} is the advertised spelling; enrich canonicalises before "
            f"indexing this table, so the key can never be reached"
        )
