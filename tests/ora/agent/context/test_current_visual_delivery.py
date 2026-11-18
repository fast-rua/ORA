"""Complete public RGB-D reaches the policy's observation and both provider wires."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PIL import Image

from ora.agent.composition.baseline import load_native_baseline
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.evidence.image_binding import PostActionObservationError, resolve_post_action_images
from ora.agent.context.observation.projector import ContextProjector
from ora.agent.context.observation.current_visual import materialize_current_visual
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.anthropic_request import translate_anthropic_request
from ora.environment.synthetic import SyntheticSession
from tests.support.agent.observation_delivery import Delivery, host, observation_text
from tests.support.agent.current_visual_delivery import ROOT, published, delivery_for
from tests.support.agent.provider_messages import provider_message


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
@pytest.mark.parametrize("after_action", [False, True])
def test_all_current_rgb_and_calibration_reach_both_native_protocols(tmp_path, backend, after_action):
    context, _, _, surface, source = published(tmp_path, backend, after_action=after_action)
    message, _ = delivery_for(tmp_path, context, surface).observation(context)
    current = observation_text(message)["current_actual"]
    cameras = load_native_baseline(ROOT)["instances"][backend]["runtime"]["policy_cameras"]
    from ora.agent.context.causal_value import causal_provider_value
    # The delivered facts equal the causal facts.
    assert current["public_robot_facts"] == causal_provider_value(context["public_robot_facts"])
    grip = current["public_robot_facts"]["end_effectors"]["right"]["gripper"]
    if backend == "libero_pro":
        assert grip["measured_aperture_mm"] == 40
        assert grip["measured_finger_displacements_m"] == [.012, -.028]
    else:
        assert grip["commanded_opening_normalized"] == 1
        assert grip["measured_opening_normalized"] == .18181818
    assert set(current["camera_geometry"]) == set(cameras)
    assert len([b for b in message["content"] if b["type"] == "image_url"]) == len(cameras)
    assert all(dim == {"width": 4, "height": 3} for dim in current["view_dimensions"].values())
    assert "object_gt_pose" not in json.dumps(current)
    for index, camera in enumerate(cameras):
        geometry = current["camera_geometry"][camera]
        assert geometry["intrinsic_cv"][0][0] == 50.25 + index
        # observation_delivery_trim_v2 keeps every pose representation
        # (v1's single-pose rule is covered in test_delivery_trim_runtime).
        assert geometry["cam2world_cv"][0][3] == pytest.approx(.123 + index)
        assert geometry["camera_origin_world_m"] == pytest.approx([.123 + index, .234, 1.345])
        assert geometry["camera_axes_world"]["forward"] == [0, 0, 1]
    from ora.agent.context.causal_value import _causal_projection_violations
    from ora.agent.evidence.trace_privacy import _forbidden_path
    assert _causal_projection_violations(message["content"]) == []
    assert _forbidden_path(message["content"]) is None
    request = {"model": "offline", "max_tokens": 64, "messages": [message]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        wire = json.dumps(body)
        for camera in cameras:
            assert base64.b64encode((source / camera / "rgb.png").read_bytes()).decode() in wire
        assert "50.25" in wire and "1.345" in wire
        assert "public_robot_facts" in wire and "measured_opening_normalized" in wire
    images = resolve_post_action_images(context, root=tmp_path, path_key="run_path")
    assert {image["modality"] for image in images} == {"rgb"}
    assert context["current_visual_observation"]["depth_delivery"] == "on_demand"


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
@pytest.mark.parametrize("after_action", [False, True])
def test_native_slots_v4_delivers_every_camera_through_the_observation_and_its_slot(
    tmp_path, backend, after_action,
):
    """D3b: every image stays in the observation; calibration sits in the round slot."""
    from ora.agent.context.delivery import round_static_facts as slot

    context, _, _, surface, source = published(tmp_path, backend, after_action=after_action)
    context = {**context, "remaining_budget": {"session_round": 1, "session_rounds": 1}}
    legacy, _ = delivery_for(tmp_path, context, surface).observation(context)
    message, slot_message = delivery_for(tmp_path, context, surface,
                                         context_composition=slot.COMPOSITION).observation(context)
    cameras = load_native_baseline(ROOT)["instances"][backend]["runtime"]["policy_cameras"]
    images = [block for block in message["content"] if block["type"] == "image_url"]
    assert images == [block for block in legacy["content"] if block["type"] == "image_url"]
    assert len(images) == len(cameras)
    current = observation_text(message)["current_actual"]
    # observation_delivery_trim_v2: an omitted field is not named; only a slot
    # field with no current value would be listed.
    assert "camera_geometry" not in current
    assert current["static_facts"] == {"round_slot": 1}
    assert slot_message is not None
    fields = json.loads(slot_message["content"][0]["text"])["round_static_facts"]["current_actual"]
    restored = slot.restore(current, fields)
    assert restored == observation_text(legacy)["current_actual"]
    assert set(restored["camera_geometry"]) == set(cameras)
    from ora.agent.context.causal_value import _causal_projection_violations
    assert _causal_projection_violations(
        [provider_message(message), provider_message(slot_message)]) == []
    request = {"model": "offline", "max_tokens": 64, "messages": [message, slot_message]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        wire = json.dumps(body)
        for camera in cameras:
            assert base64.b64encode((source / camera / "rgb.png").read_bytes()).decode() in wire
        assert "50.25" in wire and "1.345" in wire


def test_current_visual_requires_bound_unmodified_sensor_artifacts(tmp_path):
    context, _, _, _, _ = published(tmp_path, "libero_pro")
    record = context["current_visual_observation"]
    image = tmp_path / record["images"][0]["run_path"]
    image.chmod(0o644)
    image.write_bytes(b"tampered")
    with pytest.raises(PostActionObservationError, match="hash"):
        resolve_post_action_images(context, root=tmp_path, path_key="run_path")
    stale = copy.deepcopy(context)
    stale["policy_state"]["step"] += 1
    with pytest.raises(PostActionObservationError, match="current policy state"):
        resolve_post_action_images(stale, root=tmp_path, path_key="run_path")


def test_missing_declared_camera_and_modified_calibration_cannot_silently_deliver_subset(tmp_path):
    context, projector, store, _, source = published(tmp_path, "libero_pro")
    observation = json.loads((source / "observation.json").read_text())
    args = dict(run_root=tmp_path, workspace=projector.workspace, source_root=source,
        policy_state=store.steps[-1], observation_id=observation["observation_id"],
        observation_sha256=context["current_visual_observation"]["observation_sha256"],
        required_views=("agentview", "wrist"))
    with pytest.raises(PostActionObservationError, match="declared current camera"):
        materialize_current_visual(cameras={"agentview": observation["cameras"]["agentview"]}, **args)
    path = source / "wrist/calibration.json"
    path.write_text("{}")
    with pytest.raises(PostActionObservationError, match="hash mismatch"):
        materialize_current_visual(cameras=observation["cameras"], **args)


def test_robotwin_default_excludes_observer_at_collection_tools_and_actual_input(tmp_path):
    from ora.agent.composition.surface.tool_surface import validate_schema_instance, ToolSurfaceError
    from ora.orchestrator.spec.defaults import expand_native_defaults
    from tests.support.orchestrator.run_specs import native_spec
    expected = ["head_camera", "front_camera", "left_camera", "right_camera"]
    for kind in ("native_api",):
        spec = native_spec("robotwin")
        spec["policy"]["kind"] = kind
        expanded = expand_native_defaults(spec, root=ROOT)
        assert expanded["runtime"]["policy_cameras"] == expected
        assert expanded["runtime"]["policy_observer_camera"] is False
        assert expanded["runtime"]["main_camera"] == "head_camera"
        surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
        assert surface.supports("current_public_robot_facts_v1")
        from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
        descriptor_path = ROOT / surface.view_config["public_geometry_descriptor"]["path"]
        compiled_card = render_embodiment_card(load_embodiment_descriptor(descriptor_path))
        context_paths = expanded["policy_context"]["project_instructions"]
        assert any((ROOT / path).read_text() == compiled_card for path in context_paths), (
            "selected public descriptor must deliver its exact compiled card")
        assert not surface.supports("policy_observer_camera_v1")
        assert "observer_camera" not in json.dumps(surface.definitions)
        assert "observer_camera" not in json.dumps(surface.result_schemas)
        schemas = {tool["name"]: tool["inputSchema"] for tool in surface.definitions}
        for name, arguments in {
            "observe": {"views": ["observer_camera"]},
            "vision_segment": {"camera": "observer_camera", "points_uv": [[1, 1]]},
            "vision_ground": {"camera": "observer_camera", "text": "cup"},
            "mark_point": {"point_id": "p", "view": "observer_camera", "x": 1, "y": 1},
        }.items():
            with pytest.raises(ToolSurfaceError):
                validate_schema_instance(arguments, schemas[name], root_path="$input")
    context, projector, store, _, source = published(tmp_path, "robotwin")
    observation = json.loads((source / "observation.json").read_text())
    observation["cameras"]["observer_camera"] = observation["cameras"]["head_camera"]
    with pytest.raises(PostActionObservationError, match="undeclared camera"):
        materialize_current_visual(run_root=tmp_path, workspace=projector.workspace,
            source_root=source, cameras=observation["cameras"], policy_state=store.steps[-1],
            observation_id=observation["observation_id"], observation_sha256=None,
            required_views=tuple(expected))


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
def test_calibration_numbers_have_exactly_one_owner_in_the_delivered_observation(tmp_path, backend):
    """Two modules render the same calibration artifact; only one may publish it.

    Before this contract, ``camera_geometry`` and ``public_robot_facts`` each
    wrote the same intrinsic and world-to-camera matrices into the same
    ``current_actual``. They were byte-identical on every observation of every
    request, so the second copy bought the policy nothing and cost it context.
    ``camera_geometry`` is the richer producer - it alone derives
    ``cam2world_cv``, ``camera_origin_world_m`` and ``camera_axes_world`` - so
    it keeps the numbers, and the facts keep the pixel-space mapping they alone
    own plus an explicit pointer to the owner.
    """
    context, _, _, surface, _ = published(tmp_path, backend)
    message, _ = delivery_for(tmp_path, context, surface).observation(context)
    current = observation_text(message)["current_actual"]
    geometry = current["camera_geometry"]
    facts = current["public_robot_facts"]["cameras"]
    assert set(geometry) == set(facts) == set(surface.view_config["automatic_actual_views"])
    # A bare status reads as "unavailable" to a policy, so the payload names the
    # owner -- once, beside cameras, because it is the same sentence for every
    # camera but the name and this object carries one entry per camera per
    # retained observation.
    owner = current["public_robot_facts"]["calibration_owner"]
    assert owner == {"field": "camera_geometry.<camera>",
                     "intrinsic_cv": "intrinsic_cv",
                     "world_to_camera_cv": "extrinsic_cv"}
    for name, entry in facts.items():
        calibration = entry["calibration"]
        assert calibration["status"] == "reported_in_camera_geometry"
        assert set(calibration) == {"status", "pixel_basis"}
        assert calibration["pixel_basis"] == "native_rgb_before_sheet_translation"
        # The once-stated mapping resolves against this very payload, for each
        # camera, substituting that entry's own key for <camera>.
        resolved = current
        for step in owner["field"].replace("<camera>", name).split("."):
            resolved = resolved[step]
        assert isinstance(resolved[owner["intrinsic_cv"]], list)
        assert len(resolved[owner["intrinsic_cv"]]) == 3
        assert len(resolved[owner["world_to_camera_cv"]]) == 4
        # The facts keep exactly the pixel-space mapping they uniquely own.
        assert {"width", "height", "image_view", "native_pixel_to_image_offset_xy",
                "resampling", "pixel_coordinates"} <= set(entry)
    # Each matrix now appears once in the whole delivered observation.
    delivered = json.dumps(current)
    for name in geometry:
        for matrix in (geometry[name]["intrinsic_cv"], geometry[name]["extrinsic_cv"]):
            assert delivered.count(json.dumps(matrix)) == 1
    from ora.agent.context.causal_value import _causal_projection_violations
    from ora.agent.evidence.trace_privacy import _forbidden_path
    assert _causal_projection_violations(message["content"]) == []
    assert _forbidden_path(message["content"]) is None


def test_post_action_image_bytes_are_verified_and_legacy_is_unchanged(tmp_path):
    from ora.agent.context.observation.action_views import materialize_post_action_observation
    from tests.support.agent.observation_fixtures import observation_fixture

    assert resolve_post_action_images({}, root=tmp_path) == []
    observation, source = observation_fixture(tmp_path, ["head"])
    projected = materialize_post_action_observation(
        run_root=tmp_path, workspace=tmp_path / "workspace", observation=observation,
        observation_root=source, policy_state={"step": observation["sequence"],
            "state_sha256": "state-hash", "observation_id": observation["observation_id"]},
        previous_action={"schema": "ora-action.v3", "action": "move_pose", "action_id": "action-1",
            "observation_id": "observation-0", "arguments": {}},
        view_config={"automatic_actual_views": ["head"], "rgb_resolution": "native",
            "current_observation_delivery": "all_public_v1"},
        camera_roles={},
    )
    delivered = tmp_path / "workspace" / projected["images"][0]["path"]
    delivered.chmod(0o644)
    delivered.write_bytes(b"different")
    with pytest.raises(PostActionObservationError, match="hash"):
        resolve_post_action_images({"post_action_observation": projected}, root=tmp_path / "workspace")


def test_policy_turn_immutable_multi_image_extension_preserves_legacy_kwargs(tmp_path):
    from ora.agent.contracts import PolicyTurnInput

    args = dict(task_id="task", task_instruction="Act", context={}, image_path=None,
        observation_id="obs", observation_step=1)
    assert "image_paths" not in PolicyTurnInput(**args).runtime_kwargs()
    paths = [tmp_path / "head.png", tmp_path / "wrist.png"]
    turn = PolicyTurnInput(**args, image_paths=paths)
    paths.clear()
    assert turn.runtime_kwargs()["image_paths"] == (tmp_path / "head.png", tmp_path / "wrist.png")
