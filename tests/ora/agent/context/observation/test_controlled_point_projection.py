"""Where the controlled point (TCP) lands in each public camera image.

A wrist camera is mounted away from the grasp point: on ALOHA the TCP projects
outside its own wrist image, on the Panda 86 px below the image centre. A policy
that aims by centring an object in a view needs that pixel, not the flange
offset alone. Every number here is computed from public calibration and the
public TCP pose of the same observation.
"""
from __future__ import annotations

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.view_config import controlled_point_projection_delivery
from ora.agent.context.observation.public_facts import (
    project_controlled_point,
    project_current_public_facts,
)
from ora.agent.context.causal_value import _causal_projection_violations, causal_provider_value
from tests.support.agent.public_robot_facts import fixture
from tests.support.agent.controlled_point_projection import _dojo_observation

IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
K = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
LIVE = {"libero": "libero_placement_direct_overhead_v1",
        "robotwin": "robotwin_toolchain_b2_closed_noservo_v2"}


def test_a_point_in_front_projects_to_its_pixel() -> None:
    value = project_controlled_point(K, IDENTITY, [0.1, 0.05, 1.0], width=100, height=80)
    assert value["pixel_xy"] == pytest.approx([60.0, 45.0])
    assert value["inside_image"] is True
    assert value["in_front_of_camera"] is True
    assert value["depth_m"] == pytest.approx(1.0)


def test_a_point_off_the_image_is_reported_outside_with_its_pixel() -> None:
    # The measured ALOHA wrist geometry: (3.2, 4.0, 7.1) cm in camera frame.
    aloha_k = [[287.0, 0.0, 160.0], [0.0, 287.0, 120.0], [0.0, 0.0, 1.0]]
    value = project_controlled_point(aloha_k, IDENTITY, [0.032, 0.040, 0.071], width=320, height=240)
    assert value["inside_image"] is False
    assert value["pixel_xy"][1] > 240


def test_a_point_behind_the_camera_has_no_pixel() -> None:
    value = project_controlled_point(K, IDENTITY, [0.0, 0.0, -0.5], width=100, height=80)
    assert value == {"pixel_xy": None, "inside_image": False, "in_front_of_camera": False,
                     "depth_m": pytest.approx(-0.5)}


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_each_camera_carries_every_arm_only_when_delivered(tmp_path, backend) -> None:
    surface, observation, source = fixture(tmp_path, backend)
    arms = sorted(observation["proprioception"]["end_effectors"])
    delivered = project_current_public_facts(
        observation=observation, observation_root=source, view_config=surface.view_config,
        controlled_point_projection=True)
    for camera in delivered["cameras"].values():
        pixels = camera["controlled_point_pixels"]
        assert sorted(pixels) == arms
        # TCP (0, 0, 1) world; the fixture camera sits 1 m along +x.
        assert pixels[arms[0]]["pixel_xy"] == pytest.approx([16.0, 5.0])
        assert pixels[arms[0]]["inside_image"] is False
    assert not _causal_projection_violations(causal_provider_value(delivered))
    silent = project_current_public_facts(
        observation=observation, observation_root=source, view_config=surface.view_config)
    assert all("controlled_point_pixels" not in camera for camera in silent["cameras"].values())


def _camera_entry_schemas(node):
    found = []
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict) and "native_pixel_to_image_offset_xy" in properties:
            found.append(node)
        for value in node.values():
            found.extend(_camera_entry_schemas(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_camera_entry_schemas(value))
    return found


@pytest.mark.parametrize("profile,delivered", [
    (LIVE["libero"], True), (LIVE["robotwin"], True),
])
def test_only_declaring_surfaces_deliver_and_admit_the_pixels(profile, delivered) -> None:
    surface = surface_of(profile)
    assert controlled_point_projection_delivery(surface.view_config) is delivered
    entries = _camera_entry_schemas(dict(surface.result_schemas))
    assert entries
    for entry in entries:
        assert ("controlled_point_pixels" in entry["properties"]) is delivered


# RoboDojo stack_blocks layout-0 observation-0000 of run
# exp11a-robodojo-3eb59643d917--20260926T231016Z: native 640x480 cameras, the
# published intrinsic_cv / 4x4 extrinsic_cv and the measured link6 poses.
DOJO = "robodojo_placement_toolkit_v5"


def test_dojo_live_surface_projects_link6_and_the_jaw_contact_centre(tmp_path) -> None:
    from ora.agent.context.observation.contact_points import (
        jaw_contact_center_delivery, project_policy_public_facts,
    )
    from ora.agent.toolkit.host.reach.jaw_center import jaw_contact_center_offsets
    from ora.agent.composition.surface.tool_surface import validate_schema_instance

    surface, observation, source = _dojo_observation(tmp_path)
    assert controlled_point_projection_delivery(surface.view_config)
    assert jaw_contact_center_delivery(surface.view_config)
    offsets = jaw_contact_center_offsets(surface.view_config)
    assert offsets == {"left": pytest.approx([0.1501, 0, 0]), "right": pytest.approx([0.1501, 0, 0])}
    facts = project_policy_public_facts(
        observation=observation, observation_root=source, view_config=surface.view_config,
        controlled_point_projection=True)
    cameras = facts["cameras"]
    assert sorted(cameras) == ["head", "left_wrist", "right_wrist"]
    for name in cameras:
        assert sorted(cameras[name]["controlled_point_pixels"]) == ["left", "right"]
        assert sorted(cameras[name]["jaw_center_pixels"]) == ["left", "right"]
    # Both flanges and both contact centres are in the head view.
    head = cameras["head"]
    assert head["controlled_point_pixels"]["left"]["pixel_xy"] == pytest.approx([82.6, 353.6], abs=0.1)
    assert head["jaw_center_pixels"]["left"]["pixel_xy"] == pytest.approx([123.2, 248.7], abs=0.1)
    assert all(head[key][arm]["inside_image"] for key in ("controlled_point_pixels", "jaw_center_pixels")
               for arm in ("left", "right"))
    # link6 sits 4.8 cm behind its own wrist camera: no pixel. The contact
    # centre is 8.2 cm in front, 56 px below the image centre.
    for arm in ("left", "right"):
        own = cameras[f"{arm}_wrist"]
        flange = own["controlled_point_pixels"][arm]
        assert flange["pixel_xy"] is None and flange["in_front_of_camera"] is False
        assert flange["depth_m"] == pytest.approx(-0.048, abs=0.001)
        centre = own["jaw_center_pixels"][arm]
        assert centre["inside_image"] is True and centre["depth_m"] == pytest.approx(0.082, abs=0.001)
        assert centre["pixel_xy"] == pytest.approx([320.0, 295.6], abs=0.2)
    assert not _causal_projection_violations(causal_provider_value(facts))
    # The live observe result schema admits the delivered entry.
    validate_schema_instance(head, surface.result_schemas["observe"]["$defs"]["camera"] | {
        "$defs": surface.result_schemas["observe"]["$defs"]})


def test_dojo_live_surface_admits_both_pixel_fields() -> None:
    live = surface_of(DOJO)
    entries = _camera_entry_schemas(dict(live.result_schemas))
    assert entries and all({"controlled_point_pixels", "jaw_center_pixels"} <= set(entry["properties"])
                           for entry in entries)
