"""observed_clearance on the ALOHA and Panda jaws (0059 M4, ``observed_clearance_reference_v2``).

v1 reads ``reference_parameters`` only the ARX X5 descriptor publishes and two
encoder finger positions. v2 keeps v1's result shape and measurement and reads
each embodiment's own public facts: on the grip-centre frames the jaw centre is
the controlled point (zero offset), Panda fingers open by their signed
``gripper_finger_qpos`` (finger_2's opening direction is the negative joint
axis), ALOHA fingers by half the measured ``gripper_opening_m``, wrist cameras
are those attached to the gripper, finger roots lie back along each frame's
approach axis, and the jaws hold when the last command was close and the
measured jaw separation exceeds the Harness near-closed bound (3 mm).
Offline: synthetic top-down depth scenes in each backend's camera vocabulary
and the default surfaces that declare it.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.schemas.observed_clearance import PROPERTY, TOOLS
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.surface.tool_surface import validate_schema_instance
from ora.agent.toolkit.host.motion_rules import clearance as oc
from ora.agent.toolkit.host.motion_rules import clearance_runtime as rt
from ora.agent.toolkit.host.perception.localization import NEAR_CLOSED_M
from ora.agent.toolkit.host.motion_rules.clearance_geometry import load_jaw_geometry_v2
from ora.agent.composition.embodiment import jaw_models
from ora.agent.composition.embodiment.jaw_models import JAW_MODELS
from tests.support.agent.observed_clearance_v2 import (
    RIM_TOP, ALOHA_DOWN, PANDA_DOWN, _write_scene, _aloha, _panda, ROBOTWIN_CAMERAS,
    LIBERO_CAMERAS,
)

#: The capability whose field these tests hold to its runtime.
CAPABILITY = "observed_clearance_reference_v2"
ROOT = Path(__file__).resolve().parents[5]
#: The default surface of each backend whose jaw v2 adds (the X5 keeps v1's loader).
DEFAULTS = {"robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
            "libero_pro": "libero_placement_direct_overhead_v1"}


def _descriptor(profile):
    binding = surface_of(profile).view_config["public_geometry_descriptor"]
    return ROOT / binding["path"]


ALOHA = _descriptor("robotwin_toolchain_b2_closed_noservo_v2")
PANDA = _descriptor("libero_placement_direct_overhead_v1")
X5 = _descriptor("robodojo_placement_toolkit_v5")


# --------------------------------------------------------------------------
# Jaw models
# --------------------------------------------------------------------------

def test_the_aloha_jaw_is_its_tcp_with_gripper_mounted_wrist_cameras():
    geometry = load_jaw_geometry_v2(ALOHA, "aloha_grip_center_tcp")
    assert set(geometry) == {"left", "right"}
    left = geometry["left"]
    assert left.contact_center_local == (0.0, 0.0, 0.0)
    assert left.wrist_cameras == ("left_camera",) and geometry["right"].wrist_cameras == ("right_camera",)
    assert (left.finger_opening, left.hold_rule) == (jaw_models.OPENING_SYMMETRIC, jaw_models.HOLD_SEPARATION)
    assert left.near_closed_m == NEAR_CLOSED_M and left.empty_close_reference is None
    assert left.root_face == (0, 1, 2, 3)  # approach local +x, as on the X5
    assert rt._qpos(_aloha([0] * 7, opening_m=0.04), left) == pytest.approx([0.02, 0.02])
    with pytest.raises(ValueError, match="separation is unavailable"):
        rt._qpos({**_aloha([0] * 7), "gripper_opening_m": None}, left)


def test_the_panda_jaw_is_its_grip_site_and_finger_two_opens_against_its_joint_axis():
    from ora.environment.backends.libero_pro import robot_asset_geometry

    geometry = load_jaw_geometry_v2(PANDA, "panda_grip_site")
    right = geometry["right"]
    assert set(geometry) == {"right"} and right.contact_center_local == (0.0, 0.0, 0.0)
    assert right.wrist_cameras == ("wrist",)
    assert right.finger_signs == (1.0, -1.0)
    # The extractor publishes finger_2's opening direction as the negative joint axis.
    assert "for index, sign in ((1, 1), (2, -1)):" in inspect.getsource(robot_asset_geometry.panda_geometry_from_model)
    assert rt._qpos(_panda([0] * 7, finger=0.03), right) == pytest.approx([0.03, 0.03])
    assert right.root_face == (0, 2, 4, 6)  # approach local +z: the low-z face is the root
    with pytest.raises(ValueError, match="finger positions"):
        rt._qpos({"gripper_finger_qpos": [0.01]}, right)


def test_the_x5_keeps_its_released_jaw_and_v1_defaults_are_unchanged():
    assert load_jaw_geometry_v2(X5, "arx_x5_link6") == rt.load_jaw_geometry(X5)
    released = rt.load_jaw_geometry(X5)["left"]
    assert (released.finger_opening, released.hold_rule, released.root_face) == (
        jaw_models.OPENING_ENCODER, jaw_models.HOLD_ENCODER, oc.ROOT_FACE)
    with pytest.raises(ValueError, match="no observed-clearance jaw model"):
        load_jaw_geometry_v2(X5, "unknown_frame")
    assert set(JAW_MODELS) == {"arx_x5_link6", "panda_grip_site", "aloha_grip_center_tcp", "robotiq_2f85_tcp",
                               "yam_linear_4310_tcp"}


@pytest.mark.parametrize("effector,holds", [
    (_aloha([0] * 7, close=True, opening_m=0.03), True),
    (_aloha([0] * 7, close=True, opening_m=0.002), False),
    (_aloha([0] * 7, close=False, opening_m=0.03), False),
    (_panda([0] * 7, close=True, finger=0.01), True),
    (_panda([0] * 7, close=True, finger=0.001), False),
])
def test_the_jaws_hold_above_the_near_closed_separation_after_a_close(effector, holds):
    convention, path = (("panda_grip_site", PANDA) if "gripper_finger_qpos" in effector
                        else ("aloha_grip_center_tcp", ALOHA))
    geometry = next(iter(load_jaw_geometry_v2(path, convention).values()))
    assert rt.jaws_hold(effector, geometry, {}) is holds


def test_root_faces_follow_the_approach_axis():
    assert oc.root_face([1, 0, 0]) == (0, 1, 2, 3)
    assert oc.root_face([0, 0, 1]) == (0, 2, 4, 6)
    assert oc.root_face([-1, 0, 0]) == (4, 5, 6, 7)
    solids = (("fingers", oc.box_corners([0, 0, 0], [1, 1, 1])),)
    default = oc.robot_volume(solids, controlled_point=[0.5, 0.5, 2.0])
    explicit = oc.robot_volume(solids, controlled_point=[0.5, 0.5, 2.0], root_indices=oc.ROOT_FACE)
    assert np.array_equal(default.hull_xy, explicit.hull_xy)
    assert (default.floor_z, default.body_floor_z) == (explicit.floor_z, explicit.body_floor_z)
    # A finger box whose root face is its high-z face (a Panda finger pointing
    # down, seen in world): the body prism starts at the root, not the tip.
    turned = oc.robot_volume(solids, controlled_point=[0.5, 0.5, 2.0], root_indices=(1, 3, 5, 7))
    assert (default.body_floor_z, turned.body_floor_z) == (0.0, 1.0)


# --------------------------------------------------------------------------
# Measurements in each backend's vocabulary
# --------------------------------------------------------------------------

def test_an_aloha_target_below_the_observed_rim_reports_it(tmp_path):
    geometry = load_jaw_geometry_v2(ALOHA, "aloha_grip_center_tcp")
    observation = _write_scene(tmp_path / "obs", ROBOTWIN_CAMERAS, {
        "left": _aloha([-0.3, -0.3, 1.1, *ALOHA_DOWN]), "right": _aloha([0.6, -0.3, 1.1, *ALOHA_DOWN])})
    tip = max(finger.maximum[0] for finger in geometry["left"].fingers)  # fingertips beyond the TCP
    target = [0.0, 0.0, 0.80 + tip, *ALOHA_DOWN]
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="left", target_pose=target,
                                     geometry=geometry, held={})
    assert result["status"] == "measured"
    assert result["footprint_lowest_m"][2] == pytest.approx(0.80, abs=1e-4)
    assert result["clearance_m"] == pytest.approx(0.80 - RIM_TOP, abs=0.005)
    validate_schema_instance(result, PROPERTY)


def test_a_panda_target_above_the_rim_reports_positive_clearance(tmp_path):
    geometry = load_jaw_geometry_v2(PANDA, "panda_grip_site")
    observation = _write_scene(tmp_path / "obs", LIBERO_CAMERAS, {"right": _panda([0.3, 0.3, 1.1, *PANDA_DOWN])})
    tip = max(finger.maximum[2] for finger in geometry["right"].fingers)
    target = [0.0, 0.0, 0.90 + tip, *PANDA_DOWN]
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="right", target_pose=target,
                                     geometry=geometry, held={})
    assert result["status"] == "measured"
    assert result["footprint_lowest_m"][2] == pytest.approx(0.90, abs=1e-4)
    assert result["clearance_m"] == pytest.approx(0.90 - RIM_TOP, abs=0.005)


def test_the_aloha_held_extent_counts_only_while_the_jaws_measure_open_on_it(tmp_path):
    geometry = load_jaw_geometry_v2(ALOHA, "aloha_grip_center_tcp")
    attach = [0.0, 0.0, 1.0, *ALOHA_DOWN]
    localization = {"status": "localized", "aabb_min_xyz_m": [-0.02, -0.02, 0.88],
                    "aabb_max_xyz_m": [0.02, 0.02, 0.96]}
    corners, recentered = rt.held_local_corners(localization, attach, geometry["left"].contact_center_local)
    assert recentered is False  # the grip-centre TCP is over the object
    held = {"left": rt.HeldExtent(target_ref="target-abc", local_corners=corners)}
    target = [0.2, 0.0, 1.2, *ALOHA_DOWN]
    for name, opening, included in (("holding", 0.03, True), ("closed-empty", 0.001, False)):
        observation = _write_scene(tmp_path / name, ROBOTWIN_CAMERAS, {
            "left": _aloha(attach, close=True, opening_m=opening), "right": _aloha([0.6, -0.3, 1.1, *ALOHA_DOWN])})
        result = rt.annotate_observation(observation, tmp_path / name, arm="left", target_pose=target,
                                         geometry=geometry, held=held)
        assert (result["held_object"] is not None) is included, name
        assert (result["footprint_lowest_part"] == "held_object") is included, name


# --------------------------------------------------------------------------
# Surfaces and the Tool Runtime edge
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["robotwin", "libero_pro"])
def test_the_defaults_bind_the_released_field_on_every_pose_tool(backend):
    surface = surface_of(DEFAULTS[backend])
    assert surface.supports(CAPABILITY)
    for name in TOOLS:
        target = surface.result_schemas[name]["properties"]["compiled_pose"]["properties"]["targets"]
        assert target["additionalProperties"]["properties"]["observed_clearance"] == PROPERTY, name
    help_text = next(item["description"] for item in surface.definitions if item["name"] == "move_pose")
    assert f"measured jaw separation exceeds {round(NEAR_CLOSED_M * 1000)} mm" in help_text
    assert "does not confirm a grasp" in help_text


def test_the_runtime_edge_measures_on_the_robotwin_default(tmp_path):
    surface = surface_of(DEFAULTS["robotwin"])
    host = SimpleNamespace(surface=surface, artifact_root=tmp_path / "artifacts")
    observation = _write_scene(tmp_path / "obs", ROBOTWIN_CAMERAS, {
        "left": _aloha([-0.3, -0.3, 1.1, *ALOHA_DOWN]), "right": _aloha([0.6, -0.3, 1.1, *ALOHA_DOWN])})
    value = rt.reference(host, "left", [0.0, 0.0, 0.95, *ALOHA_DOWN], observation=observation,
                         root=tmp_path / "obs")["observed_clearance"]
    assert value["status"] == "measured" and value["held_object"] is None
    target = surface.result_schemas["move_pose"]["properties"]["compiled_pose"]["properties"]["targets"]
    validate_schema_instance(value, target["additionalProperties"]["properties"]["observed_clearance"])
    # A missing jaw measurement is reported, never raised into the move.
    broken = copy.deepcopy(observation)
    broken["proprioception"]["end_effectors"]["right"]["gripper_opening_m"] = None
    assert rt.reference(host, "left", [0.0, 0.0, 0.95, *ALOHA_DOWN], observation=broken,
                        root=tmp_path / "obs")["observed_clearance"]["reason"] == "reference_failed"


def test_only_the_v2_defaults_declare_v2():
    declaring = {profile for profile, path in PROFILE_CATALOG.items()
                 if CAPABILITY in json.loads(path.read_text(encoding="utf-8")).get("capabilities", [])}
    # Defaults and the restricted RoboDojo toolkit successor inherit this
    # contract (RoboDojo uses the X5 jaw loader).
    assert declaring == {"robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                         "robodojo_placement_toolkit_v5", "robodojo_placement_toolkit_v6",
                         "robocasa_mobile_direct_v1", "robolab_direct_v1"}


def test_the_yam_fingers_move_from_their_open_reference_by_half_the_measured_separation():
    """The YAM descriptor states the fingers fully open (46.95 mm each), not closed as ALOHA's and Robotiq's do."""
    from ora.agent.toolkit.lib import _clearance_inputs as lib

    for module in (rt, lib):
        geometry = SimpleNamespace(finger_opening=jaw_models.OPENING_SYMMETRIC_FROM_REFERENCE,
                                   fingers=(None, None), reference_finger_qpos=(0.04695, 0.04695))
        assert module._qpos({"gripper_opening_m": 0.0939}, geometry) == pytest.approx([0.0, 0.0])
        assert module._qpos({"gripper_opening_m": 0.0}, geometry) == pytest.approx([-0.04695, -0.04695])
    assert JAW_MODELS["yam_linear_4310_tcp"]["opening"] == jaw_models.OPENING_SYMMETRIC_FROM_REFERENCE
