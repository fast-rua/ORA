"""observed_clearance_reference_v1 on the live RoboDojo surface: inputs, binding and delivery.

The measurement itself is ``test_observed_clearance.py``. Here the Tool
Runtime gathers its inputs from what the policy already receives -- the
observation's public depth and calibration (hash-checked), measured state of
both arms, the bound public geometry descriptor, and the policy's own
localization of the target it named when the jaws closed -- and every
``move_pose``, ``move_to``, ``rotate_wrist`` and ``rotate_pitch`` target carries
the read-only ``observed_clearance`` its declared result schema admits. The
move is submitted unchanged and a failure of the reference never blocks it.
Offline: a synthetic top-down depth scene and the fake Dojo plant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.toolkit.host.motion_rules import clearance as oc
from ora.agent.toolkit.host.motion_rules import clearance_runtime as rt
from tests.support.agent.observed_clearance_runtime import DOWN, RIM_TOP, _write_scene, HOME_RIGHT

DOJO = "robodojo_placement_toolkit_v5"
ARX = Path(__file__).resolve().parents[5] / "ora/policy/embodiments/arx-x5"
#: The descriptor the live Dojo surface binds (v6: v5's geometry, its gripper source at the toolkit host).
DESCRIPTOR = ARX / "descriptor-public-geometry-v6.json"


def _geometry():
    return rt.load_jaw_geometry(DESCRIPTOR)


def test_the_live_surface_binds_descriptor_v6():
    binding = surface_of(DOJO).view_config["public_geometry_descriptor"]
    assert (Path(__file__).resolve().parents[5] / binding["path"]).resolve() == DESCRIPTOR.resolve()


def test_the_descriptor_supplies_both_jaws_and_their_wrist_cameras():
    geometry = _geometry()
    assert set(geometry) == {"left", "right"}
    left = geometry["left"]
    assert [finger.opening_direction for finger in left.fingers] == [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    assert left.empty_close_reference == pytest.approx(0.185185185)
    assert left.contact_center_local == pytest.approx((0.1501, 0.0, 0.0))
    assert left.wrist_cameras == ("left_wrist",)


def test_a_target_below_the_observed_rim_reports_it_from_the_observation(tmp_path):
    observation = _write_scene(tmp_path / "obs", left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT)
    target = [0.0, 0.0, 0.80 + 0.15757, *DOWN]
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="left", target_pose=target,
                                     geometry=_geometry(), held={})
    assert result["status"] == "measured"
    assert result["clearance_m"] == pytest.approx(0.80 - RIM_TOP, abs=0.005)
    assert result["highest_surface"]["observation_id"] == observation["observation_id"]
    assert result["held_object"] is None


def test_a_drifted_depth_file_is_not_read(tmp_path):
    observation = _write_scene(tmp_path / "obs", left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT)
    for name in observation["cameras"]:
        observation["cameras"][name]["files"]["depth_numpy_m_f32"]["sha256"] = "0" * 64
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="left",
                                     target_pose=[0.0, 0.0, 1.0, *DOWN], geometry=_geometry(), held={})
    assert (result["status"], result["reason"]) == ("unknown", "depth_unavailable")


def test_held_extent_follows_the_jaws_only_while_they_hold(tmp_path):
    # An object 8 cm tall localized under jaws that closed at 1.0 m.
    attach = [0.0, 0.0, 1.0, *DOWN]
    localization = {"status": "localized", "aabb_min_xyz_m": [-0.02, -0.02, 0.80],
                    "aabb_max_xyz_m": [0.02, 0.02, 0.88], "support_z_m": 0.79}
    corners, recentered = rt.held_local_corners(localization, attach, (0.1501, 0.0, 0.0))
    assert recentered is False
    world = corners @ oc.rotation(DOWN).T + np.array(attach[:3])
    assert world[:, 2].min() == pytest.approx(0.79)  # bottom lowered to the reported support
    target = [0.2, 0.0, 1.2, *DOWN]
    held = {"left": rt.HeldExtent(target_ref="target-abc", local_corners=corners)}
    holding = _write_scene(tmp_path / "holding", left_pose=attach, right_pose=HOME_RIGHT, left_close=True,
                           left_opening=0.5)
    result = rt.annotate_observation(holding, tmp_path / "holding", arm="left", target_pose=target,
                                     geometry=_geometry(), held=held)
    assert result["footprint_lowest_part"] == "held_object"
    assert result["footprint_lowest_m"][2] == pytest.approx(0.99, abs=1e-3)
    assert result["held_object"] == {"target_ref": "target-abc", "extent_included": True,
                                     "recentered_under_jaws": False}
    empty_close = _write_scene(tmp_path / "empty", left_pose=attach, right_pose=HOME_RIGHT, left_close=True,
                               left_opening=0.19)
    result = rt.annotate_observation(empty_close, tmp_path / "empty", arm="left", target_pose=target,
                                     geometry=_geometry(), held=held)
    assert result["footprint_lowest_part"] == "fingers" and result["held_object"] is None


def test_a_localization_away_from_the_jaws_is_moved_under_them_and_says_so():
    attach = [0.0, 0.0, 1.0, *DOWN]
    far = {"status": "localized", "aabb_min_xyz_m": [0.30, 0.30, 0.80], "aabb_max_xyz_m": [0.34, 0.34, 0.88]}
    corners, recentered = rt.held_local_corners(far, attach, (0.1501, 0.0, 0.0))
    world = corners @ oc.rotation(DOWN).T + np.array(attach[:3])
    assert recentered is True and world[:, :2].mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert rt.held_local_corners({"status": "insufficient_depth", "aabb_min_xyz_m": None,
                                  "aabb_max_xyz_m": None}, attach, (0.1501, 0.0, 0.0)) is None


# --------------------------------------------------------------------------
# The Tool Runtime on the live Dojo surface
# --------------------------------------------------------------------------


def test_grasp_execute_records_its_committed_target_and_contact_pose(tmp_path):
    from types import SimpleNamespace

    host = SimpleNamespace(surface=surface_of(DOJO), control_root=tmp_path)
    contact = {"position_m": [0.1, -0.05, 0.93], "orientation_xyzw": DOWN}
    (tmp_path / "manipulation-skill-commitment.json").write_text(json.dumps({
        "skill": "grasp_execute", "arm": "right", "source_ref": "target-0123456789abcdef01234567",
        "poses": {"approach": contact, "lift": contact}}))
    rt.note_grasp_execute(host)
    assert host._observed_clearance_holds["right"] == {
        "target_ref": "target-0123456789abcdef01234567", "source": "grasp_execute",
        "attach_pose": [0.1, -0.05, 0.93, *DOWN]}
    (tmp_path / "manipulation-skill-commitment.json").write_text("{not json")
    rt.note_grasp_execute(host)  # a record it cannot read never raises
    rt.note_gripper(host, {"right": "close"}, target_ref="target-x", observation={})
    assert "right" not in host._observed_clearance_holds


def test_the_live_dojo_surface_declares_it():
    dojo = surface_of(DOJO)
    assert dojo.supports(rt.V2_CAPABILITY)
    help_text = next(item["description"] for item in dojo.definitions if item["name"] == "move_pose")
    assert "observed_clearance" in help_text
