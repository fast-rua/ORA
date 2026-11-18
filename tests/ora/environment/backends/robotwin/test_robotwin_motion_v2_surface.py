"""Robotwin motion v2 surface contract regressions."""

from __future__ import annotations
from typing import Any
import pytest
from ora.agent.composition import resolve_tool_surface
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.contracts import compile_enhanced_move_pose

#: The kept RoboTwin default surface (primitive_runtime's default).
DEFAULT = "robotwin_toolchain_b2_closed_noservo_v2"


def test_runtime_camera_allowlist_tracks_surface_observer_capability() -> None:
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(DEFAULT)
    assert not runtime.surface.supports("policy_observer_camera_v1")
    assert runtime._robotwin_policy_cameras() == {
        "head_camera", "front_camera", "left_camera", "right_camera",
    }


def test_enhanced_move_pose_compiles_pure_translation_without_copying_quaternion() -> None:
    compiled = compile_enhanced_move_pose({
        "position": {
            "mode": "delta",
            "translation_m": [0.0, 0.0, 0.04],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }, current_pose_xyz_xyzw=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.34, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["position_mode"] == "delta"
    assert compiled["orientation_mode"] == "hold_current"


def test_enhanced_move_pose_compiles_standard_absolute_proposal_pose() -> None:
    compiled = compile_enhanced_move_pose({
        "position_m": [0.1, 0.2, 0.4],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }, current_pose_xyz_xyzw=[
        0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["position_mode"] == "absolute"
    assert compiled["orientation_mode"] == "absolute_quaternion"

    with pytest.raises(ValueError, match="either a standard absolute TCP pose"):
        compile_enhanced_move_pose({
            "position_m": [0.1, 0.2, 0.4],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "position": {"mode": "hold_current"},
        }, current_pose_xyz_xyzw=[
            0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
        ])


def test_enhanced_move_pose_compiles_rotation_only_about_tool_axis() -> None:
    compiled = compile_enhanced_move_pose({
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta",
            "axis": "z",
            "angle_deg": 90.0,
            "frame": "end_effector",
        },
    }, current_pose_xyz_xyzw=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    root_half = 2.0 ** -0.5
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.3, 0.0, 0.0, root_half, root_half,
    ])
    assert compiled["translation_distance_m"] == 0.0
    assert compiled["rotation_angle_deg"] == pytest.approx(90.0)


def test_enhanced_move_pose_resolves_tool_frame_translation() -> None:
    root_half = 2.0 ** -0.5
    compiled = compile_enhanced_move_pose({
        "position": {
            "mode": "delta",
            "translation_m": [0.1, 0.0, 0.0],
            "frame": "end_effector",
        },
        "orientation": {"mode": "hold_current"},
    }, current_pose_xyz_xyzw=[
        0.0, 0.0, 1.0, 0.0, 0.0, root_half, root_half,
    ])
    assert compiled["target_pose_xyz_xyzw"][:3] == pytest.approx([
        0.0, 0.1, 1.0,
    ])


@pytest.mark.parametrize("field,value", [
    ("no_op", {
        "position": {"mode": "hold_current"},
        "orientation": {"mode": "hold_current"},
    }),
    ("angle", {
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta", "axis": "z",
            "angle_deg": 181.0, "frame": "end_effector",
        },
    }),
    ("zero_delta", {
        "position": {
            "mode": "delta", "translation_m": [0.0, 0.0, 0.0],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }),
])
def test_enhanced_move_pose_rejects_noop_and_invalid_angle(
    field: str, value: dict[str, Any],
) -> None:
    del field
    with pytest.raises(ValueError):
        compile_enhanced_move_pose(
            value,
            current_pose_xyz_xyzw=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
