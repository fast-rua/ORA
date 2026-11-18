"""Live RoboDojo surface repairs for the D3L issue table.

D3L-R001: align_plan's closed result enum rejected a reason the Dojo
reachability adapter produces (``ik_branch_jump`` then), turning a correctable
refusal into a non-retryable internal_tool_error. D3L-R003: historical observe admitted only
LIBERO camera names on Dojo, and then recommended those names although the
observe schema rejects them.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.tool_surface import validate_schema_instance
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.contracts.reach_check import ROBODOJO_REASONS

ROOT = Path(__file__).resolve().parents[5]
LIVE = "robodojo_placement_toolkit_v5"


def _produced_reasons() -> set[str | None]:
    """Every rejection reason the Dojo reachability adapter can return: the planner's IK's, ``no_planner`` included."""
    source = (ROOT / "ora/environment/backends/robodojo/reachability.py").read_text()
    reasons: set[str | None] = {None, *ROBODOJO_REASONS}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "reason":
                    reasons |= {item.value for item in ast.walk(value)
                                if isinstance(item, ast.Constant) and isinstance(item.value, str)}
    return reasons


def _align_result(reason: str | None) -> dict:
    return {
        "schema": "ora.align-plan-result.v1", "target_ref": "target-1",
        "approach_side": "top", "requested_arm": "left", "motion_planner_called": False,
        "candidates": [{
            "source_index": 0, "arm": "left",
            "align_pose": {"position_m": [0.0, -0.3, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "reachable": reason is None, "position_error_m": None if reason == "ik_failed" else 0.001,
            "orientation_error_rad": None if reason == "ik_failed" else 0.01, "iterations": 1,
            "rejection_reason": reason,
            "claim_boundary": "SDK IK and FK only; no physics, collision-free path, contact or task-success claim",
        }],
    }


def test_the_live_align_enum_is_exactly_what_the_backend_produces() -> None:
    schema = surface_of(LIVE).result_schemas["align_plan"]
    enum = schema["properties"]["candidates"]["items"]["properties"]["rejection_reason"]["enum"]
    assert set(enum) == _produced_reasons() == {
        None, "ik_failed", "goal_below_support", "fk_residual_exceeded",
        "fk_calibration_mismatch", "stale_observation", "no_planner", "planner_error"}


@pytest.mark.parametrize("reason", sorted(_produced_reasons(), key=str))
def test_every_produced_align_result_validates_on_the_live_surface(reason) -> None:
    validate_schema_instance(_align_result(reason), surface_of(LIVE).result_schemas["align_plan"])


def test_the_live_align_help_names_the_dojo_reasons_and_recovery() -> None:
    text = next(d for d in surface_of(LIVE).definitions if d["name"] == "align_plan")["description"]
    for phrase in ("ik_failed", "no_planner", "goal_below_support", "fk_residual_exceeded",
                   "goal_below_support is a limit of the surface", "iterations (1 on this backend"):
        assert phrase in text
    assert "SDK inverse-kinematics solve" not in text
    assert "otherwise joint_limit, singular or not_converged" not in text


def _cameras(profile: str) -> set[str]:
    surface = surface_of(profile)
    host = SimpleNamespace(surface=surface)
    host._robotwin_policy_cameras = MethodType(ManipulationToolRuntime._robotwin_policy_cameras, host)
    return ManipulationToolRuntime._physical_policy_cameras(host)


def _observe_view_enum(profile: str) -> set[str]:
    schema = next(d for d in surface_of(profile).definitions if d["name"] == "observe")["inputSchema"]
    return set(schema["properties"]["views"]["items"]["enum"])


def test_historical_observe_on_dojo_admits_exactly_the_advertised_cameras() -> None:
    assert _cameras(LIVE) == _observe_view_enum(LIVE) == {"head", "left_wrist", "right_wrist"}


@pytest.mark.parametrize("profile", [
    "robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
])
def test_historical_observe_names_only_cameras_the_surface_advertises(profile) -> None:
    assert _cameras(profile) <= _observe_view_enum(profile)
    assert set(surface_of(profile).view_config["default_observe_views"]) <= _cameras(profile)
