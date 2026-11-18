"""RoboDojo's default surface carries Dojo backend facts, not RoboTwin ones.

These checks keep what a Dojo result, camera set, controlled frame and help text
must say on toolkit v5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.agent.surface_catalog import surface_of

ROOT = Path(__file__).resolve().parents[5]
SIBLING = "robotwin_toolchain_b2_closed_noservo_v2"
ROBODOJO = "robodojo_placement_toolkit_v5"
ROBODOJO_CAMERAS = ["head", "left_wrist", "right_wrist"]


@pytest.fixture(scope="module")
def robodojo():
    return surface_of(ROBODOJO)


@pytest.fixture(scope="module")
def sibling():
    return surface_of(SIBLING)


def test_the_grasp_ik_annotation_admits_exactly_what_the_dojo_check_returns(robodojo):
    from ora.contracts.reach_check import ROBODOJO_REASONS, SHARED_REASONS

    stage = robodojo.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"][
        "ik_reachability"]["properties"]["pregrasp"]["properties"]
    assert set(stage["reason"]["enum"]) == {None, *SHARED_REASONS, *ROBODOJO_REASONS}


def test_the_align_result_admits_exactly_what_the_dojo_adapter_returns(robodojo):
    enum = robodojo.result_schemas["align_plan"]["properties"]["candidates"]["items"][
        "properties"]["rejection_reason"]["enum"]
    assert set(enum) == {None, "ik_failed", "goal_below_support", "fk_residual_exceeded",
                         "fk_calibration_mismatch", "stale_observation", "no_planner", "planner_error"}


def test_view_config_binds_the_dojo_cameras_and_controlled_frame(robodojo, sibling):
    assert robodojo.view_config["automatic_actual_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["default_observe_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["controlled_frame_convention"] == "arx_x5_link6"
    shared = {key for key in sibling.view_config if key not in {
        "automatic_actual_views", "default_observe_views", "controlled_frame_convention",
        "public_geometry_descriptor"}}
    for key in shared:
        assert robodojo.view_config[key] == sibling.view_config[key], key


def test_no_robotwin_or_aloha_fact_survives_outside_the_shared_legacy_names(robodojo):
    """The substitutions are complete: no RoboTwin camera, frame, planner or evidence fact remains."""
    for definition in robodojo.definitions:
        text = definition["description"] + json.dumps(definition["inputSchema"])
        for term in ("RoboTwin", "ALOHA", "head_camera", "front_camera", "left_camera",
                     "right_camera", "CuRobo", "Top-K", "grip-center", "public TCP", "measured TCP",
                     "force/progress", "recovery.baseline_step", "execution_gate", "128-seed",
                     "not collision-checked", "consumes one robot decision."):
            assert term not in text, (definition["name"], term)


#: The controller-feedback requirement the activation contract checked while
#: RoboDojo was a developing surface (removed by integration 0059 unit A0, when
#: robodojo_placement_toolkit_v5 became the RoboDojo default instance).
DEVELOPING_REQUIREMENT = (
    "robodojo_controller_feedback_v1", "current_public_robot_facts_v1", "motion_mode_companion_help_v1",
    "optional_argument_null_as_omitted_v1", "compute_grammar_from_evaluator_v1",
)


def test_the_dojo_surface_satisfies_the_developing_robodojo_requirement(robodojo):
    assert all(robodojo.supports(cap) for cap in DEVELOPING_REQUIREMENT)
    default = json.loads((ROOT / "tests/data/default-expectations.json").read_text())["instances"]["robodojo"]
    assert set(DEVELOPING_REQUIREMENT) <= set(default["required_capabilities"])
