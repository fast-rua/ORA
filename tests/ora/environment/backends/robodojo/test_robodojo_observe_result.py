"""The live RoboDojo ``observe`` result admits what every Dojo observation carries.

RoboDojo g003 run ``b50ff4c37763`` (``imitate_sorting_sequence``) called
``observe`` twice and received ``internal_tool_error`` both times:
``$result.public_robot_facts has unknown result fields ['controller_feedback']``.
The bound result (``robodojo-live-observe-results-v2.json``) also declared the
RoboTwin camera names and no ARX jaw branch, so an observe call on that surface
could not validate at all. No g003 run ever received an ``observe`` result.

These tests drive the real dispatcher with an observation in the recorded
shape -- controller contract, per-arm gripper feedback, three Dojo cameras --
and assert that the result validates and still carries every recorded field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from tests.support.robodojo.perception import (
    BOUNDS, FakeHeadGrounding, FakeX5Proposer, _synthetic_robodojo_observation,
)
from tests.support.robodojo.observe_result import RECORDED_CONTROLLER_FEEDBACK, _runtime


def test_observe_validates_and_keeps_the_controller_contract_and_jaw_feedback(tmp_path):
    runtime = _runtime(tmp_path)
    result = runtime.invoke("observe", {})
    facts = result["public_robot_facts"]
    assert facts["controller_feedback"] == RECORDED_CONTROLLER_FEEDBACK
    assert set(facts["cameras"]) == {"head", "left_wrist", "right_wrist"}
    right = facts["end_effectors"]["right"]["gripper"]
    assert right["measured_aperture_mm"] is None
    assert right["feedback"]["drive_target_opening_normalized"] == 0.3
    assert right["command_target"] == "close"


@pytest.mark.parametrize("views", [["head"], ["left_wrist", "right_wrist"]])
def test_observe_with_selected_views_validates(tmp_path, views):
    result = _runtime(tmp_path).invoke("observe", {"views": views})
    assert set(result["public_robot_facts"]["cameras"]) == set(views)
