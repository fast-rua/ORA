"""Tool-call mode's projection of a dry run (``session_core/domain.py``): each arm's predicted pose with its world axes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.agent.code_mode import answers as code_answers
from ora.agent.composition.embodiment.geometry import public_local_axes
from ora.agent.session_core.domain import PolicyDomainMixin

PREDICTED = [0.4, 0.0, 0.2, 0.1, -0.3, 0.2, 0.9]
RECORD = {"dry_run": True, "action": "move_pose", "status": "planned", "feasible": True, "route": "line",
          "arms": {"right": {"target_pose": PREDICTED, "predicted_pose": PREDICTED, "end_joints_deg": [0.0] * 6}},
          "claim_boundary": "the robot's own motion"}


def _projected(view_config):
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = SimpleNamespace(surface=SimpleNamespace(result_schemas={}, view_config=view_config))
    return planner._causal_tool_result("move_pose", RECORD)


@pytest.mark.parametrize("convention", ["panda_grip_site", "aloha_grip_center_tcp", "arx_x5_link6", "robotiq_2f85_tcp"])
def test_a_tool_call_dry_run_gives_each_arm_s_predicted_axes_as_code_mode_does(convention) -> None:
    """User decision 2026-11-06: the predicted axes reach every surface that publishes its controlled frame, in tool-call
    mode as in code mode, by the same function (``code_mode/answers.predicted_axes``)."""
    from scipy.spatial.transform import Rotation

    arm = _projected({"controlled_frame_convention": convention})["arms"]["right"]
    keys = list(arm)
    assert keys[keys.index("predicted_pose") + 1:keys.index("predicted_pose") + 3] == [
        "predicted_approach_world", "predicted_jaw_line_world"]
    local, rotation = public_local_axes(convention), Rotation.from_quat(PREDICTED[3:])
    for name, axis in (("predicted_approach_world", "approach"), ("predicted_jaw_line_world", "closing")):
        assert max(abs(a - b) for a, b in zip(arm[name], rotation.apply(local[axis]))) <= 5e-6, name
    assert arm == code_answers.predicted_axes(RECORD, {"controlled_frame_convention": convention})["arms"]["right"]


def test_no_predicted_axes_where_the_surface_publishes_no_controlled_frame() -> None:
    arm = _projected({})["arms"]["right"]
    assert "predicted_approach_world" not in arm and "predicted_jaw_line_world" not in arm
