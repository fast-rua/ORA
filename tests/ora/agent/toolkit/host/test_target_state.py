"""``robot.submit_action`` on ``robot_runtime_v2``: the target-state v4 proposal (``tools/motion/target_state.py``).

The tool validates and normalizes one ``target_state`` proposal and writes it
as the observation's single submitted action; nothing moves here. A pose, a
gripper opening or both may be requested, the pose inside the workspace
envelope unless it moves an out-of-envelope arm strictly back toward it, and
every refusal names its field. A second submission for the same observation
is refused.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from ora.agent.error_classes import ToolError
from tests.support.orchestrator.runtime_v1 import BOUNDS, publish_synthetic, robot_tools

IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _runtime(tmp_path: Path, bounds=BOUNDS) -> tuple[Any, Path, dict]:
    """A v2 tool runtime on the published Synthetic observation (both arms at z = 1.0 m)."""

    observation, _pointer, _workspace, control, _tools, state_store = publish_synthetic(tmp_path)
    tools = robot_tools(control, tmp_path / "events-v2.jsonl", bounds, state_root=state_store.root)
    return tools, control, observation


def _submitted(control: Path) -> dict:
    return json.loads((control / "submitted-action.json").read_text(encoding="utf-8"))


def _pose(z: float = 1.02) -> dict:
    return {"position_m": [0.0, 0.0, z], "orientation_xyzw": list(IDENTITY)}


def test_a_pose_only_target_is_written_as_one_v4_action_bound_to_the_observation(tmp_path: Path) -> None:
    tools, control, observation = _runtime(tmp_path)
    result = tools.invoke("robot.submit_action", {"arm": "left", "pose": _pose()})
    assert result["accepted"] is True and result["directive"] == "action_submitted"
    action = _submitted(control)
    assert action["schema"] == "ora-action.v4" and action["action"] == "target_state"
    assert action["arguments"] == {"arm": "left", "pose": {"position_m": [0.0, 0.0, 1.02],
                                                           "orientation_xyzw": IDENTITY}}
    assert action["observation_id"] == observation["observation_id"]
    assert action["action_id"] == result["action_id"]
    assert len(action["bound_observation_sha256"]) == 64
    # The observation closes behind the submission.
    assert json.loads((control / "current-observation.json").read_text())["open"] is False


def test_a_gripper_only_target_holds_the_pose_and_a_null_gripper_is_no_gripper(tmp_path: Path) -> None:
    tools, control, _ = _runtime(tmp_path)
    tools.invoke("robot.submit_action", {"arm": "right", "gripper": 0})
    assert _submitted(control)["arguments"] == {"arm": "right", "gripper": 0.0}

    tools, control, _ = _runtime(tmp_path / "null")
    tools.invoke("robot.submit_action", {"arm": "right", "pose": _pose(), "gripper": None})
    assert "gripper" not in _submitted(control)["arguments"]


def test_a_combined_target_carries_both_and_the_orientation_is_normalized(tmp_path: Path) -> None:
    tools, control, _ = _runtime(tmp_path)
    tools.invoke("robot.submit_action", {
        "arm": "left", "pose": {"position_m": [0.1, -0.1, 1.2], "orientation_xyzw": [0.0, 0.0, 0.0, 1.01]},
        "gripper": 0.25,
    })
    arguments = _submitted(control)["arguments"]
    assert arguments["gripper"] == 0.25
    assert arguments["pose"]["position_m"] == [0.1, -0.1, 1.2]
    assert arguments["pose"]["orientation_xyzw"] == pytest.approx(IDENTITY)


@pytest.mark.parametrize("arguments, message, field", [
    ({"arm": "left", "gripper": 0.5, "speed": 1}, "target state fields are arm, optional pose and gripper", None),
    ({"arm": "middle", "gripper": 0.5}, "arm must be left or right", "arm"),
    ({"arm": "left"}, "requires pose or gripper", "arguments"),
    ({"arm": "left", "gripper": 1.5}, "gripper must be in [0,1]", "gripper"),
    ({"arm": "left", "gripper": True}, "gripper must be in [0,1]", "gripper"),
    ({"arm": "left", "gripper": math.nan}, "gripper must be in [0,1]", "gripper"),
    ({"arm": "left", "pose": {"position_m": [0, 0, 1.0]}}, "pose must contain position_m and orientation_xyzw",
     "pose"),
    ({"arm": "left", "pose": {"position_m": [0, 0], "orientation_xyzw": IDENTITY}}, "position_m", None),
    ({"arm": "left", "pose": {"position_m": [0, 0, math.inf], "orientation_xyzw": IDENTITY}}, "position_m", None),
    ({"arm": "left", "pose": {"position_m": [0, 0, 1.0], "orientation_xyzw": [0, 0, 0, 0]}}, "", "arguments"),
])
def test_an_invalid_target_is_refused_with_its_field_and_nothing_is_written(
    tmp_path: Path, arguments: dict, message: str, field: str | None,
) -> None:
    tools, control, _ = _runtime(tmp_path)
    with pytest.raises(ToolError) as refused:
        tools.invoke("robot.submit_action", arguments)
    assert message in str(refused.value)
    if field is not None:
        assert refused.value.field == field
    assert not (control / "submitted-action.json").exists()
    assert json.loads((control / "current-observation.json").read_text())["open"] is True


def test_a_pose_outside_the_envelope_is_refused_with_the_axis_interval(tmp_path: Path) -> None:
    tools, control, _ = _runtime(tmp_path)
    with pytest.raises(ToolError) as refused:
        tools.invoke("robot.submit_action", {"arm": "left", "pose": _pose(z=1.6)})
    assert refused.value.code == "out_of_bounds" and refused.value.field == "pose.position_m[2]"
    assert refused.value.constraints == {"minimum": BOUNDS["z"][0], "maximum": BOUNDS["z"][1]}
    assert not (control / "submitted-action.json").exists()


def test_an_arm_outside_the_envelope_may_only_move_back_toward_it(tmp_path: Path) -> None:
    # Both arms are at z = 1.0, below this envelope's z floor.
    bounds = {**BOUNDS, "z": [1.1, 1.45]}
    tools, control, _ = _runtime(tmp_path, bounds)
    with pytest.raises(ToolError) as refused:
        tools.invoke("robot.submit_action", {"arm": "left", "pose": _pose(z=0.95)})
    assert refused.value.code == "out_of_bounds"
    tools.invoke("robot.submit_action", {"arm": "left", "pose": _pose(z=1.05)})  # still outside, but closer
    assert _submitted(control)["arguments"]["pose"]["position_m"][2] == 1.05


def test_a_second_submission_for_the_same_observation_is_refused(tmp_path: Path) -> None:
    tools, control, _ = _runtime(tmp_path)
    tools.invoke("robot.submit_action", {"arm": "left", "gripper": 1.0})
    with pytest.raises(ToolError) as refused:
        tools.invoke("robot.submit_action", {"arm": "left", "gripper": 0.0})
    assert refused.value.code == "duplicate_action" and refused.value.retryable is False
    assert _submitted(control)["arguments"] == {"arm": "left", "gripper": 1.0}
