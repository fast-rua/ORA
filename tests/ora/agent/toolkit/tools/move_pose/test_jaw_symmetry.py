"""The jaw-symmetry rule of the default motion compiler (v4/v5, where joint trajectories run).

The wrist starts at the identity orientation; every request asks for 170 degrees about the approach axis
(link6 local +X), whose half-turn twin is 10 degrees away. An empty jaw takes the twin on every interaction
and orientation form; a jaw that holds something never does; exact_orientation keeps the request.
"""
import math
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.lib.motion_plan import orientation
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import _smooth, tool
from ora.agent.toolkit.tools.move_pose._axes import CAPABILITY as WORLD_AXES

HALF = math.radians(170) / 2
REQUESTED = [math.sin(HALF), 0.0, 0.0, math.cos(HALF)]  # 170 degrees about local +X
FORMS = {
    "absolute_quaternion": {"orientation_mode": "absolute_quaternion", "orientation_xyzw": REQUESTED},
    "world_axes": {"orientation_mode": "world_axes", "approach_world": [1.0, 0.0, 0.0],
                   "jaw_line_world": [0.0, math.cos(math.radians(170)), math.sin(math.radians(170))]},
    "axis_angle_delta": {"orientation_mode": "axis_angle_delta", "rotation_axis": "x", "rotation_angle_deg": 170.0,
                         "orientation_frame": "end_effector"},
}


def _setup(monkeypatch, *, gripper="open", held=None, trajectories=True):
    effectors = {arm: {"pose_xyz_xyzw": [0.3, y, 1.0, 0.0, 0.0, 0.0, 1.0], "arm_joint_qpos": [0.0] * 6,
                       "gripper_command_target": gripper} for arm, y in (("left", 0.2), ("right", -0.2))}
    obs = SimpleNamespace(manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": effectors}}),
                          identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    monkeypatch.setattr(tool.observation, "observe", lambda api: obs)
    monkeypatch.setattr(tool.clearance, "estimate", lambda *a, **k: SimpleNamespace(json=lambda: {}))
    monkeypatch.setattr(tool._held, "extents", lambda api, jaws: {})
    state = {"tool.held.left": held, "tool.held.right": None}
    api = SimpleNamespace(state=SimpleNamespace(read=lambda key: state[key]))
    config = {"synchronous_execution": True, "execution": {"generation": 3}, "motion": {
        "joint_trajectory": trajectories, "mutation_refusals": {}, "capabilities": [WORLD_AXES], "view_config": {},
        "local_axes": {"approach": [1.0, 0.0, 0.0], "closing": [0.0, 1.0, 0.0]}, "jaw_offsets": {},
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}, "reach_available": False,
        "jaws": {}, "held": {}, "examples": []}}
    return api, config


def _compile(api, config, target, *, exact=False, primitive="move_pose", request=None):
    action, compiled = tool.compile_action(api, {"targets": {"left": target}}, config=config, primitive=primitive,
                                           request_for_target=request, exact_orientation=exact)
    return action["arguments"]["targets"]["left"]["orientation_xyzw"], compiled["targets"]["left"]


def _turn(q):
    return orientation.turn_deg([0.0, 0.0, 0.0, 1.0], q)


@pytest.mark.parametrize("interaction", ["free_space", "precision", "contact_approach"])
@pytest.mark.parametrize("form", list(FORMS))
def test_an_empty_jaw_takes_the_twin_nearer_the_wrist_on_every_mode_and_form(monkeypatch, interaction, form):
    api, config = _setup(monkeypatch)
    target = {"position_mode": "hold_current", "interaction": interaction, **FORMS[form]}
    sent, compiled = _compile(api, config, target)
    assert compiled["jaw_symmetry"] == "twin_nearer_current" and _turn(sent) == pytest.approx(10.0, abs=1e-3)
    assert compiled["target_pose_xyz_xyzw"][3:] == pytest.approx(sent)


@pytest.mark.parametrize("form", ["absolute_quaternion", "axis_angle_delta"])
def test_a_jaw_holding_something_never_takes_the_twin_of_a_request_with_a_sign(monkeypatch, form):
    for gripper, held in (("close", None), ("open", {"target_ref": "cup"})):
        api, config = _setup(monkeypatch, gripper=gripper, held=held)
        sent, compiled = _compile(api, config, {"position_mode": "hold_current", "interaction": "free_space",
                                                **FORMS[form]})
        assert compiled["jaw_symmetry"] == "held_no_flip" and _turn(sent) == pytest.approx(170.0, abs=1e-3)


def test_a_held_object_keeps_its_grip_when_a_jaw_line_is_requested(monkeypatch):
    """A jaw line has no sign, so the same world_axes target before and after a close needs no turn: taken as given, it
    turned the hand 180 degrees, 123 times in step 8's Extended arm."""
    for gripper, held in (("close", None), ("open", {"target_ref": "cup"})):
        api, config = _setup(monkeypatch, gripper=gripper, held=held)
        sent, compiled = _compile(api, config, {"position_mode": "hold_current", "interaction": "free_space",
                                                **FORMS["world_axes"]})
        assert compiled["jaw_symmetry"] == "twin_nearer_current" and _turn(sent) == pytest.approx(10.0, abs=1e-3)


def test_exact_orientation_still_turns_a_held_object_on_a_jaw_line(monkeypatch):
    api, config = _setup(monkeypatch, gripper="close")
    sent, compiled = _compile(api, config, {"position_mode": "hold_current", "interaction": "free_space",
                                            **FORMS["world_axes"]}, exact=True)
    assert compiled["jaw_symmetry"] == "exact" and _turn(sent) == pytest.approx(170.0, abs=1e-3)


@pytest.mark.parametrize("form", list(FORMS))
def test_exact_orientation_keeps_the_request(monkeypatch, form):
    api, config = _setup(monkeypatch)
    sent, compiled = _compile(api, config, {"position_mode": "hold_current", "interaction": "precision",
                                            **FORMS[form]}, exact=True)
    assert compiled["jaw_symmetry"] == "exact" and _turn(sent) == pytest.approx(170.0, abs=1e-3)


def test_a_request_nearer_than_its_twin_and_hold_current_stay_as_requested(monkeypatch):
    api, config = _setup(monkeypatch)
    sent, compiled = _compile(api, config, {"position_mode": "absolute", "position_m": [0.32, 0.2, 1.0],
                                            "orientation_mode": "hold_current", "interaction": "free_space"})
    assert compiled["jaw_symmetry"] == "requested" and _turn(sent) == pytest.approx(0.0, abs=1e-3)


def test_move_to_and_rotate_follow_the_same_rule(monkeypatch):
    api, config = _setup(monkeypatch)
    sent, compiled = _compile(api, config, {"position_m": [0.32, 0.2, 1.0]}, primitive="move_to")
    assert compiled["jaw_symmetry"] == "requested"
    rotate = lambda value: {"position": {"mode": "hold_current"}, "orientation": {
        "mode": "axis_angle_delta", "axis": "x", "angle_deg": float(value), "frame": "end_effector"}}
    sent, compiled = _compile(api, config, 170.0, primitive="rotate_wrist", request=rotate)
    assert compiled["jaw_symmetry"] == "twin_nearer_current" and _turn(sent) == pytest.approx(10.0, abs=1e-3)
    sent, compiled = _compile(api, config, 170.0, primitive="rotate_wrist", request=rotate, exact=True)
    assert compiled["jaw_symmetry"] == "exact" and _turn(sent) == pytest.approx(170.0, abs=1e-3)


def test_without_joint_trajectories_the_released_compiler_is_unchanged(monkeypatch):
    api, config = _setup(monkeypatch, trajectories=False)
    sent, compiled = _compile(api, config, {"position_mode": "hold_current", "interaction": "precision",
                                            **FORMS["absolute_quaternion"]})
    assert "jaw_symmetry" not in compiled and _turn(sent) == pytest.approx(170.0, abs=1e-3)


def test_exact_orientation_is_a_switch():
    assert _smooth.jaw_option({"targets": {}, "exact_orientation": True}) == (True, {"targets": {}})
    assert _smooth.jaw_option({"targets": {}}) == (False, {"targets": {}})
    with pytest.raises(ToolRefusal) as error:
        _smooth.jaw_option({"targets": {}, "exact_orientation": "yes"})
    assert error.value.fields["field"] == "exact_orientation"
    assert _smooth.jaw_labels({"targets": {"left": {"jaw_symmetry": "exact"}, "right": {}}}) == {"left": "exact"}
