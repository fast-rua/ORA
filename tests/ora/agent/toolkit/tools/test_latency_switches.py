"""The model-facing latency switches on the tool side (``ora/agent/toolkit/tools/_latency.py``).

A call whose configuration has no ``latency_switches`` (every benchmark call and every sealed one) runs as before. A
realtime plan turns two things off: ``move_pose``'s reach reference, which then asks the planner's IK nothing and says
so, and the preview pictures, which a dry run then leaves out (keeping its numbers) and a refused real call never
draws. Offline: a recording stub bridge and views whose camera reads fail the test if anything reads them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.agent.toolkit.host.reach.alignment import frame_local_axes
from ora.agent.toolkit.host.reach.jaw_center import jaw_center_offsets
from ora.agent.toolkit.lib import action_preview
from ora.agent.toolkit.tools import _dry_run, _latency
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import _context, _reach
from ora.contracts.action_plan import segment
from tests.support.agent.move_pose_reach import DOJO, HOME, REQUESTED, _bridge
from tests.support.agent.reach_stacks import through_adapter
from tests.support.agent.surface_catalog import surface_of

BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
START = [0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]
OFF = {_latency.KEY: {"reach_reference": False, "preview_images": False}}


def _host(bridge, *, reach_reference):
    surface = surface_of(DOJO)
    view = surface.view_config or {}
    return SimpleNamespace(
        surface=_context.Surface(tuple(surface.capabilities), dict(view),
                                 frame_local_axes(str(view.get("controlled_frame_convention"))),
                                 jaw_center_offsets(view)),
        workspace_bounds_m=BOUNDS, align_plan_callback=through_adapter(bridge), reach_reference=reach_reference)


def _observation(arm="right"):
    return {"proprioception": {"end_effectors": {arm: {
        "arm_joint_qpos": list(HOME), "gripper_command_target": "open", "pose_xyz_xyzw": START}}}}


def test_a_call_without_the_key_has_every_switch_on():
    assert _latency.on({}, "reach_reference") and _latency.on({"motion": {}}, "preview_images")
    assert _latency.host_on(SimpleNamespace(), "preview_images")
    assert _latency.host_on(SimpleNamespace(latency_switches=None), "reach_reference")
    assert not _latency.on(OFF, "reach_reference") and not _latency.host_on(SimpleNamespace(**OFF), "preview_images")
    context = _context.from_configuration(None, {"capabilities": [], "view_config": {}, "local_axes": {},
                                                 "jaw_offsets": {}, "workspace_bounds_m": BOUNDS,
                                                 "reach_available": True})
    assert context.reach_reference is True, "a sealed call's context asks the reach reference, as it did"


def test_move_pose_asks_the_reach_reference_when_the_switch_is_on():
    requests = []
    _pose, notes = _reach.plan_target(
        _host(_bridge(requests, exact=(HOME, None)), reach_reference=True), "right", current_pose=START,
        target_pose=[0.32, -0.03, 0.93, *REQUESTED], interaction="free_space",
        orientation_mode="absolute_quaternion", observation=_observation(), jaw_turn=False)
    assert requests and notes["reach_along_request"].get("status") != "off"


def test_move_pose_asks_no_reach_reference_when_the_switch_is_off():
    requests = []
    target = [0.32, -0.03, 0.93, *REQUESTED]
    pose, notes = _reach.plan_target(
        _host(_bridge(requests, exact=(HOME, None)), reach_reference=False), "right", current_pose=START,
        target_pose=target, interaction="free_space", orientation_mode="absolute_quaternion",
        observation=_observation(), jaw_turn=False)
    assert requests == [], "no IK question is asked for the annotation"
    assert pose == target and notes == {"reach_along_request": _latency.REACH_OFF}


def test_a_semantic_primitive_asks_no_reach_reference_when_the_switch_is_off():
    requests = []
    host = _host(_bridge(requests, exact=(HOME, None)), reach_reference=False)
    compiled = {"target_pose_xyz_xyzw": [0.32, -0.03, 0.93, *REQUESTED]}
    reference = _reach.primitive_reference(host, "right", compiled, current_pose=START, interaction="free_space")
    assert requests == [] and reference == {"reach_along_request": _latency.REACH_OFF}


def _unreadable():
    pytest.fail("a preview that draws nothing reads no camera")


def _view(*, render):
    manifest = {"proprioception": {"end_effectors": {"left": {"pose_xyz_xyzw": START, "gripper": 1.0}}}}
    return _dry_run.View(manifest, _unreadable if not render else (lambda: []), {}, None, None, render=render)


def _refusal():
    error = ToolRefusal("blocked", code="path_blocked")
    error.requested = {"left": [START, [0.4, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]]}
    return error


def test_a_refused_dry_run_has_no_picture_when_previews_are_off():
    record = _dry_run.refused("move_pose", _refusal(), _view(render=False))
    assert record["feasible"] is False and record["failure_code"] == "path_blocked"
    assert record["image"] == action_preview.PICTURE_OFF and "images_base64" not in record


def test_a_feasible_dry_run_keeps_its_numbers_without_a_picture(monkeypatch):
    monkeypatch.setattr(action_preview, "render", lambda *a, **k: pytest.fail("nothing is drawn"))
    seen = _dry_run.View(_view(render=False).manifest, lambda: [], {}, None, None, render=False)
    segments = [segment("arm", "left", "planned path", planned=True, native_actions=10, duration_s=0.4,
                        poses=[START, [0.1, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]])]
    record = _dry_run.composed("move_pose", segments, seen, route="joint")
    assert record["image"] == action_preview.PICTURE_OFF and "images_base64" not in record
    assert record["arms"]["left"]["native_actions"] == 10 and record["summary"], "the plan's numbers stay"


def test_a_refused_real_call_carries_no_picture_when_previews_are_off():
    error = _dry_run.pictured("move_pose", ToolRefusal("blocked", code="path_blocked"),
                              {"left": START}, {"left": START}, lambda: _view(render=False))
    assert not error.images_base64


def test_the_view_reads_the_switch_from_the_call_configuration():
    obs = SimpleNamespace(manifest=SimpleNamespace(json=lambda: {}), cameras=())
    assert _dry_run.view(None, obs, {"motion": {}}).render is True
    assert _dry_run.view(None, obs, {"motion": {}, **OFF}).render is False
