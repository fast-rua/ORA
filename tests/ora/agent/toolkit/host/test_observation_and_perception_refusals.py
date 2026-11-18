"""Saved-step, pixel and mark refusals name the field, what arrived and the bounds.

Refusal census A3, B2, B7, C1-C5 (0059 open-bugs triage rows 7, 12, 14,
15, 16, 25, 26, 27):

* ``step`` equal to the current step is the current observation, not history;
* a step outside the saved range is an argument refusal naming the range, and
  the round reader no longer speaks of "placement";
* one pixel outside the image names its index and the image size, a reversed
  rectangle is not called "outside", and an extreme world point is refused as
  that argument instead of overflowing into an infrastructure error;
* ``history_point_ids`` refusals name the unsolved ids and the solved ones;
* the Harness's own public robot facts failing is infrastructure.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.host.shared import saved_steps
from ora.agent.error_classes import ToolError


def _refused(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


def _robotwin(tmp_path):
    from tests.support.robotwin.public_run import RoboTwinPublicRun

    run = RoboTwinPublicRun(tmp_path, bind_submitted=True, toolkit=True)
    run.runtime.invoke("set_gripper", {"targets": {"right": "open"}})
    run.publish({"right": "open"})
    return run, run.runtime


# --- saved steps (rows 7, 14, 25) ---------------------------------------------

def test_the_current_step_routes_to_the_current_path():
    host = SimpleNamespace(_current=lambda: ({"policy_state": {"step": 4}}, None, None))
    assert saved_steps.route_current(host, 4) is None
    assert saved_steps.route_current(host, 3) == 3

    def closed():
        raise ToolError("closed", code="action_pending_fresh_observation", retryable=False)

    assert saved_steps.route_current(SimpleNamespace(_current=closed), 4) == 4


def test_observe_at_the_current_step_is_current_and_draws_history(tmp_path):
    _run, tools = _robotwin(tmp_path)
    current = tools._current()[0]["policy_state"]["step"]
    result = tools.invoke("observe", {"step": current, "views": ["left_camera"], "history_point_ids": []})
    assert result.get("status") != "actual_historical_observation"
    assert tools.invoke("observe", {"step": 0, "views": ["left_camera"]})["status"] == (
        "actual_historical_observation")


def test_a_step_outside_the_round_is_an_argument_with_its_range(tmp_path):
    _run, tools = _robotwin(tmp_path)
    current = tools._current()[0]["policy_state"]["step"]
    for tool, arguments in (("observe", {"step": 99, "views": ["left_camera"]}),
                            ("vision_segment", {"step": 99, "camera": "left_camera", "points_uv": [[48, 48]]})):
        error = _refused(tools.invoke, tool, arguments)
        assert (error["code"], error["field"], error["received"]) == ("invalid_arguments", "step", 99)
        assert error["constraints"] == {"minimum": 0, "maximum": current, "scope": "current_round"}
        assert "placement" not in error["message"]


def test_a_run_without_public_state_is_infrastructure(tmp_path):
    _run, tools = _robotwin(tmp_path)
    tools.state_root = None
    error = _refused(saved_steps.round_timeline, tools, tools._current()[0])
    assert error["code"] == "artifact_integrity_fault" and error["infrastructure"] is True
    assert "placement" not in error["message"]


def test_geometry_query_states_the_saved_range(tmp_path):
    from tests.support.orchestrator.runtime_v1 import publish_synthetic

    *_rest, tools, _store = publish_synthetic(tmp_path)
    error = _refused(tools.invoke, "geometry.query", {"camera": "head_camera", "points": [[1, 1]], "step": 7})
    assert (error["code"], error["field"]) == ("invalid_arguments", "step")
    assert error["constraints"] == {"minimum": 0, "maximum": 0, "scope": "run"}


def test_a_historical_view_is_read_only_and_bound_to_its_round(tmp_path):
    run, tools = _robotwin(tmp_path)
    current = copy.deepcopy(tools._current_grip_state()[0])
    result = tools.invoke("observe", {"step": 0, "views": ["left_camera"]})
    assert result["status"] == "actual_historical_observation"
    assert result["observation_step"] == 0
    from tests.support.agent.batch_segmentation import BatchBackend
    tools.segmentation_backend = BatchBackend()  # the default surface segments by point batch
    target = tools.invoke("vision_segment", {"step": 0, "camera": "left_camera", "points_uv": [[48, 48]]})
    assert target["observation_id"] == run.frames[0]["root"].name
    assert tools._current_grip_state()[0] == current
    from ora.agent.session_core.domain import PolicyDomainMixin
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = SimpleNamespace(surface=tools.surface)
    visible = planner._causal_tool_result("observe", result)
    assert visible["status"] == "actual_historical_observation"
    assert visible["observation_step"] == 0
    assert visible["views"] == ["left_camera"]
    tools.reset_round(1)
    run.publish(reset=True)
    with pytest.raises(ToolError, match="current round"):
        tools.invoke("observe", {"step": 0, "views": ["left_camera"]})


# --- history overlays (row 27) ------------------------------------------------

def test_history_point_ids_name_the_unsolved_and_the_solved_ids(tmp_path):
    _run, tools = _robotwin(tmp_path)
    error = _refused(tools.invoke, "observe", {"views": ["left_camera"], "history_point_ids": ["rim"]})
    assert (error["field"], error["received"]) == ("history_point_ids", ["rim"])
    assert error["constraints"] == {"solved_point_ids": []}


def test_a_saved_step_refuses_overlays_by_field(tmp_path):
    _run, tools = _robotwin(tmp_path)
    error = _refused(tools.invoke, "observe", {"step": 0, "views": ["left_camera"], "history_point_ids": ["rim"]})
    assert (error["field"], error["received"]) == ("history_point_ids", ["rim"])
    assert error["constraints"] == {"allowed_with_step": ["step", "views"]}


# --- pixels, rectangles and world points (rows 15, 16) ------------------------

@pytest.mark.parametrize("arguments,field,message", [
    ({"region": [10, 10, 5, 20]}, "region", "x0 < x1 and y0 < y1"),
    ({"region": [0, 0, 99999, 5]}, "region", "is outside the"),
    ({"points": [[1, 1], [99999, 5]]}, "points[1]", "is outside the"),
])
def test_geometry_query_refusals_name_the_entry_and_the_image(tmp_path, arguments, field, message):
    from tests.support.orchestrator.runtime_v1 import publish_synthetic

    *_rest, tools, _store = publish_synthetic(tmp_path)
    error = _refused(tools.invoke, "geometry.query", {"camera": "head_camera", **arguments})
    assert (error["code"], error["field"]) == ("invalid_arguments", field)
    assert message in error["message"]


def test_an_extreme_world_point_is_out_of_frame_not_an_internal_error(tmp_path):
    from tests.support.orchestrator.runtime_v1 import publish_synthetic

    *_rest, tools, _store = publish_synthetic(tmp_path)
    # In front of the camera, but its projection overflows: no pixel exists.
    result = tools.invoke("geometry.query", {"camera": "head_camera", "world_points_m": [[1e308, 0.0, 1e-8]]})
    entry = result["world_points"][0]
    assert (entry["in_front"], entry["in_frame"]) == (True, False)
    assert "pixel_uv" not in entry


def test_a_world_point_without_finite_camera_coordinates_is_that_argument():
    from ora.agent.toolkit.tools.geometry_query import _arguments as geometry_arguments

    with pytest.raises(ToolRefusal) as caught:
        geometry_arguments.finite_camera_point([float("inf"), 0.0, 1.0], [1e308, 1e308, 0], 2)
    assert (caught.value.code, caught.value.fields["field"]) == ("invalid_arguments", "world_points_m[2]")


# --- marks (row 12) -----------------------------------------------------------

def test_a_repeated_point_id_is_named():
    from ora.agent.toolkit.tools.mark_point import _batch

    with pytest.raises(ToolRefusal) as caught:
        _batch._batch_arguments({
            "view": "agentview", "point_ids": ["a", "b", "a"], "points_uv": [[1, 1], [2, 2], [3, 3]]})
    error = caught.value.fields
    assert (error["field"], error["received"]) == ("point_ids[2]", "a")
    assert error["constraints"] == {"first_index": 0}


# --- public robot facts (row 26) ----------------------------------------------

def test_failed_public_robot_facts_are_infrastructure(tmp_path, monkeypatch):
    from ora.agent.error_classes import PublicFactsError
    from tests.support.agent.move_pose_reach import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)

    def drifted(**_kwargs):
        raise PublicFactsError("public geometry descriptor is missing or drifted")

    monkeypatch.setattr(runtime, "public_facts_projector", drifted)
    _pointer, root, observation = runtime._current()
    error = _refused(runtime._attach_observe_public_facts, {"views": []}, root, observation)
    assert error["code"] == "embodiment_binding"
    assert error["retryable"] is False and error["infrastructure"] is True
