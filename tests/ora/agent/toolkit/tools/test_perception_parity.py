"""The toolkit's perception tools answer as they did beside the handlers, on every other backend's surface.

Each call runs on the toolkit over one publication, each result through the backend's own closed schema, and its
result or refusal and the overlay files it writes are checked against the digests in ``tests/data/toolkit-goldens``,
recorded while the handlers answered each call alike. What differed for another unit's reason stays out, and says so.
"""

import json

import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.lib.pointcloud import VIEW_AXES
from ora.agent.toolkit.tools.observe import _pointcloud
from ora.agent.toolkit.host.observe import publication
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_perception import (
    BOX, SURFACES, Calls, toolkit_host, without_input_schema, without_public_facts,
)

GOLDEN = Goldens(__file__)

SINGLE_ARM = ("libero", "robocasa", "robolab")
VISION = (*SINGLE_ARM, "robotwin")


def cameras(backend):
    return SURFACES[backend][1]


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


def same(calls, name, arguments):
    return calls.call(name, arguments)


def overlays_kept(calls):
    """The overlays the toolkit wrote; the handlers kept each one, and ``mark_point``'s two more."""
    calls.check("overlays", calls.overlays(calls.new))


def code(outcome):
    return outcome["error"]["error"]["code"]


def message(outcome):
    return outcome["error"]["error"]["message"]


@pytest.mark.parametrize("backend", SURFACES)
def test_marks_are_measured_and_kept(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    calls = Calls(new, golden)
    first, second = cameras(backend)[:2]
    measured = same(calls, "mark_point", {"view": first, "point_ids": ["rim", "table"], "points_uv": [[30, 25], [5, 40]]})
    assert measured["result"]["status"] == "solved" and len(measured["result"]["images_base64"]) == 1
    same(calls, "mark_point", {"view": first, "point_ids": ["rim"], "points_uv": [[30, 25]]})  # the same pixel again
    same(calls, "mark_point", {"view": second, "point_ids": ["rim"], "points_uv": [[30, 25]]})  # another view disagrees
    same(calls, "mark_point", {"view": first, "point_ids": ["far", "off"], "points_uv": [[30, 25], [999, 5]]})
    golden("marks", new.toolkit_state_access.read("tool.marks"), new.control_root.parent)


@pytest.mark.parametrize("backend", SURFACES)
def test_a_view_outside_the_publication_fails_its_entries_and_a_cloud_view_is_refused(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    without_input_schema(new)
    calls = Calls(new, golden)
    missing = same(calls, "mark_point", {"view": "nowhere", "point_ids": ["a", "b"], "points_uv": [[1, 1], [2, 2]]})
    assert missing["result"]["status"] == "failed" and "images_base64" not in missing["result"]
    assert {mark["error_message"] for mark in missing["result"]["marks"]} == {"view is not markable"}
    refused = same(calls, "mark_point", {"view": "pointcloud_top", "point_ids": ["a"], "points_uv": [[1, 1]]})
    assert "stays on the single-pixel form" in message(refused)


@pytest.mark.parametrize("backend", SURFACES)
def test_geometry_queries_answer_each_form(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    without_input_schema(new)
    calls = Calls(new, golden)
    camera = cameras(backend)[0]
    for query in ({"points": [[30, 25], [5, 40]]}, {"points": [[30, 25]], "window_px": 0}, {"region": [0, 0, 10, 10]},
                  {"world_points_m": [[0, 0, 1.0], [0, 0, 9.0]]}, {"points": [[30, 25]], "step": 0},
                  {"points": [[999, 0]]}, {"region": [10, 0, 0, 10]}, {"points": [[30, 25]], "step": 99}):
        same(calls, "geometry_query", {"camera": camera, **query})
    for name in ("nowhere", 7):
        assert message(same(calls, "geometry_query", {"camera": name, "points": [[30, 25]]})) == (
            "camera is not policy-visible")


def test_an_uncalibrated_station_camera_still_answers_depth(golden, tmp_path, monkeypatch):
    new = toolkit_host(tmp_path, "yam", monkeypatch, uncalibrated=("front",))
    calls = Calls(new, golden)
    assert "region_result" in same(calls, "geometry_query", {"camera": "front", "region": [0, 0, 10, 10]})["result"]
    refused = same(calls, "mark_point", {"view": "front", "point_ids": ["a"], "points_uv": [[30, 25]]})
    assert code(refused) == "invalid_camera_calibration"
    # A pixel across the box's edge fails its own entry before the calibration is read.
    edge = same(calls, "mark_point", {"view": "front", "point_ids": ["a"], "points_uv": [[BOX[0], 25]]})
    assert edge["result"]["marks"][0]["error_code"] == "depth_discontinuity"


def test_a_camera_without_depth_refuses_alone(golden, tmp_path, monkeypatch):
    new = toolkit_host(tmp_path, "yam", monkeypatch, without_depth=("front",))
    calls = Calls(new, golden)
    assert code(same(calls, "mark_point", {"view": "front", "point_ids": ["a"], "points_uv": [[1, 1]]})) == (
        "artifact_integrity_fault")
    assert same(calls, "mark_point", {"view": "overhead", "point_ids": ["a"], "points_uv": [[30, 25]]})["result"]


@pytest.mark.parametrize("backend", VISION)
def test_only_the_footprint_surfaces_get_the_overhead_pose(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    calls = Calls(new, golden)
    camera = cameras(backend)[0]
    grounded = same(calls, "vision_ground", {"camera": camera, "text": "the box"})["result"]
    point = [(BOX[0] + BOX[2]) // 2, (BOX[1] + BOX[3]) // 2]
    segmented = same(calls, "vision_segment", {"camera": camera, "points_uv": [point]})["result"]
    saved = same(calls, "vision_segment", {"camera": camera, "points_uv": [point], "step": 0})["result"]
    for result in (grounded, segmented, saved):
        assert ("overhead_view_pose" in result["candidates"][0]["localization"]) == (backend in SINGLE_ARM)
    if backend in SINGLE_ARM:  # a saved step's target still gets a pose from the current wrist
        assert saved["candidates"][0]["localization"]["footprint_view"] == "oblique"
        assert grounded["candidates"][0]["localization"]["overhead_view_pose"]["camera"] == "wrist"


@pytest.mark.parametrize("wrist", [dict(absent=("wrist",)), dict(uncalibrated=("wrist",))], ids=["absent", "no_pose"])
def test_without_a_current_wrist_a_measurement_carries_no_pose(golden, tmp_path, monkeypatch, wrist):
    new = toolkit_host(tmp_path, "libero", monkeypatch, **wrist)
    calls = Calls(new, golden)
    grounded = same(calls, "vision_ground", {"camera": "agentview", "text": "the box"})["result"]
    localization = grounded["candidates"][0]["localization"]
    assert localization["footprint_view"] == "oblique" and "overhead_view_pose" not in localization


def test_a_grounding_camera_the_publication_lacks_is_not_policy_visible(golden, tmp_path, monkeypatch):
    new = toolkit_host(tmp_path, "libero", monkeypatch, absent=("wrist",))
    without_input_schema(new)
    calls = Calls(new, golden)
    assert message(same(calls, "vision_ground", {"camera": "wrist", "text": "the box"})) == "camera is not policy-visible"


@pytest.mark.parametrize("backend", ("robotwin", "yam"))
def test_dual_arm_observe_answers_views_and_history(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    without_input_schema(new)
    calls = Calls(new, golden)
    camera = cameras(backend)[0]
    same(calls, "mark_point", {"view": camera, "point_ids": ["rim"], "points_uv": [[30, 25]]})
    same(calls, "observe", {"views": [camera], "history_point_ids": ["rim"]})
    same(calls, "observe", {})
    word = {"robotwin": "RoboTwin", "yam": "station"}[backend]
    for views in (["nowhere"], [camera, camera]):
        refused = same(calls, "observe", {"views": views})
        assert message(refused) == f"views must be a non-empty unique {word} camera list"
    overlays_kept(calls)


@pytest.mark.parametrize("backend", SINGLE_ARM)
def test_single_arm_observe_refuses_unknown_and_saved_cloud_views(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch, both_arms=True)
    without_input_schema(new)
    without_public_facts(new)
    calls = Calls(new, golden)
    unknown = same(calls, "observe", {"views": ["nowhere"]})
    assert message(unknown) == "unknown observe view"
    assert set(VIEW_AXES) < set(unknown["error"]["error"]["constraints"]["enum"])
    assert message(same(calls, "observe", {"views": ["agentview", "agentview"]})) == "views must be a non-empty unique list"
    saved = same(calls, "observe", {"step": 0, "views": ["pointcloud_top"]})
    assert message(saved) == "historical observe requires physical camera views"
    assert set(saved["error"]["error"]["constraints"]["enum"]) == {"agentview", "wrist"}


@pytest.mark.parametrize("backend", SINGLE_ARM)
def test_point_cloud_views_are_drawn(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch, both_arms=True)
    without_public_facts(new)
    calls = Calls(new, golden)
    same(calls, "mark_point", {"view": "wrist", "point_ids": ["rim", "off"], "points_uv": [[30, 25], [32, 24]]})
    for arguments in ({"views": ["pointcloud_top"]}, {"views": list(VIEW_AXES), "history_point_ids": ["rim", "off"]},
                      {"views": ["pointcloud_front", "pointcloud_side"]}):
        actual = calls.call("observe", arguments, checked=False)
        # Everything but the arm fields, which are the host lane's (rows 7 and 19).
        kept = {key: actual["result"][key] for key in ("success", "observation_id", "views", "view_dimensions",
                                                      "images_base64", "image_mapping", "measurement_contract")}
        golden(f"observe-{calls.index - 1}", kept, new.control_root.parent)
    overlays_kept(calls)


@pytest.mark.parametrize("backend", SINGLE_ARM)
def test_a_call_of_only_cloud_views_gets_the_robot_facts_with_no_camera(golden, tmp_path, monkeypatch, backend):
    new = toolkit_host(tmp_path, backend, monkeypatch)
    control, root, manifest = new._current()
    identity = {key: control[key] for key in ("observation_id", "observation_sha256")}
    published = publication(new, BlobStore(new.control_root.parent), identity, root, manifest)
    facts = _pointcloud.robot_facts(json.loads(published.data)["public_robot_facts_by_view"])
    golden("facts", facts, new.control_root.parent)
    assert facts["cameras"] == {}
