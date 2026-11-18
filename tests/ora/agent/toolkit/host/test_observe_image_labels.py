"""Each image an observe call returns is named: its camera, its place in the call and the marks drawn on it.

A dual-arm observe (YAM, RoboTwin, RoboDojo) renders one image per requested
view, in the order of ``views``, with each arm's TCP drawn as ``L`` and ``R``
where it falls inside that view; a historical observe returns the saved images
without marks. ``observe_image_labels`` says so per image and the Pi tool
projection (``PiPlanner._project``) returns those labels as ``image_labels``,
which free play puts before each image (ora-pi/src/play/driver.ts).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

from ora.agent.toolkit.host.observe_labels import observe_image_labels
from tests.support.agent.observation_delivery import host

YAM = SimpleNamespace(capabilities=frozenset({"yam_rgbd_v1"}), supports=lambda name: name == "yam_rgbd_v1")


def _png() -> str:
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (4, 3), (1, 2, 3)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_a_current_observe_names_each_view_in_order_and_the_tcp_marks_on_it() -> None:
    labels = observe_image_labels(YAM, "observe", {"views": ["overhead", "front", "wrist_left", "wrist_right"]}, 4)
    assert labels == [
        f"observe image {index} of 4: {view} camera. Marks: L = left arm TCP, R = right arm TCP, each drawn "
        "only where it falls inside this view; names from history_point_ids are drawn the same way."
        for index, view in enumerate(["overhead", "front", "wrist_left", "wrist_right"], start=1)
    ]


def test_a_historical_observe_is_named_as_saved_and_unmarked() -> None:
    result = {"views": ["front"], "status": "actual_historical_observation", "observation_step": 7}
    assert observe_image_labels(YAM, "observe", result, 1) == [
        "observe image 1 of 1: front camera, the saved image of observation step 7; no marks drawn."]


def test_no_labels_where_the_images_cannot_be_named_one_to_one() -> None:
    assert observe_image_labels(YAM, "observe", {"views": ["overhead", "front"]}, 1) is None
    assert observe_image_labels(YAM, "mark_point", {"views": ["overhead"]}, 1) is None
    single_arm = SimpleNamespace(capabilities=frozenset(), supports=lambda name: False)
    assert observe_image_labels(single_arm, "observe", {"views": ["agentview"]}, 1) is None
    assert observe_image_labels(None, "observe", {"views": ["overhead"]}, 1) is None


def test_the_pi_projection_returns_the_labels_beside_the_images(tmp_path: Path) -> None:
    planner = host(tmp_path, tools=SimpleNamespace(surface=YAM), context_composition=None)
    result = {"success": True, "views": ["overhead", "front"], "images_base64": [_png(), _png()]}
    projected = planner._project("observe", result, call_id="call-1", step=3)
    assert len(projected["images"]) == 2
    assert [label.split(".")[0] for label in projected["image_labels"]] == [
        "observe image 1 of 2: overhead camera", "observe image 2 of 2: front camera"]
    other = planner._project("geometry_query", {"success": True}, call_id="call-2", step=3)
    assert other["image_labels"] is None
