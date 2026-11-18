"""A step's tool images in the Viewer, from what the toolkit logs.

The Viewer draws a step's derived images from what the tools log: ``openeta_view_derived``,
``grounding_query_materialized`` and ``grasp_plan_materialized``. The toolkit keeps all three: each test runs its calls on
a stand-in host on the toolkit (``tests/support/agent/stack_parity.py``) and checks what ``compile_trajectory`` shows and
the events it read against the digests in ``tests/data/toolkit-goldens``, recorded while the tool handlers showed the
same. A query answered from its cache logs as a new one does, since a step without the event shows no image. RoboDojo's
toolkit surfaces log none of this.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agent.stack_parity import OBSERVATION, on_toolkit, stand_in
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import install, make_host, use_toolkit_v5
from viewer.trajectory import compile_trajectory

WITH_GROUNDING = ("libero", "robotwin", "robocasa", "robolab")
MARKS = {"point_ids": ["block"], "points_uv": [[64, 64]]}
LOGGED = {"openeta_view_derived", "grounding_query_materialized", "grasp_plan_materialized"}
CAMERA = {"libero": "agentview", "robotwin": "head_camera", "robocasa": "agentview", "robolab": "agentview",
          "yam": "overhead"}
GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


def shown(golden: Any, host: Any) -> None:
    """What the Viewer shows of the step and the events it read, against the toolkit's recorded digest."""
    golden("viewer", {"images": images(host), "logged": logged(host)}, host.control_root.parent)


def stepped(tmp_path: Path, key: str, monkeypatch: Any) -> Any:
    """A stand-in on the toolkit with the Viewer's step open."""
    host = on_toolkit(stand_in(tmp_path / "toolkit", key, monkeypatch))
    with host.event_path.open("a") as stream:
        stream.write(json.dumps({"type": "observation", "observation_id": OBSERVATION,
                                 "policy_state": {"step": 0}}) + "\n")
    return host


def call(host: Any, index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    invoke = getattr(host, "invoke_with_context", None)
    if invoke is None:
        return host.invoke(name, arguments)
    return invoke(name, arguments, call_context=proposal(host, name, arguments, index=index))


def images(host: Any) -> list[dict[str, Any]]:
    """What the Viewer shows of the step besides the observation's own cameras."""
    step = compile_trajectory(host.control_root.parent)["steps"][0]
    return [image for image in step["images"] if image.get("kind") not in {"rgb", "observation", None}]


def logged(host: Any) -> list[dict[str, Any]]:
    """The events without the clock: ``time``, and a grasp manifest's bytes, which hold the service's latency."""
    rows = [json.loads(line) for line in host.event_path.read_text().splitlines()]
    rows = [{key: value for key, value in row.items() if key != "time"} for row in rows if row["type"] in LOGGED]
    for row in rows:
        if "manifest_artifact" in row:
            row["manifest_artifact"] = row["manifest_artifact"]["path"]
    return rows


def ground(host: Any, key: str, index: int = 0) -> dict[str, Any]:
    return call(host, index, "vision_ground", {"camera": CAMERA[key], "text": "block"})


@pytest.mark.parametrize("key", WITH_GROUNDING)
def test_grounding_shows_its_overview_and_candidate_cards(golden, tmp_path, monkeypatch, key) -> None:
    toolkit = stepped(tmp_path, key, monkeypatch)
    ground(toolkit, key)
    assert [image["kind"] for image in images(toolkit)] == ["grounding", "grounding_candidate"]
    shown(golden, toolkit)


@pytest.mark.parametrize("key", WITH_GROUNDING)
def test_a_cached_grounding_still_shows_in_its_step(golden, tmp_path, monkeypatch, key) -> None:
    """A repeated query is answered from its cache and publishes nothing: the result says which query it was."""
    toolkit = stepped(tmp_path, key, monkeypatch)
    ground(toolkit, key), ground(toolkit, key, index=1)
    assert len(logged(toolkit)) == 2
    shown(golden, toolkit)


def plan(host: Any, key: str, index: int) -> dict[str, Any]:
    """``grasp_plan`` of the block grounded first; a single-arm surface publishes no ``arm``."""
    arguments = {"target_ref": ground(host, key)["candidates"][0]["target_ref"]}
    return call(host, index, "grasp_plan", {**arguments, **({"arm": "right"} if key == "robotwin" else {})})


@pytest.mark.parametrize("key", WITH_GROUNDING)
def test_a_grasp_plan_shows_its_contact_sheet_and_its_review(golden, tmp_path, monkeypatch, key) -> None:
    toolkit = stepped(tmp_path, key, monkeypatch)
    plan(toolkit, key, 1)
    sheets = [image for image in images(toolkit) if image["kind"] == "grasp_plan"]
    assert {image["visibility"] for image in sheets} == {"policy_visible", "review_only"}
    shown(golden, toolkit)


def test_a_repeated_grasp_plan_still_shows_in_its_step(golden, tmp_path, monkeypatch) -> None:
    toolkit = stepped(tmp_path, "libero", monkeypatch)
    arguments = {"target_ref": ground(toolkit, "libero")["candidates"][0]["target_ref"]}
    call(toolkit, 1, "grasp_plan", arguments), call(toolkit, 2, "grasp_plan", arguments)
    assert sum(row["type"] == "grasp_plan_materialized" for row in logged(toolkit)) == 2
    shown(golden, toolkit)


@pytest.mark.parametrize("key", [*WITH_GROUNDING, "yam"])
def test_marks_show_their_annotated_view_and_their_sheet(golden, tmp_path, monkeypatch, key) -> None:
    toolkit = stepped(tmp_path, key, monkeypatch)
    call(toolkit, 0, "mark_point", {"view": CAMERA[key], **MARKS})
    assert [image["kind"] for image in images(toolkit)] == ["mark_point_batch_panel", "mark_point_batch_feedback"]
    shown(golden, toolkit)
    for image in images(toolkit):
        assert (toolkit.control_root.parent / image["path"]).is_file()


def test_a_call_that_marks_nothing_shows_nothing(tmp_path, monkeypatch) -> None:
    toolkit = stepped(tmp_path, "libero", monkeypatch)
    call(toolkit, 0, "mark_point", {"view": "agentview", "point_ids": ["far"], "points_uv": [[5000, 5000]]})
    assert logged(toolkit) == []


@pytest.fixture
def robodojo(tmp_path, monkeypatch):
    @contextmanager
    def invocation(name, context, budget):
        yield SimpleNamespace()

    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    complete_cameras(host)
    use_toolkit_v5(host)
    host.toolkit_executor = SimpleNamespace(invocation=invocation)
    install(host)
    try:
        yield host
    finally:
        client.close()


def test_robodojo_logs_and_keeps_none_of_this(robodojo) -> None:
    """Its tool contexts carry no ``environment_plans``, which is what a sealed one replays against."""
    run = robodojo.control_root.parent
    ground_arguments = {"camera": "head", "text": "object", "max_candidates": 1}
    target = call(robodojo, 0, "vision_ground", ground_arguments)["candidates"][0]["target_ref"]
    call(robodojo, 1, "grasp_plan", {"target_ref": target, "arm": "left"})
    marked = call(robodojo, 2, "mark_point", {"view": "head", "point_ids": ["block"], "points_uv": [[300, 195]]})
    assert marked["images_base64"], "the model still gets its sheet"
    written = [] if not robodojo.event_path.exists() else robodojo.event_path.read_text().splitlines()
    assert not [row for row in map(json.loads, written) if row["type"] in LOGGED]
    assert not list(run.glob("openeta-tool-artifacts/*/overlays/*"))
