"""One image for the parallel ``mark_point`` calls of one turn (``batched_mark_overlay_v1``).

Several single-pixel calls in one assistant response each pay a tool round trip, but they share one overlay instead of
returning one image each. The host starts the batch, the calls run with their images held back, and this runs once after
the last of them, as no tool call of its own. The model gets the image on the last call that succeeded.
"""

import hashlib
from collections.abc import Mapping

from ...lib import images, observation
from ...lib.images import RenderedView
from ...lib.values import JsonValue
from ..move_pose import _atomic
from ..observe import _annotated, _pointcloud
from . import _batch, _single, tool


def _selected(calls, results, points):
    """The solved marks of the batch, each in its colour, and the views they were clicked on, in call order."""
    selected, views = [], []
    for call, result in zip(calls, results):
        point_id = str(call["point_id"])
        record = points.get(point_id)
        if not (result.get("success") is True and result.get("status") in {"solved", "verified"}
                and isinstance(record, Mapping) and record.get("status") == "solved"
                and isinstance(record.get("xyz_m"), list)):
            continue
        selected.append([point_id, list(record["xyz_m"]), list(tool.COLOURS[len(selected) % len(tool.COLOURS)])])
        view = str(call["view"])
        if view not in views:
            views.append(view)
    return selected, views


def _panels(api, config, obs, views, selected):
    """Every mark on each view, as (view, PNG); a view that is neither camera nor derived has no panel."""
    atomic, manifest = config["atomic"], obs.manifest.json()
    panels, clouds = [], None
    for view in views:
        source = _single.CANDIDATE_CAMERA if view == _single.CANDIDATE_VIEW else view
        if source in manifest.get("cameras", {}):
            rendered = _annotated.camera_view(api, obs, atomic, source, selected)
        elif source in _batch.POINTCLOUD_VIEWS:
            clouds = clouds or _pointcloud.render(api, obs, atomic["pointcloud"])
            rendered = _annotated.cloud_view(clouds, source, selected)
        else:
            continue
        _annotated.published(api, rendered, source, manifest, "mark_point_batch_panel")
        panels.append((source, rendered.png))
    return panels


def run(api, request):
    """Replay: recorded observation and tool.marks; pure rendering. ``request`` has the batch's calls and results."""
    config = api.artifacts.tool_context()
    obs = observation.observe(api)
    manifest = obs.manifest.json()
    _atomic.grip_state(manifest)
    selected, views = _selected(request["calls"], request["results"], api.state.read("tool.marks"))
    if not selected:
        return {}
    panels = _panels(api, config, obs, views, selected)
    if not panels:
        return {}
    resolution = tuple(config["mark_point"]["resolution"])
    sheet = images.contact_sheet(tuple(panels), points=JsonValue.of(selected), resolution=resolution)
    identity = {"observation_id": obs.identity.json()["observation_id"],
                "panels": [{"view": view, "sha256": hashlib.sha256(png).hexdigest()} for view, png in panels],
                "points": [{"point_id": name, "xyz_m": xyz, "colour": colour} for name, xyz, colour in selected],
                "output_resolution": list(resolution)}
    rendered = RenderedView(sheet, resolution[0], resolution[1], JsonValue.of(identity))
    return {"images_base64": [_annotated.published(api, rendered, panels[0][0], manifest, "mark_point_batch_feedback")]}
