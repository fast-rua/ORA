"""``mark_point`` as one pixel per call, the form an atomic surface publishes (``config["atomic"]``).

The camera path is the batch's ``tool.measure`` at the surface's RGB resolution. What only this form has: ``candidate_agentview``,
a preview's picture of the agentview, measures on the agentview; a pixel of a point-cloud view fixes two world axes, so the
point stays ``pending`` until a complementary view fixes the third; and a solved call returns one overlay per source view,
unless it is one of several parallel calls of one turn (``_feedback.py``), which share one image.
"""

from collections.abc import Mapping

from ...lib import observation, pointcloud
from ..errors import ToolRefusal
from ..move_pose import _atomic
from ..observe import _annotated, _pointcloud
from . import _batch, tool

FIELDS = {"point_id", "view", "x", "y"}
CANDIDATE_VIEW = "candidate_agentview"
CANDIDATE_CAMERA = "agentview"
FEEDBACK_COLOUR = [255, 0, 255]
#: Two clicks of one axis on complementary views may disagree by this much, metres, before the point is refused.
CLICK_TOLERANCE_M = 0.02


def run(api, arguments, config):
    keys = set(arguments)
    if keys == FIELDS:
        return measure(api, config, point_id=arguments["point_id"], view=arguments["view"], x=arguments["x"],
                       y=arguments["y"])
    if keys & {"point_ids", "points_uv"}:
        raise _batch._reject("this tool surface measures one pixel per mark_point call; call it once per point with "
                             "point_id, view, x and y", accepted_fields=["point_id", "view", "x", "y"])
    raise ToolRefusal("mark_point requires point_id, view, x and y")


def _verified_click(previous, constraint, identity):
    """A pixel of a derived view on a solved point: it must agree with the point on both axes the view shows."""
    known = dict(zip(("x", "y", "z"), previous["xyz_m"]))
    disagreement = max(abs(float(known[axis]) - float(value)) for axis, value in constraint.items())
    if disagreement > tool.VERIFY_TOLERANCE_M:
        raise ToolRefusal(f"independent pointcloud measurements disagree by {disagreement:.4f} m, "
                          f"more than the {tool.VERIFY_TOLERANCE_M} m maximum", code="inconsistent_mark",
                          constraints={"distance_m": disagreement, "maximum_m": tool.VERIFY_TOLERANCE_M})
    return {**previous, "sources": [*previous.get("sources", []), identity],
            "last_verification": {"observation_id": identity["observation_id"], "axis_max_error_m": disagreement}}


def _new_click(previous, constraint, identity, point_id):
    """A pixel of a derived view on an unsolved point: its axes join the point's, and the third axis solves it."""
    pending = previous if isinstance(previous, Mapping) else {}
    constraints = dict(pending.get("constraints", {}))
    for axis, value in constraint.items():
        distance = abs(float(constraints.get(axis, value)) - value)
        if distance > CLICK_TOLERANCE_M:
            raise ToolRefusal(f"complementary pointcloud clicks disagree on {axis} by {distance:.4f} m, more than the "
                              f"{CLICK_TOLERANCE_M} m maximum", code="inconsistent_mark",
                              constraints={"axis": axis, "distance_m": distance, "maximum_m": CLICK_TOLERANCE_M})
        constraints[axis] = value
    fields = {"point_id": point_id, "observation_id": identity["observation_id"],
              "sources": [*pending.get("sources", []), identity]}
    if set(constraints) == {"x", "y", "z"}:
        return {"status": "solved", **fields, "xyz_m": [constraints[axis] for axis in ("x", "y", "z")]}
    return {"status": "pending", **fields, "constraints": constraints}


def _measured(api, config, obs, points, *, point_id, view, x, y, identity):
    """Measure the pixel on the view it names; the status the call reports, and the record it leaves."""
    atomic, bounds = config["atomic"], config["motion"]["workspace_bounds_m"]
    previous = points.get(point_id)
    camera = CANDIDATE_CAMERA if view == CANDIDATE_VIEW else view
    if camera in obs.manifest.json().get("cameras", {}):
        failure = obs.result.data.json().get("camera_errors", {}).get(camera)
        if failure:
            raise ToolRefusal.from_public(failure)
        size = atomic["rgb_resolution"]
        return tool.measure(points=points, point_id=point_id, view=view, pixel=(x, y), obs=obs,
                            camera=observation.load_pixels(api, obs.camera(camera)), bounds=bounds, quality={},
                            size=None if size == "native" else tuple(size))
    if view not in _batch.POINTCLOUD_VIEWS:
        raise ToolRefusal("view is not markable")
    clicked = pointcloud.click_constraint(_pointcloud.render(api, obs, atomic["pointcloud"])[view], x, y)
    solved = isinstance(previous, Mapping) and previous.get("status") == "solved"
    record = (_verified_click(previous, clicked, identity) if solved
              else _new_click(previous, clicked, identity, point_id))
    if record["status"] == "solved":
        record["action_workspace"] = tool.marked_workspace(record["xyz_m"], bounds)
    points[point_id] = record
    return "verified" if solved else record["status"], record


def _overlays(api, config, obs, point_id, record):
    """The mark on every view that measured it, on this observation's pixels."""
    atomic, manifest = config["atomic"], obs.manifest.json()
    labelled = [[point_id, list(record["xyz_m"]), FEEDBACK_COLOUR]]
    images, clouds = [], None
    for source in record["sources"]:
        view = str(source["view"])
        if view in manifest.get("cameras", {}) or view == CANDIDATE_VIEW:
            camera = CANDIDATE_CAMERA if view == CANDIDATE_VIEW else view
            rendered = _annotated.camera_view(api, obs, atomic, camera, labelled)
            images.append(_annotated.published(api, rendered, camera, manifest, "mark_point_feedback"))
        elif view in _batch.POINTCLOUD_VIEWS:
            clouds = clouds or _pointcloud.render(api, obs, atomic["pointcloud"])
            rendered = _annotated.cloud_view(clouds, view, labelled)
            images.append(_annotated.published(api, rendered, view, manifest, "mark_point_feedback"))
    return images


def measure(api, config, *, point_id, view, x, y):
    """Replay: recorded observation and tool.marks; pure measurement and overlays. World metres, pixels at the view's size."""
    if not isinstance(point_id, str) or not point_id or not isinstance(view, str):
        raise ToolRefusal("point_id and view must be non-empty strings")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
        raise ToolRefusal("x and y must be integers")
    obs = observation.observe(api)
    _atomic.grip_state(obs.manifest.json())
    points = api.state.read("tool.marks")
    identity = {"observation_id": obs.identity.json()["observation_id"], "view": view, "x": x, "y": y}
    reused = tool.reused(points, point_id, identity)
    if reused is not None:
        workspace = reused.get("action_workspace", tool.marked_workspace(reused["xyz_m"], config["motion"]["workspace_bounds_m"]))
        return {"success": True, "status": "solved", "point_id": point_id, "xyz_m": reused["xyz_m"],
                "views": [source["view"] for source in reused.get("sources", [])], "action_workspace": dict(workspace)}
    status, record = _measured(api, config, obs, points, point_id=point_id, view=view, x=x, y=y, identity=identity)
    api.state.write("tool.marks", points)
    result = {"success": True, "status": status, "point_id": point_id}
    if record["status"] != "solved":
        return {**result, "next_views": sorted(set(_batch.POINTCLOUD_VIEWS) - {source["view"] for source in record["sources"]})}
    result.update(xyz_m=record["xyz_m"], views=[source["view"] for source in record["sources"]],
                  action_workspace=record["action_workspace"])
    if not config["atomic"]["feedback_batch"]:
        result["images_base64"] = _overlays(api, config, obs, point_id, record)
    return result
