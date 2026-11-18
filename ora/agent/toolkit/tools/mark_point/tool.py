"""Named public RGB-D measurements with explicit state-owned verification."""

import base64
from collections.abc import Mapping

import numpy as np

from ...interface.types import digest
from ...lib import images, observation
from ...lib._workspace import workspace_state
from ...lib.values import JsonValue
from ..errors import ToolRefusal
from ..observe.tool import publish_overlay
from . import _batch, _measurement

COLOURS = ((255, 64, 64), (64, 255, 255), (255, 220, 64), (96, 255, 96),
           (255, 128, 255), (96, 160, 255), (255, 160, 64), (255, 255, 255))
ENTRY_ERRORS = {"invalid_arguments", "invalid_depth", "insufficient_depth_support",
                "depth_discontinuity", "inconsistent_mark", "stale_point"}
#: Two independent measurements of one name may disagree by this much, metres, before the second is refused.
VERIFY_TOLERANCE_M = .025


def reused(points, point_id, identity):
    """The solved record this exact pixel already measured, or None. A pending point of another observation is stale."""
    previous = points.get(point_id)
    solved = isinstance(previous, Mapping) and previous.get("status") == "solved"
    if solved and any(all(source.get(key) == value for key, value in identity.items())
                      for source in previous.get("sources", []) if isinstance(source, Mapping)):
        return previous
    if isinstance(previous, Mapping) and not solved and previous.get("observation_id") != identity["observation_id"]:
        raise ToolRefusal("pending point is stale after world change", code="stale_point")
    return None


def marked_workspace(xyz_m, bounds):
    """Whether a measured point is inside the action workspace, which says nothing of reach."""
    workspace = workspace_state(xyz_m, bounds)
    return {**workspace, "semantics": "measurement_not_reachability",
            "motion_use": "candidate_requires_motion_preview" if workspace["inside"]
            else "measurement_only_outside_action_envelope"}


def measure(*, points, point_id, view, pixel, obs, camera, bounds, quality, size=None):
    """Measure or verify one point, updating only the supplied explicit map. A click's pixels are ``size`` when it is given,
    else the camera's own."""
    x, y = pixel
    identity = {"observation_id": obs.identity.json()["observation_id"], "view": view, "x": x, "y": y}
    previous = points.get(point_id)
    solved = isinstance(previous, Mapping) and previous.get("status") == "solved"
    if reused(points, point_id, identity) is not None:
        return "solved", previous
    if camera is None:
        raise ToolRefusal("view is not markable")
    width, height = size or camera.rgb.numpy().shape[1::-1]
    if not (0 <= x < width and 0 <= y < height):
        raise ToolRefusal(f"click [{x}, {y}] is outside the {width}x{height} {view} view",
                          field="x" if not 0 <= x < width else "y", received=[x, y],
                          constraints={"width": width, "height": height, "coordinate_convention": "opencv_top_left_xy"})
    depth = camera.depth_m.numpy()
    source_x = int(round(x * (depth.shape[1] - 1) / max(width - 1, 1)))
    source_y = int(round(y * (depth.shape[0] - 1) / max(height - 1, 1)))
    calibration = camera.calibration.json()
    z, measured = _measurement.robust_depth_measurement(depth, source_x=source_x, source_y=source_y,
                                                       calibration=calibration, quality_sink=quality)
    intrinsic = np.asarray(calibration.get("intrinsic_cv"), dtype=float)
    if (intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic))
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ToolRefusal("camera intrinsic_cv must be a finite 3x3 OpenCV matrix with positive focal lengths",
                          code="invalid_camera_calibration")
    xyz = _measurement.opencv_camera_point_to_world(
        [(source_x-intrinsic[0, 2])*z/intrinsic[0, 0],
         (source_y-intrinsic[1, 2])*z/intrinsic[1, 1], z, 1.], calibration=calibration)
    source = {**identity, "source_x": source_x, "source_y": source_y, "measurement_quality": measured}
    if solved:
        disagreement = float(np.linalg.norm(np.asarray(previous["xyz_m"], float) - np.asarray(xyz, float)))
        if disagreement > VERIFY_TOLERANCE_M:
            raise ToolRefusal(f"independent RGB-D measurements disagree by {disagreement:.4f} m, more than the {VERIFY_TOLERANCE_M} m maximum",
                              code="inconsistent_mark", constraints={"distance_m": disagreement, "maximum_m": VERIFY_TOLERANCE_M})
        record = {**previous, "sources": [*previous.get("sources", []), source],
                  "last_verification": {"observation_id": identity["observation_id"],
                                        "distance_m": disagreement, "measurement_quality": measured}}
    else:
        record = {"status": "solved", "point_id": point_id, "xyz_m": xyz,
                  "observation_id": identity["observation_id"], "sources": [source]}
    record["action_workspace"] = marked_workspace(record["xyz_m"], bounds)
    points[point_id] = record
    return "verified" if solved else "solved", record


def keep_overlays(api, rendered, composite, *, view, observation_id, selected, resolution):
    """Keep the annotated view and the sheet under the identities the handlers give them, which the Viewer shows."""
    manifest = dict(observation_id=observation_id)
    panel = JsonValue.of({**rendered.identity.json(), "kind": "mark_point_batch_panel"})
    publish_overlay(api, images.RenderedView(rendered.png, rendered.width, rendered.height, panel), view, manifest)
    identity = dict(kind="mark_point_batch_feedback", observation_id=observation_id,
                    panels=[dict(view=view, sha256=digest(rendered.png))],
                    points=[dict(point_id=name, xyz_m=xyz, colour=colour) for name, xyz, colour in selected],
                    output_resolution=list(resolution))
    publish_overlay(api, images.RenderedView(composite, *resolution, JsonValue.of(identity)), view, manifest)


def run(api, arguments):
    """Replay: recorded observation and tool.marks; pure measurement and overlays.

    Named points use episode-local world metres. Missing recorded pixels/state
    fails; this operation performs no simulator query or physical action.
    """
    config = api.artifacts.tool_context()
    if "atomic" in config:  # one pixel per call
        from . import _single

        return _single.run(api, arguments, config)
    view, names, pixels = _batch._batch_arguments(arguments)
    obs = observation.observe(api)
    camera_error = obs.result.data.json().get("camera_errors", {}).get(view)
    if camera_error:
        raise ToolRefusal.from_public(camera_error)
    # A view the observation does not have fails each entry, as an unmeasurable pixel does.
    refs = obs.camera(view) if view in obs.manifest.json().get("cameras", {}) else None
    camera = None if refs is None else observation.load_pixels(api, refs)
    points = api.state.read("tool.marks")
    marks = []
    for index, (name, pixel) in enumerate(zip(names, pixels)):
        quality = {}
        try:
            status, record = measure(points=points, point_id=name, view=view, pixel=pixel,
                                     obs=obs, camera=camera, bounds=config["motion"]["workspace_bounds_m"], quality=quality)
        except ToolRefusal as error:
            if error.code not in ENTRY_ERRORS:
                raise
            entry = _batch._entry(source_index=index, point_id=name, point_uv=pixel, quality=quality)
            entry.update(error_code=error.code, error_message=str(error)[:200])
        else:
            api.state.write("tool.marks", points)
            entry = _batch._entry(source_index=index, point_id=name, point_uv=pixel, quality=quality)
            workspace = record["action_workspace"]
            entry.update(status=status, xyz_m=list(record["xyz_m"]),
                         views=_batch._distinct_views([source["view"] for source in record["sources"]]),
                         action_workspace_inside=bool(workspace["inside"]), action_workspace_use=workspace["motion_use"])
        marks.append(entry)
    _batch._attach_stored_evidence(points, marks)
    selected = []
    for entry in marks:
        if entry["status"] in {"solved", "verified"} and entry["xyz_m"]:
            colour = COLOURS[len(selected)]
            entry["colour_rgb"] = list(colour)
            selected.append([entry["point_id"], entry["xyz_m"], list(colour)])
    solved = sum(entry["status"] != "failed" for entry in marks)
    result = {"success": True, "status": "solved" if solved == len(marks) else "failed" if solved == 0 else "partial",
              "view": view, "observation_id": obs.identity.json()["observation_id"],
              "mark_count": len(marks), "solved_count": solved, "failed_count": len(marks)-solved,
              "marks": marks, "image_mapping": []}
    if selected and refs is not None:
        labels = JsonValue.of(selected)
        rendered = images.annotate(api.artifacts.read_blob(refs.rgb.json()), api.artifacts.read_blob(refs.calibration.json()),
                                   camera=view, points=labels)
        composite = images.contact_sheet(((view, rendered.png),), points=labels,
                                          resolution=tuple(config["mark_point"]["resolution"]))
        if config["motion"].get("environment_plans"):  # a surface moved from the handlers, which kept these
            keep_overlays(api, rendered, composite, view=view, observation_id=result["observation_id"],
                          selected=selected, resolution=config["mark_point"]["resolution"])
        result["images_base64"] = [base64.b64encode(composite).decode()]
        indices = [entry["source_index"] for entry in marks if entry["colour_rgb"] is not None]
        result["image_mapping"] = [{"image_index": 0, "role": "mark_batch_panel", "camera": view,
            "source_index": indices[0] if len(indices) == 1 else None, "candidate_id": None,
            "target_ref": None, "observation_step": config["mark_point"]["step"]}]
    return result
