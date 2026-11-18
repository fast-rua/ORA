"""Source-bound geometry measurements with the released result shapes."""
import math
from typing import Any, Mapping
import numpy as np
from ...lib import observation
from ...interface.stages import stage
from ...lib.values import JsonValue
from ..errors import ToolRefusal as ToolError
from . import _arguments as pixels
from ._arguments import _finite_vector, require_known_arguments, require_exactly_one_of

def run(api, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Replay: recorded publication bytes; pure projection/depth statistics, no simulator query."""
    require_known_arguments(
        arguments, allowed=("camera", "points", "region", "world_points_m", "window_px", "step"),
        required=("camera",), message="geometry.query fields are camera plus points or region",
    )
    require_exactly_one_of(
        arguments, ("points", "region", "world_points_m"),
        message="provide exactly one of points, region or world_points_m",
    )
    selection = api.artifacts.tool_context()["geometry_query"]
    if "error" in selection:
        raise ToolError.from_public(selection["error"])
    obs = observation.observe(api, identity=JsonValue.of(selection["identity"]))
    if not isinstance(arguments["camera"], str) or arguments["camera"] not in obs.manifest.json().get("cameras", {}):
        raise ToolError("camera is not policy-visible")
    camera_error = obs.result.data.json().get("camera_errors", {}).get(arguments["camera"])
    if camera_error:
        raise ToolError.from_public(camera_error)
    refs = obs.camera(arguments["camera"])
    camera = observation.load_pixels(api, refs)
    depth = camera.depth_m.numpy()
    calibration = camera.calibration.json()
    record = refs.declaration.json()
    width, height = int(record["width"]), int(record["height"])
    step = selection["step"]
    state = {"state_sha256": selection["source_state_sha256"]}
    if "region" in arguments:
        box = pixels.region_box(arguments["region"], width=width, height=height)
        x0, y0, x1, y1 = box
        values = depth[y0:y1, x0:x1]
        valid = values[np.isfinite(values) & (values > 1e-6)]
        result: dict[str, Any] = {
            "region": box, "valid_count": int(valid.size),
            "valid_fraction": float(valid.size / values.size),
        }
        if valid.size:
            result["depth_m"] = {
                "min": float(np.min(valid)), "median": float(np.median(valid)),
                "max": float(np.max(valid)),
            }
        return {
            "step": step,
            "source_state_sha256": state.get("state_sha256") if state else None,
            "camera": arguments["camera"], "region_result": result,
        }
    window = arguments.get("window_px", 3)
    if not isinstance(window, int) or isinstance(window, bool) or not 0 <= window <= 15:
        raise ToolError("window_px must be in [0,15]")
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or extrinsic.shape not in {(3, 4), (4, 4)}:
        raise ToolError("invalid calibration matrices", code="invalid_camera_calibration")
    homogeneous = np.vstack([extrinsic, [0, 0, 0, 1]]) if extrinsic.shape == (3, 4) else extrinsic
    if "world_points_m" in arguments:
        world_points = arguments["world_points_m"]
        if not isinstance(world_points, list) or not 1 <= len(world_points) <= 16:
            raise ToolError("world_points_m must contain 1..16 [x,y,z] points")
        projected = []
        for index, raw_point in enumerate(world_points):
            point = _finite_vector(raw_point, 3, "world_points_m point")
            camera_point = homogeneous @ np.asarray([*point, 1.0], dtype=np.float64)
            pixels.finite_camera_point(camera_point, raw_point, index)
            z = float(camera_point[2])
            if z <= 1e-9:
                projected.append({
                    "world_xyz_m": point, "camera_xyz_m": camera_point[:3].tolist(),
                    "in_front": False, "in_frame": False,
                })
                continue
            u = float(intrinsic[0, 0] * camera_point[0] / z + intrinsic[0, 2])
            v = float(intrinsic[1, 1] * camera_point[1] / z + intrinsic[1, 2])
            if not pixels.finite_pixel(u, v):  # in front, but projects to no pixel
                projected.append({"world_xyz_m": point, "in_front": True, "in_frame": False,
                                  "camera_xyz_m": [float(value) for value in camera_point[:3]]})
                continue
            rounded = [int(round(u)), int(round(v))]
            in_frame = 0 <= rounded[0] < int(record["width"]) and 0 <= rounded[1] < int(record["height"])
            scene_depth = None
            residual = None
            if in_frame:
                sampled = float(depth[rounded[1], rounded[0]])
                if math.isfinite(sampled) and sampled > 1e-6:
                    scene_depth = sampled
                    residual = z - sampled
            projected.append({
                "world_xyz_m": point,
                "camera_xyz_m": [float(value) for value in camera_point[:3]],
                "pixel_uv": [u, v],
                "pixel_uv_rounded": rounded,
                "in_front": True,
                "in_frame": in_frame,
                "scene_depth_m": scene_depth,
                "depth_residual_m": residual,
            })
        return {
            "step": step,
            "source_state_sha256": state.get("state_sha256") if state else None,
            "camera": arguments["camera"],
            "calibration_sha256": refs.calibration.json()["sha256"],
            "world_points": projected,
        }
    points = arguments["points"]
    if not isinstance(points, list) or not 1 <= len(points) <= 16:
        raise ToolError("points must contain 1..16 source [u,v] pixels")
    inverse = np.linalg.inv(homogeneous)
    stage("depth_window", window_px=window)
    results = []
    for index, point in enumerate(points):
        u, v = pixels.query_pixel(point, index, width=width, height=height)
        patch = depth[max(0, v-window):min(depth.shape[0], v+window+1), max(0, u-window):min(depth.shape[1], u+window+1)]
        valid = patch[np.isfinite(patch) & (patch > 1e-6)]
        if not valid.size:
            results.append({"pixel_uv": [u, v], "valid": False})
            continue
        z = float(np.median(valid))
        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        world = inverse @ np.asarray([x, y, z, 1.0])
        results.append({
            "pixel_uv": [u, v], "valid": True, "depth_median_m": z,
            "valid_count": int(valid.size),
            "world_xyz_m": [float(world[i]) for i in range(3)],
        })
    return {
        "step": step,
        "source_state_sha256": state.get("state_sha256") if state else None,
        "camera": arguments["camera"], "points": results,
    }
