"""Measure a bound target, synthesize approach poses, and ask the worker planner."""
import numpy as np
from ...lib import observation, geometry, vision, motion
from ...lib.blobs import read_array
from ...lib.values import Array, JsonValue
from ...interface.types import InterfaceError
from ..errors import ToolRefusal
from ..grasp_plan.tool import target_input
from ..vision_ground.tool import source_identity
from ._requests import validate_arguments, build_request


def centroid(api, obs, reference, config):
    target, record = target_input(api, reference, obs, config)
    refs = obs.camera(target["identity"]["camera"])
    errors = obs.result.data.json().get("camera_errors", {})
    if refs.name in errors:
        raise ToolRefusal.from_public(errors[refs.name])
    camera = observation.load_camera(api, refs)
    camera = observation.CameraRGBD(camera.name, camera.rgb,
        Array.of(camera.depth_m.numpy().astype(np.float64)), camera.intrinsic_px,
        camera.camera_to_world, camera.calibration)
    mask = read_array(api, JsonValue.of(record["files"]["mask.npy"]))
    try:
        raw_mask = mask.numpy().astype(bool)
        depth = camera.depth_m.numpy()
        if raw_mask.shape != depth.shape:
            raise ValueError("segmented target RGB-D calibration shape mismatch")
        quality = None
        if config["execution"]["depth_inliers"]:
            source = source_identity(obs.identity.json()["observation_id"], refs.name, refs)
            legacy = {f"{name}_sha256": source["files"][name]["sha256"] for name in ("rgb", "depth", "calibration")}
            if target["identity"].get("source") not in (source, legacy):
                raise ValueError("target_ref source artifacts do not match the selected observation")
            mask, quality = vision.measured_depth_selection(mask, camera.depth_m)
        valid = np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5)
        if np.count_nonzero(valid) < 16 or np.count_nonzero(raw_mask & valid) < 16:
            raise ValueError("segmented target has too few valid depth pixels")
        cloud = geometry.transform_points(geometry.mask_to_cloud(camera, mask), camera.camera_to_world)
        measured = geometry.visible_surface_geometry(cloud).json()
        if quality is not None:
            minimums = {"depth_valid_fraction": .7, "depth_inlier_fraction": .7, "primary_component_fraction": .6}
            values = quality.json()
            source_quality = {key: values[key] for key in (*minimums, "touches_image_boundary")}
            failed = [key for key, minimum in minimums.items() if values[key] < minimum]
            if failed:
                raise ToolRefusal("current target mask/depth quality is insufficient to measure its geometry from this view",
                    code="target_geometry_unavailable", field="target_ref", received=reference,
                    retryable=True, retry_requires_changed_arguments=True,
                    constraints={"failed_checks": failed, "measured": source_quality, "minimums": minimums})
        return measured["centroid_world_m"]
    except InterfaceError:
        raise
    except ValueError as error:
        raise ToolRefusal(str(error), code="target_geometry_unavailable", field="target_ref",
                          received=reference, retryable=True, retry_requires_changed_arguments=True) from error


def run(api, arguments):
    """Replay: recorded images/target geometry and planner request; new IK fails 未记录.

    Directions and standoff are world metres, orientations xyzw. Candidate
    generation is pure computation; planner results never imply path safety.
    """
    config = api.artifacts.tool_context()
    obs = observation.observe(api)
    settings = config["motion"]
    if not settings["view_config"].get("controlled_frame_convention"):
        raise ToolRefusal("align_plan requires the surface to declare its controlled-frame convention",
                          code="tool_unavailable", retryable=False)
    if not settings["reach_available"]:
        raise ToolRefusal("align_plan requires the environment read-only bridge", code="tool_unavailable", retryable=False)
    validated = validate_arguments(arguments)
    centre = centroid(api, obs, validated["target_ref"], config)
    request = build_request(validated, centre_xyz_m=centre, local_axes=settings["local_axes"])
    return motion.align_plan(api, JsonValue.of(request)).data.json()
