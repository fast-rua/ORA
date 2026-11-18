"""Observed depth clearance of an explicit robot footprint; no collision claim."""

import numpy as np

from ..interface.types import InterfaceError
from . import _clearance as geometry
from . import _clearance_inputs as inputs
from .blobs import read_array, read_json
from .values import JsonValue, PoseM


def _models(jaws: JsonValue) -> dict:
    return {
        name: inputs.JawGeometry(
            **dict(fields, fingers=tuple(geometry.FingerBounds(**item) for item in fields["fingers"]))
        )
        for name, fields in jaws.json().items()
    }


def _extents(held: JsonValue) -> dict:
    return {name: inputs.HeldExtent(**item) for name, item in held.json().items()}


def along_path(manifest, cameras, *, arm: str, poses, jaws: JsonValue, held: JsonValue, body=()) -> JsonValue:
    """Pure computation: the least distance from the jaws along ``poses`` to what ``cameras`` show.

    ``cameras`` are ``action_preview.Camera``s, ``poses`` world metres and xyzw. No simulator query.
    Example: along_path(manifest, cameras, arm='left', poses=path, jaws=jaws, held=held).
    """
    try:
        views = [geometry.DepthView(camera=camera.name, depth_m=camera.depth_m, intrinsic=camera.intrinsic,
                                    camera_to_world=np.linalg.inv(camera.extrinsic))
                 for camera in cameras if camera.depth_m is not None]
        positions = {view.camera: np.asarray(view.camera_to_world, float)[:3, 3] for view in views}
        value = inputs.path_clearance(manifest, arm=arm, poses=poses, geometry=_models(jaws), held=_extents(held),
                                      views=(views, positions), body=body)
    except Exception:
        value = {"status": "unknown", "reason": "reference_failed", "claim_boundary": geometry.PATH_CLAIM}
    return JsonValue.of(value)


def held(manifest, *, arm: str, jaws: JsonValue, held: JsonValue) -> JsonValue:
    """Pure computation: what one arm's jaws hold, from the measured state, or null when they hold nothing.

    The jaws hold when their last command was close and they stopped apart (``_clearance_inputs.jaws_hold``), whatever
    closed them. ``extent`` is ``localized`` (the box of the target named when they closed), ``between_jaws`` (the gap
    they measure; what reaches past them is not measured) or ``unknown``; ``local_corners`` is that box in the
    controlled-point frame. Example: held(manifest, arm='left', jaws=jaws, held=held)."""
    try:
        value = inputs.held_object(manifest, arm=arm, geometry=_models(jaws), held=_extents(held))
    except Exception:
        value = None
    return JsonValue.of(value)


def estimate(
    api, observation, *, arm: str, target: PoseM, jaws: JsonValue, held: JsonValue
) -> JsonValue:
    """Replay: read granted depth bytes and recompute geometry; missing optional views stay absent.

    Inputs use world metres and xyzw poses. No simulator query or hidden cache.
    Example: estimate(api, obs, arm='left', target=pose, jaws=jaws, held=held).
    """
    try:
        models, extents = _models(jaws), _extents(held)
        views, positions = [], {}
        for camera in sorted(observation.cameras, key=lambda item: item.name):
            if camera.depth_m is None or camera.calibration is None:
                continue
            calibration = read_json(api, camera.calibration).json()
            intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
            extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
            if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
                raise ValueError("calibration shape mismatch")
            view = geometry.DepthView(
                camera=camera.name,
                depth_m=read_array(api, camera.depth_m).numpy(),
                intrinsic=intrinsic,
                camera_to_world=np.linalg.inv(extrinsic),
            )
            views.append(view)
            positions[camera.name] = np.asarray(view.camera_to_world, float)[:3, 3]
        value = inputs.annotate_observation(
            observation.manifest.json(),
            None,
            arm=arm,
            target_pose=[*target.position_m, *target.orientation_xyzw],
            geometry=models,
            held=extents,
            views=(views, positions),
        )
    except InterfaceError:
        raise
    except Exception:
        value = inputs._unknown("reference_failed")
    return JsonValue.of(value)


def held_extent(*, localization: JsonValue, intent: JsonValue,
                contact_center_local: tuple[float, float, float]) -> JsonValue:
    """Pure computation: live/replay fix a public world AABB to a close pose.

    All positions are metres; attach_pose_xyz_xyzw is episode-local link6 pose.
    Missing geometry produces unknown extent, never an invented object shape.
    This is a clearance advisory about intended contents, not a grasp verdict.
    Example: held_extent(localization=box, intent=held, contact_center_local=(0., 0., 0.)).
    """
    record = intent.json()
    corners = inputs.held_local_corners(localization.json(), record["attach_pose_xyz_xyzw"],
                                        contact_center_local)
    return JsonValue.of({"target_ref": record["target_ref"],
                         "local_corners": None if corners is None else corners[0].tolist(),
                         "recentered": False if corners is None else corners[1]})
