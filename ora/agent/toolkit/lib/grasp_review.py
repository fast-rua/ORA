"""Pure review images of supplied candidates and public RGB-D; never selects candidates."""
from io import BytesIO
import numpy as np

from .values import Array, JsonValue, PointCloudM
from .observation import CameraRGBD
from .fusion import FusionSource, FusedCloud
from .vision import InspectionImage
from . import _grasp_review, _grasp_extent, _grasp_frames


def contact_sheet(camera: CameraRGBD, mask: Array, target_world: PointCloudM, *,
                  observation_id: str, target_ref: str, candidates: JsonValue,
                  compiler: JsonValue, mask_sha256: str, execution_tool: str) -> InspectionImage:
    """Pure computation: render caller-ordered jaws over RGB and world-metre mask points.

    Replay needs no service. Example: contact_sheet(camera, mask, world_cloud,
    observation_id=obs_id, target_ref=ref, candidates=pool, compiler=identity,
    mask_sha256=mask_hash, execution_tool='move_pose'). No reach/contact claims.
    """
    if target_world.frame != 'world':
        raise ValueError('grasp review requires world metres')
    image, metadata = _grasp_review._render_grasp_overlay(
        rgb=camera.rgb.numpy(), mask=mask.numpy(), calibration=camera.calibration.json(),
        target_world=target_world.xyz_m.numpy(), mask_sha256=mask_sha256,
        observation_id=observation_id, target_ref=target_ref, candidates=candidates.json(),
        compiler_identity=compiler.json(), execution_tool=execution_tool)
    return InspectionImage(Array.of(np.asarray(image)), JsonValue.of(metadata))


def global_review(sources: tuple[FusionSource, ...], fused: FusedCloud, target_world: PointCloudM, *,
                  overlay: InspectionImage, workspace: JsonValue, candidates: JsonValue,
                  compiler: JsonValue, labels: JsonValue, observed_extent: bool,
                  max_depth_m: float = 2.5) -> InspectionImage:
    """Pure computation: render world-metre cloud with explicit provenance and optional extent.

    Example: global_review(sources, fused, target, overlay=card, workspace=bounds,
    candidates=pool, compiler=identity, labels=labels, observed_extent=True).
    Replay needs no service. ``labels`` supplies observation_id, target_ref, camera,
    backend_input_scope, backend_cameras, backend_points_by_camera,
    backend_excluded_cameras and policy_visible. Caller chooses all ordering,
    workspace bounds and review cameras; no hidden camera or candidate selection.
    """
    from PIL import Image
    if target_world.frame != 'world' or fused.points.frame != 'world':
        raise ValueError('grasp review requires world metres')
    rows, identity = candidates.json(), compiler.json()
    extent = None
    if observed_extent:
        extent = _grasp_extent.cell(sources_by_camera={s.camera.name: dict(
            rgb=s.camera.rgb.numpy(), depth=s.camera.depth_m.numpy(), calibration=s.camera.calibration.json(),
            files=s.files.json()) for s in sources}, camera_names=tuple(s.camera.name for s in sources),
            max_depth_m=max_depth_m, workspace_bounds_m=workspace.json(), target_world=target_world.xyz_m.numpy(),
            grip_sites=[(row['rank'], colour, _grasp_frames._compiled_grasp_geometry(
                row, compiler_identity=identity)['grip_site']) for row, colour in zip(
                    rows, _grasp_review.GRASP_CANDIDATE_COLOURS, strict=False)])
    image = Image.fromarray(overlay.rgb.numpy(), mode='RGB')
    image, metadata = _grasp_review._render_grasp_global_review(
        camera_overlay=image, camera_panel_resolution=list(image.size),
        scene_world=fused.points.xyz_m.numpy(), target_world=target_world.xyz_m.numpy(),
        scene_source_index=fused.source_index.numpy(), scene_sources=fused.metadata.json()['sources'],
        workspace_bounds_m=workspace.json(), candidates=rows, compiler_identity=identity,
        observed_extent=extent, **labels.json())
    return InspectionImage(Array.of(np.asarray(image)), JsonValue.of(metadata))


def npz_bytes(arrays: tuple[tuple[str, Array], ...]) -> bytes:
    """Pure computation: lossless compressed named arrays, retaining order/dtype/shape.

    Example: npz_bytes((('xyz_world_m', cloud.xyz_m),)). Replay performs no I/O.
    Names must be unique; arrays cannot contain Python objects.
    """
    if len({name for name, _ in arrays}) != len(arrays):
        raise ValueError('duplicate NPZ array name')
    values = {name: value.numpy() for name, value in arrays}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError('object arrays are not public numeric data')
    stream = BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()
