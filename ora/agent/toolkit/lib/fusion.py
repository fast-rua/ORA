"""Deterministic RGB-D fusion with immutable inputs and full source attribution."""
from dataclasses import dataclass

from .observation import CameraRGBD
from .values import Array, JsonValue, PointCloudM
from . import _rgbd_fusion


@dataclass(frozen=True)
class FusionSource:
    """One public calibrated RGB-D camera; files carries its original hash/size identity."""
    camera: CameraRGBD
    files: JsonValue


@dataclass(frozen=True)
class FusedCloud:
    """World metres, RGB uint8 and int16 source indices, with complete counts and source metadata."""
    points: PointCloudM
    rgb: Array
    source_index: Array
    metadata: JsonValue


def fuse(sources: tuple[FusionSource, ...], workspace_bounds_m: JsonValue, *, voxel_size_m: float = .01,
         max_depth_m: float | None = None) -> FusedCloud:
    """Pure computation: back-project, bound and voxel-deduplicate observed pixels in caller order.

    Example: fuse((head_source, wrist_source), workspace, voxel_size_m=.01).
    Camera order breaks voxel ties; no implicit camera preference or robot-self
    filtering. Pixel depths are optical-axis metres, the output is world metres.
    Missing/malformed input or an empty bounded cloud raises; replay needs no simulator.
    """
    if any(s.camera.camera_to_world.target_frame != 'world' for s in sources):
        raise ValueError('fusion requires calibration into world metres')
    result = _rgbd_fusion.fuse_rgbd_pointcloud_sources(sources_by_camera={s.camera.name: dict(
        rgb=s.camera.rgb.numpy(), depth=s.camera.depth_m.numpy(), calibration=s.camera.calibration.json(),
        files=s.files.json()) for s in sources}, camera_names=tuple(s.camera.name for s in sources),
        workspace_bounds_m=workspace_bounds_m.json(), voxel_size_m=voxel_size_m, max_depth_m=max_depth_m)
    return FusedCloud(PointCloudM('world', Array.of(result.pop('xyz_world_m'))), Array.of(result.pop('rgb_u8')),
                      Array.of(result.pop('source_index')), JsonValue.of(result))
