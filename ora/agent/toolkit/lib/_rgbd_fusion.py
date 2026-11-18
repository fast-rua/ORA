"""Pure deterministic public RGB-D fusion; all inputs supplied in memory."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

def fuse_rgbd_pointcloud_sources(
    *, sources_by_camera: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]],
    camera_names: Sequence[str] = ("agentview", "wrist"),
    voxel_size_m: float = 0.01, max_depth_m: float | None = None,
) -> dict[str, Any]:
    """Fuse public RGB-D sources in world coordinates with stable attribution."""

    import numpy as np

    from . import _rigid

    if not camera_names or len(set(camera_names)) != len(camera_names):
        raise ValueError("pointcloud camera_names must be unique and non-empty")
    if not isinstance(voxel_size_m, (int, float)) or float(voxel_size_m) <= 0:
        raise ValueError("pointcloud voxel_size_m must be positive")
    if max_depth_m is not None and (
        not isinstance(max_depth_m, (int, float)) or float(max_depth_m) <= 0
    ):
        raise ValueError("pointcloud max_depth_m must be positive when provided")

    axes = {"x": 0, "y": 1, "z": 2}
    origin = np.asarray([
        float(workspace_bounds_m[axis][0]) for axis in ("x", "y", "z")
    ])
    for axis in axes:
        bounds = workspace_bounds_m.get(axis)
        if (
            not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or not float(bounds[0]) < float(bounds[1])
        ):
            raise ValueError("pointcloud workspace bounds must contain finite xyz ranges")

    points_parts = []
    colour_parts = []
    source_parts = []
    sources: list[dict[str, Any]] = []
    for source_index, camera_name in enumerate(camera_names):
        source = sources_by_camera[camera_name]
        source_files = source['files']
        rgb = source['rgb']
        depth = source['depth'].astype(np.float64)
        calibration = source['calibration']
        intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
        if (
            rgb.shape[:2] != depth.shape or intrinsic.shape != (3, 3)
            or extrinsic.shape != (4, 4)
        ):
            raise ValueError(f"{camera_name} RGB-D calibration shape mismatch")
        valid = np.isfinite(depth) & (depth > 1e-6)
        if max_depth_m is not None:
            valid &= depth <= float(max_depth_m)
        rows, columns = np.nonzero(valid)
        z = depth[rows, columns]
        camera_points = np.stack([
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ], axis=1)
        world = _rigid.apply(camera_points, _rigid.invert(extrinsic))
        workspace_mask = np.ones(world.shape[0], dtype=bool)
        for axis, axis_index in axes.items():
            low, high = workspace_bounds_m[axis]
            workspace_mask &= (
                (world[:, axis_index] >= float(low))
                & (world[:, axis_index] <= float(high))
            )
        world = world[workspace_mask]
        colours = rgb[rows, columns][workspace_mask]
        points_parts.append(world)
        colour_parts.append(colours)
        source_parts.append(np.full(world.shape[0], source_index, dtype=np.int16))
        sources.append({
            "camera": camera_name,
            "source_index": source_index,
            "source_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "files": source_files,
            "valid_depth_points": int(len(rows)),
            "workspace_points": int(world.shape[0]),
        })

    points = np.concatenate(points_parts, axis=0)
    colours = np.concatenate(colour_parts, axis=0)
    source_indices = np.concatenate(source_parts, axis=0)
    if not len(points):
        raise ValueError("fused pointcloud has no points inside workspace bounds")
    voxels = np.floor((points - origin) / float(voxel_size_m)).astype(np.int64)
    original_order = np.arange(len(points), dtype=np.int64)
    order = np.lexsort((original_order, voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)
    retained = order[first]
    points = points[retained]
    colours = colours[retained]
    source_indices = source_indices[retained]
    for source in sources:
        source["retained_voxel_points"] = int(np.count_nonzero(
            source_indices == source["source_index"]
        ))
    return {
        "xyz_world_m": points,
        "rgb_u8": colours,
        "source_index": source_indices,
        "sources": sources,
        "camera_order": list(camera_names),
        "voxel_size_m": float(voxel_size_m),
        "max_depth_m": float(max_depth_m) if max_depth_m is not None else None,
        "workspace_bounds_m": {
            axis: [float(value) for value in workspace_bounds_m[axis]]
            for axis in ("x", "y", "z")
        },
        "input_point_count": int(sum(
            source["valid_depth_points"] for source in sources
        )),
        "workspace_point_count": int(sum(
            source["workspace_points"] for source in sources
        )),
        "filtered_voxel_count": int(points.shape[0]),
        "deduplication": "stable_first_by_camera_order",
    }
