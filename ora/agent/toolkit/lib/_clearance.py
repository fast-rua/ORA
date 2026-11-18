"""Observed clearance below a commanded jaw footprint, from public depth only.

``observed_clearance_reference_v2`` annotates each ``move_pose`` target (and
the ``move_to`` / ``rotate_wrist`` / ``rotate_pitch`` targets that compile to
it) with one measurement taken before submission: how far the lowest point of
the jaw footprint at the commanded pose lies above the highest surface the
current depth images show beneath it or within ``RADIUS_M`` of it.

Inputs are only what the policy already receives: the observation's public
depth images with their per-observation calibration, the measured robot state,
the public embodiment geometry (finger collision bounds in the controlled-point
frame, opened along their published directions by the measured finger
positions) and, for a held object, the policy's own earlier localization of it
(``observed_clearance_runtime``). No scene geometry, layout or contact data.

Method. The footprint is the two finger boxes (plus the held-object box) posed
at the target; its lowest corner is the reported lowest point. Depth pixels
whose rays can reach the footprint column are back-projected to world points.
Points of the robot are excluded with the public geometry at the measured
state, as vertical prisms over the XY hull of each arm's measured finger boxes,
held box, controlled point and wrist camera: dilated ``ROBOT_CORE_MARGIN_M``
from the lowest finger or held corner upward, and ``ROBOT_MARGIN_M`` from the
lowest of the finger roots, controlled point and wrist camera upward, because
the descriptor bounds only the fingers and the housing, wrist and links rise
above them (offline over a recorded batch: with the jaw prism alone, 138 of 236 negative
readings on arrived moves were the moving arm's own housing, wrist camera or
links, all within 9 cm of link6). Of the remaining points within ``RADIUS_M``
horizontally of any footprint solid and no higher than the footprint's top, the
surface height is the ``MIN_SUPPORT_POINTS``-th highest. The result is
``unknown`` when no depth is valid, or fewer than ``MIN_SUPPORT_POINTS`` points
remain or less than ``MIN_OBSERVED_FRACTION`` of the area's 1 cm cells holds
one. Nothing here blocks or alters a move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf as math_inf
from typing import Any, Mapping, Sequence

import numpy as np

#: Horizontal reach beside the footprint within which a surface counts.
RADIUS_M = 0.02
#: Horizontal dilation of the measured jaw prism (fingers and held object).
ROBOT_CORE_MARGIN_M = 0.01
#: Horizontal dilation of the measured body prism: the gripper housing, the
#: wrist and the links above the finger roots, which the descriptor does not bound.
ROBOT_MARGIN_M = 0.05
#: Each prism starts this far below its floor.
ROBOT_FLOOR_MARGIN_M = 0.01
#: A surface height is the one at least this many depth points reach.
MIN_SUPPORT_POINTS = 5
#: Points this close below the surface height are reported as its support.
SURFACE_BAND_M = 0.005
#: Below this observed share of the area's 1 cm cells the result is unknown.
MIN_OBSERVED_FRACTION = 0.25
CELL_M = 0.01
#: The released depth cap of the observed-extent review.
MAX_DEPTH_M = 2.5
#: How far beneath the footprint a surface is looked for.
BELOW_M = 1.5
#: A pixel window larger than this (a camera inside the footprint column, in
#: practice a wrist camera near the target) is read at a pixel stride.
MAX_WINDOW_PIXELS = 80_000
REASONS = (
    "depth_unavailable",
    "footprint_unobserved",
    "geometry_unavailable",
    "reference_failed",
)
CLAIM_BOUNDARY = (
    "From the current depth images and their calibration only; points inside the measured jaws, the held "
    "object and the space above them are excluded as robot, other robot links are not; occluded surfaces "
    "and object undersides are unknown; points up to the footprint's top within radius_m count; not a "
    "collision check or a contact prediction, and the move is submitted unchanged."
)


@dataclass(frozen=True)
class FingerBounds:
    """One finger's public collision bounds in the controlled-point frame at reference opening."""

    minimum: Sequence[float]
    maximum: Sequence[float]
    opening_direction: Sequence[float]


@dataclass(frozen=True)
class DepthView:
    camera: str
    depth_m: Any
    intrinsic: Any
    camera_to_world: Any


@dataclass(frozen=True)
class Footprint:
    """World corner sets, each labelled ``fingers`` or ``held_object``."""

    solids: tuple[tuple[str, Any], ...]

    @property
    def points(self) -> Any:
        return np.vstack([corners for _part, corners in self.solids])

    def lowest(self) -> tuple[list[float], str]:
        index, offset = int(np.argmin(self.points[:, 2])), 0
        for part, corners in self.solids:
            if index < offset + len(corners):
                return [float(value) for value in corners[index - offset]], part
            offset += len(corners)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class RobotVolume:
    """Vertical prisms over one measured arm's jaws: robot, or hidden by it.

    The jaw prism (``ROBOT_CORE_MARGIN_M``) rises from the lowest finger or
    held-object corner; the wider body prism (``margin_m``) rises from the
    lowest of the finger roots, the controlled point and the wrist camera.
    """

    hull_xy: Any
    floor_z: float
    body_floor_z: float
    margin_m: float


@dataclass
class _Candidates:
    points: list[Any] = field(default_factory=list)
    cameras: list[Any] = field(default_factory=list)
    valid_views: int = 0
    excluded: int = 0


#: ``box_corners`` indices of the low-x face: a finger's root face when the
#: approach axis is local +x (ARX X5, ALOHA). ``root_face(approach)`` gives it
#: for any axis-aligned approach.
ROOT_FACE = (0, 1, 2, 3)


def root_face(approach_local: Sequence[float]) -> tuple[int, ...]:
    """The four ``box_corners`` indices on the face farthest back along an axis-aligned approach."""
    axis = int(np.argmax(np.abs(np.asarray(approach_local, float))))
    back = 0 if float(approach_local[axis]) > 0 else 1
    shift = 2 - axis  # index = 4 * ix + 2 * iy + iz
    return tuple(index for index in range(8) if (index >> shift) & 1 == back)


def box_corners(minimum: Sequence[float], maximum: Sequence[float]) -> Any:
    """The eight corners, low-x face first (a finger's root face in its controlled-point frame)."""
    low, high = np.asarray(minimum, float), np.asarray(maximum, float)
    return np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )


def rotation(quaternion_xyzw: Sequence[float]) -> Any:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if not norm:
        raise ValueError("orientation is not a rotation")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def footprint_at(
    pose_xyz_xyzw: Sequence[float],
    *,
    finger_qpos: Sequence[float],
    fingers: Sequence[FingerBounds],
    held_local_corners: Any = None,
) -> Footprint:
    """The finger boxes, opened by the measured finger positions, and the held box at one pose."""
    turn, origin = rotation(pose_xyz_xyzw[3:]), np.asarray(pose_xyz_xyzw[:3], float)
    solids = []
    for finger, opening in zip(fingers, finger_qpos, strict=True):
        shift = float(opening) * np.asarray(finger.opening_direction, float)
        local = box_corners(
            np.asarray(finger.minimum, float) + shift,
            np.asarray(finger.maximum, float) + shift,
        )
        solids.append(("fingers", local @ turn.T + origin))
    if held_local_corners is not None:
        solids.append(
            ("held_object", np.asarray(held_local_corners, float) @ turn.T + origin)
        )
    return Footprint(solids=tuple(solids))


def robot_volume(
    solids: Sequence[tuple[str, Any]],
    *,
    controlled_point: Sequence[float],
    extra_points: Sequence[Any] = (),
    margin_m: float | None = None,
    root_indices: Sequence[int] = ROOT_FACE,
) -> RobotVolume:
    """``root_indices`` pick each finger box's root face (``box_corners`` order); the default is low-x."""
    corners = np.vstack([np.asarray(item, float) for _part, item in solids])
    roots = list(root_indices)
    body = np.vstack(
        [
            np.asarray(controlled_point, float)[None],
            *[np.asarray(point, float)[None] for point in extra_points],
            *[
                np.asarray(item, float)[roots]
                for part, item in solids
                if part == "fingers"
            ],
        ]
    )
    return RobotVolume(
        hull_xy=_hull(np.vstack([corners[:, :2], body[:, :2]])),
        floor_z=float(corners[:, 2].min()),
        body_floor_z=float(body[:, 2].min()),
        margin_m=ROBOT_MARGIN_M if margin_m is None else float(margin_m),
    )


def _hull(points: Any) -> Any:
    """Counter-clockwise convex hull (monotone chain) of 2-D points."""
    unique = sorted(
        {(round(float(x), 9), round(float(y), 9)) for x, y in np.asarray(points, float)}
    )
    if len(unique) <= 2:
        return np.array(unique, float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    def half(sequence):
        chain: list[tuple[float, float]] = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower, upper = half(unique), half(reversed(unique))
    return np.array(lower[:-1] + upper[:-1], float)


#: Directions of the polygonal disk a hull is dilated by: the dilated outline
#: lies within ``reach * (1 - cos(pi / 16))`` (1 mm at 5 cm) of the exact one.
_DILATION = np.stack(
    [np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)], axis=1
)


def _within(points: Any, hull: Any, reach: float) -> Any:
    """Whether each point lies within ``reach`` horizontally of a convex polygon (polygonal dilation)."""
    outline = _hull(
        (np.asarray(hull, float)[:, None, :] + reach * _DILATION[None]).reshape(-1, 2)
    )
    edge = np.roll(outline, -1, axis=0) - outline
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    result = np.zeros(len(points), dtype=bool)
    boxed = np.flatnonzero(
        np.all(
            (points >= outline.min(axis=0)) & (points <= outline.max(axis=0)), axis=1
        )
    )
    if len(boxed):
        result[boxed] = (
            (points[boxed] @ normal.T) <= (outline * normal).sum(axis=1) + 1e-12
        ).all(axis=1)
    return result


def _near(points_xy: Any, hulls: Sequence[Any], reach: float) -> Any:
    near = np.zeros(len(points_xy), dtype=bool)
    for hull in hulls:
        near |= _within(points_xy, hull, reach)
    return near


def _pixel_window(view: DepthView, box_low: Any, box_high: Any) -> tuple[slice, slice]:
    height, width = np.asarray(view.depth_m).shape
    to_camera = np.linalg.inv(np.asarray(view.camera_to_world, float))
    corners = box_corners(box_low, box_high) @ to_camera[:3, :3].T + to_camera[:3, 3]
    if (corners[:, 2] <= 0.01).any():
        return slice(0, height), slice(0, width)
    intrinsic = np.asarray(view.intrinsic, float)
    u = intrinsic[0, 0] * corners[:, 0] / corners[:, 2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * corners[:, 1] / corners[:, 2] + intrinsic[1, 2]
    columns = slice(
        max(0, int(np.floor(u.min()))), min(width, int(np.ceil(u.max())) + 1)
    )
    rows = slice(max(0, int(np.floor(v.min()))), min(height, int(np.ceil(v.max())) + 1))
    return rows, columns


def _world_points(view: DepthView, window: tuple[slice, slice]) -> Any:
    pixels = max(0, window[0].stop - window[0].start) * max(
        0, window[1].stop - window[1].start
    )
    if not pixels:
        return np.empty((0, 3))
    stride = max(1, int(np.ceil(np.sqrt(pixels / MAX_WINDOW_PIXELS))))
    depth = np.asarray(view.depth_m, dtype=np.float64)[window][::stride, ::stride]
    rows, columns = np.nonzero(
        np.isfinite(depth) & (depth > 1e-6) & (depth <= MAX_DEPTH_M)
    )
    if not len(rows):
        return np.empty((0, 3))
    z = depth[rows, columns]
    rows, columns = rows * stride + window[0].start, columns * stride + window[1].start
    intrinsic, camera_to_world = np.asarray(view.intrinsic, float), np.asarray(
        view.camera_to_world, float
    )
    camera = np.stack(
        [
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        axis=1,
    )
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _in_robot(points: Any, robot: Sequence[RobotVolume]) -> Any:
    excluded = np.zeros(len(points), dtype=bool)
    for volume in robot:
        above = points[:, 2] >= volume.floor_z - ROBOT_FLOOR_MARGIN_M
        if not above.any():
            continue
        planar, body = (
            points[above, :2],
            points[above, 2] >= volume.body_floor_z - ROBOT_FLOOR_MARGIN_M,
        )
        inside = _within(planar, volume.hull_xy, ROBOT_CORE_MARGIN_M)
        if body.any():
            inside[body] |= _within(planar[body], volume.hull_xy, volume.margin_m)
        excluded[above] |= inside
    return excluded


def _collect(
    views: Sequence[DepthView],
    footprint: Footprint,
    robot: Sequence[RobotVolume],
    hulls: Sequence[Any],
    radius_m: float,
) -> _Candidates:
    corners = footprint.points
    low = np.array(
        [*(corners[:, :2].min(axis=0) - radius_m), corners[:, 2].min() - BELOW_M]
    )
    high = np.array([*(corners[:, :2].max(axis=0) + radius_m), corners[:, 2].max()])
    found = _Candidates()
    for view in views:
        depth = np.asarray(view.depth_m)
        if depth.ndim != 2 or not (np.isfinite(depth) & (depth > 1e-6)).any():
            continue
        found.valid_views += 1
        points = _world_points(view, _pixel_window(view, low, high))
        if not len(points):
            continue
        points = points[
            np.all((points >= low - 1e-9) & (points <= high + 1e-9), axis=1)
        ]
        if len(points):
            points = points[_near(points[:, :2], hulls, radius_m)]
        robot_mask = (
            _in_robot(points, robot)
            if robot and len(points)
            else np.zeros(len(points), bool)
        )
        points = points[~robot_mask]
        found.excluded += int(robot_mask.sum())
        found.points.append(points)
        found.cameras.append(np.full(len(points), view.camera, dtype=object))
    return found


def _observed_fraction(
    points: Any, hulls: Sequence[Any], radius_m: float, low_xy: Any, high_xy: Any
) -> float:
    counts = np.maximum(np.ceil((high_xy - low_xy) / CELL_M).astype(int), 1)
    grid = np.stack(
        np.meshgrid(
            *(
                low_xy[axis] + (np.arange(counts[axis]) + 0.5) * CELL_M
                for axis in (0, 1)
            ),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    region = _near(grid, hulls, radius_m)
    if not region.any() or not len(points):
        return 0.0
    cells = np.clip(((points[:, :2] - low_xy) // CELL_M).astype(int), 0, counts - 1)
    observed = np.zeros(int(counts.prod()), dtype=bool)
    observed[cells[:, 0] * counts[1] + cells[:, 1]] = True
    return float((observed & region).sum() / region.sum())


#: A path's clearance is looked for this far around the jaws; a surface farther away is reported as beyond it.
PATH_REACH_M = 0.10
#: A path is checked at this many poses, evenly along it.
PATH_POSES = 24
#: Depth points kept per view after the crop: a preview's advice, not a survey.
PATH_POINTS_PER_VIEW = 30_000
#: Depth this close to the moving arm's chain of joints (``action_plan.body_points``) is the arm itself.
BODY_RADIUS_M = 0.075
#: The stretch at each end of a path, along the path, where a planner clears the hand's surroundings on purpose.
PATH_END_M = 0.05
#: In an end stretch, the jaws this close to a surface are inside what the planner cleared.
CLEARED_ENTRY_M = 0.005
PATH_CLAIM = (
    "From the current depth images and their calibration only, as they show the scene, not from a planner's world: the "
    "least distance from the finger boxes, and a held object's box where it was measured, along the path to what those "
    "images show, or how far a point they show lies inside those boxes, leaving out the last 5 cm at each end of the "
    "path. There a planner clears the hand's surroundings, so "
    "the object being approached is removed on purpose and a planned path can pass through it; where the jaws enter that "
    "space is marked. Points near each hand and along the moving arm's measured links are left out as robot, so what "
    "stands right next to them is not seen; the arm's links themselves are not checked; surfaces no camera sees are "
    "unknown; nothing the motion would move is accounted for. Advice only: it blocks nothing."
)


def _box_distance(points: Any, low: Any, high: Any) -> Any:
    """Each point's distance to one axis-aligned box (0 inside it)."""
    gap = np.maximum(np.maximum(low - points, points - high), 0.0)
    return np.sqrt((gap * gap).sum(axis=1))


def _box_signed(points: Any, low: Any, high: Any) -> Any:
    """Each point's distance to one axis-aligned box, negative inside it: minus its distance to the nearest face."""
    outside = _box_distance(points, low, high)
    inside = np.minimum(points - low, high - points).min(axis=1)
    return np.where(outside > 0.0, outside, -np.maximum(inside, 0.0))


def _local_boxes(fingers: Sequence[FingerBounds], finger_qpos: Sequence[float], held_local_corners: Any) -> list[tuple]:
    """The hand's solids in the controlled-point frame as ``(part, low, high)``, fingers opened as measured."""
    boxes = []
    for finger, opening in zip(fingers, finger_qpos, strict=True):
        shift = float(opening) * np.asarray(finger.opening_direction, float)
        boxes.append(("fingers", np.asarray(finger.minimum, float) + shift, np.asarray(finger.maximum, float) + shift))
    if held_local_corners is not None:
        corners = np.asarray(held_local_corners, float)
        boxes.append(("held_object", corners.min(axis=0), corners.max(axis=0)))
    return boxes


def near_path_ends(poses: Sequence[Sequence[float]], end_m: float = PATH_END_M) -> list[bool]:
    """Whether each pose lies within ``end_m`` of either end of the path, along it. The one place that says so."""
    steps = [float(np.linalg.norm(np.asarray(second[:3], float) - np.asarray(first[:3], float)))
             for first, second in zip(poses, poses[1:])]
    along = [0.0]
    for step in steps:
        along.append(along[-1] + step)
    return [travelled < end_m or along[-1] - travelled < end_m for travelled in along]


def _near_chain(points: Any, chain: Sequence[Sequence[float]], radius: float) -> Any:
    """Whether each point lies within ``radius`` of the polyline ``chain``."""
    near = np.zeros(len(points), dtype=bool)
    nodes = np.asarray(chain, float)
    for first, second in zip(nodes, nodes[1:]):
        along = second - first
        length = float(along @ along)
        part = np.clip(((points - first) @ along) / length, 0.0, 1.0) if length > 0 else np.zeros(len(points))
        gap = points - (first + part[:, None] * along)
        near |= (gap * gap).sum(axis=1) <= radius * radius
    return near


def along_path(
    views: Sequence[DepthView],
    *,
    poses: Sequence[Sequence[float]],
    finger_qpos: Sequence[float],
    fingers: Sequence[FingerBounds],
    held_local_corners: Any = None,
    robot: Sequence[RobotVolume] = (),
    body: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """The least distance from the jaws along ``poses`` to what ``views`` show, beyond the path's ends (``PATH_CLAIM``),
    and where inside an end the jaws first enter a surface there. Where the jaws' boxes reach into what the images show,
    ``clearance_m`` is 0 and ``inside_m`` is how far the deepest point seen lies inside them, which the position and
    fraction then name."""
    ends = near_path_ends(poses)
    step = max(1, (len(poses) - 1) // (PATH_POSES - 1))
    indices = sorted({*range(0, len(poses), step), len(poses) - 1})
    boxes = _local_boxes(fingers, finger_qpos, held_local_corners)
    corners = np.vstack([footprint_at(poses[index], finger_qpos=finger_qpos, fingers=fingers,
                                      held_local_corners=held_local_corners).points for index in indices])
    low, high = corners.min(axis=0) - PATH_REACH_M, corners.max(axis=0) + PATH_REACH_M
    sets, names, seen = [], [], 0
    for view in views:
        depth = np.asarray(view.depth_m)
        if depth.ndim != 2 or not (np.isfinite(depth) & (depth > 1e-6)).any():
            continue
        seen += 1
        points = _world_points(view, _pixel_window(view, low, high))
        if len(points):
            points = points[np.all((points >= low) & (points <= high), axis=1)]
        if len(points) and robot:
            points = points[~_in_robot(points, robot)]
        if len(points) and len(body) > 1:
            points = points[~_near_chain(points, body, BODY_RADIUS_M)]
        if len(points) > PATH_POINTS_PER_VIEW:
            points = points[:: len(points) // PATH_POINTS_PER_VIEW + 1]
        if len(points):
            sets.append(points)
            names.append(np.full(len(points), view.camera, dtype=object))
    base = {"poses_checked": sum(not ends[index] for index in indices), "claim_boundary": PATH_CLAIM}
    if not seen:
        return {"status": "unknown", "reason": "depth_unavailable", **base}
    if not sets:
        return {"status": "beyond", "clearance_at_least_m": PATH_REACH_M, **base}
    points, cameras = np.vstack(sets), np.concatenate(names)
    best, entered = (math_inf, 0, 0, "fingers"), None
    for index in indices:
        pose = poses[index]
        local = (points - np.asarray(pose[:3], float)) @ rotation(pose[3:])
        for part, box_low, box_high in boxes:
            distance = _box_signed(local, box_low, box_high)
            nearest = int(np.argmin(distance))
            value = float(distance[nearest])
            if not ends[index] and value < best[0]:
                best = (value, index, nearest, part)
            if ends[index] and value <= CLEARED_ENTRY_M and entered is None:
                entered = {"at_fraction": round(index / max(len(poses) - 1, 1), 3), "part": part,
                           "position_m": [round(float(item), 4) for item in points[nearest]]}
    base = {**base, **({"enters_cleared_space": entered} if entered else {})}
    if all(ends):
        return {"status": "ends_only", "path_end_m": PATH_END_M, **base}
    value, index, nearest, part = best
    if value >= PATH_REACH_M:
        return {"status": "beyond", "clearance_at_least_m": PATH_REACH_M, **base}
    inside = {"inside_m": round(-value, 4)} if value < 0.0 else {}
    return {"status": "measured", "clearance_m": round(max(value, 0.0), 4), **inside,
            "at_fraction": round(index / max(len(poses) - 1, 1), 3),
            "position_m": [round(float(item), 4) for item in points[nearest]], "part": part,
            "camera": str(cameras[nearest]), "depth_points": len(points), **base}


def measure(
    views: Sequence[DepthView],
    *,
    observation_id: str,
    footprint: Footprint,
    robot: Sequence[RobotVolume] = (),
    radius_m: float | None = None,
) -> dict[str, Any]:
    """The observed clearance of one footprint (see the module docstring)."""
    radius_m = RADIUS_M if radius_m is None else float(radius_m)
    lowest, part = footprint.lowest()
    hulls = [_hull(corners[:, :2]) for _part, corners in footprint.solids]
    found = _collect(views, footprint, robot, hulls, radius_m)
    points = np.vstack(found.points) if found.points else np.empty((0, 3))
    cameras = (
        np.concatenate(found.cameras) if found.cameras else np.empty(0, dtype=object)
    )
    corners = footprint.points
    fraction = _observed_fraction(
        points,
        hulls,
        radius_m,
        corners[:, :2].min(axis=0) - radius_m,
        corners[:, :2].max(axis=0) + radius_m,
    )
    base = {
        "footprint_lowest_m": [round(value, 4) for value in lowest],
        "footprint_lowest_part": part,
        "radius_m": radius_m,
        "footprint_observed_fraction": round(fraction, 2),
        "robot_points_excluded": found.excluded,
    }
    if not found.valid_views:
        return {
            "status": "unknown",
            "reason": "depth_unavailable",
            **base,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    if len(points) < MIN_SUPPORT_POINTS or fraction < MIN_OBSERVED_FRACTION:
        return {
            "status": "unknown",
            "reason": "footprint_unobserved",
            **base,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    order = np.argsort(-points[:, 2], kind="stable")
    chosen = int(order[MIN_SUPPORT_POINTS - 1])
    height, camera = float(points[chosen, 2]), str(cameras[chosen])
    support = int(
        ((cameras == camera) & (points[:, 2] >= height - SURFACE_BAND_M)).sum()
    )
    return {
        "status": "measured",
        "clearance_m": round(lowest[2] - height, 4),
        **base,
        "highest_surface": {
            "position_m": [round(float(value), 4) for value in points[chosen]],
            "camera": camera,
            "observation_id": observation_id,
            "depth_points": support,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
