"""Where a mobile manipulator could stand to reach a target, and whether its base can drive there: the shared base.

A refusal of an arm target at the measured base pose (out of reach, unreachable, in collision, or with no path) is
answered with the nearest base poses from which the same move is admitted, clear of the public depth seen now. The
search is the same for every simulator. A backend supplies its robot (``MobileBase``: the footprint, the base servo,
the lift's range and the arm's reach), the measured state (``Here``: the base pose, the torso height and the arm
mount), the robot's own body at that state (``OwnBody``: the planner's collision spheres and the boxes fixed to it) and
two questions of the shared planner (``search``'s ``solve`` and ``admits``):

1. Stands. Arm-mount positions around the target, at the standoffs of ``Search`` and the one the mount has now, on
   bearings around the bearing it has now; at each, the base yaw now, the yaw that faces the target and the one
   between; at each, the torso height now and, where a public action sets the lift (``MobileBase.torso_range_m``), the
   heights around the one that puts the arm's shoulder level with the target. A stand within ``max_travel_m`` of the
   base whose mount has the target within the arm's admission reach is kept.
2. Footprint. The base's boxes at the stand, grown by ``search_margin_m``, hold none of the public depth points
   (``PublicPoints``: the published cameras' depth in their own calibration, without the floor, the robot's own body
   and what lies within ``own_margin_m`` of the base where it stands now), screened on floor cells of ``raster_m`` per
   box height band; and at least ``floor_seen_fraction`` of the floor under it was seen, or lies under the base now.
   A ``floor_cell_m`` cell is seen when a camera ray ended in it, or passed over it, below the top of the boxes that
   stand on the floor. Unseen space is no obstacle, but a stand on floor no camera saw may be inside furniture.
3. Ranking: the base's travel plus ``turn_cost_m_per_rad`` per radian of turn plus ``lift_cost`` per metre of lift,
   and ``nudge_cost_m`` more for a move shorter than ``nudge_m`` that turns less than ``nudge_rad``: a nudge in place
   is offered only when no real move is.
4. Reach: the cheapest ``blank_queries`` stands are asked of the shared IK in a blank world (kinematics alone).
5. Clear: in the same order, a stand at least ``separation_m`` or ``separation_rad`` from those kept, whose base move
   there passes ``sweep`` ``suggestion_slack_m`` stricter than ``move_base``'s own check (the servo's own path, its
   footprint grown by ``admission_margin_m`` and the slack, and the robot's body carried at its joints), so a move to
   the rounded stand is admitted too, from which the shared IK still reaches the target wherever the servo may stop
   (``arrival_band``: its position and yaw tolerances either way), and from which the move itself is admitted as its
   real call would admit it (``admits``: its path in the world seen from there, and its timing), up to
   ``observed_checks`` admissions, until ``suggestions`` are kept.

``sweep`` is also ``move_base``'s admission: a base move whose footprint, or whose carried arm or load, along the
servo's path would hold a depth point is refused before anything moves, by the real call and its dry run alike. The
load is the planner's own and the depth linked to it (``linked_load``): a load the planner's world bounds short of its
size moves with the hand whole, and its own depth is no obstacle.

Only public inputs enter: the published cameras' depth and calibration, the measured base pose, torso height and arm
joints, and the robot's own kinematics. No object, contact, task source or success predicate does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.contracts.pose import matrix_quaternion

from .depth_load import fit_held_spheres
from .held_depth import edge_pixels, load_voxels
from .depth_world_inputs import pose_matrix

SCHEMA = "ora.base-placement.v1"
#: The codes of an arm refusal a base move may answer.
SEARCHED_CODES = ("target_out_of_reach_from_base", "target_unreachable_at_base_pose",
                  "target_in_collision_at_base_pose", "path_in_collision_at_base_pose")
#: ``move_base``'s refusal when its footprint would meet the depth on the way.
BASE_PATH_CODE = "base_path_in_collision"
#: What ``arm_path`` says of a kept stand: since 2026-11-06 every kept stand's move is ``admitted`` (``admits``); a
#: receipt from before says whether a joint path from the arm's current joints ``planned`` there or not.
ADMITTED, PLANNED, NOT_PLANNED = "admitted", "planned", "not_planned"
#: The public points a footprint is checked against keep one point per voxel of this size (the depth world's).
OBSTACLE_VOXEL_M = 0.01


@dataclass(frozen=True)
class Box:
    """One box of the base's footprint in its own frame (x forward, y left, z up from the floor), metres."""

    x_m: tuple[float, float]
    y_m: tuple[float, float]
    z_m: tuple[float, float]
    #: The box's top rises with the torso lift (the column that carries the arm).
    top_lifts: bool = False


@dataclass(frozen=True)
class Servo:
    """How the base servo drives a holonomic base: a proportional step toward the goal, capped, one per tick."""

    linear_gain: float
    max_linear_m_s: float
    yaw_gain: float
    max_yaw_rad_s: float
    tick_s: float
    position_tolerance_m: float
    yaw_tolerance_rad: float
    max_ticks: int


@dataclass(frozen=True)
class MobileBase:
    """What a backend tells the search of its robot."""

    footprint: tuple[Box, ...]
    servo: Servo
    #: The lift's travel a stand may choose, or None: the stands keep the measured height (a robot without a lift, or
    #: one whose lift no public action sets, which ``move_base`` does not move).
    torso_range_m: tuple[float, float] | None
    torso_step_m: float
    #: The admission's reach from the arm mount.
    reach_m: float
    #: The arm's shoulder above its mount: a reach level with it goes farthest.
    shoulder_m: float
    #: Depth points at or below this height are the floor.
    floor_m: float


@dataclass(frozen=True)
class Search:
    """The search's own numbers: the same for every robot."""

    standoffs_m: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70)
    bearing_offsets_rad: tuple[float, ...] = tuple(math.radians(value) for value in (0, 30, -30, 60, -60, 90, -90))
    max_travel_m: float = 1.5
    turn_cost_m_per_rad: float = 0.3
    lift_cost: float = 0.5
    #: A move shorter than ``nudge_m`` that turns less than ``nudge_rad`` costs ``nudge_cost_m`` more: in 107 base
    #: suggestions of the 2026-11-06 RoboCasa smoke, 25 firsts were turns of 5 to 15 degrees in place.
    nudge_m: float = 0.05
    nudge_rad: float = math.radians(15.0)
    nudge_cost_m: float = 0.5
    #: A stand keeps this much room around its footprint; a move only ``admission_margin_m``.
    search_margin_m: float = 0.05
    admission_margin_m: float = 0.01
    #: How deep the robot's body, carried along a base move, may lie in the depth: the depth world's own voxel, as deep
    #: as a straight arm line's links may (RoboCasa's ``LINE_WORLD_TOLERANCE_M``). Of the 2026-11-06 smoke's base moves
    #: a world input replays, the carried arm lay 19 to 51 mm deep in three of the four that stalled, and 6 and 24 mm
    #: deep in two of the 19 that arrived.
    carried_tolerance_m: float = 0.01
    #: A suggested stand's move passes the sweep this much stricter than ``move_base`` admits it (its footprint grown
    #: this much more, the carried body this much less deep), so the move a policy copies from the printed, rounded
    #: suggestion is admitted too: in smoke ``rc-ab-6864e57de`` a move 3 mm off its suggestion (SteamInMicrowave [59])
    #: was refused, its held load 1.6 mm deeper than the tolerance.
    suggestion_slack_m: float = 0.01
    #: The depth within ``load_reach_m`` of the held load's spheres that ``linked_load`` links to them is the load too:
    #: the robot's own, carried by a base move as ``load_spheres`` spheres fitted to it as the planner fits its load's.
    #: Where the linked depth runs on past the reach (a wall the load hangs beside, the door a handle belongs to), the
    #: part nearer the depth past the reach stays scene, next to the load, so a move that would drag it is refused.
    #: The planner's world bounds its load 0.15 m past the jaws and leaves one that stands on a support to the scene:
    #: in smoke ``rc-ab-6864e57de`` two bowls held by the rim and lifted (StoreLeftoversInBowl, SteamInMicrowave) kept
    #: only the jaws' slice as their load, and the rest of them, left in the depth, refused 67 base moves as the load
    #: meeting itself; in ``rc-ab-6d8eb1acf`` a bowl hanging within 2 cm of a wall (StoreLeftoversInBowl, step 2700)
    #: was all left to the scene, wall and bowl, until the split.
    load_reach_m: float = 0.30
    load_spheres: int = 32
    #: How far past the reach the depth is followed, to tell which way a chain that runs on past it runs.
    load_shell_m: float = 0.05
    #: A pixel next to one of the robot's own and more than this behind it, and more than this in front of another
    #: neighbour, is the robot's edge (``held_depth.edge_pixels``): in smoke ``rc-ab-6d8eb1acf`` such mixed pixels, 0.12 to
    #: 1.06 m behind the robot or its load and 0.3 to 2.4 m in front of the scene beyond, were met by the carried bowl
    #: (SteamInMicrowave [45]), the carried arm and the footprint.
    edge_step_m: float = 0.05
    #: What lies this close to the base where it stands now is the base itself, or already touches it.
    own_margin_m: float = 0.03
    blank_queries: int = 48
    observed_checks: int = 4
    suggestions: int = 2
    #: A stand this close to a kept one, in both position and yaw, adds nothing.
    separation_m: float = 0.15
    separation_rad: float = math.radians(20.0)
    #: A move shorter than this, and turning less, is the pose the base already has.
    here_m: float = 0.03
    here_rad: float = math.radians(3.0)
    #: The servo's path is checked at least this often.
    sweep_step_m: float = 0.05
    sweep_step_rad: float = math.radians(5.0)
    #: A stand's footprint is screened on floor cells this size (well inside ``search_margin_m``); a move's is exact.
    raster_m: float = 0.02
    #: The floor under a stand's footprint counts as seen per cell of this size, sampled this often along each camera
    #: ray, and this much of it must be seen. In the 2026-11-06 smoke the over-shoulder camera saw little floor: the
    #: stands behind DeliverStraw's cabinet face had at most 6 % of theirs seen, those in front of it 10 to 81 %, and
    #: the suggestions the model drove to 28 to 96 %.
    floor_cell_m: float = 0.05
    floor_ray_samples: int = 48
    floor_seen_fraction: float = 0.1


SEARCH = Search()


@dataclass(frozen=True)
class Here:
    """The measured state: base ``(x, y, yaw)`` in the world, torso height, and the arm mount's world pose."""

    base: tuple[float, float, float]
    torso_m: float
    mount: tuple[float, ...]
    #: The point the base turns about, in its own frame.
    turn_centre_m: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Stand:
    """A base pose and torso height to stand at."""

    x_m: float
    y_m: float
    yaw_rad: float
    torso_m: float
    cost: float = 0.0


@dataclass(frozen=True)
class OwnBody:
    """The robot at its measured state: what of the depth is its own, and what a base move carries.

    ``spheres`` (world ``[x, y, z, radius]``) are the planner's collision spheres at the measured joints, the held
    load's last (``held`` of them; ``public_points`` adds those of the depth linked to the load): a depth point within a
    sphere's radius plus ``margin_m`` is the robot's own, as the planner's own world takes it out. ``boxes`` are boxes
    fixed to the robot, each a world 4x4 frame and its ``((x0, x1), (y0, y1), (z0, z1))`` in that frame, such as the
    hand's, which holds the fingers whatever their opening."""

    spheres: Any
    margin_m: float
    boxes: tuple[tuple[Any, tuple[tuple[float, float], ...]], ...] = ()
    held: int = 0

    def holds(self, points: Any) -> Any:
        """Which world ``points`` show the robot itself."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        mine = within_spheres(points, self.spheres, self.margin_m)
        for frame, box in self.boxes:
            mine |= in_box(points, frame, box)
        return mine

    def carrying(self, load: Any) -> OwnBody:
        """This body with ``load`` (world ``[x, y, z, radius]``) added to what it holds."""
        load = np.asarray(load, dtype=np.float64).reshape(-1, 4)
        return OwnBody(np.vstack([np.asarray(self.spheres, dtype=np.float64).reshape(-1, 4), load]), self.margin_m,
                       self.boxes, self.held + len(load))


@dataclass(frozen=True)
class PublicPoints:
    """The public depth of one moment: what a footprint may meet, which pixels are the robot's own, the floor seen and
    the robot's body a base move carries."""

    obstacles: Any
    #: Per camera, the flat indices of pixels that show the robot itself (its arm, or its base where it stands).
    own_pixels: Mapping[str, Any] = field(default_factory=dict)
    #: The floor seen: one ``(x, y)`` per ``Search.floor_cell_m`` cell a camera ray ended in or passed over low enough.
    floor: Any = None
    body: OwnBody | None = None


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def base_matrix(x: float, y: float, yaw: float) -> Any:
    """The 4x4 world transform of a base pose on the floor."""
    transform = np.eye(4)
    transform[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    transform[:2, 3] = (x, y)
    return transform


def shift(here: Here, stand: Stand) -> Any:
    """The world transform that carries the robot, arm joints held, from where it stands to ``stand``."""
    moved = base_matrix(stand.x_m, stand.y_m, stand.yaw_rad) @ np.linalg.inv(base_matrix(*here.base))
    moved[2, 3] += stand.torso_m - here.torso_m
    return moved


def carried(here: Here, stand: Stand, pose: Sequence[float]) -> list[float]:
    """A world pose fixed to the robot (the mount, the grip site with the arm held) once it stands at ``stand``."""
    transform = shift(here, stand) @ pose_matrix(list(pose))
    return [*(float(value) for value in transform[:3, 3]), *matrix_quaternion(transform[:3, :3].tolist())]


def lifted(here: Here, torso_m: float) -> Stand:
    """The base where it stands, the torso at ``torso_m``."""
    return Stand(*here.base, torso_m)


# ------------------------------------------------------------------ public depth

def back_project(camera: Mapping[str, Any], max_depth_m: float) -> tuple[Any, Any]:
    """World points of a camera's valid depth pixels (inside its published range and ``max_depth_m``) and their flat
    pixel indices; the camera is ``depth_m``, ``intrinsic_cv``, ``cam2world_cv`` and ``depth_range_m``."""
    depth = np.asarray(camera["depth_m"], dtype=np.float64)
    near, far = (float(value) for value in camera["depth_range_m"])
    valid = np.isfinite(depth) & (depth > near) & (depth < far) & (depth <= max_depth_m)
    rows, columns = np.nonzero(valid)
    values = depth[rows, columns]
    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
    local = np.stack([(columns - intrinsic[0, 2]) * values / intrinsic[0, 0],
                      (rows - intrinsic[1, 2]) * values / intrinsic[1, 1], values], axis=1)
    pose = np.asarray(camera["cam2world_cv"], dtype=np.float64)
    return local @ pose[:3, :3].T + pose[:3, 3], rows * depth.shape[1] + columns


#: Points per block of a sphere test, and the coarse cell that keeps the test to the points near a sphere.
_BLOCK = 8192
_NEAR_CELL_M = 0.05


def _sphere_depth(points: Any, spheres: Any, margin: float) -> tuple[Any, Any]:
    """Per point, how deep it lies inside the deepest of ``spheres`` grown by ``margin`` (negative: outside all) and
    which sphere that is (``-inf`` and -1 for a point no sphere comes near). Only the points in a coarse cell a sphere's
    bounds touch are measured, as ``depth_world`` screens its robot points."""
    depth, which = np.full(len(points), -np.inf), np.full(len(points), -1, dtype=np.int64)
    spheres = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    reach = spheres[:, 3] + margin
    spheres, reach = spheres[reach > 0.0], reach[reach > 0.0]
    if not len(points) or not len(spheres):
        return depth, which
    low = points.min(axis=0)
    shape = np.floor((points.max(axis=0) - low) / _NEAR_CELL_M).astype(np.int64) + 1
    grid = np.zeros(tuple(shape), dtype=bool)
    first = np.clip(np.floor((spheres[:, :3] - reach[:, None] - low) / _NEAR_CELL_M).astype(np.int64), 0, shape)
    last = np.clip(np.floor((spheres[:, :3] + reach[:, None] - low) / _NEAR_CELL_M).astype(np.int64) + 1, 0, shape)
    for (a, b, c), (d, e, f) in zip(first.tolist(), last.tolist()):
        grid[a:d, b:e, c:f] = True
    near = np.nonzero(grid[tuple(np.floor((points - low) / _NEAR_CELL_M).astype(np.int64).T)])[0]
    for start in range(0, len(near), _BLOCK):
        chosen = near[start:start + _BLOCK]
        inside = reach[None, :] - np.sqrt(((points[chosen, None, :] - spheres[None, :, :3]) ** 2).sum(axis=2))
        which[chosen] = np.argmax(inside, axis=1)
        depth[chosen] = inside[np.arange(len(chosen)), which[chosen]]
    return depth, which


def within_spheres(points: Any, spheres: Any, margin: float) -> Any:
    """Which world ``points`` lie within a sphere ``[x, y, z, radius]`` of ``spheres`` grown by ``margin``."""
    return _sphere_depth(np.asarray(points, dtype=np.float64).reshape(-1, 3), spheres, margin)[0] >= 0.0


def in_box(points: Any, frame: Any, box: Sequence[Sequence[float]]) -> Any:
    """Which world ``points`` lie in ``box`` (``(low, high)`` per axis) of the 4x4 world ``frame``."""
    frame = np.asarray(frame, dtype=np.float64)
    local = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - frame[:3, 3]) @ frame[:3, :3]
    bounds = np.asarray(box, dtype=np.float64)
    return ((local >= bounds[:, 0]) & (local <= bounds[:, 1])).all(axis=1)


def _radius(base: MobileBase) -> float:
    """The farthest corner of the footprint from the base origin, in the floor plane."""
    return max(math.hypot(x, y) for box in base.footprint for x in box.x_m for y in box.y_m)


def _deepest(points: Any, base: MobileBase, pose: Sequence[float], torso_m: float, margin_m: float) -> tuple[float, int]:
    """The deepest any point lies inside the grown footprint at ``pose`` and its index (``-inf`` when none is near)."""
    near = np.nonzero(((points[:, :2] - np.asarray(pose[:2], dtype=np.float64)) ** 2).sum(axis=1)
                      <= (_radius(base) + margin_m) ** 2)[0]
    if not len(near):
        return -math.inf, -1
    depth = footprint_depth(points[near], base, pose, torso_m, margin_m)
    hit = int(np.argmax(depth))
    return float(depth[hit]), int(near[hit])


def footprint_depth(points: Any, base: MobileBase, pose: Sequence[float], torso_m: float, margin_m: float) -> Any:
    """How deep each point lies inside the base's footprint at ``pose`` grown by ``margin_m`` (negative: outside)."""
    x, y, yaw = (float(value) for value in pose)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    local_x, local_y = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    deepest = np.full(len(points), -np.inf)
    for box in base.footprint:
        top = box.z_m[1] + (torso_m if box.top_lifts else 0.0)
        depth = np.minimum.reduce([local_x - box.x_m[0], box.x_m[1] - local_x, local_y - box.y_m[0],
                                   box.y_m[1] - local_y, points[:, 2] - box.z_m[0], top - points[:, 2]]) + margin_m
        deepest = np.maximum(deepest, depth)
    return deepest


def _one_per_cell(points: Any, cell_m: float) -> Any:
    """The first of ``points`` in each cell of ``cell_m`` (over all their coordinates), in their order."""
    if not len(points):
        return points
    _cells, first = np.unique(np.floor(points / cell_m).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def _seen_floor(points: Any, origin: Any, top_m: float, search: Search) -> Any:
    """The ``floor_cell_m`` cells (one ``(x, y)`` each) where rays from ``origin`` to ``points`` run at or below
    ``top_m``, ``floor_ray_samples`` samples along each."""
    steps = np.arange(1, search.floor_ray_samples + 1, dtype=np.float64)[None, :, None] / search.floor_ray_samples
    found = [np.zeros((0, 2))]
    for start in range(0, len(points), _BLOCK):
        samples = origin + steps * (points[start:start + _BLOCK, None, :] - origin)
        found.append(_one_per_cell(samples[samples[..., 2] <= top_m][:, :2], search.floor_cell_m))
    return _one_per_cell(np.concatenate(found), search.floor_cell_m)


@dataclass(frozen=True)
class _Fit:
    """What ``depth_load.fit_held_spheres`` reads: the voxel its centres stand for and how many spheres it fits."""

    voxel_size_m: float
    held_sphere_slots: int


def linked_load(points: Any, own: OwnBody, search: Search = SEARCH) -> Any:
    """Spheres (world ``[x, y, z, radius]``) for the depth ``points`` that is part of the load ``own`` holds
    (``held_depth.load_voxels`` from the load's spheres, within ``Search.load_reach_m`` of their centre), fitted as the
    planner fits its load's; none when nothing is held or nothing links to it."""
    spheres = np.asarray(own.spheres, dtype=np.float64).reshape(-1, 4)
    held = spheres[len(spheres) - own.held:] if own.held else np.zeros((0, 4))
    voxels = load_voxels(points, held, own.margin_m, search.load_reach_m, search.load_shell_m, OBSTACLE_VOXEL_M)
    return fit_held_spheres(voxels, _Fit(OBSTACLE_VOXEL_M, search.load_spheres)) if len(voxels) else np.zeros((0, 4))


def public_points(cameras: Mapping[str, Mapping[str, Any]], *, base: MobileBase, here: Here, own: OwnBody,
                  max_depth_m: float, search: Search = SEARCH) -> PublicPoints:
    """The public depth of one moment: obstacles above the floor, the floor seen (``_seen_floor``, below the top of the
    boxes that stand on it), and per camera the pixels of the robot itself: those ``own`` holds, those of the depth
    linked to its held load (``linked_load``, whose spheres the returned ``body`` carries as its load too), and the base
    where it stands, grown by ``own_margin_m``."""
    seen_by_camera, floor = {}, []
    top = max((box.z_m[1] for box in base.footprint if box.z_m[0] <= base.floor_m), default=base.floor_m)
    for name, camera in cameras.items():
        points, pixels = back_project(camera, max_depth_m)
        mine = own.holds(points) | (footprint_depth(points, base, here.base, here.torso_m, search.own_margin_m) >= 0.0)
        seen_by_camera[name] = (points, pixels, mine)
        floor.append(_seen_floor(points, np.asarray(camera["cam2world_cv"], dtype=np.float64)[:3, 3], top, search))
    load = linked_load(np.concatenate([points[~mine & (points[:, 2] > base.floor_m)]
                                       for points, _, mine in seen_by_camera.values()] or [np.zeros((0, 3))]),
                       own, search)
    if len(load):
        own = own.carrying(load)
    obstacles, mine_by_camera = [], {}
    for name, (points, pixels, mine) in seen_by_camera.items():
        if len(load):
            mine = mine | within_spheres(points, load, own.margin_m)
        mine = mine | edge_pixels(cameras[name], pixels, mine, search.edge_step_m)
        mine_by_camera[name] = pixels[mine]
        obstacles.append(points[~mine & (points[:, 2] > base.floor_m)])
    found = np.concatenate(obstacles) if obstacles else np.zeros((0, 3))
    seen = np.concatenate(floor) if floor else np.zeros((0, 2))
    # One point per voxel of the depth world's size is all a footprint can tell apart.
    return PublicPoints(_one_per_cell(found, OBSTACLE_VOXEL_M), mine_by_camera,
                        _one_per_cell(seen, search.floor_cell_m), own)


def without_own(cameras: Mapping[str, Mapping[str, Any]], points: PublicPoints) -> dict[str, dict[str, Any]]:
    """The cameras with the robot's own pixels taken out (no depth there, so free), for a world seen from elsewhere:
    the robot moves away from where the cameras saw it."""
    result = {}
    for name, camera in cameras.items():
        depth = np.array(camera["depth_m"], dtype=np.float32, copy=True)
        depth.reshape(-1)[np.asarray(points.own_pixels.get(name, ()), dtype=np.int64)] = 0.0
        result[name] = {**dict(camera), "depth_m": depth}
    return result


# ------------------------------------------------------------------ the base's move

def servo_path(here: Here, base: MobileBase, goal: Sequence[float]) -> list[tuple[float, float, float]]:
    """The base poses the servo passes on its way to ``goal`` ``(x, y, yaw)``: the turn centre driven straight at the
    servo's speed while the base turns at its own rate, as the backend's servo commands them, ``goal`` last."""
    servo = base.servo
    cx, cy = here.turn_centre_m

    def centre(x: float, y: float, yaw: float) -> tuple[float, float]:
        return x + math.cos(yaw) * cx - math.sin(yaw) * cy, y + math.sin(yaw) * cx + math.cos(yaw) * cy

    gx, gy, gyaw = (float(value) for value in goal)
    target = centre(gx, gy, gyaw)
    x, y, yaw = here.base
    ax, ay = centre(x, y, yaw)
    poses = []
    for _ in range(servo.max_ticks):
        dx, dy, dyaw = target[0] - ax, target[1] - ay, wrap(gyaw - yaw)
        distance = math.hypot(dx, dy)
        if distance <= servo.position_tolerance_m and abs(dyaw) <= servo.yaw_tolerance_rad:
            break
        step = min(servo.max_linear_m_s, servo.linear_gain * distance) * servo.tick_s
        if distance > 1e-9:
            ax, ay = ax + dx / distance * min(step, distance), ay + dy / distance * min(step, distance)
        turn = max(-servo.max_yaw_rad_s, min(servo.max_yaw_rad_s, servo.yaw_gain * dyaw)) * servo.tick_s
        yaw = wrap(yaw + (turn if abs(turn) < abs(dyaw) else dyaw))
        poses.append((ax - math.cos(yaw) * cx + math.sin(yaw) * cy, ay - math.sin(yaw) * cx - math.cos(yaw) * cy, yaw))
    poses.append((gx, gy, gyaw))
    return poses


def _thinned(poses: Sequence[tuple[float, float, float]], search: Search) -> list[tuple[float, float, float]]:
    """Every pose at least ``sweep_step_m`` or ``sweep_step_rad`` past the last one kept, and the last."""
    kept: list[tuple[float, float, float]] = []
    for pose in poses[:-1]:
        last = kept[-1] if kept else None
        if last is None or math.hypot(pose[0] - last[0], pose[1] - last[1]) >= search.sweep_step_m \
                or abs(wrap(pose[2] - last[2])) >= search.sweep_step_rad:
            kept.append(pose)
    return [*kept, poses[-1]]


def _carried(obstacles: Any, body: OwnBody, here: Here, pose: Sequence[float], torso_m: float,
             margin_m: float) -> tuple[float, int, int]:
    """How deep a depth point lies inside the robot's body, carried at its joints to ``pose``, its spheres grown by
    ``margin_m`` (negative: shrunk; ``-inf`` when none is near), that point's index and the sphere's."""
    spheres = np.asarray(body.spheres, dtype=np.float64).reshape(-1, 4)
    if not len(spheres):
        return -math.inf, -1, -1
    moved = shift(here, Stand(*(float(value) for value in pose), torso_m))
    depth, which = _sphere_depth(obstacles, np.hstack([spheres[:, :3] @ moved[:3, :3].T + moved[:3, 3],
                                                       spheres[:, 3:]]), margin_m)
    hit = int(np.argmax(depth)) if len(depth) else -1
    return (float(depth[hit]), hit, int(which[hit])) if hit >= 0 else (-math.inf, -1, -1)


def sweep(points: PublicPoints, base: MobileBase, here: Here, goal: Sequence[float], torso_m: float | None = None, *,
          margin_m: float | None = None, slack_m: float = 0.0, search: Search = SEARCH) -> dict[str, Any] | None:
    """Where the base, driven by its servo to ``goal`` ``(x, y, yaw)``, first meets the depth, or ``None``.

    At every pose the servo passes, the base's footprint grown by ``margin_m`` (``admission_margin_m``) must hold no
    depth point, and the robot's body as ``points.body`` holds it, carried at its joints, none deeper than
    ``carried_tolerance_m``; ``slack_m`` grows the footprint and shrinks that tolerance by as much (a suggested stand's
    move, ``suggestion_slack_m``). Returns ``robot_part`` (``base``, ``arm`` or ``held_object``), ``path_fraction`` (of
    the poses the servo passes), ``penetration_m`` (how far the deepest point lies inside the footprint or the sphere
    itself, 0 when only inside the footprint's margin), ``obstacle_height_m`` (that point's height above the floor) and
    ``obstacle_xy_m`` (where it is)."""
    margin = (search.admission_margin_m if margin_m is None else margin_m) + slack_m
    tolerance = search.carried_tolerance_m - slack_m
    torso = here.torso_m if torso_m is None else torso_m
    obstacles = np.asarray(points.obstacles, dtype=np.float64).reshape(-1, 3)
    poses = _thinned(servo_path(here, base, goal), search)
    if not len(obstacles):
        return None
    body = points.body
    for index, pose in enumerate(poses):
        depth, hit = _deepest(obstacles, base, pose, torso, margin)
        part, inside = "base", depth - margin
        if depth < 0.0 and body is not None:
            depth, hit, sphere = _carried(obstacles, body, here, pose, torso, -tolerance)
            part = "held_object" if sphere >= len(body.spheres) - body.held else "arm"
            inside = depth + tolerance
        if depth >= 0.0:
            point = obstacles[hit]
            return {"robot_part": part, "path_fraction": round((index + 1) / len(poses), 3),
                    "penetration_m": round(max(0.0, inside), 4),
                    "obstacle_height_m": round(float(point[2]), 3),
                    "obstacle_xy_m": [round(float(point[0]), 3), round(float(point[1]), 3)]}
    return None


# ------------------------------------------------------------------ the search

def _heights(base: MobileBase, here: Here, target_z: float, mount_z: float) -> list[float]:
    """The torso now, and, where a stand may choose the lift, its steps around the height that levels the shoulder
    with the target."""
    if base.torso_range_m is None:
        return [here.torso_m]
    low, high = base.torso_range_m
    grid = [low + index * base.torso_step_m for index in range(int(math.floor((high - low) / base.torso_step_m)) + 1)]
    grid = sorted({round(value, 6) for value in (*grid, high)})
    level = min(max(target_z - (mount_z - here.torso_m) - base.shoulder_m, low), high)
    nearest = min(range(len(grid)), key=lambda index: abs(grid[index] - level))
    around = grid[max(0, nearest - 1):nearest + 2]
    return [here.torso_m, *(value for value in around if abs(value - here.torso_m) > 1e-6)]


def stands(target: Sequence[float], here: Here, base: MobileBase, search: Search = SEARCH) -> list[Stand]:
    """Every stand step 1 keeps (module docstring), cheapest first."""
    tx, ty, tz = (float(value) for value in target[:3])
    x0, y0, yaw0 = here.base
    mount = np.linalg.inv(base_matrix(*here.base)) @ np.array([*here.mount[:3], 1.0])
    mx, my = float(mount[0]), float(mount[1])
    bearing = math.atan2(here.mount[1] - ty, here.mount[0] - tx)
    now = math.hypot(here.mount[0] - tx, here.mount[1] - ty)
    standoffs = sorted({*search.standoffs_m, *((round(now, 3),) if min(search.standoffs_m) <= now <= max(
        search.standoffs_m) else ())})
    heights = _heights(base, here, tz, here.mount[2])
    found: dict[tuple[int, ...], Stand] = {}
    for standoff in standoffs:
        for offset in search.bearing_offsets_rad:
            angle = bearing + offset
            px, py = tx + standoff * math.cos(angle), ty + standoff * math.sin(angle)
            facing = math.atan2(ty - py, tx - px)
            yaws: list[float] = []
            for yaw in (yaw0, facing, yaw0 + wrap(facing - yaw0) / 2.0):
                if all(abs(wrap(yaw - other)) > math.radians(5.0) for other in yaws):
                    yaws.append(wrap(yaw))
            for yaw in yaws:
                bx = px - (math.cos(yaw) * mx - math.sin(yaw) * my)
                by = py - (math.sin(yaw) * mx + math.cos(yaw) * my)
                travel, turn = math.hypot(bx - x0, by - y0), abs(wrap(yaw - yaw0))
                if travel > search.max_travel_m or (travel < search.here_m and turn < search.here_rad):
                    continue
                nudge = search.nudge_cost_m if travel < search.nudge_m and turn < search.nudge_rad else 0.0
                for height in heights:
                    if math.dist((px, py, here.mount[2] + height - here.torso_m), (tx, ty, tz)) > base.reach_m:
                        continue
                    stand = Stand(bx, by, yaw, height, travel + search.turn_cost_m_per_rad * turn
                                  + search.lift_cost * abs(height - here.torso_m) + nudge)
                    key = (round(bx * 100), round(by * 100), round(math.degrees(yaw)), round(height * 1000))
                    if key not in found or found[key].cost > stand.cost:
                        found[key] = stand
    return sorted(found.values(), key=lambda stand: (stand.cost, stand.x_m, stand.y_m, stand.yaw_rad, stand.torso_m))


class _Raster:
    """The obstacles as floor cells per footprint box: a cell is taken when a point in it lies within the box's height
    band, grown by the margin. A stand's footprint is then its boxes' cells, sampled finer than a cell."""

    def __init__(self, obstacles: Any, base: MobileBase, torsos: Sequence[float], margin_m: float, cell_m: float) -> None:
        self.base, self.margin, self.cell = base, margin_m, cell_m
        self.low = obstacles[:, :2].min(axis=0) if len(obstacles) else np.zeros(2)
        shape = (np.floor((obstacles[:, :2].max(axis=0) - self.low) / cell_m).astype(np.int64) + 1
                 if len(obstacles) else np.ones(2, dtype=np.int64))
        cells = np.floor((obstacles[:, :2] - self.low) / cell_m).astype(np.int64)
        self.taken: dict[tuple[int, float], Any] = {}
        for index, box in enumerate(base.footprint):
            for torso in (torsos if box.top_lifts else (0.0,)):
                top = box.z_m[1] + (torso if box.top_lifts else 0.0)
                band = (obstacles[:, 2] >= box.z_m[0] - margin_m) & (obstacles[:, 2] <= top + margin_m)
                grid = np.zeros(tuple(shape), dtype=bool)
                grid[tuple(cells[band].T)] = True
                self.taken[(index, round(torso, 6) if box.top_lifts else 0.0)] = grid

    def clear(self, pose: Sequence[float], torso_m: float) -> bool:
        x, y, yaw = (float(value) for value in pose)
        step, margin = self.cell / 2.0, self.margin
        for index, box in enumerate(self.base.footprint):
            grid = self.taken[(index, round(torso_m, 6) if box.top_lifts else 0.0)]
            xs = np.arange(box.x_m[0] - margin, box.x_m[1] + margin + step / 2, step)
            ys = np.arange(box.y_m[0] - margin, box.y_m[1] + margin + step / 2, step)
            local = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
            world = np.stack([x + math.cos(yaw) * local[:, 0] - math.sin(yaw) * local[:, 1],
                              y + math.sin(yaw) * local[:, 0] + math.cos(yaw) * local[:, 1]], axis=1)
            cells = np.floor((world - self.low) / self.cell).astype(np.int64)
            inside = ((cells >= 0) & (cells < np.asarray(grid.shape))).all(axis=1)
            if grid[tuple(cells[inside].T)].any():
                return False
        return True


class _Floor:
    """The floor cells a camera saw, or that the base stands on now (its own footprint hides that floor, and the base is
    there): a stand's floor is the footprint's boxes that stand on the floor, sampled finer than a cell."""

    def __init__(self, floor: Any, base: MobileBase, here: Here, cell_m: float) -> None:
        self.base, self.cell = base, cell_m
        self.boxes = [box for box in base.footprint if box.z_m[0] <= base.floor_m]
        seen = np.asarray(floor if floor is not None else np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)
        self.seen = set(map(tuple, np.floor(np.vstack([seen, self._samples(here.base)]) / cell_m).astype(np.int64)
                            .tolist()))

    def _samples(self, pose: Sequence[float]) -> Any:
        x, y, yaw = (float(value) for value in pose)
        step = self.cell / 2.0
        rows = []
        for box in self.boxes:
            xs = np.arange(box.x_m[0] + step / 2, box.x_m[1], step)
            ys = np.arange(box.y_m[0] + step / 2, box.y_m[1], step)
            local = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
            rows.append(np.stack([x + math.cos(yaw) * local[:, 0] - math.sin(yaw) * local[:, 1],
                                  y + math.sin(yaw) * local[:, 0] + math.cos(yaw) * local[:, 1]], axis=1))
        return np.vstack(rows) if rows else np.zeros((0, 2))

    def fraction(self, pose: Sequence[float]) -> float:
        """How much of the floor under the footprint at ``pose`` is seen."""
        samples = self._samples(pose)
        if not len(samples):
            return 1.0
        cells = np.floor(samples / self.cell).astype(np.int64).tolist()
        return sum(tuple(cell) in self.seen for cell in cells) / len(cells)


def _apart(stand: Stand, kept: Sequence[Stand], search: Search) -> bool:
    return all(math.hypot(stand.x_m - other.x_m, stand.y_m - other.y_m) >= search.separation_m
               or abs(wrap(stand.yaw_rad - other.yaw_rad)) >= search.separation_rad for other in kept)


def arrival_band(stand: Stand, servo: Servo) -> list[Stand]:
    """Where the base may stop when its servo drives it to ``stand``: the position tolerance either way along the
    world's x and y, and the yaw tolerance either way."""
    distance, turn = servo.position_tolerance_m, servo.yaw_tolerance_rad
    return [Stand(stand.x_m + dx, stand.y_m + dy, wrap(stand.yaw_rad + dyaw), stand.torso_m)
            for dx, dy, dyaw in ((distance, 0.0, 0.0), (-distance, 0.0, 0.0), (0.0, distance, 0.0),
                                 (0.0, -distance, 0.0), (0.0, 0.0, turn), (0.0, 0.0, -turn))]


def suggestion(here: Here, stand: Stand, arm_path: str | None, *, torso: bool = True) -> dict[str, Any]:
    """A kept stand as the policy reads it: a ready ``move_base`` target, with the torso height the arm reaches from
    when the stand chose it (``torso``)."""
    return {"x_m": round(stand.x_m, 3), "y_m": round(stand.y_m, 3), "yaw_deg": round(math.degrees(stand.yaw_rad), 1),
            **({"torso_height_m": round(stand.torso_m, 3)} if torso else {}),
            "distance_m": round(math.hypot(stand.x_m - here.base[0], stand.y_m - here.base[1]), 3),
            "turn_deg": round(math.degrees(wrap(stand.yaw_rad - here.base[2])), 1), "arm_path": arm_path}


def search(target: Sequence[float], here: Here, base: MobileBase, points: PublicPoints, *,
           solve: Callable[[Sequence[Stand]], Sequence[bool]], admits: Callable[[Stand], bool | None],
           settings: Search = SEARCH) -> dict[str, Any]:
    """The nearest stands from which the move to ``target`` (world xyz) is admitted, as the module docstring orders
    the checks.

    ``solve(stands)`` answers, per stand, whether the shared IK reaches the target from there in a blank world
    (kinematics alone; one question for many stands). ``admits(stand)`` says whether the refused move itself is
    admitted from there as its real call admits it (its path in the world seen from there, then its timing); ``None``,
    when the planner could not answer, keeps no stand."""
    candidates = stands(target, here, base, settings)
    obstacles = np.asarray(points.obstacles, dtype=np.float64).reshape(-1, 3)
    raster = _Raster(obstacles, base, sorted({stand.torso_m for stand in candidates}), settings.search_margin_m,
                     settings.raster_m) if len(obstacles) else None
    floor = _Floor(points.floor, base, here, settings.floor_cell_m)
    clear, unseen = [], 0
    for stand in candidates:
        pose = (stand.x_m, stand.y_m, stand.yaw_rad)
        if raster is not None and not raster.clear(pose, stand.torso_m):
            continue
        if floor.fraction(pose) < settings.floor_seen_fraction:
            unseen += 1
            continue
        clear.append(stand)
    asked = clear[:settings.blank_queries]
    reached = [stand for stand, solved in zip(asked, solve(asked) if asked else ()) if solved]
    kept: list[Stand] = []
    checked = blocked = unreached = 0
    for stand in reached:
        if len(kept) >= settings.suggestions or checked >= settings.observed_checks:
            break
        if not _apart(stand, kept, settings):
            continue
        swept = sweep(points, base, here, (stand.x_m, stand.y_m, stand.yaw_rad), slack_m=settings.suggestion_slack_m,
                      search=settings)
        if swept is not None:
            blocked += swept["robot_part"] != "base"
            continue
        if not all(solve(arrival_band(stand, base.servo))):
            unreached += 1
            continue
        checked += 1
        if admits(stand):
            kept.append(stand)
    return {"schema": SCHEMA,
            "suggestions": [suggestion(here, stand, ADMITTED, torso=base.torso_range_m is not None) for stand in kept],
            "searched": {"stands": len(candidates), "footprint_clear": len(clear), "floor_unseen": unseen,
                         "asked": len(asked), "ik_reached": len(reached), "arm_blocked": blocked,
                         "arrival_unreached": unreached, "checked_in_world": checked,
                         "max_travel_m": settings.max_travel_m}}


__all__ = ["ADMITTED", "BASE_PATH_CODE", "Box", "Here", "MobileBase", "NOT_PLANNED", "OwnBody", "PLANNED", "PublicPoints",
           "SCHEMA", "SEARCH", "SEARCHED_CODES", "Search", "Servo", "Stand", "arrival_band", "back_project",
           "base_matrix", "carried", "footprint_depth", "in_box", "lifted", "linked_load", "public_points", "search",
           "servo_path", "shift", "stands", "suggestion", "sweep", "within_spheres", "without_own", "wrap"]
