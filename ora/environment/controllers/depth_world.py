"""Public-depth collision world of the shared motion planner (``public_depth_voxel_v1``).

A pure function of public evidence: both cameras' metric depth with their
published calibration, the measured base pose, joints and finger opening, and
the robot's collision spheres as the planner itself reports them at those
joints. No simulator state, segmentation or object identity enters it. The
isolated CuRobo worker builds it for every plan and IK request from one input
file the session writes (``write_world_inputs``); a host tool reading the same
file gets the same world and hash.

The steps, all in the robot base frame:

1. Back-project every pixel of both cameras whose depth lies inside the
   camera's published range (and ``max_depth_m``).
2. Classify each point. Robot: inside a robot sphere plus
   ``robot_point_margin_m``, or inside the hand box at the measured hand pose
   (CuRobo locks the fingers open, so the box covers the real fingers whatever
   their opening), except a hand-box point at or below the support under the
   start hand (step 4) plus half a voxel, which is that support. Held seed:
   between the jaws (the pad slab across the hand box's width) while the
   gripper is commanded closed with a gap between ``held_min_gap_m`` and
   ``held_max_gap_m``; the world inputs carry no jaw-stall outcome, so
   commanded-closed jaws that stopped short of closing, but did close, are the
   evidence of a load. World: the rest.
3. Thicken each world and seed point along its camera ray (the unseen back of
   a surface), keep what falls inside the grid and mark those voxels occupied.
   Space behind a surface beyond the thickness, occluded or not, stays free.
   The held object is the connected component (26-neighbourhood) of occupied
   voxels grown from the seed voxels, within ``held_component_reach_m`` of the
   pad slab and never through the slab of a support below the seed (step 4),
   so a table the object stands on stays world. A component that reaches that
   bound is joined to the scene (a container, a shelf), so only the seed
   voxels are held then (``held_component: seed_only``) and the scene keeps
   its structure. So is a component that reaches more than
   ``held_overhang_m`` past the jaws along the closing axis and stands on that
   support in at least ``held_rest_columns`` columns: a held object touching a
   cup on a plate, not a bowl held by its rim, which leaves the support as it
   lifts. Its voxels leave the world and
   become at most ``held_sphere_slots`` spheres in the ``panda_hand`` frame
   (deterministic farthest-point k-means, each sphere covering its voxels),
   which the planner attaches to the hand: no ghost of the load stays behind.
4. Support: the dominant horizontal surface under a hand pose
   (``support``). Within ``support_window_m`` of the fingertips, each grid
   column's highest voxel holding a seen world point is its top; the support
   is the lowest pair of adjacent levels holding at least
   ``support_min_columns`` tops, at the median height of the depth points in it.
   When the arm hides that window (no level qualifies there), the whole grid
   is the window: the table is seen around the arm. Objects stand on it; a plate, box
   or bowl rim above it is not a support.
5. Start carve: clear the voxels within the robot spheres and the held spheres
   (each plus ``start_carve_margin_m``) and within the carve box (the hand box
   ending ``carve_box_tip_m`` along the approach, about 1 cm past the
   fingertips, plus the margin on its other faces), so the measured start is
   never in collision with the depth of the robot and its load. The carve box
   leaves the support under the start hand: a support voxel in the slab
   ``support_slab_m`` deep whose centre lies below the support surface.
6. Goal carve: clear the carve box at the goal hand pose, extended by
   ``goal_standoff_m`` against the approach direction, the held spheres swept
   over that standoff, and, for a joint-space goal, the robot spheres there.
   All of that leaves the support under the goal hand. The spheres attached for
   this goal stop one voxel above it (``depth_load.clip_to_floor``), so a load
   set down on it is not refused for what they over-cover, unless the load's
   own seen points lie more than ``support_tolerance_m`` below it: then they
   stop there, in the support. A goal whose fingertips (``fingertip_m`` along the
   approach, ``finger_span_m`` either side of the jaw centre, as CuRobo locks the
   fingers open) or whose load lie more than ``support_tolerance_m`` below the
   support surface is refused (``goal_refusal``) before any solver runs, unless
   the columns under that fingertip or those points show a lower surface (the
   support ends there). The tolerance only spares the refusal; the carve still
   leaves the support, so a goal within it is solved against the support's
   voxels like any other obstacle.
7. ESDF: Euclidean distance between voxel centres, positive inside occupied
   space and negative outside, half a voxel at the boundary, clamped to
   ``esdf_clamp_m``. The windowed separable transform is exact within the clamp.

Array layout follows CuRobo v0.7.8's voxel kernel: shape
``round(dims / voxel) + 1`` per axis, C order (x slowest), voxel ``i`` spanning
``[centre - dims / 2 + i * voxel, ... + voxel)``.

The kernel treats the outer ``CUROBO_UNOBSERVED_BORDER_VOXELS`` voxels of each
face as unobserved: a sphere there reads -100 (free) whatever the grid holds.
The grid is therefore the intended volume grown by that many voxels per face
around the same centre; ``effective_low_m`` and ``effective_high_m`` bound the
part the planner actually collides with (x -0.20..1.01, y -0.60..0.61,
z -0.25..0.76 m in the base frame for ``public_depth_voxel_v1``). Depth in the
border is still voxelized and enters the hash, but never blocks a plan.
``curobo_v2_grid`` exports exactly that part in cuRobo V2's layout.

The steps above read the Panda's numbers; a robot's own are its entry of
``ROBOT_WORLDS``: its parameter set and the fixed turn from its tool frame to the
hand convention these steps use (approach +z, jaw line y). A two-arm robot writes
``ora.depth-world-input.v2``, with its own named cameras and the other arm
(joints and base pose): that arm's spheres (``other_arm_spheres_base``, the same
robot model at those joints) take their depth out of the scene and go back in as
occupied voxels after every carve, so a plan keeps clear of the whole arm, seen or
not, and no goal carve can clear it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .depth_esdf import signed_distance
from .depth_load import clip_to_floor, fingertip_refusal, fit_held_spheres, load_refusal, stands_past_jaws
from .depth_world_inputs import (
    CAMERAS, INPUT_FIELDS, WORLD_INPUT_SCHEMA, WORLD_INPUT_SCHEMA_V2, DepthCamera, DepthWorldError, WorldInputs,
    other_arm_spheres_base, pose_matrix, read_world_inputs, valid_spheres, write_world_inputs,
)

WORLD_MODEL = "public_depth_voxel_v1"
WORLD_HASH_DOMAIN = b"ora.public-depth-world.v1\0"
#: The coarse cell that keeps a sphere test to the points near a sphere.
NEAR_CELL_M = 0.05
#: How far a sphere's search box reaches past the sphere on every axis: far above any rounding of the squared
#: distance at these scales, so the box never leaves out a point the sphere test takes.
SPHERE_BOX_PAD_M = 1e-3
#: Voxels per face that CuRobo v0.7.8's voxel kernel reports as unobserved (-100).
CUROBO_UNOBSERVED_BORDER_VOXELS = 3
@dataclass(frozen=True)
class WorldParameters:
    """One named, versioned parameter set; its values enter every world hash."""

    world_model: str
    voxel_size_m: float
    #: The grid box in the robot base frame (identity rotation); also the crop.
    #: It includes the kernel's unobserved border (``effective_low_m``).
    grid_center_base_m: tuple[float, float, float]
    grid_dims_m: tuple[float, float, float]
    esdf_clamp_m: float
    max_depth_m: float
    surface_thickness_m: float
    robot_point_margin_m: float
    start_carve_margin_m: float
    #: ``panda_hand`` frame (approach +z, jaw line y): CuRobo's hand and open
    #: finger spheres with their buffer plus about 2 cm.
    hand_box_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    #: The carve box is the hand box ending here along the approach.
    carve_box_tip_m: float
    #: The fingertips in the ``panda_hand`` frame: along the approach, and either
    #: side of the jaw centre with CuRobo's fingers locked open.
    fingertip_m: float
    finger_span_m: float
    support_window_m: float
    support_min_columns: int
    support_slab_m: float
    #: A goal is refused only when a fingertip lies more than this below the
    #: support surface (a median of depth points, so it carries depth noise).
    support_tolerance_m: float
    goal_standoff_m: float
    held_min_gap_m: float
    #: Jaws this far apart or more never closed on anything (the Panda opens to 8 cm).
    held_max_gap_m: float
    held_gap_inset_m: float
    #: The seed slab between the jaws, in the ``panda_hand`` frame: the hand
    #: box's width and the pads' span along the approach.
    held_box_x_m: tuple[float, float]
    held_box_z_m: tuple[float, float]
    held_component_reach_m: float
    #: A component reaching this far past the jaws along the closing axis, and standing on
    #: the support in this many columns, is the scene it touches, not the load.
    held_overhang_m: float
    held_rest_columns: int
    held_sphere_slots: int

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(round(dim / self.voxel_size_m)) + 1 for dim in self.grid_dims_m)

    @property
    def grid_low_m(self) -> Any:
        return (np.asarray(self.grid_center_base_m, dtype=np.float64)
                - 0.5 * np.asarray(self.grid_dims_m, dtype=np.float64))

    @property
    def effective_low_m(self) -> Any:
        """The low corner of the volume the kernel collides with (inside the border)."""
        return self.grid_low_m + CUROBO_UNOBSERVED_BORDER_VOXELS * self.voxel_size_m

    @property
    def effective_high_m(self) -> Any:
        """The high corner of that volume (exclusive)."""
        shape = np.asarray(self.grid_shape)
        return self.grid_low_m + (shape - CUROBO_UNOBSERVED_BORDER_VOXELS) * self.voxel_size_m

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


PUBLIC_DEPTH_VOXEL_V1 = WorldParameters(
    world_model=WORLD_MODEL,
    voxel_size_m=0.01,
    grid_center_base_m=(0.40, 0.0, 0.25),
    # 1.20 x 1.20 x 1.00 m of collision volume plus the kernel's 3-voxel border per face.
    grid_dims_m=(1.26, 1.26, 1.06),
    esdf_clamp_m=0.15,
    max_depth_m=2.5,
    surface_thickness_m=0.02,
    # The arm's own depth reaches 2.7 cm past its collision spheres; left in, it blocks the arm's next path.
    robot_point_margin_m=0.03,
    start_carve_margin_m=0.02,
    hand_box_m=((-0.045, 0.045), (-0.125, 0.125), (-0.035, 0.13)),
    carve_box_tip_m=0.115,
    fingertip_m=0.1034,
    finger_span_m=0.04,
    support_window_m=0.15,
    support_min_columns=100,
    support_slab_m=0.03,
    support_tolerance_m=0.01,
    goal_standoff_m=0.05,
    held_min_gap_m=0.003,
    held_max_gap_m=0.075,
    held_gap_inset_m=0.002,
    held_box_x_m=(-0.045, 0.045),
    held_box_z_m=(0.06, 0.13),
    held_component_reach_m=0.15,
    held_overhang_m=0.06,
    held_rest_columns=40,
    held_sphere_slots=32,
)
#: The ARX X5, in its arm base frame. Both arms stand 0.45 m behind the table's middle, 0.3 m either side of it, so
#: the box reaches the far edge, past the other arm's base and just below the table top. The numbers along the
#: approach come from ``agent/tools/grasp/arx_x5_gripper.py`` (the pads end 0.1576 m along +x of ``link6``, the
#: finger rails sit at +/-0.0249 m with up to 0.044 m of travel); the planner holds the jaws open.
PUBLIC_DEPTH_VOXEL_X5_V1 = WorldParameters(
    world_model=WORLD_MODEL,
    voxel_size_m=0.01,
    grid_center_base_m=(0.38, 0.0, 0.25),
    # 1.40 x 1.80 x 0.80 m of collision volume plus the kernel's 3-voxel border per face.
    grid_dims_m=(1.46, 1.86, 0.86),
    esdf_clamp_m=0.15,
    max_depth_m=2.5,
    surface_thickness_m=0.02,
    robot_point_margin_m=0.03,
    start_carve_margin_m=0.02,
    hand_box_m=((-0.06, 0.06), (-0.105, 0.105), (-0.05, 0.18)),
    carve_box_tip_m=0.17,
    fingertip_m=0.1576,
    finger_span_m=0.05,
    support_window_m=0.15,
    support_min_columns=100,
    support_slab_m=0.03,
    support_tolerance_m=0.01,
    goal_standoff_m=0.05,
    held_min_gap_m=0.003,
    held_max_gap_m=0.084,
    held_gap_inset_m=0.002,
    held_box_x_m=(-0.06, 0.06),
    held_box_z_m=(0.12, 0.18),
    held_component_reach_m=0.15,
    held_overhang_m=0.06,
    held_rest_columns=40,
    held_sphere_slots=32,
)
#: The ALOHA, in an arm's base frame. Each base stands 4 cm above the table top at its near edge, 0.3 m either side of
#: the robot's middle, so the box reaches the table's far edge, past the other arm's base and below the table top. The
#: numbers along the approach come from ``fl_link6``: the finger rails sit 0.0846 m along +x, the grip centre at 0.12 m
#: and the fingertips at 0.155 m; each jaw opens to about 0.05 m off the centreline and the jaws to 9.5 cm.
PUBLIC_DEPTH_VOXEL_ALOHA_V1 = WorldParameters(
    world_model=WORLD_MODEL,
    voxel_size_m=0.01,
    grid_center_base_m=(0.45, 0.0, 0.25),
    # 1.40 x 1.80 x 0.80 m of collision volume plus the kernel's 3-voxel border per face.
    grid_dims_m=(1.46, 1.86, 0.86),
    esdf_clamp_m=0.15,
    max_depth_m=2.5,
    surface_thickness_m=0.02,
    robot_point_margin_m=0.03,
    start_carve_margin_m=0.02,
    hand_box_m=((-0.06, 0.06), (-0.105, 0.105), (-0.05, 0.17)),
    carve_box_tip_m=0.16,
    fingertip_m=0.155,
    finger_span_m=0.05,
    support_window_m=0.15,
    support_min_columns=100,
    support_slab_m=0.03,
    support_tolerance_m=0.01,
    goal_standoff_m=0.05,
    held_min_gap_m=0.003,
    held_max_gap_m=0.085,
    held_gap_inset_m=0.002,
    held_box_x_m=(-0.06, 0.06),
    held_box_z_m=(0.09, 0.16),
    held_component_reach_m=0.15,
    held_overhang_m=0.06,
    held_rest_columns=40,
    held_sphere_slots=32,
)
#: RoboLab's Droid, in the arm's root frame: the table top is z = 0 and the public depth reaches about x -0.2 to 0.9 m,
#: y +/-0.5 m. Lengths along the approach are from ``panda_hand``: the flange ``base_link`` sits 18.17 mm along and the
#: fingertips 150.3 mm past it (RoboLab's USD, the open jaw); the jaws open to 8.5 cm.
PUBLIC_DEPTH_VOXEL_ROBOTIQ_V1 = WorldParameters(
    world_model=WORLD_MODEL,
    voxel_size_m=0.01,
    grid_center_base_m=(0.35, 0.0, 0.30),
    # 1.40 x 1.40 x 0.80 m of collision volume plus the kernel's 3-voxel border per face.
    grid_dims_m=(1.46, 1.46, 0.86),
    esdf_clamp_m=0.15,
    max_depth_m=2.5,
    surface_thickness_m=0.02,
    robot_point_margin_m=0.03,
    start_carve_margin_m=0.02,
    hand_box_m=((-0.045, 0.045), (-0.095, 0.095), (-0.035, 0.19)),
    carve_box_tip_m=0.18,
    fingertip_m=0.1685,
    finger_span_m=0.075,
    support_window_m=0.15,
    support_min_columns=100,
    support_slab_m=0.03,
    support_tolerance_m=0.01,
    goal_standoff_m=0.05,
    held_min_gap_m=0.003,
    held_max_gap_m=0.085,
    held_gap_inset_m=0.002,
    held_box_x_m=(-0.045, 0.045),
    held_box_z_m=(0.11, 0.19),
    held_component_reach_m=0.15,
    held_overhang_m=0.06,
    held_rest_columns=40,
    held_sphere_slots=32,
)
#: The YAM station's arm, in its base frame (the station world's axes): the box holds its 0.62 m reach, the other base
#: 0.61 m to either side and the table top (the base plane). The hand's origin is the public TCP at the fingertips; open,
#: gripper and fingertips span x +/-0.034, y +/-0.071, z -0.155..0.002 m there (station model, 2026-11-15).
PUBLIC_DEPTH_VOXEL_YAM_V1 = replace(
    PUBLIC_DEPTH_VOXEL_ALOHA_V1, grid_center_base_m=(0.25, 0.0, 0.30), grid_dims_m=(1.06, 1.56, 0.86),
    hand_box_m=((-0.055, 0.055), (-0.09, 0.09), (-0.175, 0.02)), carve_box_tip_m=0.012, fingertip_m=0.002,
    finger_span_m=0.055, held_max_gap_m=0.088, held_box_x_m=(-0.055, 0.055), held_box_z_m=(-0.05, 0.012))
WORLD_PARAMETERS = {WORLD_MODEL: PUBLIC_DEPTH_VOXEL_V1}
#: Each robot's parameter set and the turn ``[qx, qy, qz, qw]`` from its tool frame to the hand convention.
ROBOT_WORLDS: dict[str, tuple[WorldParameters, tuple[float, float, float, float]]] = {
    "franka_panda": (PUBLIC_DEPTH_VOXEL_V1, (0.0, 0.0, 0.0, 1.0)),
    # link6: approach +x, jaw line y; the hand's +z is link6's +x and its +x is link6's -z.
    "arx_x5": (PUBLIC_DEPTH_VOXEL_X5_V1, (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))),
    # fl_link6: approach +x, jaw line y, as link6.
    "aloha_agilex": (PUBLIC_DEPTH_VOXEL_ALOHA_V1, (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))),
    # panda_hand sits 45 degrees off panda_link8; link8's own axes are the hand's, the Robotiq closing along y.
    "franka_robotiq": (PUBLIC_DEPTH_VOXEL_ROBOTIQ_V1, (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))),
    # left_tcp, the YAM's public TCP: approach +z, closing line +x; the hand's jaw line +y is the TCP's +x.
    "yam_linear_4310": (PUBLIC_DEPTH_VOXEL_YAM_V1, (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))),
}


def world_parameters(world_model: str, robot: str = "franka_panda") -> WorldParameters | None:
    """``robot``'s parameters in ``world_model``, or None for the robot-only model."""
    if world_model != WORLD_MODEL:
        return None
    if robot not in ROBOT_WORLDS:
        raise ValueError(f"{robot} has no {WORLD_MODEL} parameters yet; it plans robot-only")
    return ROBOT_WORLDS[robot][0]


def tool_from_hand(robot: str = "franka_panda") -> Any:
    """The 4x4 turn from ``robot``'s hand convention to its tool frame."""
    turn = ROBOT_WORLDS[robot][1]
    return pose_matrix([0.0, 0.0, 0.0, *turn])


_spheres = valid_spheres


def _within_spheres(points: Any, spheres: Any, margin: float) -> Any:
    inside = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(spheres):
        return inside
    # Only points in a coarse cell a sphere touches are tested: three 640x480 cameras are close to a million points.
    low = points.min(axis=0)
    shape = np.floor((points.max(axis=0) - low) / NEAR_CELL_M).astype(np.int64) + 1
    near = np.zeros(tuple(shape), dtype=bool)
    reach = spheres[:, 3:4] + margin
    first = np.clip(np.floor((spheres[:, :3] - reach - low) / NEAR_CELL_M).astype(np.int64), 0, shape)
    last = np.clip(np.floor((spheres[:, :3] + reach - low) / NEAR_CELL_M).astype(np.int64) + 1, 0, shape)
    for (a, b, c), (d, e, f) in zip(first.tolist(), last.tolist()):
        near[a:d, b:e, c:f] = True
    index = np.nonzero(near[tuple(np.floor((points - low) / NEAR_CELL_M).astype(np.int64).T)])[0]
    # Each sphere is tested only on the near points in its box (sorted by x, then cut by y and z), with the
    # all-pairs expression for that one sphere, so a point is taken exactly when that test takes it.
    index = index[np.argsort(points[index, 0], kind="stable")]
    xs = points[index, 0]
    bounds = np.abs(reach[:, 0]) + SPHERE_BOX_PAD_M
    starts = np.searchsorted(xs, spheres[:, 0] - bounds, "left").tolist()
    stops = np.searchsorted(xs, spheres[:, 0] + bounds, "right").tolist()
    for number, (bound, start, stop) in enumerate(zip(bounds.tolist(), starts, stops)):
        sphere = spheres[number:number + 1]
        chosen = index[start:stop]
        chosen = chosen[(np.abs(points[chosen, 1] - sphere[0, 1]) <= bound)
                        & (np.abs(points[chosen, 2] - sphere[0, 2]) <= bound)]
        squared = ((points[chosen, None, :] - sphere[None, :, :3]) ** 2).sum(axis=2)
        inside[chosen[(squared <= (sphere[None, :, 3] + margin) ** 2)[:, 0]]] = True
    return inside


def _in_box(points: Any, frame: Any, box: Any, margin: float = 0.0) -> Any:
    local = (points - frame[:3, 3]) @ frame[:3, :3]
    bounds = np.asarray(box, dtype=np.float64)
    return ((local >= bounds[:, 0] - margin) & (local <= bounds[:, 1] + margin)).all(axis=1)


def _held_box(parameters: WorldParameters, gap: float) -> Any:
    half = 0.5 * gap - parameters.held_gap_inset_m
    return [parameters.held_box_x_m, (-half, half), parameters.held_box_z_m]


def _dilate(mask: Any) -> Any:
    """One step of 26-neighbourhood dilation."""
    grown = mask.copy()
    for axis in range(3):
        step = grown.copy()
        low = tuple(slice(0, -1) if index == axis else slice(None) for index in range(3))
        high = tuple(slice(1, None) if index == axis else slice(None) for index in range(3))
        step[low] |= grown[high]
        step[high] |= grown[low]
        grown = step
    return grown


def _back_project(camera: DepthCamera, base_from_world: Any, parameters: WorldParameters) -> tuple[Any, Any]:
    """Valid depth pixels as base-frame points, with the camera origin in the base frame."""
    depth = np.asarray(camera.depth_m, dtype=np.float64)
    near, far = camera.depth_range_m
    rows, columns = np.nonzero(np.isfinite(depth) & (depth > near) & (depth < far)
                               & (depth <= parameters.max_depth_m))
    values = depth[rows, columns]
    intrinsic = np.asarray(camera.intrinsic_cv, dtype=np.float64)
    local = np.stack([
        (columns - intrinsic[0, 2]) * values / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * values / intrinsic[1, 1],
        values,
    ], axis=1)
    base_from_camera = base_from_world @ np.asarray(camera.cam2world_cv, dtype=np.float64)
    return local @ base_from_camera[:3, :3].T + base_from_camera[:3, 3], base_from_camera[:3, 3]


@dataclass(frozen=True)
class DepthWorld:
    esdf: Any
    occupied: Any
    held_spheres_hand: Any
    sha256: str
    summary: dict[str, Any]


class Scene:
    """The goal-independent part of one world: classified points and the start-carved grid."""

    def __init__(
        self, inputs: WorldInputs, *, robot_spheres_base: Any, start_hand_base: Any,
        parameters: WorldParameters = PUBLIC_DEPTH_VOXEL_V1, other_arm_spheres: Any = None,
    ) -> None:
        self.parameters = parameters
        self.inputs_sha256 = inputs.content_sha256
        size = parameters.voxel_size_m
        shape = parameters.grid_shape
        low = parameters.grid_low_m
        high = low + np.asarray(shape) * size
        spheres = _spheres(robot_spheres_base)
        other = np.zeros((0, 4)) if other_arm_spheres is None else _spheres(other_arm_spheres)
        hand = np.asarray(start_hand_base, dtype=np.float64)
        if hand.shape != (4, 4) or not np.isfinite(hand).all():
            raise DepthWorldError("start hand pose is malformed")
        self.finger_gap_m = inputs.finger_gap_m
        self.holding = (inputs.gripper_directive > 0
                        and parameters.held_min_gap_m < self.finger_gap_m < parameters.held_max_gap_m)
        base_from_world = np.linalg.inv(pose_matrix(inputs.base_pose_world_xyzw))
        counts = {"valid": 0, "robot": 0, "held": 0, "world": 0, "world_in_grid": 0}
        reach = parameters.surface_thickness_m
        cameras = []
        for camera in inputs.cameras:
            points, origin = _back_project(camera, base_from_world, parameters)
            counts["valid"] += len(points)
            points = points[((points >= low - reach) & (points <= high + reach)).all(axis=1)]
            in_hand = _in_box(points, hand, parameters.hand_box_m)
            held = (in_hand & _in_box(points, hand, _held_box(parameters, self.finger_gap_m))
                    if self.holding else np.zeros(len(points), dtype=bool))
            by_sphere = _within_spheres(points, np.vstack([spheres, other]), parameters.robot_point_margin_m)
            cameras.append((points, origin, in_hand & ~by_sphere & ~held, (in_hand | by_sphere) & ~held, held))
        # The hand box and the jaw slab reach past the fingertips: what they hold
        # at or below the support under the start hand is that support, not the
        # robot and not the load.
        self._index_surface(np.concatenate([points[~robot & ~held] for points, _, _, robot, held in cameras]))
        support = self.support(hand)
        if support is not None:
            ceiling = support["z_m"] + 0.5 * size
            cameras = [(points, origin, box_only, robot & ~(box_only & (points[:, 2] <= ceiling)),
                        held & (points[:, 2] > ceiling))
                       for points, origin, box_only, robot, held in cameras]
        world_points, seed_points, surface_points, seed_raw = [], [], [], []
        steps = int(round(parameters.surface_thickness_m / size))
        for points, origin, _, robot, held in cameras:
            world = points[~robot & ~held]
            counts["robot"] += int(robot.sum())
            counts["held"] += int(held.sum())
            counts["world"] += len(world)
            counts["world_in_grid"] += int(((world >= low) & (world < high)).all(axis=1).sum())
            surface_points.append(world)
            seed_raw.append(points[held])
            for target, chosen in ((world_points, ~robot & ~held), (seed_points, held)):
                rays = points[chosen] - origin
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                target.extend(points[chosen] + rays * (step * size) for step in range(steps + 1))
        occupied = self._voxelize(world_points)
        seed = self._voxelize(seed_points)
        occupied |= seed
        self.point_counts = counts
        surface = np.concatenate(surface_points)
        self._index_surface(surface)
        held = self._held_component(occupied, seed, np.concatenate(seed_raw), hand)
        occupied &= ~held
        centres = low + (np.argwhere(held) + 0.5) * size
        self.held_voxels = int(held.sum())
        self.held_spheres_hand = fit_held_spheres((centres - hand[:3, 3]) @ hand[:3, :3], parameters)
        # The load moves with the hand: it is no surface a later goal stands on.
        index = np.floor((surface - low) / size).astype(np.int64)
        inside = ((index >= 0) & (index < np.asarray(shape))).all(axis=1)
        keep = np.ones(len(surface), dtype=bool)
        keep[inside] = ~held[tuple(index[inside].T)]
        self._index_surface(surface[keep])
        # The load's own seen points, unthickened: where it really ends at a goal.
        load = np.concatenate([surface[~keep], *seed_raw])
        self._load_points_hand = (load - hand[:3, 3]) @ hand[:3, :3]
        self.start_support = self.support(hand)
        margin = parameters.start_carve_margin_m
        box = [[low_ - margin, high_ + margin] for low_, high_ in parameters.hand_box_m]
        box[2][1] = parameters.carve_box_tip_m
        start = self._carve(occupied, spheres=np.vstack([spheres, self._held_base(hand)]),
                            sphere_margin=margin, frame=hand, box=box,
                            protect=self.start_support)
        self.carved_start = start
        self.other_arm = self._sphere_voxels(other)
        occupied |= self.other_arm
        self.occupied = occupied
        self.esdf = signed_distance(occupied, parameters)

    def _sphere_voxels(self, spheres: Any) -> Any:
        """The voxels whose centre lies inside any of ``spheres``."""
        parameters = self.parameters
        grid = np.zeros(parameters.grid_shape, dtype=bool)
        size, low = parameters.voxel_size_m, parameters.grid_low_m
        for x, y, z, radius in np.asarray(spheres, dtype=np.float64).reshape(-1, 4):
            first = np.clip(np.floor((np.array([x, y, z]) - radius - low) / size).astype(int), 0, grid.shape)
            last = np.clip(np.ceil((np.array([x, y, z]) + radius - low) / size).astype(int) + 1, 0, grid.shape)
            axes = [low[axis] + (np.arange(first[axis], last[axis]) + 0.5) * size for axis in range(3)]
            inside = ((axes[0][:, None, None] - x) ** 2 + (axes[1][None, :, None] - y) ** 2
                      + (axes[2][None, None, :] - z) ** 2 <= radius ** 2)
            grid[first[0]:last[0], first[1]:last[1], first[2]:last[2]] |= inside
        return grid

    def _voxelize(self, chunks: list[Any]) -> Any:
        parameters = self.parameters
        shape = parameters.grid_shape
        grid = np.zeros(shape, dtype=bool)
        if chunks:
            index = np.floor((np.concatenate(chunks) - parameters.grid_low_m)
                             / parameters.voxel_size_m).astype(np.int64)
            grid[tuple(index[((index >= 0) & (index < np.asarray(shape))).all(axis=1)].T)] = True
        return grid

    def _held_component(self, occupied: Any, seed: Any, seed_points: Any, hand: Any) -> Any:
        """The occupied voxels connected to the seed, within reach, off every support below it."""
        parameters = self.parameters
        size, low = parameters.voxel_size_m, parameters.grid_low_m
        held = np.zeros_like(occupied)
        self.held_component = None
        if not self.holding or not seed.any() or not len(seed_points):
            return held
        reach = parameters.held_component_reach_m
        box = np.array([parameters.held_box_x_m, (-0.5 * self.finger_gap_m, 0.5 * self.finger_gap_m),
                        parameters.held_box_z_m]) + np.array([-reach, reach])
        corners = np.array([[x, y, z] for x in box[0] for y in box[1] for z in box[2]])
        corners = corners @ hand[:3, :3].T + hand[:3, 3]
        first = np.clip(np.floor((corners.min(axis=0) - low) / size).astype(int), 0, None)
        last = np.minimum(np.floor((corners.max(axis=0) - low) / size).astype(int) + 1, occupied.shape)
        crop = tuple(slice(a, b) for a, b in zip(first, last))
        index = np.indices(tuple(last - first)).reshape(3, -1).T + first
        allowed = _in_box(low + (index + 0.5) * size, hand, box).reshape(tuple(last - first))
        allowed &= occupied[crop]
        # Never through the slab of a support at or below the seen seed: the load stands on it.
        bottom = int(np.floor((seed_points[:, 2].min() - low[2]) / size))
        steps = int(round(parameters.support_slab_m / size))
        below = self.supports(hand, below_index=bottom + 1)
        for support in below:
            allowed[:, :, max(0, support["top_index"] - steps - first[2]):
                    max(0, support["top_index"] + 1 - first[2])] = False
        grown = seed[crop] & allowed
        while True:
            step = _dilate(grown) & allowed
            if np.array_equal(step, grown):
                break
            grown = step
        # A component that reaches the bound is joined to the scene (a container,
        # a shelf, an object it leans on): taking it would delete that structure
        # from the world. Then only the seed is held and the rest stays world.
        inner = box + np.array([1.5 * size, -1.5 * size])
        voxels = np.argwhere(grown) + first
        centres = low + (voxels + 0.5) * size
        self.held_component = "whole"
        wide = bool(below) and stands_past_jaws(voxels, centres, hand, self.finger_gap_m, below[-1], parameters)
        if wide or not _in_box(centres, hand, inner).all():
            grown = seed[crop] & allowed
            self.held_component = "seed_only"
        held[crop] = grown
        return held

    def _index_surface(self, points: Any) -> None:
        """The unthickened world points in the grid and their voxel indices (support heights)."""
        parameters = self.parameters
        index = np.floor((points - parameters.grid_low_m) / parameters.voxel_size_m).astype(np.int64)
        inside = ((index >= 0) & (index < np.asarray(parameters.grid_shape))).all(axis=1)
        self._surface_z = points[inside, 2]
        self._surface_index = index = index[inside]
        # Column tops from the seen surface only: a thickened ray drifts into
        # neighbouring columns below the surface and would fake a lower level.
        tops = np.full(parameters.grid_shape[:2], -1, dtype=np.int64)
        np.maximum.at(tops, (index[:, 0], index[:, 1]), index[:, 2])
        self._tops = tops

    def fingertips(self, hand: Any) -> Any:
        """CuRobo's two open fingertips and the jaw centre between them, in the base frame."""
        parameters = self.parameters
        local = np.array([[0.0, side * parameters.finger_span_m, parameters.fingertip_m]
                          for side in (-1.0, 0.0, 1.0)])
        return local @ hand[:3, :3].T + hand[:3, 3]

    def support(self, hand: Any) -> dict[str, Any] | None:
        """The dominant horizontal surface under ``hand``'s fingertips, or ``None``.

        The lowest qualifying level. Its ``z_m`` is the median height of the
        depth points in its two levels.
        """
        found = self.supports(hand)
        return found[0] if found else None

    def supports(self, hand: Any, *, below_index: int | None = None) -> list[dict[str, Any]]:
        """Every qualifying level under ``hand``, lowest first (top voxel below ``below_index``)."""
        parameters = self.parameters
        size, low = parameters.voxel_size_m, parameters.grid_low_m
        tips = self.fingertips(hand)
        centre = tips[:, :2].mean(axis=0)
        tops = self._tops
        xs = low[0] + (np.arange(tops.shape[0]) + 0.5) * size
        ys = low[1] + (np.arange(tops.shape[1]) + 0.5) * size
        local = ((xs[:, None] - centre[0]) ** 2 + (ys[None, :] - centre[1]) ** 2
                 <= parameters.support_window_m ** 2)
        index = self._surface_index
        for name, window in (("local", local), ("grid", np.ones_like(local))):
            counts = np.bincount(tops[window & (tops >= 0)], minlength=tops.max(initial=0) + 2)
            pairs = counts[:-1] + counts[1:]
            levels = np.nonzero(pairs >= parameters.support_min_columns)[0]
            tops_of = np.where(counts[levels + 1] > 0, levels + 1, levels)
            if below_index is not None:
                levels, tops_of = levels[tops_of < below_index], tops_of[tops_of < below_index]
            if len(levels):
                break
        found = []
        for level, top in zip(levels.tolist(), tops_of.tolist()):
            mine = window[index[:, 0], index[:, 1]] & ((index[:, 2] == level) | (index[:, 2] == level + 1))
            z = (float(np.median(self._surface_z[mine])) if mine.any()
                 else float(low[2] + (top + 1) * size))
            found.append({"z_m": z, "level_index": level, "top_index": top, "columns": int(pairs[level]),
                          "window": name})
        return found

    def _protected(self, support: dict[str, Any] | None) -> Any:
        """Grid z indices of the support slab a carve box leaves in place."""
        if support is None:
            return np.zeros(0, dtype=np.int64)
        parameters = self.parameters
        size = parameters.voxel_size_m
        steps = int(round(parameters.support_slab_m / size))
        candidates = np.arange(max(0, support["top_index"] - steps), support["top_index"] + 1)
        centres = parameters.grid_low_m[2] + (candidates + 0.5) * size
        return candidates[centres < support["z_m"]]

    def _goal_refusal(self, goal: Any, support: dict[str, Any] | None) -> dict[str, Any] | None:
        """Why a goal is refused before any solver runs, or ``None``."""
        if support is None:
            return None
        parameters, tops = self.parameters, self._tops
        load = self._load_points_hand if len(self.held_spheres_hand) else np.zeros((0, 3))
        return (fingertip_refusal(self.fingertips(goal)[[0, 2]], support, tops, parameters)
                or load_refusal(load @ goal[:3, :3].T + goal[:3, 3], support, tops, parameters))

    def _held_base(self, hand: Any) -> Any:
        held = self.held_spheres_hand
        return np.hstack([held[:, :3] @ hand[:3, :3].T + hand[:3, 3], held[:, 3:]])

    def _carve(self, occupied: Any, *, spheres: Any, sphere_margin: float, frame: Any, box: Any,
               protect: dict[str, Any] | None, goal: bool = False) -> int:
        """Clear the spheres (plus margin) and the box, sparing ``protect``'s slab; a start's spheres clear it too."""
        index = np.argwhere(occupied)
        centres = self.parameters.grid_low_m + (index + 0.5) * self.parameters.voxel_size_m
        spare = np.isin(index[:, 2], self._protected(protect))
        clear = (_within_spheres(centres, _spheres(spheres), sphere_margin) & ~(spare & goal)
                 | _in_box(centres, frame, box) & ~spare)
        occupied[tuple(index[clear].T)] = False
        return int(clear.sum())

    def world(self, goal_hand_base: Any, goal_spheres_base: Any = None) -> DepthWorld:
        """The world for one goal: the scene with the goal carve and its own hash."""
        parameters = self.parameters
        goal = np.asarray(goal_hand_base, dtype=np.float64)
        if goal.shape != (4, 4) or not np.isfinite(goal).all():
            raise DepthWorldError("goal hand pose is malformed")
        approach = goal[:3, 2]
        standoff = parameters.goal_standoff_m
        steps = int(math.ceil(standoff / parameters.voxel_size_m))
        swept = []
        for step in range(steps + 1):
            frame = goal.copy()
            frame[:3, 3] -= approach * (standoff * step / steps)
            swept.append(self._held_base(frame))
        spheres = np.vstack([*swept, np.zeros((0, 4)) if goal_spheres_base is None
                             else _spheres(goal_spheres_base)])
        box = [list(parameters.hand_box_m[0]), list(parameters.hand_box_m[1]),
               [parameters.hand_box_m[2][0] - standoff, parameters.carve_box_tip_m]]
        support = self.support(goal)
        refusal = self._goal_refusal(goal, support)
        occupied = self.occupied.copy()
        # The box holds its own clearance; spheres take the start margin.
        carved = self._carve(occupied, spheres=spheres, sphere_margin=parameters.start_carve_margin_m,
                             frame=goal, box=box, protect=support, goal=True)
        occupied |= self.other_arm
        esdf = self.esdf if not carved else signed_distance(occupied, parameters)
        held = self.held_spheres_hand
        if support is not None and len(held) and len(self._load_points_hand):
            lowest = float((self._load_points_hand @ goal[2, :3]).min() + goal[2, 3])
            resting = lowest >= support["z_m"] - parameters.support_tolerance_m
            held = clip_to_floor(held, goal, support["z_m"] + parameters.voxel_size_m if resting else lowest)
        digest = hashlib.sha256(WORLD_HASH_DOMAIN)
        digest.update(json.dumps({
            "parameters": parameters.as_dict(), "inputs_sha256": self.inputs_sha256,
            "grid_shape": list(parameters.grid_shape),
        }, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.packbits(occupied).tobytes())
        digest.update(np.ascontiguousarray(held, dtype=np.float64).tobytes())
        summary = {
            "world_model": parameters.world_model,
            "inputs_sha256": self.inputs_sha256,
            "sha256": digest.hexdigest(),
            "voxel_size_m": parameters.voxel_size_m, "grid_shape": list(parameters.grid_shape),
            "points": dict(self.point_counts),
            "occupied_voxels": int(occupied.sum()),
            "carved_voxels": {"start": self.carved_start, "goal": carved},
            "support": {"start": self.start_support, "goal": support},
            "goal_refusal": refusal,
            "holding": bool(self.holding), "finger_gap_m": self.finger_gap_m,
            "held_voxels": self.held_voxels, "held_component": self.held_component,
            "held_spheres": len(held),
            **({"other_arm_voxels": int(self.other_arm.sum())} if self.other_arm.any() else {}),
        }
        return DepthWorld(esdf=esdf, occupied=occupied, held_spheres_hand=held.copy(),
                          sha256=summary["sha256"], summary=summary)


def curobo_v2_grid(esdf: Any, parameters: WorldParameters = PUBLIC_DEPTH_VOXEL_V1) -> dict[str, Any]:
    """``esdf`` in cuRobo V2's voxel layout, over exactly the volume the v0.7.8 kernel collides with.

    V2 reads distances negative inside, as float16, from ``round(dims / voxel)`` voxels per axis centred
    at ``(i + 1/2) voxel - dims / 2``, with no border (a query outside reads free): V2 voxel ``i`` is
    voxel ``i + border`` here, at the same centre, so both kernels see the same occupancy.
    """
    border, size = CUROBO_UNOBSERVED_BORDER_VOXELS, parameters.voxel_size_m
    inner = np.asarray(esdf, dtype=np.float32)[border:-border, border:-border, border:-border]
    dims = [count * size for count in inner.shape]
    if [int(np.float32(dim) / np.float32(size)) for dim in dims] != list(inner.shape):  # the kernel truncates
        raise DepthWorldError("the V2 grid dimensions are not whole voxels in float32")
    layout = {"dims_m": dims, "voxel_size_m": size,
              "center_base_m": ((parameters.effective_low_m + parameters.effective_high_m) / 2).tolist()}
    features = np.ascontiguousarray(-inner, dtype=np.float16)
    digest = hashlib.sha256(b"ora.curobo-v2-voxel-grid.v1\0" + json.dumps(layout, sort_keys=True).encode())
    digest.update(features.tobytes())
    return {**layout, "features": features, "sha256": digest.hexdigest()}


def build_world(
    inputs: WorldInputs, *, robot_spheres_base: Any, start_hand_base: Any, goal_hand_base: Any,
    goal_spheres_base: Any = None, parameters: WorldParameters = PUBLIC_DEPTH_VOXEL_V1,
) -> DepthWorld:
    """One scene, one goal."""
    return Scene(inputs, robot_spheres_base=robot_spheres_base, start_hand_base=start_hand_base,
                 parameters=parameters).world(goal_hand_base, goal_spheres_base)


__all__ = [
    "CAMERAS", "CUROBO_UNOBSERVED_BORDER_VOXELS", "DepthCamera", "DepthWorld", "DepthWorldError", "INPUT_FIELDS",
    "PUBLIC_DEPTH_VOXEL_ALOHA_V1", "PUBLIC_DEPTH_VOXEL_ROBOTIQ_V1", "PUBLIC_DEPTH_VOXEL_V1", "PUBLIC_DEPTH_VOXEL_X5_V1", "PUBLIC_DEPTH_VOXEL_YAM_V1", "ROBOT_WORLDS", "Scene", "WORLD_INPUT_SCHEMA", "WORLD_INPUT_SCHEMA_V2", "WORLD_MODEL",
    "WORLD_PARAMETERS", "WorldInputs", "WorldParameters", "build_world", "curobo_v2_grid", "other_arm_spheres_base",
    "pose_matrix", "read_world_inputs", "signed_distance", "tool_from_hand", "world_parameters", "write_world_inputs",
]
