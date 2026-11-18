"""A RoboLab scene's trace (``ora/environment/scene_trace.py``) over a replay program's re-execution.

The Isaac part is RoboDojo's ``isaac_scene.py``, which names no benchmark; this says where RoboLab keeps its things.
The bodies are the rigid prims under the env root: the Panda's and the Robotiq's links (``robot``), and the task scene's
objects, its table among them, which RoboLab's scenes build of cubes and cylinders, so those are tessellated here. What
has no rigid body and is drawn (the robot's table fixture, a scene's fixed parts) is ``static``. Every pose is relative to
the robot's root, the frame of the receipts and the cameras' calibration, and the manifest's ``tcp`` is the receipts' TCP
on the Robotiq's flange.

A row is one native action, 8 PhysX steps at 120 Hz: the replay reads the scene between native actions only. RoboLab has
no execution capture, so the clock is the receipts': ``physx_step_since_round_start`` starts at the state the policy
first sees, after the reset's two held native actions, and ``physics_step / 8`` is the receipts' native actions
accumulated in action order (``robolab_replay.receipt_clock_failures`` checks every action).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.environment.backends.robodojo.isaac_scene import (
    GREY, RigidPoses, body_mesh, material_look, part_mesh, pose_matrix, scene_bodies,
)
from ora.environment.benchmarks.robolab import DECIMATION, PHYSICS_DT
from ora.environment.controllers import robolab_droid as D
from ora.environment.scene_trace import Body, Mesh, SceneTraceError, SceneTraceWriter, merge

from .common import array, quat_xyzw_from_matrix

CLOCK_UNIT = "physx_step_since_round_start"
#: One row per native action.
STRIDE = DECIMATION
CYLINDER_SIDES = 32
#: The robot's prim and the task scene's, under the env root.
ROBOT, SCENE = "robot", "scene"


def cube_faces(size: float) -> tuple[Any, Any, Any]:
    """A ``UsdGeom.Cube``: points, polygon counts and indices, as a mesh's arrays."""
    half = size / 2.0
    points = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    quads = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    return points, np.full(6, 4), np.array(quads)


def cylinder_faces(radius: float, height: float, axis: str = "Z", sides: int = CYLINDER_SIDES) -> tuple[Any, Any, Any]:
    """A ``UsdGeom.Cylinder`` as ``sides`` quads and two capping polygons: points, counts and indices."""
    angle = np.arange(sides) * 2.0 * np.pi / sides
    ring = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    points = np.r_[np.c_[ring, np.full(sides, -height / 2.0)], np.c_[ring, np.full(sides, height / 2.0)]]
    following = (np.arange(sides) + 1) % sides
    walls = np.c_[np.arange(sides), following, following + sides, np.arange(sides) + sides].ravel()
    indices = np.r_[walls, np.arange(sides)[::-1], np.arange(sides) + sides]
    return points[:, {"X": [2, 0, 1], "Y": [1, 2, 0], "Z": [0, 1, 2]}[axis]], np.r_[np.full(sides, 4), sides, sides], indices


def _drawn_shape(prim: Any) -> bool:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    return (prim.IsA(UsdGeom.Gprim) and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render))


def _shapes(prim: Any) -> tuple[Any, Any, Any] | None:
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Cube):
        return cube_faces(float(UsdGeom.Cube(prim).GetSizeAttr().Get()))
    if prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        return cylinder_faces(float(cylinder.GetRadiusAttr().Get()), float(cylinder.GetHeightAttr().Get()),
                              str(cylinder.GetAxisAttr().Get()))
    return None


def scene_mesh(body: Any, cache: Any) -> Mesh | None:
    """``body_mesh`` and the cubes and cylinders under the body's prim, one colour per shape."""
    from pxr import Usd

    world_to_body = np.linalg.inv(pose_matrix(cache.GetLocalToWorldTransform(body)))
    parts = [body_mesh(body, cache)]
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        shape = _shapes(prim) if _drawn_shape(prim) else None
        if shape is not None:
            image, color = material_look(prim)
            to_body = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ world_to_body
            parts.append(part_mesh(*shape, to_body, image=image, rgb=color or GREY))
    return merge([part for part in parts if part is not None])


def static_prims(stage: Any, root: str) -> list[str]:
    """The env root's and the task scene's children drawn with no rigid body under them."""
    from pxr import Usd, UsdPhysics

    def rigid(prim: Any) -> bool:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False

    children = [prim for prim in stage.GetPrimAtPath(root).GetChildren() if prim.GetName() not in (ROBOT, SCENE)]
    children += list(stage.GetPrimAtPath(f"{root}/{SCENE}").GetChildren())
    found = []
    for prim in children:
        below = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
        if any(_drawn_shape(item) for item in below) and not any(rigid(item) for item in below):
            found.append(str(prim.GetPath()))
    return found


def bodies_of(session: Any) -> tuple[list[Body], RigidPoses]:
    """The round's bodies and their poses, from the session's live stage, in the robot root's frame."""
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()
    bridge = session.bridge
    root = bridge.env.scene.env_prim_paths[0]
    data = bridge.robot.data
    if not np.allclose(array(data.root_quat_w[0]), (1.0, 0.0, 0.0, 0.0), atol=1e-6):
        raise SceneTraceError("a pose carries no frame rotation: RoboLab's robot root must be unrotated")
    return scene_bodies(stage, root, SimulationManager.get_physics_sim_view(), robots=[f"{root}/{ROBOT}"],
                        statics=static_prims(stage, root), origin=data.root_pos_w[0], mesh_of=scene_mesh)


def tcp_link(bodies: Any) -> dict[str, Any]:
    """The receipts' TCP as a body and an offset: the flange's name in the trace and the TCP in the flange's frame."""
    flange = next(body.key for body in bodies if body.key.rsplit("/", 1)[-1] == D.FLANGE)
    rotation = np.array(D.TCP_AXES_IN_FLANGE, dtype=float).T
    return {"right": {"body": flange,
                      "offset_xyz_xyzw": [*map(float, D.TCP_OFFSET_FROM_FLANGE_M), *quat_xyzw_from_matrix(rotation)]}}


class RobolabSceneTrace:
    """What the replay calls around each native action; reading the scene changes no state."""

    def __init__(self, directory: Path, *, run_id: str) -> None:
        self.writer = SceneTraceWriter(directory, run_id=run_id, backend="robolab", clock_unit=CLOCK_UNIT, stride=STRIDE)
        self.poses: RigidPoses | None = None

    def begin_round(self, session: Any, round_index: int) -> None:
        bodies, self.poses = bodies_of(session)
        self.writer.tcp = tcp_link(bodies)
        self.writer.begin_round(round_index, bodies, timestep_s=PHYSICS_DT)

    def begin_action(self, entry: Mapping[str, Any]) -> None:
        self.writer.begin_action(entry)

    def after_tick(self) -> None:
        """The native action's steps on the clock; the read happens at the row, after its last step."""
        self.poses.refresh()
        for _ in range(STRIDE):
            self.writer.after_step()

    def end_action(self) -> None:
        self.writer.end_action()

    def finish(self) -> dict[str, Any]:
        return self.writer.finish()


__all__ = ["CLOCK_UNIT", "STRIDE", "RobolabSceneTrace", "bodies_of", "cube_faces", "cylinder_faces", "scene_mesh",
           "static_prims", "tcp_link"]
