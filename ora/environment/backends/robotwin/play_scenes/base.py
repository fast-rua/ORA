"""Worker-only base of ORA's RoboTwin free-play scenes (imports RoboTwin's ``envs``).

A play scene is a RoboTwin ``Base_Task`` without a task. ``setup_demo`` runs
RoboTwin's own initialization (table, wall, lights, the ALOHA embodiment, its
cameras and RoboTwin's stability check), ``load_actors`` places the scene's
objects through ``build_scene``, ``check_success`` is always false and there is
no expert (``play_once`` refuses). A scene places objects only with these
helpers, so every object is recorded in ``play_objects`` with its asset and
files for the play-scene audit:

- ``add_model``: a model from RoboTwin's asset library (``assets/objects``),
  through RoboTwin's ``create_actor`` at the model's calibrated scale, or at an
  explicit scale for a library model whose calibration is missing or unsuited
  to a table setting (the same builder calls ``create_actor`` makes);
- ``add_box``: RoboTwin's ``create_box`` primitive, as the official block tasks
  build their blocks.

Each object is placed by its measured bounding box: centred on the requested
table position and dropped from 2 mm above the table top, so a model's mesh
origin never decides where it rests. The seed (``np.random``, seeded by
``Base_Task._init_task_env_``) decides each scene's jitter.

A play scene also has ORA's operator camera (``OPERATOR_CAMERA``): 1280 x 720,
from the operator's side of the table at three quarters, both arms and the
objects in frame (``OPERATOR_VIEW``). It is added to the SAPIEN scene after the
objects (their entity ids stay as before) and never to RoboTwin's camera
manager, so no observation, point cloud or policy camera includes it; only the
operator's view (``/view``) and the execution capture read it
(``operator_rgb``, ``capture_images``). It ray-traces like RoboTwin's cameras,
at 8 samples per pixel (the policy cameras keep RoboTwin's 32): measured on
H200 at the pinned SAPIEN, 0.105 s per 1280 x 720 frame against 0.33 s at 32.
``capture_images`` renders only the cameras the capture records (RoboTwin's
``update_picture`` renders both wrist cameras too), and ``object_poses`` gives
the dynamic objects' poses the post-action settle compares.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sapien

from envs._base_task import Base_Task  # type: ignore[import-not-found]
from envs.utils import Actor, create_actor, create_box  # type: ignore[import-not-found]
from envs.utils.create_actor import get_glb_or_obj_file  # type: ignore[import-not-found]

from ora.environment.benchmarks.robotwin_play_scenes import TABLE_TOP_Z_M

#: wxyz rotations that stand a library model on the table. RoboTwin's meshes are y-up.
Y_UP = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
#: A scene ends at this many semantic actions (RoboTwin's default for a task without a step limit).
STEP_LIMIT = 1000
#: Gap between a dynamic object's lowest point and the table top when it is placed.
DROP_CLEARANCE_M = 0.002
#: ORA's operator camera, the operator's own view of a play scene; never a policy camera.
OPERATOR_CAMERA = "operator_camera"
#: World metres (the table centred at the origin, ``y`` away from the robot, its top at z 0.74): from the
#: operator's side of the table, up and to the robot's right, looking at the table just in front of the arms.
#: Chosen from rendered candidates on table_setting_sequence (2026-10-21): both arms, the near row and the
#: far row (the coaster included) in frame at 44 degrees of vertical field.
OPERATOR_VIEW = {"eye": (0.62, 0.62, 1.42), "target": (0.0, -0.18, 0.84), "up": (0.0, 0.0, 1.0),
                 "fovy_deg": 44.0, "width": 1280, "height": 720, "near_m": 0.1, "far_m": 100.0,
                 "samples_per_pixel": 8}
#: The cameras a play scene's execution capture records: the robot's two fixed cameras, as before, and the
#: operator camera (the operator's page and a video play the motion through it).
CAPTURE_CAMERAS = ("head_camera", "front_camera", OPERATOR_CAMERA)


def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """The wxyz product ``a * b`` (``b`` first, then ``a``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def jitter(limit: float) -> float:
    """A seeded uniform offset in ``[-limit, limit]``."""
    return float(np.random.uniform(-limit, limit))


@dataclass
class PlayObject:
    """One placed object: its name, RoboTwin actor, asset, body type and asset files."""

    name: str
    actor: Any
    asset: str
    dynamic: bool
    files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def entity(self) -> Any:
        return self.actor.actor


def _model_files(model: str, model_id: int) -> tuple[Path, Path, Path]:
    """The collision mesh, visual mesh and model data ``create_actor`` reads (relative to the RoboTwin root)."""
    directory = Path("assets/objects") / model
    collision = get_glb_or_obj_file(directory / "collision", model_id) if (directory / "collision").exists() else None
    if collision is None or not collision.exists():
        collision = get_glb_or_obj_file(directory, model_id)
    visual = get_glb_or_obj_file(directory / "visual", model_id) if (directory / "visual").exists() else None
    if visual is None or not visual.exists():
        visual = get_glb_or_obj_file(directory, model_id)
    return Path(collision), Path(visual), directory / f"model_data{model_id}.json"


def rigid_component(entity: Any) -> Any:
    """The entity's PhysX rigid body (dynamic or static)."""
    for component in entity.get_components():
        if isinstance(component, sapien.physx.PhysxRigidBaseComponent):
            return component
    raise RuntimeError(f"play scene object {entity.get_name()} has no rigid body")


def look_at_pose(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> Any:
    """A SAPIEN camera pose at ``eye`` looking at ``target`` (camera x forward, y left, z up, as RoboTwin's)."""
    eye_m, target_m, up_m = (np.asarray(value, dtype=float) for value in (eye, target, up))
    forward = target_m - eye_m
    forward /= np.linalg.norm(forward)
    left = np.cross(up_m, forward)
    left /= np.linalg.norm(left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    matrix[:3, 3] = eye_m
    return sapien.Pose(matrix)


def camera_rgb(camera: Any) -> np.ndarray:
    """A camera's last picture as uint8 RGB, as RoboTwin's ``get_rgb`` converts it."""
    return (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)


class PlaySceneTask(Base_Task):
    """A RoboTwin task with no task: objects on the table, nothing to succeed at."""

    def setup_demo(self, **kwargs: Any) -> None:
        super()._init_task_env_(**kwargs)
        self.step_lim = STEP_LIMIT

    def load_actors(self) -> None:
        self.play_objects: list[PlayObject] = []
        self.build_scene()
        # After the objects: their SAPIEN entity ids stay what they were before the camera existed.
        self.operator_camera = self._add_operator_camera()

    def build_scene(self) -> None:
        raise NotImplementedError

    def play_once(self) -> None:
        raise RuntimeError("a play scene has no expert")

    def check_success(self) -> bool:
        return False

    @property
    def table_top_z(self) -> float:
        return TABLE_TOP_Z_M + float(self.table_z_bias)

    # --- the operator's camera and what the execution capture reads -----------------------------

    #: The camera ``operator_rgb`` renders, as the operator view names it.
    operator_camera_name = OPERATOR_CAMERA
    #: The cameras ``capture_images`` renders for a play scene's execution capture.
    capture_cameras = CAPTURE_CAMERAS

    def _add_operator_camera(self) -> Any:
        """ORA's operator camera in the SAPIEN scene; RoboTwin's camera manager never lists it."""
        view = OPERATOR_VIEW
        camera = self.scene.add_camera(name=OPERATOR_CAMERA, width=int(view["width"]), height=int(view["height"]),
                                       fovy=math.radians(float(view["fovy_deg"])), near=float(view["near_m"]),
                                       far=float(view["far_m"]))
        # Per camera: RoboTwin's global ray-tracing setting (32 samples) stays the policy cameras'.
        camera.set_property("spp", int(view["samples_per_pixel"]))
        camera.entity.set_pose(look_at_pose(view["eye"], view["target"], view["up"]))
        return camera

    def operator_rgb(self) -> np.ndarray:
        """The operator camera now (the scene's render transforms synchronized first; no light changes)."""
        self.scene.update_render()
        self.operator_camera.take_picture()
        return camera_rgb(self.operator_camera)

    def capture_images(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        """Render only ``names`` (RoboTwin's fixed cameras by name, or the operator camera) and read them.

        Unlike ``Base_Task._update_render`` nothing here randomizes lights or consumes RNG; unlike the
        camera manager's ``update_picture`` the wrist cameras are not rendered.
        """
        fixed = dict(zip(self.cameras.static_camera_name, self.cameras.static_camera_list))
        cameras = {}
        for name in names:
            camera = self.operator_camera if name == OPERATOR_CAMERA else fixed.get(name)
            if camera is None:
                raise RuntimeError(f"play scene has no capture camera {name!r}")
            cameras[name] = camera
        self.scene.update_render()
        for camera in cameras.values():
            camera.take_picture()
        return {name: camera_rgb(camera) for name, camera in cameras.items()}

    def object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each dynamic object's world position (m) and wxyz orientation."""
        poses = {}
        for item in self.play_objects:
            if item.dynamic:
                pose = item.entity.get_pose()
                poses[item.name] = (np.asarray(pose.p, dtype=float), np.asarray(pose.q, dtype=float))
        return poses

    # --- placing objects ------------------------------------------------------------------------

    def add_model(self, name: str, model: str, model_id: int, xy: Sequence[float], *, yaw: float = 0.0,
                  upright: Sequence[float] = Y_UP, scale: float | None = None, static: bool = False) -> PlayObject:
        """A library model resting on the table at ``xy``, turned by ``yaw`` about the vertical."""
        orientation = quaternion_multiply(yaw_quaternion(yaw), upright)
        pose = sapien.Pose([float(xy[0]), float(xy[1]), TABLE_TOP_Z_M + 0.3], list(orientation))
        collision, visual, data = _model_files(model, model_id)
        if scale is None:
            actor = create_actor(self, pose=pose, modelname=model, model_id=model_id, convex=True,
                                 is_static=static)
        else:
            actor = self._scaled_model(pose, model, collision, visual, data, scale, static)
        if actor is None:
            raise RuntimeError(f"RoboTwin asset {model}/base{model_id} is missing")
        actor.set_name(name)
        self._rest_on_table(actor.actor, xy, clearance=0.0 if static else DROP_CLEARANCE_M)
        files = tuple(sorted({path.as_posix() for path in (collision, visual, data) if path.exists()}))
        return self._register(PlayObject(name, actor, f"{model}/base{model_id}", not static, files))

    def add_box(self, name: str, xy: Sequence[float], half_size: Sequence[float], color: Sequence[float], *,
                yaw: float = 0.0, static: bool = False) -> PlayObject:
        """RoboTwin's box primitive resting on the table at ``xy``."""
        clearance = 0.0 if static else DROP_CLEARANCE_M
        # create_box adds the table's height bias itself.
        pose = sapien.Pose([float(xy[0]), float(xy[1]), TABLE_TOP_Z_M + float(half_size[2]) + clearance],
                           list(yaw_quaternion(yaw)))
        actor = create_box(self, pose=pose, half_size=[float(value) for value in half_size],
                           color=tuple(color), is_static=static, name=name)
        return self._register(PlayObject(name, actor, "create_box", not static))

    def _scaled_model(self, pose: Any, model: str, collision: Path, visual: Path, data: Path, scale: float,
                      static: bool) -> Any:
        """``create_actor``'s builder calls at an explicit scale."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static" if static else "dynamic")
        scales = [float(scale)] * 3
        builder.add_multiple_convex_collisions_from_file(filename=str(collision), scale=scales)
        builder.add_visual_from_file(filename=str(visual), scale=scales)
        entity = builder.build(name=model)
        entity.set_pose(pose)
        model_data = json.loads(data.read_text(encoding="utf-8")) if data.exists() else {}
        return Actor(entity, {**model_data, "scale": scales})

    def _rest_on_table(self, entity: Any, xy: Sequence[float], *, clearance: float) -> None:
        low, high = (np.asarray(value, dtype=float) for value in rigid_component(entity).compute_global_aabb_tight())
        pose = entity.get_pose()
        shift = np.array([float(xy[0]) - (low[0] + high[0]) / 2, float(xy[1]) - (low[1] + high[1]) / 2,
                          self.table_top_z + clearance - low[2]])
        entity.set_pose(sapien.Pose(np.asarray(pose.p, dtype=float) + shift, pose.q))

    def _register(self, item: PlayObject) -> PlayObject:
        if any(other.name == item.name for other in self.play_objects):
            raise RuntimeError(f"play scene object name {item.name} is used twice")
        self.play_objects.append(item)
        return item


__all__ = ["CAPTURE_CAMERAS", "DROP_CLEARANCE_M", "IDENTITY", "OPERATOR_CAMERA", "OPERATOR_VIEW", "PlayObject",
           "PlaySceneTask", "STEP_LIMIT", "Y_UP", "camera_rgb", "jitter", "look_at_pose", "quaternion_multiply",
           "rigid_component", "yaw_quaternion"]
