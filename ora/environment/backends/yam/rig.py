"""The device layer of the YAM worker: one ``Rig`` per mode.

A rig owns the joints, the grippers and the cameras. The motion layer only sends it
joint and finger targets one control tick at a time and reads back what was measured,
so the same planner, checks and receipts run on the simulated station and on the real
arms (``real_rig.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ora.environment.controllers import yam_station as Y

from .common import YamError
from .kinematics import StationKinematics, pose_matrix
from .scene import (
    CV_FROM_MUJOCO_CAMERA, DEFAULT_FIXED_CAMERAS, DEFAULT_OBJECTS, DEFAULT_TABLE, SIM_FOVY_DEG, SIM_IMAGE_SIZE,
    build_cell_model, camera_intrinsic,
)

# The pose every episode starts from: both TCPs 0.16 m above the base plane, 0.28 m out,
# approaching straight down. Pointing down a YAM reaches about 0.21 m there, so this
# leaves room for a free-space lift.
READY_TCP_XYZ_M = {"left": (0.28, 0.22, 0.16), "right": (0.28, -0.22, 0.16)}
READY_TCP_XYZW = (1.0, 0.0, 0.0, 0.0)       # 180 deg about x: approach (z) points down
# The station model's cameras: (public name, MJCF camera, mount).
SIM_CAMERAS = (("overhead", "ora_overhead", "fixed_world"), ("front", "ora_front", "fixed_world"),
               ("wrist_left", "ora_wrist_left", "end_effector"), ("wrist_right", "ora_wrist_right", "end_effector"))


@dataclass
class TickResult:
    """What one control tick measured. ``events`` are guard stops (real arms only)."""

    arm_q: dict[str, np.ndarray]
    finger_m: dict[str, float]
    events: list[dict[str, Any]] = field(default_factory=list)


class Rig(Protocol):
    mode: str
    kinematics: StationKinematics
    table_top_z_m: float | None

    def arm_q(self, arm: str) -> np.ndarray: ...
    def finger_m(self, arm: str) -> float: ...
    def step(self, arm_targets: Mapping[str, np.ndarray], finger_targets: Mapping[str, float]) -> TickResult: ...
    def cameras(self) -> dict[str, dict[str, Any]]: ...
    def operator_approval(self, summary: Mapping[str, Any]) -> tuple[bool, str | None]: ...
    def close(self) -> None: ...


def ready_configuration(kinematics: StationKinematics) -> dict[str, np.ndarray]:
    """Joint angles of the ready pose, solved once by IK from the zero pose."""

    seed = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    out = {}
    for arm in Y.ARMS:
        result = kinematics.ik(arm, pose_matrix(READY_TCP_XYZ_M[arm], READY_TCP_XYZW), seed, max_iterations=400)
        if not result.ok:
            raise YamError(f"ready pose of the {arm} arm is unreachable on the station model")
        out[arm] = result.q
    return out


class SimRig:
    """The station model as a kinematic cell: joints follow commands at the speed limit."""

    mode = "sim"
    FINGER_SPEED_M_S = 0.06

    def __init__(self, config: Mapping[str, Any]) -> None:
        import mujoco

        self.mujoco = mujoco
        table = dict(config.get("table") or DEFAULT_TABLE)
        self.objects = [dict(item) for item in (config.get("objects") or DEFAULT_OBJECTS)]
        self.model = build_cell_model(table=table, objects=self.objects,
                                      fixed_cameras=DEFAULT_FIXED_CAMERAS, wrist_cameras=True)
        self.kinematics = StationKinematics(self.model)
        self.table_top_z_m = float(table["top_z_m"])
        self.data = mujoco.MjData(self.model)
        self._q = ready_configuration(self.kinematics)
        self._finger = {arm: Y.FINGER_OPEN_M for arm in Y.ARMS}
        self._held: dict[str, tuple[str, np.ndarray] | None] = {arm: None for arm in Y.ARMS}
        self._object_pose = {item["name"]: pose_matrix(item["position_m"], (0, 0, 0, 1)) for item in self.objects}
        self._renderers: dict[tuple[int, int], Any] = {}
        self._sync()

    # ----------------------------------------------------------- state
    def arm_q(self, arm: str) -> np.ndarray:
        return self._q[arm].copy()

    def finger_m(self, arm: str) -> float:
        return float(self._finger[arm])

    def _sync(self) -> None:
        from scipy.spatial.transform import Rotation

        self.kinematics.set_state(self._q, self._finger)
        self.data.qpos[:] = self.kinematics.qpos
        for item in self.objects:
            body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"ora_object_{item['name']}")
            adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
            pose = self._object_pose[item["name"]]
            x, y, z, w = Rotation.from_matrix(pose[:3, :3]).as_quat()
            self.data.qpos[adr:adr + 7] = [*(pose[:3, 3] - np.asarray(Y.STATION_TO_WORLD_XYZ_M)), w, x, y, z]
        self.mujoco.mj_forward(self.model, self.data)

    # ----------------------------------------------------------- control
    def step(self, arm_targets: Mapping[str, np.ndarray], finger_targets: Mapping[str, float]) -> TickResult:
        dt = Y.CONTROL_PERIOD_S
        for arm, target in arm_targets.items():
            delta = np.asarray(target, float) - self._q[arm]
            limit = Y.MAX_JOINT_SPEED_RAD_S * dt
            self._q[arm] = self._q[arm] + np.clip(delta, -limit, limit)
        for arm, target in finger_targets.items():
            current = self._finger[arm]
            goal = float(np.clip(target, Y.FINGER_CLOSED_M, Y.FINGER_OPEN_M))
            stop = self._grasp_stop(arm) if goal < current else None
            if stop is not None:
                goal = max(goal, stop)
            step = self.FINGER_SPEED_M_S * dt
            self._finger[arm] = current + float(np.clip(goal - current, -step, step))
            if self._held[arm] is not None and self._finger[arm] > self._grasp_width(self._held[arm][0]) / 2 + 0.004:
                self._release(arm)
        self.kinematics.set_state(self._q, self._finger)
        for arm, held in self._held.items():
            if held is not None:
                self._object_pose[held[0]] = self.kinematics.tcp(arm) @ held[1]
        self._sync()
        return TickResult({arm: self._q[arm].copy() for arm in Y.ARMS},
                          {arm: float(self._finger[arm]) for arm in Y.ARMS})

    def _grasp_width(self, name: str) -> float:
        item = next(item for item in self.objects if item["name"] == name)
        return float(min(item["size_m"][:2]))

    def _grasp_stop(self, arm: str) -> float | None:
        """Finger joint at which closing jaws meet an object between the fingertips."""

        held = self._held[arm]
        if held is not None:
            return self._grasp_width(held[0]) / 2
        tcp = self.kinematics.tcp(arm)
        for item in self.objects:
            local = np.linalg.inv(tcp) @ np.append(self._object_pose[item["name"]][:3, 3], 1.0)
            half = np.asarray(item["size_m"], float) / 2
            if abs(local[0]) < self._finger[arm] and abs(local[1]) < 0.02 and abs(local[2]) < half[2] + 0.015:
                width = self._grasp_width(item["name"])
                if self._finger[arm] - 0.002 <= width / 2:
                    self._held[arm] = (item["name"], np.linalg.inv(tcp) @ self._object_pose[item["name"]])
                    return width / 2
        return None

    def _release(self, arm: str) -> None:
        name, _ = self._held[arm]
        self._held[arm] = None
        item = next(item for item in self.objects if item["name"] == name)
        pose = self._object_pose[name]
        pose[:3, :3] = np.eye(3)
        pose[2, 3] = self.table_top_z_m + float(item["size_m"][2]) / 2

    def operator_approval(self, summary: Mapping[str, Any]) -> tuple[bool, str | None]:
        return True, None

    def home(self) -> str:
        """The ready pose at once (a kinematic cell needs no drive); held objects drop in place."""

        for arm in Y.ARMS:
            if self._held[arm] is not None:
                self._release(arm)
        self._q = ready_configuration(self.kinematics)
        self._sync()
        return "at rest"

    # ----------------------------------------------------------- cameras
    def _scene_option(self):
        option = self.mujoco.MjvOption()
        option.sitegroup[:] = 0        # the model's TCP markers are not in the world
        return option

    def cameras(self) -> dict[str, dict[str, Any]]:
        width, height = SIM_IMAGE_SIZE
        option = self._scene_option()
        out = {}
        for public, name, mount in SIM_CAMERAS:
            cam = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, name)
            renderer = self._renderer(width, height)
            renderer.disable_depth_rendering()
            renderer.update_scene(self.data, camera=cam, scene_option=option)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=cam, scene_option=option)
            depth = renderer.render().astype(np.float32).copy()
            renderer.disable_depth_rendering()
            depth[(depth <= 0.0) | (depth > 3.0)] = 0.0
            cam2world = np.eye(4)
            cam2world[:3, :3] = self.data.cam_xmat[cam].reshape(3, 3) @ CV_FROM_MUJOCO_CAMERA
            cam2world[:3, 3] = self.data.cam_xpos[cam] + np.asarray(Y.STATION_TO_WORLD_XYZ_M)
            fovy = SIM_FOVY_DEG["wrist" if mount == "end_effector" else "fixed"]
            out[public] = {"rgb": rgb, "depth": depth, "intrinsic_cv": camera_intrinsic(fovy, width, height),
                           "cam2world_cv": cam2world, "mount": mount, "source_camera": f"sim:{name}",
                           "calibration_source": "station_model"}
        return out

    def camera_rgb(self) -> dict[str, np.ndarray]:
        """The four cameras' colour images alone, for the world view's frames (no depth, no calibration)."""

        width, height = SIM_IMAGE_SIZE
        option = self._scene_option()
        out = {}
        for public, name, _mount in SIM_CAMERAS:
            cam = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, name)
            renderer = self._renderer(width, height)
            renderer.disable_depth_rendering()
            renderer.update_scene(self.data, camera=cam, scene_option=option)
            out[public] = renderer.render().copy()
        return out

    def _renderer(self, width: int, height: int):
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = self.mujoco.Renderer(self.model, height, width)
        return self._renderers[key]

    def close(self) -> None:
        for renderer in self._renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._renderers.clear()


def build_rig(config: Mapping[str, Any]) -> Rig:
    mode = config.get("rig_mode", "sim")
    if mode == "sim":
        return SimRig(config)
    if mode in {"read", "on"}:
        from .real_rig import RealRig

        return RealRig(config)
    raise YamError(f"unknown YAM rig mode {mode!r}")
