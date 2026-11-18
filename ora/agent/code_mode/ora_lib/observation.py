"""The current published observation, as code mode mounts it read-only at ``/ora/observation``.

The host (``ora/agent/code_mode/library.py``) copies the public artifacts ``geometry_query`` reads, after
checking each one's hash: every camera's RGB image, depth (metres along the optical axis) and calibration,
and the measured robot state; and the mask of every segmented target recorded on this observation
(``vision_ground`` and ``vision_segment`` calls, the model's own included); and the public robot facts the
observation message shows, on a surface that delivers them. ``current.json``
(``ora.code-mode-observation.v1``) names them. After a world-changing call the host publishes the next
observation there before the call returns, so ``current()`` always gives the observation the robot's tools see
now.

    obs = current()
    camera = obs.camera("agentview")      # K, extrinsic, depth_m, rgb, width, height
    obs.end_effectors["right"]            # the measured public fields, as published
    obs.end_effectors["right"]["gripper"]["measured_opening_normalized"]  # the fields the observation message shows
    obs.robot_facts.get("mobile_base")    # x_m, y_m, yaw_deg, torso_height_m, on a mobile base
    obs.pose("right")                     # the measured pose of tcp:right in world (a frames.Pose)
    obs.target("target-...").mask         # the mask of a target vision_ground or vision_segment returned
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from . import geometry

MOUNT = "/ora/observation"
SCHEMA = "ora.code-mode-observation.v1"


class ObservationUnavailable(RuntimeError):
    """No published observation is mounted here, or the current one could not be read."""


class ListsAttributes:
    """An ``AttributeError`` that says which public attributes there are. ``__getattr__`` runs only for a name normal
    lookup did not find, so ``hasattr`` still answers False; a ``_`` name is what Python or NumPy looks for itself
    (pickling, ``__array_wrap__``), and gets the usual message."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        message = f"{type(self).__name__!r} object has no attribute {name!r}"
        if not name.startswith("_"):
            message += f"; it has: {', '.join(item for item in dir(self) if not item.startswith('_'))}"
        raise AttributeError(message, name=name, obj=self)


class Camera(ListsAttributes):
    """One published camera: OpenCV intrinsics ``K`` (pixels), ``extrinsic`` (world -> camera, 4x4),
    ``camera_to_world`` (its inverse), image size, and its depth (HxW float32 metres) and RGB (HxWx3 uint8),
    read when first used. ``mount`` is where it is fixed (``fixed_world``, or the arm it moves with)."""

    def __init__(self, root: str, name: str, record: dict[str, Any]) -> None:
        self.name = name
        self.width, self.height = int(record["width"]), int(record["height"])
        self.mount = record.get("mount")
        self._root, self._files = root, record["files"]
        with open(os.path.join(root, self._files["calibration"]), encoding="utf-8") as stream:
            self.calibration: dict[str, Any] = json.load(stream)
        self.K = np.asarray(self.calibration["intrinsic_cv"], dtype=np.float64)
        self.extrinsic = geometry.rigid(self.calibration["extrinsic_cv"])
        self.camera_to_world = geometry.invert(self.extrinsic)
        self._depth: np.ndarray | None = None
        self._rgb: np.ndarray | None = None

    @property
    def depth_m(self) -> np.ndarray:
        if self._depth is None:
            depth = np.load(os.path.join(self._root, self._files["depth_m"]), allow_pickle=False)
            depth.setflags(write=False)
            self._depth = depth
        return self._depth

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            rgb = np.load(os.path.join(self._root, self._files["rgb"]), allow_pickle=False)
            rgb.setflags(write=False)
            self._rgb = rgb
        return self._rgb

    def __repr__(self) -> str:
        return f"Camera({self.name!r}, {self.width}x{self.height}, mount={self.mount!r})"


class Target:
    """One segmented target of this observation (``vision_ground``, ``vision_segment``): the ``camera`` it was
    segmented on and its ``mask`` (HxW bool, the tool's, read when first used)."""

    def __init__(self, root: str, reference: str, record: dict[str, Any]) -> None:
        self.target_ref, self.camera = reference, str(record["camera"])
        self._root, self._file = root, record["mask"]
        self._mask: np.ndarray | None = None

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            mask = np.load(os.path.join(self._root, self._file), allow_pickle=False).astype(bool)
            mask.setflags(write=False)
            self._mask = mask
        return self._mask

    def __repr__(self) -> str:
        return f"Target({self.target_ref!r}, camera={self.camera!r})"


class Observation(ListsAttributes):
    """One published observation: ``observation_id``, ``step``, ``main_camera``, ``cameras`` (by name),
    ``camera_errors`` (cameras whose artifacts did not verify), ``targets`` (the segmented targets recorded on it,
    by ``target_ref``; ``target_errors`` those whose artifacts did not verify), ``robot_facts`` and ``end_effectors``.

    ``robot_facts`` is the public robot facts this observation's message shows, as it shows them: ``end_effectors``
    (each arm's pose, axes and ``gripper``), ``mobile_base`` on a mobile base (``x_m``, ``y_m``, ``yaw_deg``,
    ``torso_height_m``), ``cameras``; empty on a surface that delivers none.

    ``end_effectors`` is the measured state as published, with each arm's ``gripper`` the facts' where they state one
    (``robot_facts["end_effectors"][arm]["gripper"]``, in the backend's fields: ``measured_opening_normalized``,
    ``commanded_opening_normalized``, ``command_target``, ``measured_aperture_mm``). Whether a grasp holds is not one
    of them."""

    def __init__(self, root: str, index: dict[str, Any]) -> None:
        if index.get("schema") != SCHEMA:
            raise ObservationUnavailable(f"{root}: not an {SCHEMA} index")
        error = index.get("error")
        if error:
            raise ObservationUnavailable(f"no current observation: {error.get('message') or error}")
        self.root = root
        self.observation_id: str = index["observation_id"]
        self.step: int | None = index.get("step")
        self.main_camera: str | None = index.get("main_camera")
        self.camera_errors: dict[str, Any] = dict(index.get("camera_errors") or {})
        self.cameras = {name: Camera(root, name, record) for name, record in index["cameras"].items()}
        self.targets = {reference: Target(root, reference, record)
                        for reference, record in (index.get("targets") or {}).items()}
        self.target_errors: dict[str, Any] = dict(index.get("target_errors") or {})
        with open(os.path.join(root, index["proprioception"]), encoding="utf-8") as stream:
            proprioception = json.load(stream)
        self.robot_facts: dict[str, Any] = {}
        if isinstance(index.get("robot_facts"), str):
            with open(os.path.join(root, index["robot_facts"]), encoding="utf-8") as stream:
                self.robot_facts = json.load(stream)
        self.end_effectors: dict[str, dict[str, Any]] = dict(proprioception.get("end_effectors") or {})
        for arm, facts in (self.robot_facts.get("end_effectors") or {}).items():
            if isinstance(self.end_effectors.get(arm), dict) and "gripper" in facts:
                self.end_effectors[arm] = {**self.end_effectors[arm], "gripper": facts["gripper"]}

    def camera(self, name: str) -> Camera:
        """A camera by name (``"agentview"``; ``"camera:agentview"`` and ``"pixel:agentview"`` also name it)."""

        name = str(name).split(":", 1)[1] if str(name).startswith(("camera:", "pixel:")) else str(name)
        if name in self.cameras:
            return self.cameras[name]
        reason = f" ({self.camera_errors[name].get('message')})" if name in self.camera_errors else ""
        raise KeyError(f"no camera {name!r} in observation {self.observation_id}{reason}; "
                       f"cameras: {', '.join(sorted(self.cameras)) or 'none'}")

    def target(self, reference: str) -> Target:
        """A segmented target of this observation by its ``target_ref``, one segmented since this object was read
        included."""

        if reference in self.targets:
            return self.targets[reference]
        if reference not in self.target_errors:
            try:
                latest = load(self.root)
            except ObservationUnavailable:
                latest = self
            if latest.observation_id == self.observation_id and reference in latest.targets:
                return latest.targets[reference]
            if latest.observation_id == self.observation_id:
                self.target_errors = latest.target_errors
        if reference in self.target_errors:
            raise KeyError(f"the mask of {reference} did not verify: {self.target_errors[reference].get('message')}")
        raise KeyError(f"no mask for {reference} in observation {self.observation_id}; a target_ref is valid on the "
                       "observation vision_ground or vision_segment segmented it on, so after an action segment "
                       f"again; targets of this one: {', '.join(sorted(self.targets)) or 'none'}")

    def tcp_matrix(self, arm: str) -> np.ndarray:
        """The 4x4 ``world <- tcp:<arm>`` transform: the arm's measured end-effector pose."""

        record = self.end_effectors.get(arm)
        pose = record.get("pose_xyz_xyzw") if isinstance(record, dict) else None
        if not isinstance(pose, list) or len(pose) != 7:
            raise KeyError(f"no measured pose for arm {arm!r} in observation {self.observation_id}; "
                           f"arms with one: {', '.join(self.arms()) or 'none'}")
        return geometry.pose_matrix(pose[:3], pose[3:])

    def arms(self) -> list[str]:
        return sorted(name for name, record in self.end_effectors.items()
                      if isinstance(record, dict) and isinstance(record.get("pose_xyz_xyzw"), list))

    def pose(self, arm: str):
        """The measured pose of ``tcp:<arm>`` in ``world``, a ``frames.Pose``."""

        from .frames import Pose

        record = self.end_effectors.get(arm) or {}
        self.tcp_matrix(arm)  # names the arms when there is none
        pose = record["pose_xyz_xyzw"]
        return Pose(pose[:3], pose[3:], frame="world")

    def to_world(self, frame: str) -> np.ndarray:
        """The 4x4 ``world <- frame`` transform for ``world``, ``tcp:<arm>`` or ``camera:<name>``."""

        if frame == "world":
            return np.eye(4)
        kind, _, name = frame.partition(":")
        if kind == "tcp" and name:
            return self.tcp_matrix(name)
        if kind == "camera" and name:
            return self.camera(name).camera_to_world
        raise ValueError(f"unknown frame {frame!r}: frames are world, tcp:<arm>, camera:<name> and pixel:<name>")

    def from_world(self, frame: str) -> np.ndarray:
        """The 4x4 ``frame <- world`` transform (a camera's is its published ``extrinsic_cv``)."""

        if frame.startswith("camera:"):
            return self.camera(frame).extrinsic
        return geometry.invert(self.to_world(frame))

    def transform(self, target: str, source: str) -> np.ndarray:
        """The 4x4 ``target <- source`` transform between two of this observation's frames."""

        if target == source:
            return np.eye(4)
        if source == "world":
            return self.from_world(target)
        if target == "world":
            return self.to_world(source)
        return self.from_world(target) @ self.to_world(source)

    def __repr__(self) -> str:
        return (f"Observation({self.observation_id!r}, step={self.step}, cameras={sorted(self.cameras)}, "
                f"arms={self.arms()})")


_cache: dict[str, tuple[bytes, Observation]] = {}


def load(root: str = MOUNT) -> Observation:
    """The observation ``root/current.json`` names, read again only when it changed."""

    path = os.path.join(root, "current.json")
    try:
        with open(path, "rb") as stream:
            raw = stream.read()
    except OSError:
        raise ObservationUnavailable(
            f"no published observation is mounted at {root} (the extended library mounts the current one "
            "for code mode)") from None
    cached = _cache.get(root)
    if cached is not None and cached[0] == raw:
        return cached[1]
    observation = Observation(root, json.loads(raw))
    _cache[root] = (raw, observation)
    return observation


def current() -> Observation:
    """The current published observation (``/ora/observation``)."""

    return load(MOUNT)


__all__ = ["Camera", "MOUNT", "Observation", "ObservationUnavailable", "SCHEMA", "Target", "current", "load"]
