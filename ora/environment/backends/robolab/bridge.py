"""Pinned upstream RoboLab lifecycle with simulator imports confined to startup.

Ported from a first binding (single RubiksCubeTask, seed 0) and
generalized to the admitted roster and seeds. Poses are in the fixed robot-root
frame; the public controlled point is the Robotiq TCP (controllers/robolab_droid.py).
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.official_limit import official_limit_multiple
from ora.environment.benchmarks.robolab import (
    DECIMATION, PACKAGES, PHYSICS_DT, PUBLIC_CAMERAS, REVISION, TASKS,
)
from ora.environment.controllers import robolab_droid as D

from .common import RobolabError, array, file_hash, quat_xyzw_from_matrix, rotation_wxyz, write_json
from .control_record import JointTargetTap

_TCP_ROTATION = np.array(D.TCP_AXES_IN_FLANGE, dtype=float).T  # columns: the TCP axes in the flange frame
_TCP_OFFSET = np.array(D.TCP_OFFSET_FROM_FLANGE_M, dtype=float)


def verify_source(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if commit != REVISION or dirty:
        raise RobolabError(f"RoboLab requires the clean pinned source {REVISION}; found {commit} dirty={bool(dirty)}")
    return {"root": str(root), "commit": commit, "dirty": False}


def _prepare_framework_shutdown() -> None:
    """Release Kit instrumentation while native plugins still exist."""

    import faulthandler
    import signal

    kit = sys.modules.get("omni.kit.app._impl")
    if kit is not None:
        kit._toggle_gc_profiling(False)
    for fatal_signal in (signal.SIGABRT, signal.SIGSEGV):
        signal.signal(fatal_signal, signal.SIG_DFL)
    faulthandler.enable()


def _matrix_from_gf(matrix: Any) -> np.ndarray:
    return np.array([[matrix[row][column] for column in range(4)] for row in range(4)], dtype=float).T


class RobolabBridge:
    """Private Isaac handles never enter the session's public observation."""

    def __init__(self, environment: Any, app: Any, task: str) -> None:
        self.env, self.app, self.task = environment, app, task
        self.robot = self.env.scene["robot"]
        self.arm_ids = [self.robot.joint_names.index(name) for name in D.ARM_JOINT_NAMES]
        self.finger_id = self.robot.joint_names.index(D.GRIPPER_JOINT_NAME)
        self.flange_id = self.robot.body_names.index(D.FLANGE)
        self.pads = self._bind_pads()
        self.tap = JointTargetTap(self.robot)
        self.ticks: list[tuple[Any, Any]] = []

    # ------------------------------------------------------------------ pads
    def _bind_pads(self) -> dict[str, dict[str, Any]]:
        """Static collision boxes of each Robotiq finger in its bodies' frames, read once from USD.

        Per side: the fingertip pad (``fingertips`` prims under the inner
        finger) and the whole finger (every collision Gprim under the inner
        and outer finger bodies), each stored as box corners in the frame of
        the body that carries it.
        """

        from pxr import Usd, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])

        def body_prim(name: str) -> Any:
            found = [prim for prim in stage.Traverse()
                     if str(prim.GetPath()).endswith("/" + name) and "/Robot" in str(prim.GetPath())]
            if len(found) != 1:
                raise RobolabError(f"Robotiq body is not unique: {name}")
            return found[0]

        def corners(prims: list[Any], body: Any) -> np.ndarray:
            body_world = _matrix_from_gf(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            out = []
            for prim in prims:
                bound = cache.ComputeWorldBound(prim)
                box, matrix = bound.GetRange(), _matrix_from_gf(bound.GetMatrix())
                for index in range(8):
                    local = np.array([box.GetMin()[k] if not (index >> k) & 1 else box.GetMax()[k]
                                      for k in range(3)] + [1.0])
                    out.append((np.linalg.inv(body_world) @ (matrix @ local))[:3])
            return np.array(out)

        def gprims(body: Any, token: str | None = None) -> list[Any]:
            """Collision prims under ``body`` (instance proxies included); their bounds cover the meshes."""

            from pxr import UsdPhysics

            prims = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if prim.HasAPI(UsdPhysics.CollisionAPI) and (token is None or token in prim.GetName())]
            if not prims:
                raise RobolabError(f"no collision prim{' ' + token if token else ''} under {body.GetPath()}")
            return prims

        sides: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            inner, outer = body_prim(f"{side}_inner_finger"), body_prim(f"{side}_outer_finger")
            pads = gprims(inner, "fingertips")
            sides[side] = {
                "pad": [(self.robot.body_names.index(f"{side}_inner_finger"), corners(pads, inner))],
                "finger": [(self.robot.body_names.index(f"{side}_inner_finger"), corners(gprims(inner), inner)),
                           (self.robot.body_names.index(f"{side}_outer_finger"), corners(gprims(outer), outer))],
            }
        return sides

    def _corners_flange(self, parts: list[tuple[int, np.ndarray]]) -> np.ndarray:
        data = self.robot.data
        flange_pos, flange_rot = array(data.body_pos_w[0, self.flange_id]), rotation_wxyz(data.body_quat_w[0, self.flange_id])
        out = []
        for body_index, local in parts:
            body_pos = array(data.body_pos_w[0, body_index])
            body_rot = rotation_wxyz(data.body_quat_w[0, body_index])
            out.append(((local @ body_rot.T + body_pos) - flange_pos) @ flange_rot)
        return np.vstack(out)

    def jaw_separation_m(self) -> float:
        """Distance between the inner faces of the two fingertip pads along the closing line (flange y)."""

        left = self._corners_flange(self.pads["left"]["pad"])
        right = self._corners_flange(self.pads["right"]["pad"])
        return float(max(0.0, left[:, 1].min() - right[:, 1].max()))

    def contact_geometry(self) -> dict[str, Any]:
        """Finger and pad extents in the TCP frame at the current finger state."""

        def box(corners: np.ndarray) -> dict[str, list[float]]:
            local = (corners - _TCP_OFFSET) @ _TCP_ROTATION
            return {"minimum": local.min(axis=0).round(6).tolist(), "maximum": local.max(axis=0).round(6).tolist()}

        fingers = []
        for index, (side, direction) in enumerate((("right", -1.0), ("left", 1.0)), start=1):
            fingers.append({"id": f"finger_{index}", "opening_direction_local": [direction, 0.0, 0.0],
                            "collision_bounds_m": box(self._corners_flange(self.pads[side]["finger"])),
                            "pad_bounds_m": box(self._corners_flange(self.pads[side]["pad"]))})
        return {"controlled_point": "right_tcp", "unit": "metre", "reference_finger_qpos_m": [0.0, 0.0],
                "fingers": fingers, "claim": "robot_geometry_not_contact_or_grasp_guarantee"}

    # ----------------------------------------------------------------- state
    def clock(self) -> int:
        return int(self.env.sim.current_time_step_index)

    def reset(self, seed: int) -> dict[str, Any]:
        self.env.reset_eval_state()
        self.env.reset(seed=int(seed))
        return {"frozen": bool(array(self.env._frozen_envs).any()),
                "results": self.env.get_env_results(), "episode_steps": int(self.env.episode_length_buf[0]),
                "native_action_limit": TASKS[self.task]["native_action_limit"]}

    def flange_pose_root(self) -> tuple[np.ndarray, np.ndarray]:
        from isaaclab.utils.math import subtract_frame_transforms

        data = self.robot.data
        pos, quat = subtract_frame_transforms(data.root_pos_w, data.root_quat_w,
                                              data.body_pos_w[:, self.flange_id], data.body_quat_w[:, self.flange_id])
        position, quaternion = array(pos[0]).astype(float), array(quat[0]).astype(float)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise RobolabError("non-finite flange pose")
        return position, quaternion

    def tcp_pose(self) -> list[float]:
        """TCP pose (xyz + xyzw) in the robot-root frame."""

        position, quaternion = self.flange_pose_root()
        flange_rot = rotation_wxyz(quaternion)
        tcp_position = position + flange_rot @ _TCP_OFFSET
        return [*tcp_position.tolist(), *quat_xyzw_from_matrix(flange_rot @ _TCP_ROTATION)]

    def flange_target_from_tcp(self, tcp_xyzw: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """The flange pose (position, wxyz) that places the TCP at ``tcp_xyzw``."""

        from scipy.spatial.transform import Rotation

        tcp_rot = Rotation.from_quat(tcp_xyzw[3:7]).as_matrix()
        flange_rot = tcp_rot @ _TCP_ROTATION.T
        flange_pos = np.asarray(tcp_xyzw[:3]) - flange_rot @ _TCP_OFFSET
        xyzw = Rotation.from_matrix(flange_rot).as_quat()
        return flange_pos, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    def finger_qpos(self) -> float:
        return float(self.robot.data.joint_pos[0, self.finger_id])

    def arm_qpos(self) -> list[float]:
        return array(self.robot.data.joint_pos[0, self.arm_ids]).astype(float).tolist()

    def gripper_contact_force(self) -> float:
        try:
            sensor = self.env.scene.sensors["gripper__all_objs"]
            forces = array(sensor.data.net_forces_w[0])
            return float(np.linalg.norm(forces, axis=-1).max())
        except Exception:  # noqa: BLE001 - a scene without the sensor reports no contact evidence
            return 0.0

    def cameras(self) -> dict[str, Any]:
        root_pos = array(self.robot.data.root_pos_w[0])
        root_rotation = rotation_wxyz(self.robot.data.root_quat_w[0])
        result = {}
        for public, name in PUBLIC_CAMERAS.items():
            camera = self.env.scene[name]
            data = camera.data
            rgb = array(data.output["rgb"][0])[..., :3].copy()
            depth = array(data.output["distance_to_image_plane"][0]).squeeze().copy()
            cam2root = np.eye(4)
            cam2root[:3, :3] = root_rotation.T @ rotation_wxyz(data.quat_w_ros[0])
            cam2root[:3, 3] = root_rotation.T @ (array(data.pos_w[0]) - root_pos)
            result[public] = {"rgb": rgb, "depth": depth, "intrinsic_cv": array(data.intrinsic_matrices[0]).copy(),
                              "cam2world_cv": cam2root, "source_camera": name, "frame": int(array(camera.frame)[0])}
        return result

    # --------------------------------------------------------------- command
    def step_joints(self, arm_joints: list[float], finger_action: float) -> dict[str, Any]:
        """One native action that drives the arm to joint targets.

        The registered arm action is RoboLab's absolute differential IK, whose term
        turns a pose into joint targets each physics step. A planned path already is
        joint targets, so for this action that term applies them instead; decimation,
        the gripper term, terminations and the official verdict are the native step's.
        """

        import torch

        term = self.env.action_manager.get_term("arm_action")
        target = torch.tensor([arm_joints], dtype=torch.float32, device=self.env.device)
        term.apply_actions = lambda: self.robot.set_joint_position_target(target, joint_ids=self.arm_ids)
        try:
            return self.step(self.tcp_pose(), finger_action)
        finally:
            del term.apply_actions

    def step(self, tcp_target_xyzw: list[float], finger_action: float) -> dict[str, Any]:
        flange_pos, flange_wxyz = self.flange_target_from_tcp(tcp_target_xyzw)
        return self._native_step([*flange_pos.tolist(), *flange_wxyz.tolist(), float(finger_action)])

    def replay_tick(self, native: Any, joint_targets: Any) -> dict[str, Any]:
        """One recorded native action: its tensor, and each physics step's recorded joint targets (``control_record``)."""

        import torch

        self.tap.feed = iter(torch.tensor(joint_targets, device=self.env.device).unsqueeze(1))
        try:
            return self._native_step(np.asarray(native).tolist())
        finally:
            self.tap.feed = None

    def take_ticks(self) -> list[tuple[Any, Any]]:
        """The native actions sent since the last call, each as ``(action tensor, joint targets per physics step)``."""

        ticks, self.ticks = self.ticks, []
        return ticks

    def _native_step(self, native: list[float]) -> dict[str, Any]:
        import torch

        command = torch.tensor([native], dtype=torch.float32, device=self.env.device)
        _, _, terminated, truncated, _ = self.env.step(command)
        self.ticks.append((array(command[0]).copy(), self.tap.take()))
        results = self.env.get_env_results()
        return {"terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
                "frozen": bool(self.env._frozen_envs[0]), "official_result": bool(results[0]["success"]),
                "episode_steps": int(self.env.episode_length_buf[0]), "subtask_progress": self.subtask_progress()}

    def subtask_progress(self) -> dict[str, Any] | None:
        """RoboLab's own subtask score after this native action (its SubtaskCompletionRecorder), or None.

        Evaluation-only: the session keeps it for the official verdict and no observation or receipt carries it.
        The recorder keeps a terminated episode's last values until the next step, so the read after the final
        native action is that episode's score.
        """

        from robolab.core.logging.results import get_current_subtask_info

        info = get_current_subtask_info(self.env, 0)
        if not info:
            return None
        return {"score": float(info.get("score", 0.0)), "completed": int(info.get("completed", 0)),
                "total": int(info.get("total", 0))}

    def close(self) -> None:
        import gc

        _prepare_framework_shutdown()
        try:
            self.env.close()
        finally:
            world = sys.modules.get("robolab.core.world.world_state")
            if world is not None:
                world.clear_world_cache()
            self.robot = None
            self.env = None
            gc.collect()
            self.app.close()


def build_environment(config: Mapping[str, Any], output_root: Path) -> tuple[RobolabBridge, dict[str, Any]]:
    import faulthandler

    faulthandler.enable()
    task, seed = str(config["task"]), int(config["seed"])
    if task not in TASKS:
        raise RobolabError(f"RoboLab task is not admitted: {task}")
    root = Path(config["robolab_root"]).resolve()
    source = verify_source(root)
    source["packages"] = {name: importlib.metadata.version(name)
                          for name in ("robolab", "isaacsim", "isaaclab", "torch", "numpy", "warp-lang")}
    for name, version in PACKAGES.items():
        if source["packages"][name] != version:
            raise RobolabError(f"RoboLab requires {name} {version}; found {source['packages'][name]}")
    sys.path.insert(0, str(root))
    runtime_directory = Path(config.get("runtime_directory") or output_root / "runtime")
    for key, relative in (("XDG_CACHE_HOME", "cache/xdg"), ("XDG_DATA_HOME", "data/xdg"),
                          ("XDG_CONFIG_HOME", "config/xdg"), ("XDG_RUNTIME_DIR", "runtime/xdg"),
                          ("CUDA_CACHE_PATH", "cache/cuda"), ("__GL_SHADER_DISK_CACHE_PATH", "cache/gl"),
                          ("OPTIX_CACHE_PATH", "cache/optix")):
        path = runtime_directory / relative
        path.mkdir(parents=True, exist_ok=True)
        if key == "XDG_RUNTIME_DIR":
            path.chmod(0o700)
        os.environ[key] = str(path)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from ora.environment.backends.robodojo import graphics_compat
    from ora.environment.backends.robodojo.bridge import _configure_crash_reporting, renderer_table_binding

    graphics_root = Path(config["graphics_runtime_root"])
    source["graphics_compatibility"] = graphics_compat.activate(graphics_root)
    import cv2  # noqa: F401 -- upstream requires OpenCV before IsaacLab.
    from isaaclab.app import AppLauncher

    gpu = int(config["renderer_gpu"])
    startup_log = output_root / "kit-startup.log"
    crash_args, _ = _configure_crash_reporting(output_root)
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", fast_shutdown=False,
                      kit_args=(f"--/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false "
                                f"--/renderer/multiGpu/maxGpuCount=1 --/renderer/activeCudaGpus=0 "
                                f"--/renderer/useCudaDeviceOrder=true --/log/file={startup_log} "
                                f"--portable-root {runtime_directory / 'kit'} {crash_args}")).app
    env = None
    try:
        graphics_compat.verify_activation(graphics_root, source["graphics_compatibility"])
        import torch
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RobolabError("RoboLab worker must have one logical CUDA device")
        inventory = []
        for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
                                            text=True, timeout=15).splitlines():
            index, uuid, bus = [part.strip() for part in line.split(",")]
            inventory.append({"index": int(index), "uuid": uuid, "pci_bus_id": bus})
        source["device_binding"] = renderer_table_binding(startup_log.read_text(), selected_index=gpu,
                                                          cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),
                                                          inventory=inventory)
        source["device_binding"]["schema"] = "ora.robolab-device-binding.v1"
        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

        if Path(robolab.constants.PACKAGE_DIR).resolve() != root:
            raise RobolabError("imported RoboLab differs from the bound checkout")
        robolab.constants.VERBOSE = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.set_output_dir(str(output_root / "private-upstream"))
        auto_register_droid_abs_ik_envs(task=task)
        names = get_envs(task=task)
        if len(names) != 1:
            raise RobolabError("RoboLab task registration is ambiguous")
        cfg = parse_env_cfg(names[0], device="cuda:0", seed=seed, num_envs=1, use_fabric=True)
        for name in PUBLIC_CAMERAS.values():
            camera_cfg = getattr(cfg.scene, name)
            camera_cfg.data_types = ["rgb", "distance_to_image_plane"]
            camera_cfg.update_latest_camera_pose = True
        # Isaac Lab truncates at this length; the official limit stays the roster's.
        multiple = official_limit_multiple(config)
        cfg.episode_length_s = cfg.episode_length_s * multiple
        env, cfg = create_env(cfg)
        if cfg.decimation != DECIMATION or abs(cfg.sim.dt - PHYSICS_DT) > 1e-12:
            raise RobolabError("RoboLab native cadence differs")
        if int(env.max_episode_length) != TASKS[task]["native_action_limit"] * multiple:
            raise RobolabError("RoboLab native timeout differs from the admitted roster")
        expected = config.get("expected_task_language")
        if expected and cfg.instruction != expected:
            raise RobolabError(f"RoboLab task language differs: {cfg.instruction!r} != {expected!r}")
        bridge = RobolabBridge(env, app, task)
        import omni.usd
        layers = {}
        for layer in omni.usd.get_context().get_stage().GetUsedLayers():
            path = Path(layer.realPath)
            if layer.realPath and path.is_file():
                layers[str(path.resolve())] = file_hash(path)
        source.update({"loaded_usd_layers": layers, "python_executable": sys.executable,
                       "task": task, "seed": seed, "instruction": cfg.instruction,
                       "native_action_limit": TASKS[task]["native_action_limit"],
                       "physics_dt": PHYSICS_DT, "decimation": DECIMATION})
        write_json(output_root / "source-manifest.json", source)
        return bridge, source
    except BaseException:
        if env is not None:
            env.close()
        app.close()
        raise
