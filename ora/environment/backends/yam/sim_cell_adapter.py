"""The simulated YAM station's ``CellAdapter``: the real-cell kit with no hardware.

A cell config with ``robot.sim`` loads this instead of ``cell_adapter.py``. The arms are
kinematic: held, they follow every command at once; released, they stay where they are
(the person folds them back, ``SimCell``). Their encoders read the truth's angles minus its
zero errors, and the cameras are rendered by MuJoCo from the truth's model of the cell
(``sim_cell.py``'s ``world.json``: the right base moved, the joint zeros and the wrist
mounts off, the fixed cameras where they stand, the table and the checkerboard), while the
kit plans and fits on the plain robot model, as it does at a real cell. So ``ora cell
setup`` and the hand-eye calibration measure an injected error and ``ora cell sim status``
shows how close they came.

The buses are files in the state directory (``SimBus``): a reader or held arms count as a
receiver while open, every command and read is a frame, and the cell lock is the state
directory's (``ora.cell.config.isolate``); this adapter refuses the real cell's lock.
Rendering needs a headless OpenGL: the adapter sets ``MUJOCO_GL`` to ``egl`` unless it is set.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.cell.adapter import ArmReading
from ora.cell.errors import CellError
from ora.environment.controllers import yam_station as Y

from .cell_adapter import YamCellAdapter
from .sim_cell import SimBus, SimCell, SimState, rigid

FRAME_PERIOD_S = 1.0 / 30.0                    # a camera streams at most this fast, like a RealSense
DEPTH_RANGE_M = (0.07, 3.0)
GRIPPER_MOVE_S = 0.3                           # how long ``grip`` and the gripper calibration take
CV_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])


def _opening(raw: float, limits: Sequence[float] | None) -> float | None:
    if not limits:
        return None
    closed, opened = (float(v) for v in limits)
    return float(np.clip((raw - closed) / (opened - closed), 0.0, 1.0))


class SimReader:
    def __init__(self, state: SimState, buses: Mapping[str, str], gripper_limits: Mapping[str, Sequence[float]]) -> None:
        self.state, self.buses, self.limits = state, dict(buses), dict(gripper_limits)
        self.handles = [state.open_handle(channel) for channel in self.buses.values()]

    def read(self) -> dict[str, ArmReading]:
        with self.state.changing() as value:
            for channel in self.buses.values():
                self.state.count_frames(value, channel)
            arms = {arm: dict(value["arms"][arm]) for arm in self.buses}
        return {arm: ArmReading(q=np.asarray(s["q"], float), gripper_raw=float(s["gripper_raw"]),
                                gripper_opening=_opening(float(s["gripper_raw"]), self.limits.get(arm)),
                                max_temperature_c=30.0, torque_nm=[0.0] * 6)
                for arm, s in arms.items()}

    def close(self) -> None:
        for handle in self.handles:
            handle.unlink(missing_ok=True)
        self.handles = []


class SimHeld:
    """Arms whose motors this process enabled: commands move them exactly; ``release`` lets go."""

    def __init__(self, state: SimState, arms: Sequence[str], buses: Mapping[str, str],
                 gripper_limits: Mapping[str, Sequence[float]]) -> None:
        self.state, self.arms = state, tuple(arms)
        self.buses = {arm: buses[arm] for arm in self.arms}
        self.limits = dict(gripper_limits)
        self.pid = os.getpid()
        with state.changing() as value:
            for arm in self.arms:
                if SimState._enabled(value["arms"][arm]):
                    raise CellError(f"the {arm} arm is held by pid {value['arms'][arm]['enabled_by']}")
            for arm in self.arms:
                value["arms"][arm]["enabled_by"] = self.pid
        self.handles = [state.open_handle(channel) for channel in self.buses.values()]

    def _mine(self, value: Mapping[str, Any]) -> list[str]:
        return [arm for arm in self.arms if value["arms"][arm].get("enabled_by") == self.pid]

    def joints(self) -> dict[str, np.ndarray]:
        arms = self.state.arms()["arms"]
        return {arm: np.asarray(arms[arm]["q"], float) for arm in self.arms}

    def command(self, targets: Mapping[str, np.ndarray]) -> None:
        with self.state.changing() as value:
            mine = self._mine(value)
            for arm, q in targets.items():
                self.state.count_frames(value, self.buses[arm])
                if arm in mine:                    # a disabled motor follows nothing
                    value["arms"][arm]["q"] = np.asarray(q, float).tolist()

    def alive(self) -> dict[str, bool]:
        mine = self._mine(self.state.arms())
        return {arm: arm in mine for arm in self.arms}

    def set_mode(self, mode: str) -> None:
        if mode not in ("hold", "float"):
            raise CellError(f"unknown hold mode {mode!r}")       # a kinematic arm floats where it is

    def grip(self, openings: Mapping[str, float]) -> dict[str, float]:
        measured = {}
        with self.state.changing() as value:
            mine = self._mine(value)
            for arm, opening in openings.items():
                limits = self.limits.get(arm)
                if arm not in mine or not limits:
                    continue
                closed, opened = (float(v) for v in limits)
                target = closed + float(np.clip(opening, *Y.GRIPPER_HOLD_RANGE)) * (opened - closed)
                value["arms"][arm]["gripper_raw"] = target
                self.state.count_frames(value, self.buses[arm])
                measured[arm] = _opening(target, limits)
        time.sleep(GRIPPER_MOVE_S)
        return measured

    def release(self) -> None:
        with self.state.changing() as value:
            for arm in self._mine(value):
                value["arms"][arm]["enabled_by"] = None
        for handle in self.handles:
            handle.unlink(missing_ok=True)
        self.handles = []

    def alerts(self) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------- the rendered cameras
class _Scene:
    """The truth's model of the cell, posed from the shared state and rendered; one per process."""

    def __init__(self, state: SimState) -> None:
        import mujoco

        from .kinematics import StationKinematics
        from .scene import CHECKERBOARD_BODY, build_cell_model

        self.mujoco = mujoco
        self.version = state.world_version()
        world = state.world()
        board = world["board"]
        model = build_cell_model(table=world["table"], fixed_cameras=world["fixed_cameras"], wrist_cameras=True,
                                 checkerboard={"square_m": board["square_mm"] / 1000.0, "pattern": board["pattern"],
                                               "margin_m": board["margin_mm"] / 1000.0,
                                               "thickness_m": board["thickness_mm"] / 1000.0})
        self.kinematics = StationKinematics(model)
        self.kinematics.set_base_corrections({arm: rigid(**c) for arm, c in world["arm_base_corrections"].items()})
        self.kinematics.joint_offsets = {arm: np.asarray(v, float) for arm, v in world["joint_zero_offsets"].items()}
        for arm, correction in world["wrist_camera_corrections"].items():
            # The camera's frame in its body: the optical frame (the body) moved by the mount
            # error, then turned into MuJoCo's camera axes.
            local = rigid(**correction) @ CV_FROM_MUJOCO
            camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"ora_wrist_{arm}")
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, local[:3, :3].ravel())
            model.cam_pos[camera], model.cam_quat[camera] = local[:3, 3], quat
        self.model, self.data = model, self.kinematics.data
        self.board_mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CHECKERBOARD_BODY)])
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0             # the model's TCP markers are not in the world
        self.lock = threading.Lock()

    def pose(self, arms: Mapping[str, Any], board2world: np.ndarray | None, finger_open_m: float,
             grippers: Mapping[str, Mapping[str, float]]) -> None:
        if board2world is None:
            self.data.mocap_pos[self.board_mocap] = [0.0, 0.0, -10.0]
        else:
            quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(quat, np.asarray(board2world[:3, :3], float).ravel())
            self.data.mocap_pos[self.board_mocap] = board2world[:3, 3] - np.asarray(Y.STATION_TO_WORLD_XYZ_M)
            self.data.mocap_quat[self.board_mocap] = quat
        fingers = {arm: finger_open_m * (_opening(float(s["gripper_raw"]), (grippers[arm]["closed"], grippers[arm]["open"]))
                                         or 0.0) for arm, s in arms.items()}
        self.kinematics.set_state({arm: s["q"] for arm, s in arms.items()}, fingers)
        self.mujoco.mj_camlight(self.model, self.data)     # set_state places the bodies, not the cameras


_SCENES: dict[str, _Scene] = {}
_SCENES_LOCK = threading.Lock()


def _scene(state: SimState) -> _Scene:
    with _SCENES_LOCK:
        scene = _SCENES.get(str(state.directory))
        if scene is None or scene.version != state.world_version():
            scene = _SCENES[str(state.directory)] = _Scene(state)
        return scene


class SimCamera:
    """One simulated camera: what ``RealSenseCamera.capture`` returns, rendered; a frame of an
    unchanged cell is not rendered again."""

    def __init__(self, state: SimState, role: str, serial: str, mount: str) -> None:
        from .scene import SIM_FOVY_DEG, SIM_IMAGE_SIZE, camera_intrinsic

        self.state, self.role, self.serial, self.mount = state, role, serial, mount
        self.size = SIM_IMAGE_SIZE
        fovy = SIM_FOVY_DEG["wrist" if mount == "end_effector" else "fixed"]
        self.intrinsic = camera_intrinsic(fovy, *self.size)
        self.renderer = None
        self.last: tuple[Any, dict[str, Any]] | None = None
        self.at = 0.0

    def _render(self, scene: _Scene) -> tuple[np.ndarray, np.ndarray]:
        mujoco = scene.mujoco
        if self.renderer is None:
            self.renderer = mujoco.Renderer(scene.model, self.size[1], self.size[0])
        camera = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, f"ora_{self.role}")
        if camera < 0:
            raise CellError(f"the simulated cell has no camera '{self.role}'")
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(scene.data, camera=camera, scene_option=scene.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(scene.data, camera=camera, scene_option=scene.option)
        depth = self.renderer.render().astype(np.float32).copy()
        self.renderer.disable_depth_rendering()
        depth[(depth < DEPTH_RANGE_M[0]) | (depth > DEPTH_RANGE_M[1])] = 0.0
        return rgb, depth

    def capture(self) -> dict[str, Any]:
        wait = FRAME_PERIOD_S - (time.monotonic() - self.at)
        if wait > 0:
            time.sleep(wait)
        self.at = time.monotonic()
        scene = _scene(self.state)
        arms = self.state.arms()["arms"]
        board = self.state.board()
        key = (scene.version, json_key(arms), None if board is None else np.round(board, 6).tobytes())
        if self.last is None or self.last[0] != key:
            grippers = self.state.world()["grippers"]
            with scene.lock:
                scene.pose(arms, board, Y.FINGER_OPEN_M, grippers)
                rgb, depth = self._render(scene)
            valid = float(np.count_nonzero(depth > 0) / depth.size)
            self.last = (key, {"rgb": rgb, "depth": depth, "intrinsic_cv": self.intrinsic, "mount": self.mount,
                               "source_camera": f"sim:ora_{self.role}:{self.serial}", "depth_valid_fraction": valid,
                               "depth_pipeline_restarted": False, "depth_below_floor": False,
                               "distortion": {"model": "none", "coeffs": [0.0] * 5}})
        frame = self.last[1]
        return {**frame, "rgb": frame["rgb"].copy(), "depth": frame["depth"].copy()}

    def close(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001 - a renderer that fails to close holds nothing of the cell
                pass
            self.renderer = None


def json_key(arms: Mapping[str, Any]) -> tuple:
    return tuple((arm, tuple(np.round(s["q"], 6)), round(float(s["gripper_raw"]), 4)) for arm, s in sorted(arms.items()))


class SimCameraDriver:
    """The cell's cameras, rendered (``ora.cell.adapter.CameraDriver``)."""

    def __init__(self, state: SimState, serials: Mapping[str, str]) -> None:
        self.state, self.serials = state, dict(serials)

    def open(self, role: str, serial: str, mount: str) -> SimCamera:
        return SimCamera(self.state, role, serial, mount)

    def connected(self) -> set[str]:
        return set(self.serials.values())

    def busy(self, serials: Sequence[str]) -> dict[str, str]:
        return {}

    def reset(self, serials: Sequence[str]) -> str:
        return "simulated cameras: nothing to reset"


# ---------------------------------------------------------------- the adapter
class YamSimCellAdapter(YamCellAdapter):
    """The YAM station's constants, model and planning; the simulated cell's arms, buses and cameras."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        from ora.cell import lock

        if lock.LOCK_PATH.resolve() == lock.DEFAULT_LOCK_PATH.resolve():
            raise CellError(f"a simulated cell does not take the real cell's lock {lock.DEFAULT_LOCK_PATH}: "
                            "run it through `ora cell`, which keeps its lock in the cell's state directory, "
                            "or set ORA_CELL_LOCK")
        os.environ.setdefault("MUJOCO_GL", "egl")    # read when MuJoCo is first imported
        self.state = SimState(config.sim_state)
        self.bus = SimBus(self.state)
        self.camera_driver = SimCameraDriver(self.state, config.serials)
        self.sim = SimCell(self.state)
        self.state.world()                       # the truth exists from the first command on

    def open_reader(self, gripper_limits: Mapping[str, Sequence[float]]) -> SimReader:
        return SimReader(self.state, self.buses, gripper_limits)

    def _quiet(self, arms: Sequence[str]) -> None:
        for arm in arms:
            channel = self.buses[arm]
            if self.bus.receivers(channel) or self.bus.traffic(channel, 0.3):
                raise CellError(f"{channel} is in use by another program: refusing to take over")

    def take_over(self, arms: Sequence[str], *, mode: str, gripper_limits: Mapping[str, Sequence[float]]) -> SimHeld:
        self._quiet(arms)
        return SimHeld(self.state, arms, self.buses, gripper_limits)

    def disable_all(self) -> None:
        with self.state.changing() as value:
            for arm in value["arms"].values():
                arm["enabled_by"] = None

    def calibrate_grippers(self, arms: Sequence[str], should_stop: Callable[[], bool]) -> dict[str, list[float]]:
        """Each gripper closes and opens once; its stops come back named by the motor's nominal
        direction (the larger reading first), whichever way the gripper really opens."""

        stops, found = self.state.world()["grippers"], {}
        for arm in arms:
            if should_stop():
                break
            self._quiet([arm])
            print(f"calibrating the {arm} gripper (it opens and closes once)", flush=True)
            for end in ("closed", "open"):
                with self.state.changing() as value:
                    value["arms"][arm]["gripper_raw"] = float(stops[arm][end])
                    self.state.count_frames(value, self.buses[arm], 10)
                time.sleep(GRIPPER_MOVE_S)
            found[arm] = sorted((float(stops[arm]["closed"]), float(stops[arm]["open"])), reverse=True)
        return found

    def facts(self) -> list[tuple[str, str]]:
        world = self.state.world()
        injected = "；".join(
            [f"{arm} 底座 {np.linalg.norm(c['translation_m']) * 1000:.1f} mm / "
             f"{np.degrees(np.linalg.norm(c['rotvec_rad'])):.2f}°" for arm, c in world["arm_base_corrections"].items()]
            + [f"{arm} 关节零点最大 {np.degrees(np.max(np.abs(v))):.2f}°" for arm, v in world["joint_zero_offsets"].items()])
        return [("仿真", f"MuJoCo 运动学仿真，没有硬件；状态目录 {self.state.directory}"),
                ("注入误差", injected + "（ora cell sim status 对照标定结果）")] + [
            fact for fact in super().facts() if fact[0] not in ("CAN", "禁发帧", "i2rt 版本锁")]


def make_adapter(config: Any) -> YamSimCellAdapter:
    return YamSimCellAdapter(config)
