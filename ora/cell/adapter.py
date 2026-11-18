"""What the real-cell kit needs from a robot family, and nothing more.

The kit (cell board, live view, reset, calibration tools and motions) is written once
against these protocols. A robot family provides one ``CellAdapter`` in
``ora/environment/backends/<environment>/cell_adapter.py`` as ``make_adapter(config)``, and
may provide a simulated one (no hardware) in ``sim_cell_adapter.py``, which a cell config
with ``robot.sim`` loads instead.

An adapter may also give the kit its own ``bus`` (``Bus``) and ``camera_driver``
(``CameraDriver``); without them the arms' buses are socketcan interfaces
(``occupancy.SOCKETCAN``) and the cameras RealSense devices (``cameras.realsense.REALSENSE``).

Provisional: drawn from one robot family (I2RT YAM on DM motors over socketcan). Expect
it to change when a second family is added.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


@dataclass
class ArmReading:
    """One arm read without enabling anything."""

    q: np.ndarray                               # joint angles, rad
    gripper_raw: float | None = None            # the gripper motor's own reading
    gripper_opening: float | None = None        # 0 closed .. 1 open, None without gripper limits
    faults: dict[str, str] = field(default_factory=dict)     # documented fault codes, per motor
    status: dict[str, str] = field(default_factory=dict)     # other non-normal states, per motor
    max_temperature_c: float = 0.0
    torque_nm: list[float] = field(default_factory=list)
    motors: list[dict[str, Any]] = field(default_factory=list)


class Reader(Protocol):
    """Both arms read passively, holding the buses open between reads (live view)."""

    def read(self) -> dict[str, ArmReading]: ...
    def close(self) -> None: ...


class Held(Protocol):
    """Arms the kit has enabled. Nothing here lets go except ``release``. A gripper gives
    torque only while ``grip`` moves it: held arms never keep a gripper pushing."""

    arms: tuple[str, ...]

    def joints(self) -> dict[str, np.ndarray]: ...
    def command(self, targets: Mapping[str, np.ndarray]) -> None: ...   # joint targets; grippers held
    def alive(self) -> dict[str, bool]: ...                            # each arm's control loop
    def set_mode(self, mode: str) -> None: ...                         # "hold" the measured pose, or "float"
    def grip(self, openings: Mapping[str, float]) -> dict[str, float]: ...  # 0 closed .. 1 open; arms held
    def release(self) -> None: ...                                     # motors off: the arms go limp
    def alerts(self) -> list[dict[str, Any]]: ...                      # what is wrong now (``ora.cell.alerts`` entries)


class Bus(Protocol):
    """The arms' buses as the kit checks them, by channel name: who else has one open, whether
    anything talks on it, its link. Reads only; ``receivers`` counts this process's own handles too."""

    def receivers(self, channel: str) -> int: ...
    def traffic(self, channel: str, seconds: float) -> int: ...       # frames heard while listening
    def interface(self, channel: str) -> dict[str, Any]: ...          # present, up, state, rx/tx_packets


class Camera(Protocol):
    """One open camera: ``capture`` returns aligned metric RGB-D (``cameras/realsense.py`` names the keys)."""

    def capture(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class CameraDriver(Protocol):
    def open(self, role: str, serial: str, mount: str) -> Camera: ...
    def connected(self) -> set[str]: ...                              # serials present now
    def busy(self, serials: Sequence[str]) -> dict[str, str]: ...     # serial -> why it cannot be opened
    def reset(self, serials: Sequence[str]) -> str: ...


class Kinematics(Protocol):
    """Forward kinematics and clearances of the robot model, MuJoCo-backed."""

    mujoco: Any
    model: Any
    data: Any
    world_offset: np.ndarray                    # model frame -> the cell's world frame
    camera_corrections: dict[str, np.ndarray]   # per arm, a measured wrist-camera mount correction
    joint_offsets: dict[str, np.ndarray]        # per arm, measured encoder zero errors: model = reading + offset
    arm_limits: Mapping[str, np.ndarray]        # as encoder readings

    def set_base_corrections(self, corrections: Mapping[str, np.ndarray]) -> None: ...  # world 4x4 per arm
    def ik(self, arm: str, target_world: np.ndarray, seed: Sequence[float]) -> Any: ...  # .q, .position_error_m, .orientation_error_rad
    def within_limits(self, arm: str, q: Sequence[float]) -> bool: ...
    def self_distances(self, arm: str) -> np.ndarray: ...  # the arm's wrist against its own lower links, per pair
    def set_state(self, arm_q: Mapping[str, Sequence[float]], finger_m: Mapping[str, float] | None = None) -> None: ...
    def tcp(self, arm: str) -> np.ndarray: ...
    def chain(self, arm: str) -> list[list[float]]: ...
    def wrist_camera(self, arm: str) -> np.ndarray: ...
    def inter_arm_distance(self) -> float: ...
    def table_distance(self, arm: str) -> float | None: ...


class CellAdapter(Protocol):
    environment: str
    model: str
    arms: tuple[str, ...]
    buses: Mapping[str, str]                    # arm -> bus interface (for occupancy checks)
    camera_mounts: Mapping[str, str]            # public camera role -> "fixed_world" | "end_effector"
    wrist_cameras: Mapping[str, str]            # wrist camera role -> arm
    joint_names: tuple[str, ...]                # plain names, base to wrist
    control_hz: float
    tracking_stop_rad: float
    finger_open_m: float
    foreign_controller_markers: tuple[str, ...] # command-line fragments of programs that drive these arms
    # Hand-eye planning: the IK seed's offset from rest that brings the wrist out over the
    # table, the furthest any joint is planned from rest, and the boxes (x0, y0, x1, y1 px)
    # of the wrist image the jaws hide.
    reach_seed_offset: np.ndarray
    reach_offset_limit_rad: np.ndarray
    wrist_image_occluded: Sequence[Sequence[float]]

    def open_reader(self, gripper_limits: Mapping[str, Sequence[float]]) -> Reader: ...
    def take_over(self, arms: Sequence[str], *, mode: str,
                  gripper_limits: Mapping[str, Sequence[float]]) -> Held: ...
    def disable_all(self) -> None: ...
    def calibrate_grippers(self, arms: Sequence[str], should_stop: Callable[[], bool]) -> dict[str, list[float]]: ...
    def kinematics(self, table: Mapping[str, Any] | None = None) -> Kinematics: ...
    def calibration_offsets(self) -> list[np.ndarray]: ...  # joint offsets from rest for calibration poses
    def facts(self) -> list[tuple[str, str]]: ...           # what the operator should know, plain words


def load_adapter(config: Any) -> CellAdapter:
    """The adapter of the config's robot: ``backends/<environment>/cell_adapter.make_adapter``,
    or ``sim_cell_adapter`` for a simulated cell (``robot.sim``)."""

    name = "sim_cell_adapter" if getattr(config, "sim_state", None) is not None else "cell_adapter"
    try:
        module = importlib.import_module(f"ora.environment.backends.{config.environment}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"ora.environment.backends.{config.environment}.{name}":
            raise
        from .errors import CellError

        kind = "simulated cell" if name == "sim_cell_adapter" else "cell adapter"
        raise CellError(f"the robot '{config.environment}' has no {kind}") from None
    return module.make_adapter(config)


def bus_of(adapter: Any) -> Bus:
    """The adapter's ``bus``, else the socketcan interfaces."""

    bus = getattr(adapter, "bus", None)
    if bus is None:
        from .occupancy import SOCKETCAN

        return SOCKETCAN
    return bus


def camera_driver_of(adapter: Any) -> CameraDriver:
    """The adapter's ``camera_driver``, else RealSense."""

    driver = getattr(adapter, "camera_driver", None)
    if driver is None:
        from .cameras.realsense import REALSENSE

        return REALSENSE
    return driver
