"""The simulated YAM cell's shared state: what the simulation knows, the arms, the board, the buses.

A simulated cell (``robot.sim`` in the cell config) keeps everything in one state directory
(``CellConfig.sim_state``) that every process of the cell shares: the cell board, the
``ora cell`` commands, a calibration the board starts, the simulated operator. In it:

- ``world.json``: the truth the tools must find, written once from ``DEFAULT_WORLD``: where
  the installation departs from the robot model (the right arm's base, the inner joints'
  encoder zeros, the wrist cameras' mounts), where the fixed cameras stand, the table, the
  checkerboard, each gripper's motor readings at its stops and the arms' mechanical rest.
  Edit it (before the first ``ora cell`` command, or with the cell idle) to inject another
  error. The left arm's base is the calibration's world (``bundle.py``) and the first and
  last joints' zeros are its base's and its camera's: errors there cannot be told apart.
- ``arms.json``: each arm's encoder readings, its gripper motor's reading and which process
  has its motors enabled; changed only under ``arms.lock`` (``flock``).
- ``board.json``: where the checkerboard lies (``place_board``); absent while it is off the table.
- ``bus/<channel>/``: one file per handle a process holds open on that simulated bus
  (``receivers``); ``arms.json`` counts the frames each bus carried (``traffic``).

``SimCell`` is the person at the simulated cell: it places the board, puts the arms at rest
and answers the questions a setup step asks through the approval broker (``operate``), and
``recovery`` says how close a calibration came to the truth. Nothing here needs MuJoCo; the
cameras and the robot model are ``sim_cell_adapter.py``'s.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ora.cell.errors import CellError
from ora.cell.io import write_json
from ora.environment.controllers import yam_station as Y

SCHEMA = "ora.yam-sim-cell.v1"
SIM_OPERATOR = "simulated operator"
# The truth of a new simulated cell: a front camera and a table top about where the real
# station has them, an overhead camera that sees every board placement the simulated
# operator makes (the real one sees only the left half), and a table that reaches under the
# fixed-camera step's board. The errors are of the size the station's own hand-eye
# calibration found.
DEFAULT_WORLD: dict[str, Any] = {
    "schema": SCHEMA,
    "arm_base_corrections": {"right": {"rotvec_rad": [0.0, 0.004, 0.012], "translation_m": [0.006, -0.004, 0.002]}},
    "joint_zero_offsets": {"left": [0.0, -0.020, -0.012, 0.025, -0.018, 0.0],
                           "right": [0.0, -0.015, -0.022, -0.015, 0.020, 0.0]},
    "wrist_camera_corrections": {"left": {"rotvec_rad": [0.004, -0.008, 0.006], "translation_m": [0.002, -0.001, 0.0015]},
                                 "right": {"rotvec_rad": [-0.006, 0.005, -0.004], "translation_m": [-0.0015, 0.002, 0.001]}},
    "fixed_cameras": {
        "overhead": {"position_m": [0.45, 0.0, 1.0], "look_at_m": [0.45, 0.0, -0.09], "up_hint": [1.0, 0.0, 0.0]},
        "front": {"position_m": [0.06, 0.09, 0.98], "look_at_m": [0.43, 0.01, -0.09], "up_hint": [1.0, 0.0, 0.0]},
    },
    "table": {"center_xy_m": [0.5, 0.06], "size_xy_m": [0.9, 1.33], "top_z_m": -0.093},
    "board": {"square_mm": 20.0, "pattern": [11, 8], "margin_mm": 20.0, "thickness_mm": 3.0},
    # Motor readings at each gripper's stops. The driver names the stops by the motor's
    # nominal direction, so the right gripper's come back swapped (``tools.orient_grippers``).
    "grippers": {"left": {"closed": 6.37, "open": 1.11}, "right": {"closed": -5.16, "open": 0.21}},
    "rest_q": {"left": [0.0] * 6, "right": [0.0] * 6},       # true joint angles at the mechanical rest
}
# Where the person puts the board, by turn (``handeye.placement_hint``), as (forward of the
# arm's base m, outward of it m, turn deg toward the arm's outside); "both" is between the arms.
BOTH_SPOT = (0.36, 0.0, 0.0)
ARM_SPOTS = ((0.30, 0.0, 30.0), (0.42, 0.0, -20.0), (0.40, 0.12, 0.0))
# The fixed-camera step's board: where both wrist cameras see it as the arms rise through
# the calibration poses (they look far forward from there).
FIXED_SPOT = (0.74, 0.0, 0.0)
# A placement asked again (too few safe views) moves the board this much nearer the arm;
# after this many tries the person declines it, which ends the captures.
NUDGE_M = 0.03
TRIES = 4


def rigid(rotvec_rad: Sequence[float], translation_m: Sequence[float]) -> np.ndarray:
    """A 4x4 from a correction as the calibration stores it."""

    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec([float(v) for v in rotvec_rad]).as_matrix()
    matrix[:3, 3] = [float(v) for v in translation_m]
    return matrix


def fixed_camera_pose(spec: Mapping[str, Any]) -> np.ndarray:
    """cam2world (OpenCV optical frame) of a fixed camera given as position, look_at and up_hint."""

    position = np.asarray(spec["position_m"], float)
    forward = np.asarray(spec["look_at_m"], float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(spec["up_hint"], float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def board_pose(world: Mapping[str, Any], center_xy: Sequence[float], yaw_deg: float) -> np.ndarray:
    """board2world of the checkerboard lying flat on the table, its middle at ``center_xy``."""

    board = world["board"]
    square = float(board["square_mm"]) / 1000.0
    columns, rows = board["pattern"]
    yaw = np.radians(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    middle = np.array([(columns - 1) * square / 2, (rows - 1) * square / 2, 0.0])
    top = float(world["table"]["top_z_m"]) + float(board["thickness_mm"]) / 1000.0
    pose[:3, 3] = np.array([center_xy[0], center_xy[1], top]) - pose[:3, :3] @ middle
    return pose


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SimState:
    """The files of one simulated cell (see the module's docstring)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles = 0

    # ------------------------------------------------------------ the truth
    def world(self) -> dict[str, Any]:
        path = self.directory / "world.json"
        if not path.is_file():
            write_json(path, DEFAULT_WORLD)
        value = json.loads(path.read_text())
        if value.get("schema") != SCHEMA:
            raise CellError(f"{path} is not a {SCHEMA} file")
        return value

    def world_version(self) -> float:
        self.world()
        return (self.directory / "world.json").stat().st_mtime

    # ------------------------------------------------------------ the arms
    def _initial_arms(self) -> dict[str, Any]:
        world = self.world()
        arms = {}
        for arm in Y.ARMS:
            stops = world["grippers"][arm]
            arms[arm] = {"q": self.rest_readings(arm, world), "gripper_raw": float(stops["open"]),
                         "enabled_by": None}
        return {"arms": arms, "frames": {}}

    def rest_readings(self, arm: str, world: Mapping[str, Any] | None = None) -> list[float]:
        """The encoders' readings at the mechanical rest: reading = true angle - zero error."""

        world = world or self.world()
        offsets = np.asarray(world["joint_zero_offsets"].get(arm, [0.0] * 6), float)
        return (np.asarray(world["rest_q"][arm], float) - offsets).tolist()

    def arms(self) -> dict[str, Any]:
        path = self.directory / "arms.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            with self.changing() as value:
                return json.loads(json.dumps(value))

    @contextlib.contextmanager
    def changing(self) -> Iterator[dict[str, Any]]:
        """The arms' state, read, changed by the caller and written back, under ``arms.lock``."""

        path = self.directory / "arms.json"
        with open(self.directory / "arms.lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                value = json.loads(path.read_text()) if path.is_file() else self._initial_arms()
                yield value
                write_json(path, value)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def count_frames(self, value: dict[str, Any], channel: str, frames: int = 1) -> None:
        value.setdefault("frames", {})[channel] = int(value.get("frames", {}).get(channel, 0)) + frames

    def at_rest(self) -> None:
        """The person folds every arm whose motors are off back to its rest (they move by hand)."""

        with self.changing() as value:
            for arm, state in value["arms"].items():
                if not self._enabled(state):
                    state["q"] = self.rest_readings(arm)

    @staticmethod
    def _enabled(state: Mapping[str, Any]) -> bool:
        pid = state.get("enabled_by")
        return pid is not None and _alive(int(pid))

    # ------------------------------------------------------------ the board
    def board(self) -> np.ndarray | None:
        try:
            return np.asarray(json.loads((self.directory / "board.json").read_text())["board2world"], float)
        except (OSError, ValueError, KeyError):
            return None

    def place_board(self, center_xy: Sequence[float], yaw_deg: float = 0.0, why: str = "") -> np.ndarray:
        pose = board_pose(self.world(), center_xy, yaw_deg)
        write_json(self.directory / "board.json", {"board2world": pose.tolist(), "center_xy_m": list(center_xy),
                                                   "yaw_deg": float(yaw_deg), "why": why, "at": time.time()})
        return pose

    def clear_board(self) -> None:
        (self.directory / "board.json").unlink(missing_ok=True)

    # ------------------------------------------------------------ the buses
    def _bus(self, channel: str) -> Path:
        folder = self.directory / "bus" / channel
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def open_handle(self, channel: str) -> Path:
        self._handles += 1
        path = self._bus(channel) / f"{os.getpid()}-{threading.get_ident()}-{self._handles}"
        path.write_text("")
        return path

    def receivers(self, channel: str) -> int:
        count = 0
        for path in self._bus(channel).iterdir():
            try:
                pid = int(path.name.split("-")[0])
            except ValueError:
                continue
            if _alive(pid):
                count += 1
            else:
                path.unlink(missing_ok=True)          # a process that died with the bus open
        return count

    def frames(self, channel: str) -> int:
        return int(self.arms().get("frames", {}).get(channel, 0))


class SimBus:
    """The simulated buses as the kit checks them (``ora.cell.adapter.Bus``)."""

    def __init__(self, state: SimState) -> None:
        self.state = state

    def receivers(self, channel: str) -> int:
        return self.state.receivers(channel)

    def traffic(self, channel: str, seconds: float) -> int:
        before = self.state.frames(channel)
        time.sleep(seconds)
        return self.state.frames(channel) - before

    def interface(self, channel: str) -> dict[str, Any]:
        frames = self.state.frames(channel)
        return {"present": True, "up": True, "state": "ERROR-ACTIVE", "bitrate": 1_000_000, "simulated": True,
                "rx_packets": frames, "tx_packets": frames, "rx_errors": 0, "tx_errors": 0}


# ---------------------------------------------------------------- how close a calibration came
def _gap(found: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    turn = np.clip((np.trace(found[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1.0, 1.0)
    return {"mm": round(float(np.linalg.norm(found[:3, 3] - truth[:3, 3])) * 1000, 2),
            "deg": round(float(np.degrees(np.arccos(turn))), 3)}


def recovery(world: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    """The calibration against the truth: each arm's base, inner joint zeros and wrist mount, each fixed camera."""

    from ora.cell import calibration as cell_calibration

    bases = cell_calibration.base_corrections(calibration)
    mounts = cell_calibration.camera_corrections(calibration)
    zeros = cell_calibration.joint_offsets(calibration)
    out: dict[str, Any] = {"base": {}, "joint_zero_offsets_deg": {}, "wrist_mount": {}, "fixed_cameras": {}}
    for arm in Y.ARMS:
        truth = world["arm_base_corrections"].get(arm)
        out["base"][arm] = _gap(bases.get(arm, np.eye(4)), rigid(**truth) if truth else np.eye(4))
        truth = world["wrist_camera_corrections"].get(arm)
        out["wrist_mount"][arm] = _gap(mounts.get(arm, np.eye(4)), rigid(**truth) if truth else np.eye(4))
        injected = np.asarray(world["joint_zero_offsets"].get(arm, [0.0] * 6), float)
        found = np.asarray(zeros.get(arm, np.zeros(6)), float)
        out["joint_zero_offsets_deg"][arm] = {"injected": np.degrees(injected).round(3).tolist(),
                                              "found": np.degrees(found).round(3).tolist(),
                                              "error": np.degrees(found - injected).round(3).tolist()}
    for role, spec in world["fixed_cameras"].items():
        camera = (calibration.get("cameras") or {}).get(role)
        out["fixed_cameras"][role] = None if not camera else _gap(np.asarray(camera["cam2world_cv"], float),
                                                                  fixed_camera_pose(spec))
    table = calibration.get("table")
    out["table_mm"] = None if not table else round((float(table["top_z_m"]) - world["table"]["top_z_m"]) * 1000, 2)
    return out


# ---------------------------------------------------------------- the person at the simulated cell
class SimCell:
    """The simulated person: places the board, folds the arms, answers the broker's questions."""

    def __init__(self, state: SimState) -> None:
        self.state = state
        self.spots: dict[Any, tuple[float, float, float]] = {}      # a placement's number -> where it went
        self.turns: dict[tuple[str, ...], int] = {}

    def status(self, calibration: Mapping[str, Any]) -> dict[str, Any]:
        world = self.state.world()
        board = json.loads((self.state.directory / "board.json").read_text()) \
            if (self.state.directory / "board.json").is_file() else None
        return {"state": str(self.state.directory), "injected": {k: world[k] for k in (
                    "arm_base_corrections", "joint_zero_offsets", "wrist_camera_corrections")},
                "board": board and {k: board[k] for k in ("center_xy_m", "yaw_deg", "why")},
                "arms": {arm: {"q": np.round(s["q"], 4).tolist(), "gripper_raw": round(s["gripper_raw"], 3),
                               "enabled": SimState._enabled(s)} for arm, s in self.state.arms()["arms"].items()},
                "recovery": recovery(world, calibration)}

    def place(self, arms: Sequence[str], turn: int | None, attempt: int = 0, why: str = "") -> np.ndarray:
        """The board where ``handeye.placement_hint`` asks for it: between the arms for several,
        else the ``turn``-th spot of the one arm; an attempt after the first nudges it nearer."""

        if turn is None:
            forward, outward, turned = FIXED_SPOT
            centre, yaw = np.array([forward, 0.0]), turned
        elif len(arms) > 1:
            forward, outward, turned = BOTH_SPOT
            centre, yaw = np.array([forward, 0.0]), turned
        else:
            arm = arms[0]
            side = 1.0 if Y.BASE_IN_WORLD_XYZ_M[arm][1] > 0 else -1.0
            forward, outward, turned = ARM_SPOTS[turn % len(ARM_SPOTS)]
            base = np.asarray(Y.BASE_IN_WORLD_XYZ_M[arm][:2], float)
            centre, yaw = base + [forward, side * outward], side * turned
            if attempt:
                toward = base - centre
                centre = centre + min(attempt * NUDGE_M, 0.5 * np.linalg.norm(toward)) * toward / np.linalg.norm(toward)
        if len(arms) > 1:
            centre = centre + [-attempt * NUDGE_M, 0.0]
        return self.state.place_board(centre.tolist(), yaw, why)

    def answer(self, request: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
        """One question answered as the person would: ``(approved, reason, value)``."""

        kind, details = request.get("kind"), request.get("details") or {}
        world = self.state.world()
        if kind == "value":
            if details.get("unit") == "mm":
                return True, None, f"{float(world['board']['square_mm']):g}"
            if details.get("unit") == "columns x rows":
                columns, rows = world["board"]["pattern"]
                return True, None, f"{columns}x{rows}"
            arm = details.get("arm")
            if arm in world["grippers"] and details.get("reading") is not None:
                stops = world["grippers"][arm]
                raw = float(self.state.arms()["arms"][arm]["gripper_raw"])
                opened = abs(raw - stops["open"]) < abs(raw - stops["closed"])
                return True, None, "开" if opened else "合"
            return False, f"the simulated operator cannot answer {request.get('title')!r}", None
        if kind == "task":
            purpose = details.get("for")
            if purpose == "devices":
                return False, "a simulated cell's devices are always there", None
            if purpose == "rest":
                self.state.at_rest()
                return True, None, None
            if purpose == "fixed_cameras":
                self.place(details.get("arms") or Y.ARMS, None, why="fixed cameras")
                return True, None, None
            if "placement" in details:
                shown = tuple(details.get("arms") or Y.ARMS)
                number = details["placement"]
                if number in self.spots:
                    attempt = self.spots[number][1] + 1
                    turn = self.spots[number][0]
                    if attempt >= TRIES:
                        return False, f"placement {number} tried {TRIES} times", None
                else:
                    attempt, turn = 0, self.turns.get(shown, 0)
                    self.turns[shown] = turn + 1
                self.spots[number] = (turn, attempt)
                self.place(shown, turn, attempt, why=f"placement {number}")
            return True, None, None
        return True, None, None                    # motions, enabling, writing: yes

    def operate(self, stop: threading.Event, *, poll_s: float = 0.2,
                say: Callable[[str], None] = lambda _m: None) -> int:
        """Answer every question of the broker until ``stop`` is set; returns how many."""

        from ora.cell.operator import approval

        answered = 0
        while not stop.is_set():
            for request in approval.pending():
                approved, reason, value = self.answer(request)
                try:
                    approval.answer(request["id"], approved, reason, by=SIM_OPERATOR, value=value)
                except CellError:
                    continue                       # answered or withdrawn meanwhile
                answered += 1
                say(f"{'approved' if approved else 'declined'}: {request.get('title')}"
                    + (f" -> {value}" if value else "") + (f" ({reason})" if reason else ""))
            stop.wait(poll_s)
        return answered
