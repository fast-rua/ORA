"""The cell config: one file the cell's owner writes, validated at every start (``ora.cell-config.v1``).

It names the robot, its buses and cameras, where the tool-written calibration lives, the
safety limits the owner wants (which may only narrow the robot model's) and where the
optional perception services run. It never holds a secret (keys stay in ``ora providers``)
or a measured value (those are the calibration's). Example: ``docs/usage/real-cells.md``.

The file is found at ``--cell``, else ``$ORA_CELL``, else ``~/.config/ora/cell.yaml``.

``robot.sim: true`` makes it a simulated cell: the robot's ``sim_cell_adapter`` instead of
its hardware, with its state (the simulated arms and cameras, what the simulation knows
and the tools must find) in ``<config>.sim/`` beside the file, or ``robot.sim: {state: DIR}``.
A simulated cell keeps its own lock, approvals and alerts there (``isolate``), so it never
shares them with a real cell on the same machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import CellError

SCHEMA = "ora.cell-config.v1"
DEFAULT_PATH = Path("~/.config/ora/cell.yaml")
MOUNTS = ("fixed_world", "end_effector")


@dataclass(frozen=True)
class CameraConfig:
    role: str
    serial: str
    mount: str


@dataclass(frozen=True)
class CellConfig:
    path: Path
    cell_id: str
    environment: str
    model: str
    buses: Mapping[str, str]
    cameras: Mapping[str, CameraConfig]
    calibration_path: Path
    python: Path | None = None                  # the interpreter with the robot's drivers
    max_joint_speed_rad_s: float | None = None
    operator_present_required: bool = True
    ignore_processes: tuple[str, ...] = ()
    # The owner's widening of the start-up wrist-table check while the arm chain is being
    # recalibrated: {max_tilt_deg, max_height_gap_m, reason}; reported in every start-up.
    wrist_table_check: Mapping[str, Any] = field(default_factory=dict)
    perception: Mapping[str, str | None] = field(default_factory=dict)
    policy: Mapping[str, Any] = field(default_factory=dict)
    sim_state: Path | None = None               # a simulated cell's state directory; None for a real cell

    @property
    def serials(self) -> dict[str, str]:
        return {role: camera.serial for role, camera in self.cameras.items()}


def default_path() -> Path:
    return Path(os.environ.get("ORA_CELL") or DEFAULT_PATH).expanduser()


def _need(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise CellError(f"{where}: '{key}' is required")
    return mapping[key]


def load(path: str | Path | None = None) -> CellConfig:
    import yaml

    path = Path(path).expanduser() if path else default_path()
    if not path.is_file():
        raise CellError(f"no cell config at {path}: write one (docs/usage/real-cells.md) or pass --cell")
    raw = yaml.safe_load(path.read_text()) or {}
    where = str(path)
    if raw.get("schema") != SCHEMA:
        raise CellError(f"{where}: schema must be {SCHEMA}")
    robot = _need(raw, "robot", where)
    buses = dict(_need(robot, "buses", f"{where} robot"))
    cameras = {}
    for role, spec in (raw.get("cameras") or {}).items():
        mount = _need(spec, "mount", f"{where} cameras.{role}")
        if mount not in MOUNTS:
            raise CellError(f"{where} cameras.{role}: mount must be one of {MOUNTS}")
        cameras[role] = CameraConfig(role=role, serial=str(_need(spec, "serial", f"{where} cameras.{role}")), mount=mount)
    safety = raw.get("safety") or {}
    speed = safety.get("max_joint_speed_rad_s")
    if speed is not None and not (isinstance(speed, (int, float)) and speed > 0):
        raise CellError(f"{where} safety.max_joint_speed_rad_s must be a positive number")
    widened = dict(safety.get("wrist_table_check") or {})
    if widened:
        if not str(widened.get("reason") or "").strip():
            raise CellError(f"{where} safety.wrist_table_check needs a 'reason': it widens a start-up check")
        for key in ("max_tilt_deg", "max_height_gap_m"):
            value = widened.get(key)
            if value is not None and not (isinstance(value, (int, float)) and value > 0):
                raise CellError(f"{where} safety.wrist_table_check.{key} must be a positive number")
        if (widened.get("max_tilt_deg") or 0) > 6.0 or (widened.get("max_height_gap_m") or 0) > 0.05:
            raise CellError(f"{where} safety.wrist_table_check may widen to at most 6 deg and 0.05 m")
    sim = robot.get("sim")
    if isinstance(sim, Mapping):
        sim_state = Path(str(_need(sim, "state", f"{where} robot.sim"))).expanduser()
        if not sim_state.is_absolute():
            sim_state = path.parent / sim_state
    elif sim in (None, False):
        sim_state = None
    elif sim is True:
        sim_state = path.with_suffix(".sim")
    else:
        raise CellError(f"{where} robot.sim must be true, false or {{state: DIR}}")
    config = CellConfig(
        path=path, cell_id=str(_need(raw, "cell_id", where)), environment=str(_need(robot, "environment", f"{where} robot")),
        model=str(_need(robot, "model", f"{where} robot")), buses=buses, cameras=cameras,
        calibration_path=Path(str(_need(raw, "calibration", where))).expanduser(),
        python=Path(str(robot["python"])).expanduser() if robot.get("python") else None,
        max_joint_speed_rad_s=None if speed is None else float(speed),
        operator_present_required=bool(safety.get("operator_present_required", True)),
        ignore_processes=tuple((safety.get("foreign_controllers") or {}).get("ignore") or ()),
        wrist_table_check=widened,
        perception=dict(raw.get("perception") or {}), policy=dict(raw.get("policy") or {}), sim_state=sim_state)
    return config


#: What a simulated cell keeps in its state directory instead of beside the real cell's lock.
ISOLATED = {"ORA_CELL_LOCK": "cell.lock", "ORA_CELL_APPROVALS": "approvals", "ORA_CELL_ALERTS": "alerts.json"}


def isolate(config: CellConfig) -> dict[str, str]:
    """For a simulated cell, point the cell lock, approvals and alerts not set in the
    environment at its state directory; returns what was set. Call it before ``ora.cell.lock``
    is imported (the lock path is read once, at import)."""

    if config.sim_state is None:
        return {}
    config.sim_state.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for name, leaf in ISOLATED.items():
        if not os.environ.get(name):
            os.environ[name] = chosen[name] = str(config.sim_state / leaf)
    return chosen


def check_against(config: CellConfig, adapter: Any) -> list[str]:
    """What the config gets wrong about its robot: unknown arms, cameras or mounts."""

    problems = []
    if set(config.buses) != set(adapter.arms):
        problems.append(f"robot.buses must name exactly the arms {list(adapter.arms)}")
    for role, camera in config.cameras.items():
        if role not in adapter.camera_mounts:
            problems.append(f"camera '{role}' is not one of the robot's cameras {sorted(adapter.camera_mounts)}")
        elif adapter.camera_mounts[role] != camera.mount:
            problems.append(f"camera '{role}' is mounted {adapter.camera_mounts[role]} on this robot, not {camera.mount}")
    return problems
