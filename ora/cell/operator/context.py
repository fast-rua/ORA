"""One cell as the operator tools see it: its config, its robot's adapter, its calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import calibration as cell_calibration
from .. import config as cell_config
from ..adapter import ArmReading, Bus, Camera, CameraDriver, CellAdapter, bus_of, camera_driver_of, load_adapter
from ..errors import CellError


@dataclass
class Cell:
    config: cell_config.CellConfig
    adapter: CellAdapter

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Cell":
        config = cell_config.load(path)
        adapter = load_adapter(config)
        problems = cell_config.check_against(config, adapter)
        if problems:
            raise CellError(f"{config.path}: " + "; ".join(problems))
        return cls(config, adapter)

    # ------------------------------------------------------------ calibration
    def calibration(self) -> dict[str, Any]:
        return cell_calibration.load(self.config.calibration_path)

    def save_calibration(self, value: dict[str, Any]) -> None:
        cell_calibration.save(self.config.calibration_path, value)

    def gripper_limits(self) -> dict[str, list[float]]:
        return dict(self.calibration().get("gripper_limits") or {})

    def rest(self) -> dict[str, np.ndarray]:
        rest = self.calibration().get("rest_reference")
        if not rest:
            raise CellError("no rest pose recorded: `ora cell record-rest` with the arms at rest")
        return {arm: np.asarray(rest[arm], float) for arm in self.adapter.arms}

    def table(self) -> dict[str, Any] | None:
        return cell_calibration.table_for_model(self.calibration())

    def kinematics(self, *, with_table: bool = False):
        kinematics = self.adapter.kinematics(self.table() if with_table else None)
        return cell_calibration.apply(kinematics, self.calibration())

    # ------------------------------------------------------------ buses and arms
    @property
    def channels(self) -> list[str]:
        return list(self.adapter.buses.values())

    @property
    def bus(self) -> Bus:
        return bus_of(self.adapter)

    def quiet(self, listen_s: float = 0.3) -> None:
        """Refuse unless no other program reads or drives the arms' buses."""

        for channel in self.channels:
            if self.bus.receivers(channel):
                raise CellError(f"{channel} has a reader: another program holds the arms")
            heard = self.bus.traffic(channel, listen_s)
            if heard:
                raise CellError(f"{channel} carried {heard} frames: another program drives the arms")

    def read_arms(self) -> dict[str, ArmReading]:
        """Both arms read passively once; the caller holds the cell lock."""

        self.quiet()
        reader = self.adapter.open_reader(self.gripper_limits())
        try:
            return reader.read()
        finally:
            reader.close()

    # ------------------------------------------------------------ cameras
    @property
    def camera_driver(self) -> CameraDriver:
        return camera_driver_of(self.adapter)

    def open_camera(self, role: str) -> Camera:
        spec = self.config.cameras[role]
        return self.camera_driver.open(spec.role, spec.serial, spec.mount)

    def fixed_cameras(self) -> list[str]:
        return [role for role, camera in self.config.cameras.items() if camera.mount == "fixed_world"]
