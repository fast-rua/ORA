"""Snapshot-only replay cannot query a worker planner, even with recorded answers."""
from dataclasses import dataclass
from ..types import SimulatorRequired


@dataclass(frozen=True, slots=True)
class Planner:
    def probe_motion(self, *, request: dict):
        raise SimulatorRequired('需要仿真器: probe_motion')

    def align_plan(self, *, request: dict):
        raise SimulatorRequired('需要仿真器: align_plan')

    def reach_check(self, *, request: dict):
        raise SimulatorRequired('需要仿真器: reach_check')

    def kinematics(self, *, request: dict):
        raise SimulatorRequired('需要仿真器: kinematics')
