"""Planner calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Planner:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def probe_motion(self, *, request: dict) -> Response:
        return self._backend.probe_motion(request=request)

    def align_plan(self, *, request: dict) -> Response:
        return self._backend.align_plan(request=request)

    def reach_check(self, *, request: dict) -> Response:
        return self._backend.reach_check(request=request)

    def kinematics(self, *, request: dict) -> Response:
        return self._backend.kinematics(request=request)
