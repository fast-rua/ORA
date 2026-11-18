"""Perception calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Perception:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def segment(self, *, rgb: dict, depth: dict, point_uv: tuple[int, int]) -> Response:
        return self._backend.segment(rgb=rgb, depth=depth, point_uv=point_uv)

    def segment_points(self, *, rgb: dict, depth: dict, points_uv: tuple[tuple[int, int], ...]) -> Response:
        return self._backend.segment_points(rgb=rgb, depth=depth, points_uv=points_uv)

    def ground(self, *, rgb: dict, text: str, max_candidates: int = 3) -> Response:
        return self._backend.ground(rgb=rgb, text=text, max_candidates=max_candidates)

    def grasp_candidates(self, *, scene: dict, target: dict, camera_to_world: list,
                         workspace_bounds_m: dict, max_candidates: int = 32,
                         forward_passes: int = 1) -> Response:
        return self._backend.grasp_candidates(scene=scene, target=target, camera_to_world=camera_to_world, workspace_bounds_m=workspace_bounds_m, max_candidates=max_candidates, forward_passes=forward_passes)
