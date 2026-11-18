"""The point-segmentation and text-grounding backend protocols the perception tools
call. The resident SAM3 client (``model_services.clients.Sam3ApiSegmenter``) and
the deterministic ``fallback_segmenter`` implement them.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class PointSegmentationBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]: ...


class BatchPointSegmentationBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def segment_points(
        self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


class TextGroundingBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def ground(
        self, *, rgb_u8: Any, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...
