"""Released point-query argument refusals."""
from typing import Any
from ..errors import ToolRefusal as ToolError

COORDINATES = "opencv_top_left_xy"


def _is_pixel(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(type(item) is int for item in value)


def _outside(field: str, value: Any, width: int, height: int) -> ToolError:
    return ToolError(
        f"{field} {value} is outside the {width}x{height} source image",
        field=field, received=value,
        constraints={"width": width, "height": height, "coordinate_convention": COORDINATES,
                     "rule": "0 <= u < width and 0 <= v < height"},
    )


def batch_point_arguments(
    points: Any, max_candidates: Any, *, width: int, height: int,
    max_points: int = 8, max_per_point: int = 3,
) -> int:
    """Check ``vision_segment``'s ``points_uv`` and ``max_candidates``; return the latter."""

    if not isinstance(points, list) or not 1 <= len(points) <= max_points:
        raise ToolError(
            f"points_uv must contain 1..{max_points} [u,v] points",
            field="points_uv", received=len(points) if isinstance(points, list) else points,
            constraints={"minItems": 1, "maxItems": max_points},
        )
    for index, point in enumerate(points):
        if not _is_pixel(point):
            raise ToolError(
                "each points_uv entry must be one integer [u,v] pixel",
                field=f"points_uv[{index}]", received=point,
                constraints={"items": "integer [u,v]", "coordinate_convention": COORDINATES},
            )
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise _outside(f"points_uv[{index}]", point, width, height)
    if type(max_candidates) is not int or not 1 <= max_candidates <= max_per_point:
        raise ToolError(
            f"max_candidates must be an integer in [1,{max_per_point}]",
            field="max_candidates", received=max_candidates,
            constraints={"minimum": 1, "maximum": max_per_point},
        )
    return max_candidates

