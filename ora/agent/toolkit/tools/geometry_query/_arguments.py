"""Public geometry argument checks, preserving the released refusal fields."""
import math
from typing import Any, Mapping, Sequence, Collection
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

def region_box(box: Any, *, width: int, height: int) -> list[int]:
    """``geometry_query``'s ``region``: ordered, integer and inside the image."""

    if not isinstance(box, list) or len(box) != 4 or any(type(item) is not int for item in box):
        raise ToolError(
            "region must be four integers [x0,y0,x1,y1]", field="region", received=box,
            constraints={"items": "integer", "minItems": 4, "maxItems": 4},
        )
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise ToolError(
            "region must satisfy x0 < x1 and y0 < y1", field="region", received=box,
            constraints={"rule": "x0 < x1 and y0 < y1"},
        )
    if not (0 <= x0 and x1 <= width and 0 <= y0 and y1 <= height):
        raise ToolError(
            f"region {box} is outside the {width}x{height} source image",
            field="region", received=box,
            constraints={"width": width, "height": height, "coordinate_convention": COORDINATES,
                         "rule": "0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height"},
        )
    return box

def query_pixel(point: Any, index: int, *, width: int, height: int) -> list[int]:
    """One ``geometry_query`` ``points`` entry."""

    field = f"points[{index}]"
    if not _is_pixel(point):
        raise ToolError(
            "each point must be one integer [u,v] pixel", field=field, received=point,
            constraints={"items": "integer [u,v]", "coordinate_convention": COORDINATES},
        )
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        raise _outside(field, point, width, height)
    return point

def finite_camera_point(camera_point: Any, raw_point: Any, index: int) -> None:
    """Refuse a world point whose camera-frame coordinates overflow."""

    if not all(math.isfinite(float(value)) for value in camera_point[:3]):
        raise ToolError(
            "world_points_m entry has no finite camera-frame coordinates",
            field=f"world_points_m[{index}]", received=raw_point,
            constraints={"finite_camera_xyz_m": True},
        )

def finite_pixel(u: float, v: float) -> bool:
    """Whether a projection can be rounded to a pixel at all."""

    return math.isfinite(u) and math.isfinite(v)

def require_exactly_one_of(
    arguments: Mapping[str, Any], names: Sequence[str], *, message: str,
) -> str:
    """Return the one present name; a present key counts even when null."""

    present = [name for name in names if name in arguments]
    if len(present) != 1:
        raise _refuse(arguments, message, {
            "exactly_one_of": list(names), "received": present,
        })
    return present[0]

def require_known_arguments(
    arguments: Mapping[str, Any], *, allowed: Collection[str],
    required: Collection[str], message: str,
) -> None:
    """Refuse unknown or missing argument names, naming both sets."""

    unknown = sorted(set(arguments) - set(allowed))
    missing = sorted(set(required) - set(arguments))
    if unknown or missing:
        raise _refuse(arguments, message, {
            "allowed": sorted(allowed), "required": sorted(required),
            "unknown": unknown, "missing": missing,
        })

def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    """``value`` as ``length`` finite floats, or the refusal naming ``label``."""

    if not isinstance(value, list) or len(value) != length:
        raise ToolError(f"{label} must contain {length} numbers")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ToolError(f"{label} must contain finite numbers")
        output.append(float(item))
    return output


def _refuse(
    arguments: Mapping[str, Any], message: str, constraints: Mapping[str, Any],
) -> ToolError:
    return ToolError(
        message, code="invalid_arguments", received=sorted(str(name) for name in arguments),
        constraints=constraints,
    )
