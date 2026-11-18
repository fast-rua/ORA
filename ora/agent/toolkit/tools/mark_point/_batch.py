"""Released batch validation and public entry projection, with explicit point records."""
from typing import Any, Mapping, Sequence
from ..errors import ToolRefusal as ToolError
MARK_POINT_MAX_BATCH = 8
_MAX_POINT_ID_CHARS = 64
MAX_REPORTED_VIEWS = 8
UNMEASURED_METHOD = "not_measured"
POINTCLOUD_VIEWS = ("pointcloud_top", "pointcloud_front", "pointcloud_side")

def _reject(
    message: str, *, field: str | None = None, received: Any = None,
    **constraints: Any,
) -> ToolError:
    """A model-correctable call error, in the shared structured shape.

    ``field`` and ``received`` go to their own slots rather than into
    ``constraints``: the projection layer carries all three to the provider,
    and a caller reading ``error.field`` should not have to know that this tool
    hid it one level down.
    """

    return ToolError(
        message, code="invalid_arguments", retryable=True,
        field=field, received=received, constraints=constraints or None,
    )

def _batch_arguments(
    arguments: Mapping[str, Any],
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """Validate the whole batch before measuring any of it.

    A malformed *call* is rejected as a call; a pixel that cannot be measured
    is not.  The difference matters: the model can only fix a length mismatch
    or a duplicate name by rewriting the call, whereas it fixes a bad pixel by
    re-aiming that one entry.
    """

    if set(arguments) != {"view", "point_ids", "points_uv"}:
        raise _reject(
            "batched mark_point takes exactly view, point_ids and points_uv",
            accepted_fields=["view", "point_ids", "points_uv"],
            received=sorted(arguments),
        )
    view = arguments["view"]
    if not isinstance(view, str) or not view:
        raise _reject("view must be a non-empty view name", field="view")
    if view in POINTCLOUD_VIEWS:
        raise _reject(
            "a point-cloud view accumulates axis constraints across views, so "
            "it stays on the single-pixel form; batch a camera view instead",
            field="view", received=view,
        )
    raw_ids, raw_uv = arguments["point_ids"], arguments["points_uv"]
    for field, value in (("point_ids", raw_ids), ("points_uv", raw_uv)):
        if not isinstance(value, list) or not value:
            raise _reject(f"{field} must be a non-empty array", field=field)
        if len(value) > MARK_POINT_MAX_BATCH:
            raise _reject(
                f"{field} accepts at most {MARK_POINT_MAX_BATCH} entries, one "
                "per distinguishable overlay colour",
                field=field, received=len(value), maximum=MARK_POINT_MAX_BATCH,
            )
    if len(raw_ids) != len(raw_uv):
        raise _reject(
            "point_ids and points_uv must have the same length: entry i names "
            "the point measured at pixel i",
            point_ids=len(raw_ids), points_uv=len(raw_uv),
        )
    point_ids: list[str] = []
    for index, value in enumerate(raw_ids):
        if (
            not isinstance(value, str) or not value
            or len(value) > _MAX_POINT_ID_CHARS
        ):
            raise _reject(
                "each point_id must be a name of 1 to "
                f"{_MAX_POINT_ID_CHARS} characters",
                field=f"point_ids[{index}]", received=value,
                minLength=1, maxLength=_MAX_POINT_ID_CHARS,
            )
        point_ids.append(value)
    if len(set(point_ids)) != len(point_ids):
        repeat = next(index for index, value in enumerate(point_ids) if value in point_ids[:index])
        raise _reject(
            f"point_ids must be distinct within one call: {point_ids[repeat]!r} "
            f"repeats at indices {point_ids.index(point_ids[repeat])} and {repeat}",
            field=f"point_ids[{repeat}]", received=point_ids[repeat],
            first_index=point_ids.index(point_ids[repeat]),
        )
    points_uv: list[tuple[int, int]] = []
    for index, value in enumerate(raw_uv):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        ):
            raise _reject(
                "each pixel must be [u,v] with non-negative integers in the "
                "selected view's own top-left OpenCV frame",
                field=f"points_uv[{index}]",
            )
        points_uv.append((int(value[0]), int(value[1])))
    return view, point_ids, points_uv

def _entry(
    *, source_index: int, point_id: str, point_uv: tuple[int, int],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """One uniformly shaped result entry, so every key is always readable."""

    entry = {
        "source_index": source_index,
        "point_id": point_id,
        "point_uv": [point_uv[0], point_uv[1]],
        "status": "failed",
        "error_code": None,
        "error_message": None,
        "xyz_m": None,
        "views": [],
        "next_views": [],
        "colour_rgb": None,
        "verification_distance_m": None,
        "quality_method": UNMEASURED_METHOD,
        "quality_status": "unmeasured",
        "quality_depth_m": None,
        "quality_window_radius_px": None,
        "quality_valid_samples": None,
        "quality_supporting_samples": None,
        "quality_rejected_samples": None,
        "quality_max_support_deviation_m": None,
        "action_workspace_inside": None,
        "action_workspace_use": None,
    }
    entry.update({
        key: value for key, value in quality.items() if key in entry
    })
    return entry

def _distinct_views(values: Sequence[Any]) -> list[str]:
    """Report which views measured this name, not how many times each did.

    ``sources`` grows by one entry per measurement, so a name re-verified
    through a long episode would otherwise return an unbounded array of
    repeats. The distinct names are the decision-relevant part, and the tail is
    capped so the entry cannot drift over its declared bound.
    """

    seen: list[str] = []
    for value in values:
        name = str(value)
        if name not in seen:
            seen.append(name)
    return seen[-MAX_REPORTED_VIEWS:]

def _attach_stored_evidence(
    points: Mapping[str, Any], marks: Sequence[dict[str, Any]],
) -> None:
    """Fill in what the stored record knows, reading the point file once.

    Two entries need it. A verified entry knows it agreed with a stored
    measurement but not by how much. And re-marking the exact pixel of a point
    already solved in this observation short-circuits before the estimator
    runs, so the entry is genuinely solved while the estimator reported
    nothing: reporting "solved" beside "unmeasured" would be a contradiction,
    when the honest answer is the quality of the measurement being reused.
    """

    reusing = [
        mark for mark in marks
        if mark["status"] in {"solved", "verified"}
        and mark["quality_status"] == "unmeasured"
    ]
    verified = [mark for mark in marks if mark["status"] == "verified"]
    if not reusing and not verified:
        return

    for mark in verified:
        record = points.get(mark["point_id"])
        verification = record.get("last_verification") if isinstance(
            record, Mapping,
        ) else None
        if not isinstance(verification, Mapping):
            continue
        for key in ("distance_m", "axis_max_error_m"):
            value = verification.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mark["verification_distance_m"] = float(value)
                break

    for mark in reusing:
        record = points.get(mark["point_id"])
        sources = record.get("sources") if isinstance(record, Mapping) else None
        quality = next(
            (
                source["measurement_quality"]
                for source in reversed(sources or [])
                if isinstance(source, Mapping)
                and isinstance(source.get("measurement_quality"), Mapping)
            ),
            None,
        )
        if quality is None:
            continue
        mark.update({
            "quality_status": "measured",
            "quality_method": str(quality.get("method", UNMEASURED_METHOD)),
            "quality_depth_m": quality.get("depth_m"),
            "quality_window_radius_px": quality.get("window_radius_px"),
            "quality_valid_samples": quality.get("valid_samples"),
            "quality_supporting_samples": quality.get("supporting_samples"),
            "quality_rejected_samples": quality.get("rejected_samples"),
            "quality_max_support_deviation_m": quality.get(
                "max_support_deviation_m",
            ),
        })

