"""Released local-depth acceptance and camera projection, no I/O."""
import math
from typing import Any, Mapping
from ..errors import ToolRefusal as ToolError
MARK_POINT_DEPTH_WINDOW_RADIUS = 2
UNMEASURED_METHOD = "not_measured"

def _quality(
    sink: dict[str, Any] | None, *, status: str, method: str, **fields: Any,
) -> None:
    """Record what the estimator learned, including on the paths that fail.

    The single-pixel form discards this on the error path, so a rejected pixel
    tells the policy only its error code.  For a batch that is not enough: the
    remedy for ``depth_discontinuity`` (aim further inside the surface) is not
    the remedy for ``invalid_depth`` (the pixel sees the far plane), and a
    model correcting seven pixels at once needs to tell them apart.  The sink
    is filled in place so a raising estimator can still report what it saw.
    """

    if sink is None:
        return
    sink.clear()
    sink.update({"quality_status": status, "quality_method": method})
    sink.update(fields)

def robust_depth_measurement(
    depth: Any, *, source_x: int, source_y: int,
    calibration: Mapping[str, Any],
    quality_sink: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Measure one local surface or fail closed at depth boundaries.

    A clicked integer pixel is not sufficient evidence near silhouettes.  This
    estimator requires a locally supported cluster around the clicked depth,
    rejects mixed foreground/background windows, and rejects camera far-plane
    values.  It intentionally prefers asking the policy to click a safer pixel
    over returning a precise-looking but arbitrary world point.

    ``quality_sink``, when given, receives the diagnostics gathered so far even
    when this raises.  Passing nothing keeps the historical behavior exactly.
    """

    import numpy as np

    _quality(quality_sink, status="unmeasured", method=UNMEASURED_METHOD)
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or not (
        0 <= source_y < values.shape[0] and 0 <= source_x < values.shape[1]
    ):
        raise ToolError("selected RGB-D pixel is outside the depth array")
    bounds = calibration.get("valid_depth_range_m", {})
    near = float(bounds.get("near", 0.0)) if isinstance(bounds, Mapping) else 0.0
    far_raw = bounds.get("far") if isinstance(bounds, Mapping) else None
    far = float(far_raw) if isinstance(far_raw, (int, float)) else math.inf
    center = float(values[source_y, source_x])
    far_limit = far * 0.98 if math.isfinite(far) else math.inf
    radius = MARK_POINT_DEPTH_WINDOW_RADIUS
    if not math.isfinite(center) or center <= max(near, 1e-6) or center >= far_limit:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            quality_window_radius_px=radius,
            quality_depth_m=center if math.isfinite(center) else None,
        )
        raise ToolError(
            "selected RGB-D pixel is invalid or at the camera far plane",
            code="invalid_depth",
        )
    y0, y1 = max(0, source_y-radius), min(values.shape[0], source_y+radius+1)
    x0, x1 = max(0, source_x-radius), min(values.shape[1], source_x+radius+1)
    window = values[y0:y1, x0:x1].reshape(-1)
    valid = window[
        np.isfinite(window)
        & (window > max(near, 1e-6))
        & (window < far_limit)
    ]
    partial = {
        "quality_window_radius_px": radius,
        "quality_depth_m": center,
        "quality_valid_samples": int(valid.size),
    }
    if valid.size < 3:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            **partial,
        )
        raise ToolError(
            "selected RGB-D neighborhood has insufficient valid depth support",
            code="insufficient_depth_support",
        )
    tolerance = max(0.015, 0.03 * center)
    compatible = valid[np.abs(valid - center) <= tolerance]
    incompatible = valid[np.abs(valid - center) > tolerance]
    partial["quality_supporting_samples"] = int(compatible.size)
    partial["quality_rejected_samples"] = int(incompatible.size)
    if compatible.size < 3:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            **partial,
        )
        raise ToolError(
            "selected RGB-D pixel is not supported by a local surface",
            code="insufficient_depth_support",
        )
    if incompatible.size >= max(3, int(math.ceil(valid.size * 0.20))):
        separation = abs(float(np.median(incompatible)) - float(np.median(compatible)))
        if separation > max(0.05, 0.08 * center):
            _quality(
                quality_sink, status="partial",
                method="local_supported_depth_v1", **partial,
            )
            raise ToolError(
                "selected RGB-D neighborhood crosses a depth discontinuity; "
                "click farther inside one visible surface",
                code="depth_discontinuity",
            )
    measured = float(np.median(compatible))
    deviations = np.abs(compatible - measured)
    quality = {
        "method": "local_supported_depth_v1",
        "window_radius_px": radius,
        "valid_samples": int(valid.size),
        "supporting_samples": int(compatible.size),
        "rejected_samples": int(incompatible.size),
        "depth_m": measured,
        "max_support_deviation_m": float(np.max(deviations)),
    }
    _quality(
        quality_sink, status="measured", method="local_supported_depth_v1",
        quality_window_radius_px=radius,
        quality_depth_m=measured,
        quality_valid_samples=int(valid.size),
        quality_supporting_samples=int(compatible.size),
        quality_rejected_samples=int(incompatible.size),
        quality_max_support_deviation_m=float(np.max(deviations)),
    )
    return measured, quality

def opencv_camera_point_to_world(
    camera_point_xyzw: Any, *, calibration: Mapping[str, Any],
) -> list[float]:
    """Transform one OpenCV camera point through a validated world extrinsic.

    OpenCV commonly serializes a world-to-camera extrinsic as ``[R|t]`` with
    shape 3x4.  A homogeneous 4x4 matrix is equally valid.  Keeping this
    normalization at the RGB-D measurement boundary prevents callers from
    treating the standard 3x4 representation as directly invertible.
    """

    import numpy as np

    intrinsic = np.asarray(calibration.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(calibration.get("extrinsic_cv"), dtype=np.float64)
    point = np.asarray(camera_point_xyzw, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.all(np.isfinite(intrinsic))
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ToolError(
            "camera intrinsic_cv must be a finite 3x3 OpenCV matrix with "
            "positive focal lengths",
            code="invalid_camera_calibration",
        )
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, [0.0, 0.0, 0.0, 1.0]))
    if extrinsic.shape != (4, 4) or not np.all(np.isfinite(extrinsic)):
        raise ToolError(
            "camera extrinsic_cv must be a finite OpenCV 3x4 [R|t] or "
            "homogeneous 4x4 matrix",
            code="invalid_camera_calibration",
        )
    if point.shape != (4,) or not np.all(np.isfinite(point)):
        raise ToolError(
            "camera point must be one finite homogeneous XYZ1 vector",
            code="invalid_camera_calibration",
        )
    try:
        world = np.linalg.solve(extrinsic, point)
    except np.linalg.LinAlgError as exc:
        raise ToolError(
            "camera extrinsic_cv is singular",
            code="invalid_camera_calibration",
        ) from exc
    if not np.all(np.isfinite(world[:3])) or abs(float(world[3])) < 1e-12:
        raise ToolError(
            "camera calibration produced an invalid homogeneous world point",
            code="invalid_camera_calibration",
        )
    return (world[:3] / world[3]).tolist()

