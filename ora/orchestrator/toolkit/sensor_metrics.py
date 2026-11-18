"""Diagnostic sensor differences; these numbers never grant simulator authority."""

from io import BytesIO
import math

import numpy as np
from PIL import Image

from ora.agent.toolkit.interface.types import decode


def rgb(left, right):
    """PSNR over RGB channels; a pixel differs if any channel exceeds 16/255."""
    with Image.open(BytesIO(left)) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(BytesIO(right)) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return dict(
            status="shape_mismatch",
            original_shape=list(a.shape),
            fresh_shape=list(b.shape),
        )
    delta = np.abs(a - b)
    mse = float(np.mean(delta**2))
    return dict(
        status="ok",
        pixel_count=int(a.shape[0] * a.shape[1]),
        mse=mse,
        psnr_db=None if mse == 0 else 10 * math.log10(255**2 / mse),
        psnr_infinite=mse == 0,
        max_absolute_difference=float(delta.max()),
        fraction_pixels_gt_16=float(np.mean(np.any(delta > 16, axis=-1))),
    )


def depth(left, right):
    """Absolute metre differences on finite pairs; invalid pairs stay explicit."""
    a = np.load(BytesIO(left), allow_pickle=False).astype(np.float64)
    b = np.load(BytesIO(right), allow_pickle=False).astype(np.float64)
    if a.shape != b.shape:
        return dict(
            status="shape_mismatch",
            original_shape=list(a.shape),
            fresh_shape=list(b.shape),
        )
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    valid = finite_a & finite_b
    delta = np.abs(a[valid] - b[valid])
    return dict(
        status="ok",
        pixel_count=int(a.size),
        finite_pair_count=int(valid.sum()),
        original_nonfinite_count=int((~finite_a).sum()),
        fresh_nonfinite_count=int((~finite_b).sum()),
        finite_mask_difference_count=int((finite_a != finite_b).sum()),
        absolute_difference_m={
            name: float(np.quantile(delta, q)) if delta.size else None
            for name, q in (
                ("p50", 0.5),
                ("p90", 0.9),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1.0),
            )
        },
    )


def calibration(left, right):
    """Per-field maximum numeric delta and raw renderer reference-time counters."""
    a, b = decode(left), decode(right)
    fields, changed = {}, []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                child = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in x or key not in y:
                    changed.append(child)
                else:
                    walk(x[key], y[key], child)
        elif (
            isinstance(x, (int, float, list))
            and isinstance(y, (int, float, list))
            and not isinstance(x, bool)
            and not isinstance(y, bool)
        ):
            try:
                first, second = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                if (
                    first.shape != second.shape
                    or not np.isfinite(first).all()
                    or not np.isfinite(second).all()
                ):
                    changed.append(path)
                    return
                maximum = float(np.max(np.abs(first - second))) if first.size else 0.0
                fields[path] = maximum
                if maximum:
                    changed.append(path)
            except (TypeError, ValueError):
                if x != y:
                    changed.append(path)
        elif x != y:
            changed.append(path)

    walk(a, b, "")

    def rendering(value):
        renderer = value.get("renderer_calibration") or {}
        return dict(
            reference_time=renderer.get("reference_time"),
            simulation_time_s=renderer.get("simulation_time_s"),
        )

    return dict(
        status="ok",
        max_absolute_difference_by_field=fields,
        changed_fields=changed,
        rendering=dict(original=rendering(a), fresh=rendering(b)),
    )


def measure(role, left, right):
    if role == "rgb_lossless_png":
        return rgb(left, right)
    if role == "depth_numpy_m_f32":
        return depth(left, right)
    if role == "calibration":
        return calibration(left, right)
    return dict(status="unsupported_role")
