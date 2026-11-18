"""Pure metric geometry on NumPy arrays: rigid transforms, pinhole projection, depth lifting, point-set shape.

The arithmetic of ``ora/agent/toolkit/lib`` (``geometry``, ``values``) and of ``geometry_query``, written for
the code-mode sandbox, which has NumPy and SciPy but no ORA package. Lengths are metres, angles radians,
quaternions ``[x, y, z, w]``. A transform is a 4x4 ``target <- source`` matrix in the column-vector convention.
A camera follows OpenCV: ``x`` right, ``y`` down, ``z`` forward, ``K`` in native pixels, depth along the optical
axis (not the ray). Nothing here reads a file or knows a frame's name; ``frames.py`` builds on it.

The sandbox's interpreter has no OpenCV, so the two image operations the lib takes from it are written here
with SciPy: 8-connected components (``depth_inliers``) and the elliptical dilation of a mask (``dilate_disc``).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

#: Depth at or below this is no measurement (``geometry_query``'s rule).
MIN_DEPTH_M = 1e-6
#: The largest depth window ``geometry_query`` accepts (``window_px``).
MAX_WINDOW_PX = 15


def rigid(matrix) -> np.ndarray:
    """A finite rigid 4x4 transform from a 4x4 or 3x4 matrix, else ``ValueError`` (``TransformM``'s checks)."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.shape == (3, 4):
        a = np.vstack([a, [0.0, 0.0, 0.0, 1.0]])
    if (a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("a transform must be a finite rigid 4x4 (or 3x4) matrix")
    return a


def invert(matrix) -> np.ndarray:
    """The inverse transform (``source <- target``), as ``observation.load_camera`` computes it."""

    return np.linalg.inv(rigid(matrix))


def transform_points(xyz, matrix) -> np.ndarray:
    """Points ``(3,)`` or ``(N, 3)`` mapped by a ``target <- source`` transform (``geometry.transform_points``)."""

    a = rigid(matrix)
    p = np.asarray(xyz, dtype=np.float64)
    # Elementwise, as the toolkit lib's ``_rigid.apply``: a matrix product's last bit depends on the BLAS kernel.
    flat = p.reshape(-1, 3)
    out = flat[:, 0:1] * a[:3, 0] + flat[:, 1:2] * a[:3, 1] + flat[:, 2:3] * a[:3, 2] + a[:3, 3]
    return out.reshape(p.shape)


def rotate_vectors(v, matrix) -> np.ndarray:
    """Free vectors ``(3,)`` or ``(N, 3)`` mapped by a transform's rotation only."""

    return np.asarray(v, dtype=np.float64) @ rigid(matrix)[:3, :3].T


def unit_quaternion(xyzw) -> np.ndarray:
    """``xyzw`` normalized; a norm outside 0.98-1.02 is refused, as robot inputs are (frames and units)."""

    q = np.asarray(xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(q)) if q.shape == (4,) and np.isfinite(q).all() else 0.0
    if not 0.98 <= norm <= 1.02:
        raise ValueError(f"an orientation must be a unit quaternion [x, y, z, w]; got {q.tolist()}")
    return q / norm


def pose_matrix(xyz, xyzw) -> np.ndarray:
    """The 4x4 ``parent <- pose`` transform of a position (metres) and an ``xyzw`` orientation."""

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(unit_quaternion(xyzw)).as_matrix()
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def transform_pose(xyz, xyzw, matrix) -> tuple[np.ndarray, np.ndarray]:
    """A pose mapped by a ``target <- source`` transform; the quaternion has ``w >= 0`` (``geometry.transform_pose``)."""

    a = rigid(matrix)
    p = a[:3, :3] @ np.asarray(xyz, dtype=np.float64) + a[:3, 3]
    q = (Rotation.from_matrix(a[:3, :3]) * Rotation.from_quat(unit_quaternion(xyzw))).as_quat()
    if q[3] < 0:
        q = -q
    return p, q


def project(xyz_camera, intrinsic) -> np.ndarray:
    """Camera-frame points ``(3,)`` or ``(N, 3)`` to pixels ``[u, v]`` (floats); a point not in front is NaN.

    ``u = fx * x / z + cx``, ``v = fy * y / z + cy``, as ``geometry_query``'s ``world_points_m``.
    """

    k = np.asarray(intrinsic, dtype=np.float64)
    p = np.atleast_2d(np.asarray(xyz_camera, dtype=np.float64))
    z = p[:, 2]
    front = z > 1e-9
    safe = np.where(front, z, 1.0)
    uv = np.stack([k[0, 0] * p[:, 0] / safe + k[0, 2], k[1, 1] * p[:, 1] / safe + k[1, 2]], axis=1)
    uv[~front] = np.nan
    return uv[0] if np.ndim(xyz_camera) == 1 else uv


def back_project(uv, depth_m, intrinsic) -> np.ndarray:
    """Pixels ``[u, v]`` at optical-axis depths (metres) to camera-frame points, as ``geometry_query``'s ``points``."""

    k = np.asarray(intrinsic, dtype=np.float64)
    pixels = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    z = np.broadcast_to(np.asarray(depth_m, dtype=np.float64), pixels.shape[:1])
    xyz = np.stack([(pixels[:, 0] - k[0, 2]) * z / k[0, 0], (pixels[:, 1] - k[1, 2]) * z / k[1, 1], z], axis=1)
    return xyz[0] if np.ndim(uv) == 1 else xyz


def window_depth(depth_m, pixels, window_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The median depth and the count of valid values in a square window around each integer pixel.

    ``depth_m`` is HxW metres; ``pixels`` ``(N, 2)`` integer ``[u, v]``. The window spans ``window_px`` pixels on
    each side, clipped at the image border; values that are not finite or not above ``MIN_DEPTH_M`` are
    skipped, as ``geometry_query`` does. A pixel outside the image, or a window with no valid value, has
    median NaN and count 0. Any number of pixels.
    """

    if isinstance(window_px, bool) or not isinstance(window_px, (int, np.integer)) \
            or not 0 <= window_px <= MAX_WINDOW_PX:
        raise ValueError(f"window_px must be an integer from 0 to {MAX_WINDOW_PX}")
    depth = np.asarray(depth_m)
    if depth.dtype.kind != "f":
        depth = depth.astype(np.float64)
    pixels = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
    height, width = depth.shape
    w = int(window_px)
    # The median in the depth's own precision, as np.median takes it (the mean of the two middle values).
    padded = np.full((height + 2 * w, width + 2 * w), np.nan, dtype=depth.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # comparisons with NaN
        padded[w:w + height, w:w + width] = np.where(np.isfinite(depth) & (depth > MIN_DEPTH_M), depth, np.nan)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    median = np.full(len(pixels), np.nan)
    count = np.zeros(len(pixels), dtype=np.int64)
    offsets = np.arange(2 * w + 1)
    rows_offset, columns_offset = np.meshgrid(offsets, offsets, indexing="ij")
    rows_offset, columns_offset = rows_offset.ravel(), columns_offset.ravel()
    chosen = np.flatnonzero(inside)
    chunk = max(1, 4_000_000 // len(rows_offset))
    for start in range(0, len(chosen), chunk):
        index = chosen[start:start + chunk]
        values = np.sort(padded[pixels[index, 1, None] + rows_offset, pixels[index, 0, None] + columns_offset],
                         axis=1)  # NaN last
        valid = np.isfinite(values).sum(axis=1)
        some = valid > 0
        rows = np.flatnonzero(some)
        low = values[rows, (valid[some] - 1) // 2]
        high = values[rows, valid[some] // 2]
        count[index] = valid
        median[index[some]] = ((low + high) / values.dtype.type(2)).astype(np.float64)
    return median, count


#: 8-connectivity (``cv2.connectedComponents(..., connectivity=8)``).
_EIGHT = np.ones((3, 3), dtype=bool)


def _largest_component(valid) -> tuple[np.ndarray, int]:
    """The pixels of the largest 8-connected component of a boolean image, and how many components there are.

    Components of equal size are told apart as ``cv2.connectedComponents`` numbers them (OpenCV 4): by the first
    2x2 block, scanning block rows then block columns, that holds one of their pixels; the lowest number wins.
    """

    labels, count = ndimage.label(valid, structure=_EIGHT)
    if not count:
        return np.zeros_like(valid), 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    best = np.flatnonzero(sizes == sizes.max()) + 1
    if len(best) > 1:
        rows, columns = np.indices(valid.shape)
        blocks = ndimage.minimum((rows // 2) * valid.shape[1] + columns // 2, labels=labels, index=best)
        best = best[[int(np.argmin(blocks))]]
    return labels == int(best[0]), count


def depth_inliers(mask, depth_m, *, min_depth_m: float = MIN_DEPTH_M, max_depth_m: float = 2.5,
                  minimum_points: int = 16, minimum_half_width_m: float = 0.03,
                  mad_scale: float = 4.0 * 1.4826) -> dict[str, np.ndarray | float | int | None]:
    """The mask's pixels a depth gate keeps, as ``geometry.depth_inliers`` selects them (the old localization recipe).

    The valid pixels of the mask (finite depth in ``(min_depth_m, max_depth_m]``) are cut to their largest
    8-connected component; when it has at least ``minimum_points`` pixels, those within
    ``max(minimum_half_width_m, mad_scale * MAD)`` of its median depth are kept. Returns ``mask`` (HxW bool) and the
    counts and thresholds (``mask_pixels``, ``valid_pixels``, ``components``, ``primary_pixels``, ``inlier_pixels``,
    ``median_depth_m``, ``mad_depth_m``, ``half_width_m``, the parameters and ``connectivity``); too few pixels give
    an empty mask and the counts, and whether a thin or sparse mask measures one object is the caller's to judge
    from them. Depth is used in its own precision, as the lib does.
    """

    selected = np.asarray(mask).astype(bool)
    depth = np.asarray(depth_m)
    if (selected.ndim != 2 or selected.shape != depth.shape or not selected.size or not 0 <= min_depth_m < max_depth_m
            or minimum_points < 1 or minimum_half_width_m < 0 or mad_scale < 0
            or not np.isfinite([max_depth_m, minimum_half_width_m, mad_scale]).all()):
        raise ValueError("invalid mask/depth selection parameters")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # comparisons with NaN
        valid = selected & np.isfinite(depth) & (depth > min_depth_m) & (depth <= max_depth_m)
    primary, components = _largest_component(valid)
    result = np.zeros_like(valid)
    median = mad = half = None
    if primary.sum() >= minimum_points:
        median = float(np.median(depth[primary]))
        mad = float(np.median(np.abs(depth[primary] - median)))
        half = max(minimum_half_width_m, mad_scale * mad)
        result = primary & (np.abs(depth - median) <= half)
    return {"mask": result, "mask_pixels": int(selected.sum()), "valid_pixels": int(valid.sum()),
            "components": components, "primary_pixels": int(primary.sum()), "inlier_pixels": int(result.sum()),
            "median_depth_m": median, "mad_depth_m": mad, "half_width_m": half, "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m, "minimum_points": minimum_points,
            "minimum_half_width_m": minimum_half_width_m, "mad_scale": mad_scale, "connectivity": 8}


def dilate_disc(mask, radius: int) -> np.ndarray:
    """``mask`` (HxW bool) dilated by the elliptical element of size ``2 * radius + 1`` that
    ``cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)`` builds: row ``dy`` of it is a run of half-width
    ``round(radius * sqrt((radius**2 - dy**2) / radius**2))``. Pixels beyond the border count as empty.
    """

    selected = np.asarray(mask).astype(bool)
    if selected.ndim != 2 or isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError("dilate_disc takes an HxW mask and an integer radius of at least 1")
    radius = int(radius)
    out = np.zeros_like(selected)
    rows, columns = np.nonzero(selected)
    if not len(rows):
        return out
    # Only the mask's bounding box, grown by the radius, can change.
    top, bottom = max(0, rows.min() - radius), min(selected.shape[0], rows.max() + radius + 1)
    left, right = max(0, columns.min() - radius), min(selected.shape[1], columns.max() + radius + 1)
    crop = selected[top:bottom, left:right]
    height, width = crop.shape
    sums = np.concatenate([np.zeros((height, 1), dtype=np.int32), np.cumsum(crop, axis=1, dtype=np.int32)], axis=1)
    inverse, across = 1.0 / (radius * radius), np.arange(width)
    reach: dict[int, np.ndarray] = {}  # the crop dilated sideways by a half-width
    grown = np.zeros_like(crop)
    for dy in range(1 - height, height):  # a shift of the whole height or more leaves nothing in the crop
        if abs(dy) > radius:
            continue
        half = int(round(radius * math.sqrt((radius * radius - dy * dy) * inverse)))
        if half not in reach:
            reach[half] = (sums[:, np.minimum(across + half + 1, width)] - sums[:, np.maximum(across - half, 0)]) > 0
        if dy >= 0:
            grown[dy:] |= reach[half][:height - dy]
        else:
            grown[:height + dy] |= reach[half][-dy:]
    out[top:bottom, left:right] = grown
    return out


def mask_to_cloud(depth_m, intrinsic, mask, *, min_depth_m: float = MIN_DEPTH_M,
                  max_depth_m: float = 2.5) -> np.ndarray:
    """The camera-frame points ``(N, 3)`` of a mask's pixels with valid depth, row-major (``geometry.mask_to_cloud``)."""

    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask)
    if selected.shape != depth.shape or not 0 <= min_depth_m < max_depth_m or not np.isfinite(max_depth_m):
        raise ValueError("the mask must have the depth image's shape, and 0 <= min_depth_m < max_depth_m")
    valid = selected.astype(bool) & np.isfinite(depth) & (depth > min_depth_m) & (depth <= max_depth_m)
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    k = np.asarray(intrinsic, dtype=np.float64)
    return np.stack(((columns - k[0, 2]) * z / k[0, 0], (rows - k[1, 2]) * z / k[1, 1], z), axis=1)


def principal_geometry(points, *, lower_percentile: float = 2.5, upper_percentile: float = 97.5,
                       degeneracy_atol_m2: float = 1e-12) -> dict[str, np.ndarray | bool]:
    """Median centre, percentile box and PCA of a point set, as ``geometry.principal_geometry`` computes them.

    Returns ``center_m``, ``lower_m``, ``upper_m`` (per axis of the points' frame), ``axes`` (rows, ordered by
    variance, each signed so its largest component is positive), ``extents_m`` (percentile extent along each
    axis), ``variance_m2`` and ``ambiguous_axes`` (repeated variances: those axes are not a stable orientation).
    A visible point set is not an object's whole shape.
    """

    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not 0 <= lower_percentile < upper_percentile <= 100 \
            or not np.isfinite(degeneracy_atol_m2) or degeneracy_atol_m2 < 0:
        raise ValueError("a nonempty (N, 3) point set and valid percentile and degeneracy limits are required")
    center = np.median(p, axis=0)
    centered = p - center
    values, vectors = _principal_axes(centered.T @ centered / max(1, len(p) - 1))
    axes = vectors.T.copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    lower, upper = np.percentile(p, [lower_percentile, upper_percentile], axis=0)
    a, b = np.percentile(centered @ axes.T, [lower_percentile, upper_percentile], axis=0)
    return {"center_m": center, "lower_m": lower, "upper_m": upper, "axes": axes, "extents_m": b - a,
            "variance_m2": values,
            "ambiguous_axes": bool(np.any(np.abs(np.diff(values)) <= degeneracy_atol_m2))}


#: ``ora/agent/toolkit/lib/_principal_axes.py``, copied for the sandbox: the eigenvectors where the variances
#: differ (eigh for a 2 x 2, closed-form in LAPACK; Jacobi rotations in Python floats for a 3 x 3, whose LAPACK
#: path rounds by BLAS kernel); inside a repeated eigenspace, whose basis eigh picks by BLAS kernel, the world axes
#: projected onto it in ``x, y, z`` order and made orthonormal. Greatest variance first; a repeated variance as the
#: mean of its run.
_REPEATED_RTOL = 1e-9
_KEEP_NORM = 0.5
_SWEEPS = 50


def _principal_axes(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if len(matrix) > 2 and np.isfinite(matrix).all():
        values, vectors = _jacobi(matrix)
    else:
        values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    tolerance = _REPEATED_RTOL * float(np.max(np.abs(values), initial=0.0))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end - 1] - values[end] <= tolerance:
            end += 1
        if end - start > 1:
            values[start:end] = np.mean(values[start:end])
            vectors[:, start:end] = _world_aligned(vectors[:, start:end])
        start = end
    return values, vectors


def _jacobi(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = [[float(value) for value in row] for row in matrix]
    n = len(a)
    v = [[1.0 if row == column else 0.0 for column in range(n)] for row in range(n)]
    for sweep in range(_SWEEPS):
        if not any(a[p][q] for p in range(n) for q in range(p + 1, n)):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                g = 100.0 * abs(apq)
                if sweep > 3 and abs(a[p][p]) + g == abs(a[p][p]) and abs(a[q][q]) + g == abs(a[q][q]):
                    a[p][q] = a[q][p] = 0.0
                    continue
                h = a[q][q] - a[p][p]
                if abs(h) + g == abs(h):
                    t = apq / h
                else:
                    theta = 0.5 * h / apq
                    t = 1.0 / (abs(theta) + math.sqrt(1.0 + theta * theta))
                    if theta < 0.0:
                        t = -t
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                a[p][p] -= t * apq
                a[q][q] += t * apq
                a[p][q] = a[q][p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = a[r][p], a[r][q]
                        a[r][p] = a[p][r] = c * arp - s * arq
                        a[r][q] = a[q][r] = s * arp + c * arq
                for row in v:
                    vp, vq = row[p], row[q]
                    row[p], row[q] = c * vp - s * vq, s * vp + c * vq
    order = sorted(range(n), key=lambda index: a[index][index])
    return np.array([a[i][i] for i in order]), np.array([[row[i] for i in order] for row in v])


def _world_aligned(basis: np.ndarray) -> np.ndarray:
    projector = basis @ basis.T
    chosen: list[np.ndarray] = []
    for world_axis in np.eye(len(basis)):
        axis = projector @ world_axis
        for previous in chosen:
            axis = axis - np.sum(previous * axis) * previous
        norm = math.sqrt(np.sum(axis * axis))
        if norm >= _KEEP_NORM:
            chosen.append(axis / norm)
        if len(chosen) == basis.shape[1]:
            break
    return np.stack(chosen, axis=1)


__all__ = ["MAX_WINDOW_PX", "MIN_DEPTH_M", "back_project", "depth_inliers", "dilate_disc", "invert", "mask_to_cloud",
           "pose_matrix", "principal_geometry", "project", "rigid", "rotate_vectors", "transform_points",
           "transform_pose", "unit_quaternion", "window_depth"]
