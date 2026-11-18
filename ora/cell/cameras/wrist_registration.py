"""A fixed camera's pose from what the wrist cameras see of the same table: provisional.

Used when the fixed camera sees too little of the arms for ``register_fixed_camera``
(the arms at rest sit at the edge of both fixed views on this cell). The wrist cameras
are placed by the encoders and the CAD chain, so their points are already in
the cell's world frame; a fixed camera is placed by matching its view to theirs:

1. The table: the plane, in each view, that holds the most whitish points (the playing
   surface). It levels the fixed camera and sets its height; the wrist views give that
   height in the world.
2. Heading and position: both views are drawn from above as colour maps (surface, red,
   blue, dark, wood) and every heading, 1 deg apart, is correlated over all shifts by
   FFT; the best few are refined at 0.1 deg. The marks on an air-hockey table repeat
   under a half turn, so the loose things on it (mallets, puck), the frame and the arms
   decide between the two; when they do not, the result says so.

The accuracy is the wrist cameras' (their CAD extrinsic leaves a few degrees of tilt
unconstrained), so the result is marked provisional; registering the arms themselves
(``geometry.register_fixed_camera``) replaces it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..geometry import _level, _rz, fit_plane, transform

CELL_M = 0.01
FINE_CELL_M = 0.005
GRID = 256
BAND_M = (-0.03, 0.25)          # around the table top: the surface and what stands on it
COARSE_STEP_DEG = 1.0
FINE_SPAN_DEG, FINE_STEP_DEG = 1.5, 0.1
CANDIDATES = 6
AMBIGUOUS_RATIO = 0.92
CHANNELS = ("surface", "red", "blue", "dark", "wood")


def coloured_points(depth: np.ndarray, rgb: np.ndarray, intrinsic: Sequence[Sequence[float]], *, near: float,
                    far: float, stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(intrinsic, float)
    sampled = np.asarray(depth, float)[::stride, ::stride]
    v, u = np.nonzero((sampled > near) & (sampled < far))
    z = sampled[v, u]
    u, v = u * stride, v * stride
    return np.c_[(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], np.asarray(rgb)[v, u]


def colour_classes(rgb: np.ndarray) -> np.ndarray:
    """(n, 5) membership in surface, red, blue, dark, wood."""

    import cv2

    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return np.stack([(s < 45) & (v > 160),
                     ((h < 9) | (h > 165)) & (s > 90) & (v > 60),
                     (h > 95) & (h < 130) & (s > 70) & (v > 40),
                     v < 55,
                     (h >= 9) & (h <= 25) & (s > 60) & (v > 90)], axis=1).astype(np.float32)


def surface_plane(points: np.ndarray, classes: np.ndarray, *, tries: int = 4) -> tuple[np.ndarray, float, np.ndarray]:
    """The plane, among the few largest, with the largest share of whitish points."""

    remaining = np.ones(len(points), bool)
    best = None
    for attempt in range(tries):
        index = np.flatnonzero(remaining)
        if len(index) < 500:
            break
        normal, offset, inliers = fit_plane(points[index][np.random.default_rng(attempt).permutation(len(index))[:20000]],
                                            seed=attempt)
        members = np.abs(points @ normal - offset) < 0.008
        white = float(classes[members, 0].mean()) if members.any() else 0.0
        score = white * members.sum()
        if best is None or score > best[0]:
            best = (score, normal, offset, members)
        remaining &= ~members
    _, normal, offset, members = best
    return normal, offset, members


def _maps(points_xy: np.ndarray, classes: np.ndarray, origin: np.ndarray, cell: float) -> np.ndarray:
    import cv2

    ij = np.floor((points_xy - origin) / cell).astype(int)
    keep = (ij >= 0).all(1) & (ij < GRID).all(1)
    maps = np.zeros((len(CHANNELS), GRID, GRID), np.float32)
    for c in range(len(CHANNELS)):
        np.add.at(maps[c], (ij[keep, 1], ij[keep, 0]), classes[keep, c])
        maps[c] = cv2.GaussianBlur(np.minimum(maps[c], 3.0), (0, 0), 1.5)
    return maps


def _correlate(reference: np.ndarray, scene: np.ndarray, reference_fft: np.ndarray) -> tuple[float, int, int]:
    total = np.zeros((GRID, GRID))
    for c in range(len(CHANNELS)):
        energy = np.sqrt((scene[c] ** 2).sum() * (reference[c] ** 2).sum())
        if energy > 0:
            total += np.real(np.fft.ifft2(reference_fft[c] * np.conj(np.fft.fft2(scene[c])))) / energy
    peak = np.unravel_index(int(np.argmax(total)), total.shape)
    sy, sx = (int(p) if p < GRID // 2 else int(p) - GRID for p in peak)
    return float(total.max()), sx, sy


def _search(reference_xy, reference_classes, scene_xy, scene_classes, yaws, cell):
    centre = reference_xy.mean(0)
    origin = centre - GRID * cell / 2
    reference = _maps(reference_xy, reference_classes, origin, cell)
    reference_fft = np.fft.fft2(reference)
    out = []
    for yaw in yaws:
        rotated = scene_xy @ _rz(yaw)[:2, :2].T
        shift = centre - rotated.mean(0)
        score, sx, sy = _correlate(reference, _maps(rotated + shift, scene_classes, origin, cell), reference_fft)
        out.append((score, float(yaw), shift + np.array([sx, sy]) * cell))
    return out


def register_to_wrists(fixed_points: np.ndarray, fixed_rgb: np.ndarray, reference_world: np.ndarray,
                       reference_rgb: np.ndarray, *, prior=None) -> dict[str, Any]:
    """cam2world of a fixed camera from its coloured points (camera frame) and the wrists'
    coloured points (world). ``prior(cam2world) -> bool`` drops implausible camera poses."""

    fixed_classes, reference_classes = colour_classes(fixed_rgb), colour_classes(reference_rgb)
    ref_normal, ref_offset, ref_members = surface_plane(reference_world, reference_classes)
    if ref_normal[2] < 0:
        ref_normal, ref_offset = -ref_normal, -ref_offset
    tilt = float(np.degrees(np.arccos(min(1.0, ref_normal[2]))))
    height = float(np.median(reference_world[ref_members, 2]))
    normal, offset, members = surface_plane(fixed_points, fixed_classes)
    if offset > 0:
        normal, offset = -normal, -offset
    level = _level(normal)
    levelled = fixed_points @ level.T
    levelled[:, 2] += height - offset
    in_band = (levelled[:, 2] > height + BAND_M[0]) & (levelled[:, 2] < height + BAND_M[1])
    ref_band = (reference_world[:, 2] > height + BAND_M[0]) & (reference_world[:, 2] < height + BAND_M[1])
    if in_band.sum() < 2000 or ref_band.sum() < 2000:
        return {"accepted": False, "why": "too little of the table in one of the views"}
    scene_xy, scene_classes = levelled[in_band, :2], fixed_classes[in_band]
    ref_xy, ref_classes = reference_world[ref_band, :2], reference_classes[ref_band]

    def pose(yaw: float, shift: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = _rz(yaw) @ level
        matrix[:3, 3] = [shift[0], shift[1], height - offset]
        return matrix

    coarse = _search(ref_xy, ref_classes, scene_xy, scene_classes,
                     np.radians(np.arange(0.0, 360.0, COARSE_STEP_DEG)), CELL_M)
    coarse.sort(key=lambda item: item[0], reverse=True)
    picked: list[tuple[float, float, np.ndarray]] = []
    for item in coarse:
        if prior is not None and not prior(pose(item[1], item[2])):
            continue
        if all(abs(np.angle(np.exp(1j * (item[1] - p[1])))) > np.radians(10) for p in picked):
            picked.append(item)
        if len(picked) == CANDIDATES:
            break
    refined = []
    for _, yaw, _ in picked:
        fine = _search(ref_xy, ref_classes, scene_xy, scene_classes,
                       yaw + np.radians(np.arange(-FINE_SPAN_DEG, FINE_SPAN_DEG + 1e-9, FINE_STEP_DEG)), FINE_CELL_M)
        refined.append(max(fine, key=lambda item: item[0]))
    refined.sort(key=lambda item: item[0], reverse=True)
    if not refined:
        return {"accepted": False, "why": "no heading passes the prior"}
    best = refined[0]
    rival = refined[1] if len(refined) > 1 else None
    cam2world = pose(best[1], best[2])
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(reference_world[ref_band]).query(transform(fixed_points[in_band], cam2world),
                                                          distance_upper_bound=0.05)
    found = np.isfinite(distance)
    ratio = None if rival is None else rival[0] / best[0]
    ambiguous = ratio is not None and ratio > AMBIGUOUS_RATIO
    return {"accepted": not ambiguous, "cam2world_cv": cam2world, "score": best[0], "yaw_deg": float(np.degrees(best[1])),
            "rival_yaw_deg": None if rival is None else float(np.degrees(rival[1])), "rival_ratio": ratio,
            "table_height_m": height, "wrist_table_tilt_deg": tilt,
            "median_distance_to_wrist_points_m": float(np.median(distance[found])) if found.any() else None,
            "overlap_fraction": float(found.mean()),
            "why": (f"ambiguous: a heading {np.degrees(rival[1]):.0f} deg scores {ratio:.0%} of the best"
                    if ambiguous else None)}
