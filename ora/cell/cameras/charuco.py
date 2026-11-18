"""Place every camera of the cell from one ChArUco board seen by all of them.

The board lies flat on the table (printed, or shown full screen on a tablet). Each camera
that sees it gives the board's pose in its own frame by PnP on the ChArUco corners. A
wrist camera's own pose is known from the encoders and the CAD chain, so the board is
placed in the cell's world frame through the wrist cameras, and each fixed camera through
the board. The fixed cameras then agree with each other to the precision of the corner
detection, whatever the wrist chain's own error, which only moves all of them together.

The square size must be the board as it is shown: measure one black square on the screen
with a ruler. The aligned depth checks it: the board's distance by PnP and by depth must
agree, or the size is wrong.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

SQUARES = (5, 4)                 # columns, rows: big squares so a camera a metre away reads the markers
MARKER_RATIO = 0.72
DICTIONARY = "DICT_4X4_50"
MIN_CORNERS = 6


def board(square_m: float):
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.CharucoBoard(SQUARES, square_m, square_m * MARKER_RATIO, dictionary)


def image(width_px: int, height_px: int) -> np.ndarray:
    """The board drawn to fill a ``width_px`` x ``height_px`` screen, with a white margin."""

    import cv2

    pattern = board(1.0)                      # the drawing does not depend on the physical size
    square = int(min(width_px / (SQUARES[0] + 0.6), height_px / (SQUARES[1] + 0.6)))
    drawn = pattern.generateImage((square * SQUARES[0], square * SQUARES[1]), marginSize=0, borderBits=1)
    canvas = np.full((height_px, width_px), 255, np.uint8)
    top, left = (height_px - drawn.shape[0]) // 2, (width_px - drawn.shape[1]) // 2
    canvas[top:top + drawn.shape[0], left:left + drawn.shape[1]] = drawn
    cv2.putText(canvas, f"ORA cell ChArUco {SQUARES[0]}x{SQUARES[1]} {DICTIONARY}  -  measure one black square",
                (left, max(20, top - 12)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, square / 260), 128, 2, cv2.LINE_AA)
    return canvas


def board_pose(rgb: np.ndarray, intrinsic: Sequence[Sequence[float]], distortion: Sequence[float] | None,
               square_m: float, depth: np.ndarray | None = None) -> dict[str, Any] | None:
    """The board's pose in the camera's OpenCV frame (board2cam), or None when it is not seen."""

    import cv2

    pattern = board(square_m)
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(pattern).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    object_points = pattern.getChessboardCorners()[ids.ravel()].astype(np.float64)
    image_points = corners.reshape(-1, 2).astype(np.float64)
    k = np.asarray(intrinsic, float)
    coefficients = np.zeros(5) if distortion is None else np.asarray(distortion, float)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k, coefficients, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, k, coefficients, rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, coefficients)
    error = float(np.median(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.ravel()
    result = {"board2cam": pose, "corners": int(len(ids)), "reprojection_px": error}
    if depth is not None:
        # Depth at the corners against the PnP depth of the same corners: their ratio is the
        # error of the square size (a tablet's glass may leave the depth sparse).
        in_camera = object_points @ pose[:3, :3].T + pose[:3, 3]
        u, v = np.round(image_points).astype(int).T
        inside = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        measured = np.where(inside, depth[np.clip(v, 0, depth.shape[0] - 1), np.clip(u, 0, depth.shape[1] - 1)], 0)
        valid = measured > 0
        result["depth_scale_ratio"] = (float(np.median(measured[valid] / in_camera[valid, 2]))
                                       if valid.sum() >= 4 else None)
    return result


# ---------------------------------------------------------------- plain checkerboards
# A printed checkerboard (no markers) works too, but only when a camera sees all of it, and
# its frame is defined up to a half turn: ``flip_variants`` gives both, and the caller picks
# the one that agrees with what it already knows (a rough camera pose, another view).
CHESSBOARD_FLAGS_NAME = "CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY"


def chessboard_pose(rgb: np.ndarray, intrinsic: Sequence[Sequence[float]], distortion: Sequence[float] | None,
                    square_m: float, pattern: tuple[int, int], depth: np.ndarray | None = None) -> dict[str, Any] | None:
    """A full checkerboard of ``pattern`` inner corners (columns, rows): board2cam, or None."""

    import cv2

    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    columns, rows = pattern
    object_points = np.array([[c * square_m, r * square_m, 0.0] for r in range(rows) for c in range(columns)])
    image_points = corners.reshape(-1, 2).astype(np.float64)
    k = np.asarray(intrinsic, float)
    coefficients = np.zeros(5) if distortion is None else np.asarray(distortion, float)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, k, coefficients, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, k, coefficients, rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, coefficients)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.ravel()
    result = {"board2cam": pose, "corners": int(len(image_points)),
              "reprojection_px": float(np.median(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))}
    if depth is not None:
        in_camera = object_points @ pose[:3, :3].T + pose[:3, 3]
        u, v = np.round(image_points).astype(int).T
        measured = depth[np.clip(v, 0, depth.shape[0] - 1), np.clip(u, 0, depth.shape[1] - 1)]
        valid = measured > 0
        result["depth_scale_ratio"] = (float(np.median(measured[valid] / in_camera[valid, 2]))
                                       if valid.sum() >= 4 else None)
    return result


def flip_variants(board2cam: np.ndarray, square_m: float, pattern: tuple[int, int]) -> list[np.ndarray]:
    """The board pose and its half turn about the board's centre normal: the two frames a
    plain checkerboard cannot tell apart."""

    columns, rows = pattern
    centre = np.array([(columns - 1) * square_m / 2, (rows - 1) * square_m / 2, 0.0])
    turn = np.eye(4)
    turn[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    turn[:3, 3] = 2 * centre
    return [board2cam, board2cam @ turn]


def by_colour(rgb: np.ndarray, board2cam: np.ndarray, intrinsic: Sequence[Sequence[float]],
              distortion: Sequence[float] | None, square_m: float, pattern: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """The board pose whose first corner has the dark outer square beyond it, and whether the
    colours told. A board whose columns and rows of inner corners add up to an odd number
    (11x8) has a dark and a light square at its two far corners, so every camera names its
    corners alike; any other board, or a far corner out of the image, cannot tell."""

    import cv2

    columns, rows = pattern
    if (columns + rows) % 2 == 0:
        return board2cam, False
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
    k = np.asarray(intrinsic, float)
    coefficients = np.zeros(5) if distortion is None else np.asarray(distortion, float)
    outer = np.array([[-0.5, -0.5, 0.0], [columns - 0.5, rows - 0.5, 0.0]]) * square_m
    uv, _ = cv2.projectPoints(outer, cv2.Rodrigues(board2cam[:3, :3])[0], board2cam[:3, 3], k, coefficients)
    shades = []
    for u, v in np.round(uv.reshape(-1, 2)).astype(int):
        if not (2 <= u < gray.shape[1] - 2 and 2 <= v < gray.shape[0] - 2):
            return board2cam, False
        shades.append(gray[v - 2:v + 3, u - 2:u + 3].mean())
    if abs(shades[0] - shades[1]) < 40:
        return board2cam, False
    return (board2cam if shades[0] < shades[1] else flip_variants(board2cam, square_m, pattern)[1]), True
