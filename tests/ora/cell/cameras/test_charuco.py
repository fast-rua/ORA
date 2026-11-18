"""A plain 11x8 checkerboard names its corners by its colours: the first corner is the one with the
dark outer square beyond it, whichever way the camera sees the board; a board whose two far
corners have the same colour cannot tell."""

from __future__ import annotations

import numpy as np

from ora.cell.cameras.charuco import by_colour, chessboard_pose

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
SQUARE = 0.02


def _image(pattern: tuple[int, int]) -> np.ndarray:
    """The board straight in front of the camera, 0.5 m away, its middle on the optical axis."""

    columns, rows = pattern
    image = np.full((480, 640, 3), 235, np.uint8)
    scale = K[0, 0] / 0.5 * SQUARE
    origin = np.array([320.0, 240.0]) - scale * np.array([(columns - 1) / 2, (rows - 1) / 2])
    for i in range(-1, columns):
        for j in range(-1, rows):
            if (i + j) % 2 == 0:
                x0, y0 = origin + scale * np.array([i, j])
                image[int(round(y0)):int(round(y0 + scale)), int(round(x0)):int(round(x0 + scale))] = 15
    return image


def _first_corner(image: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray, bool]:
    pose = chessboard_pose(image, K, None, SQUARE, pattern)
    board2cam, told = by_colour(image, pose["board2cam"], K, None, SQUARE, pattern)
    corner = K @ board2cam[:3, 3]
    return corner[:2] / corner[2], told


def test_the_first_corner_is_the_same_physical_corner_from_either_side() -> None:
    image = _image((11, 8))
    seen, told = _first_corner(image, (11, 8))
    turned, told_turned = _first_corner(image[::-1, ::-1].copy(), (11, 8))
    assert told and told_turned
    assert np.allclose(turned, [639, 479] - seen, atol=1.0)


def test_a_board_whose_far_corners_match_cannot_tell() -> None:
    assert _first_corner(_image((9, 7)), (9, 7))[1] is False
