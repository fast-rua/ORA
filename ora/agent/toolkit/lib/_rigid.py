"""Affine point transforms whose last bit is the same on every machine.

``points @ R.T + t`` and ``np.linalg.inv`` go through BLAS and LAPACK, whose kernels order and fuse the
multiply-adds by CPU, so the same cloud lands a last bit apart from one machine to the next. That is invisible in a
measurement, but not in what hashes the bytes: the grasp service's inference seed is the sha256 of the scene points,
and the review artifacts are recorded by their bytes. These spell the arithmetic out elementwise (IEEE multiplies and
adds in a fixed order) and invert by cofactors, which agree with the BLAS forms to within rounding.
"""

from __future__ import annotations

import numpy as np


def apply(points, matrix) -> np.ndarray:
    """``points`` (N x 3) under the 4 x 4 (or 3 x 4) affine ``matrix``: ``R p + t`` for each point."""
    m = np.asarray(matrix, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p[:, 0:1] * m[:3, 0] + p[:, 1:2] * m[:3, 1] + p[:, 2:3] * m[:3, 2] + m[:3, 3]


def invert(matrix) -> np.ndarray:
    """The inverse of a 4 x 4 affine ``matrix`` (last row ``0 0 0 1``); any other matrix by ``np.linalg.inv``."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.array_equal(m[3], (0.0, 0.0, 0.0, 1.0)):
        return np.linalg.inv(m)
    c0, c1, c2 = m[:3, 0], m[:3, 1], m[:3, 2]
    rows = np.stack((np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)))
    determinant = c0[0] * rows[0, 0] + c0[1] * rows[0, 1] + c0[2] * rows[0, 2]
    if determinant == 0.0:
        return np.linalg.inv(m)
    inverse = np.eye(4)
    inverse[:3, :3] = rows / determinant
    t = m[:3, 3]
    inverse[:3, 3] = -(inverse[:3, 0] * t[0] + inverse[:3, 1] * t[1] + inverse[:3, 2] * t[2])
    return inverse
