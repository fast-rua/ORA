"""Principal axes of a covariance that are the same on every machine.

Where a variance repeats (a square, a disc, a sphere), ``np.linalg.eigh`` may return any orthonormal basis of
that eigenspace, and which one it returns depends on the BLAS kernel the machine runs. :func:`principal_axes`
keeps the eigenvectors where the variances differ and, inside a repeated eigenspace, takes the world axes projected
onto it in ``x, y, z`` order and made orthonormal. That basis depends only on the eigenspace, which the points
fix, so the same points give the same axes on every machine.

The eigenvectors themselves must not depend on the kernel either. LAPACK solves a 2 x 2 in closed form, the same
everywhere, so eigh keeps it; a 3 x 3 goes through a Householder reduction whose BLAS calls round by kernel, so
:func:`principal_axes` diagonalises it by Jacobi rotations in Python floats instead (the same eigenvectors to
rounding). :func:`scatter` and :func:`covariance` sum the points' products in numpy, not in a matrix product.

``ora/agent/code_mode/ora_lib/geometry.py`` carries a copy for the sandbox, which has no ORA package.
"""

from __future__ import annotations

import math

import numpy as np

#: Variances closer than this fraction of the largest one are one repeated variance.
REPEATED_RTOL = 1e-9

#: A world axis whose projection onto what is left of the eigenspace is shorter than this is skipped. Some
#: projection is always at least ``1/sqrt(3)`` long, so a repeated eigenspace always gets its full basis.
_KEEP_NORM = 0.5

#: Jacobi sweeps allowed; a 3 x 3 converges in five or six.
_SWEEPS = 50


def principal_axes(covariance) -> tuple[np.ndarray, np.ndarray]:
    """``(variances, axes)`` of a symmetric covariance, greatest variance first, axes as unit columns.

    A repeated variance is reported once for each of its axes, as their mean, so that equal variances compare
    equal; its axes are the world axes projected onto its eigenspace (module docstring). The axes are not
    sign-normalised: callers keep their own sign rule.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if len(matrix) > 2 and np.isfinite(matrix).all():
        values, vectors = _jacobi(matrix)
    else:
        values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    tolerance = REPEATED_RTOL * float(np.max(np.abs(values), initial=0.0))
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


def scatter(centered) -> np.ndarray:
    """``centered.T @ centered`` of ``(N, d)`` rows, each entry one numpy sum of elementwise products.

    A matrix product over the points sums them in an order, fused or not, that the BLAS kernel chooses, so its
    last bit differs from machine to machine; numpy's own sum adds in one order everywhere.
    """
    p = np.asarray(centered, dtype=np.float64)
    return np.array([[np.sum(p[:, i] * p[:, j]) for j in range(p.shape[1])] for i in range(p.shape[1])])


def covariance(points) -> np.ndarray:
    """The sample covariance of ``(N, d)`` rows, as ``np.cov(points, rowvar=False)``, by :func:`scatter`."""
    p = np.asarray(points, dtype=np.float64)
    return scatter(p - p.mean(axis=0)) / (len(p) - 1)


def _jacobi(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eigh's ``(values, vectors)``, values ascending, of a finite symmetric matrix by cyclic Jacobi rotations.

    Each rotation zeroes one off-diagonal entry; an entry too small to move either diagonal one is dropped
    (Rutishauser's rule). Python floats round every multiply and add on its own, in this order, on any machine.
    """
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
    """An orthonormal basis of the span of ``basis``'s columns, built from the world axes in order."""
    projector = basis @ basis.T
    chosen: list[np.ndarray] = []
    for world_axis in np.eye(len(basis)):
        axis = projector @ world_axis
        for previous in chosen:
            axis = axis - np.sum(previous * axis) * previous  # not ``previous @ axis``: BLAS dot rounds by kernel
        norm = math.sqrt(np.sum(axis * axis))
        if norm >= _KEEP_NORM:
            chosen.append(axis / norm)
        if len(chosen) == basis.shape[1]:
            break
    return np.stack(chosen, axis=1)
