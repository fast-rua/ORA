"""Test helpers for observed clearance, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/test_observed_clearance.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import numpy as np


TABLE_Z = 0.75


def _box_hits(origin, directions, low, high):
    """Slab-method ray/AABB distances along each direction (inf when missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        t1 = (np.asarray(low) - origin) * inverse
        t2 = (np.asarray(high) - origin) * inverse
    near = np.nanmax(np.minimum(t1, t2), axis=1)
    far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (far >= near) & (far > 0)
    return np.where(hit, np.where(near > 0, near, far), np.inf)


def _camera(position, *, look=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), size=(160, 120), focal=110.0):
    forward = np.asarray(look, float) / np.linalg.norm(look)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, 0], camera_to_world[:3, 1], camera_to_world[:3, 2] = right, down, forward
    camera_to_world[:3, 3] = position
    width, height = size
    intrinsic = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    return camera_to_world, intrinsic


def _render(camera_to_world, intrinsic, size, boxes, *, plane_z=TABLE_Z):
    """Depth (to the image plane) of a table plane plus axis-aligned boxes."""
    width, height = size
    rows, columns = np.mgrid[0:height, 0:width]
    rays_camera = np.stack([(columns - intrinsic[0, 2]) / intrinsic[0, 0],
                            (rows - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(rows, float)], axis=-1)
    rays = rays_camera.reshape(-1, 3) @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        plane = (plane_z - origin[2]) / rays[:, 2]
    distance = np.where(plane > 0, plane, np.inf)
    for low, high in boxes:
        distance = np.minimum(distance, _box_hits(origin, rays, low, high))
    depth = np.where(np.isfinite(distance), distance, 0.0)  # z of rays_camera is 1: t is image-plane depth
    return depth.reshape(height, width).astype(np.float32)
