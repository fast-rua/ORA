"""The shared planner's collision spheres of the Panda at measured joints, on the host, without asking the planner.

A host that has to tell the robot's own depth from the scene's (a world seen from another base pose,
``base_placement``), or carry the arm along a base move, needs the robot's body where the planner has it. cuRobo V2
(``planners/curobo_v2``) models the Panda as spheres on its links: ``franka.yml`` of the pinned V2 revision with ORA's
changes (``planners/curobo_v2/robots.py``: the fingers 6.5 mm toward the hand, a 4 mm buffer on every sphere), placed by
the URDF chain from the arm's base link (``panda_link0``) with the fingers locked open at 0.04 m. This module holds the
same table and chain, so ``spheres`` at a joint vector is what the planner's ``kinematics`` returns there, its unused
held-load slots left out, and ``hand`` its tool frame (``panda_hand``).

``SPHERE_SET_SHA256`` hashes the table as the planner names its own set (``collision_spheres_sha256`` in its ready
answer, ``robots.sphere_set_sha256``): a session compares the two at start, and a test recomputes the planner's from the
table here. numpy only.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

ROBOT = "franka_panda"
#: The URDF's revolute joints 1 to 7 from ``panda_link0`` (``franka_panda.urdf`` of V2's ``franka_description``): each
#: child link, its joint origin (xyz, rpy) in the parent; every joint turns about its local z.
_HALF_TURN = 1.57079632679
_CHAIN = (
    ("panda_link1", (0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ("panda_link2", (0.0, 0.0, 0.0), (-_HALF_TURN, 0.0, 0.0)),
    ("panda_link3", (0.0, -0.316, 0.0), (_HALF_TURN, 0.0, 0.0)),
    ("panda_link4", (0.0825, 0.0, 0.0), (_HALF_TURN, 0.0, 0.0)),
    ("panda_link5", (-0.0825, 0.384, 0.0), (-_HALF_TURN, 0.0, 0.0)),
    ("panda_link6", (0.0, 0.0, 0.0), (_HALF_TURN, 0.0, 0.0)),
    ("panda_link7", (0.088, 0.0, 0.0), (_HALF_TURN, 0.0, 0.0)),
)
#: ``panda_link8`` 0.107 m along link 7, ``panda_hand`` turned -45 degrees about it; the fingers 0.0584 m along the
#: hand, locked 0.04 m out (``lock_joints``), the right one along -y.
_LINK8_M = 0.107
_HAND_YAW_RAD = -0.785398163397
_FINGER_ALONG_M = 0.0584
FINGER_LOCK_M = 0.04
#: The planner's collision links in its order; ``attached_object`` holds the held load's slots, unused here.
COLLISION_LINKS = ("panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
                   "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger",
                   "attached_object")
HELD_SLOTS = 32
#: Every sphere's buffer (``robots.SPHERE_BUFFER_M``), added to its radius as the planner adds it.
BUFFER_M = 0.004
#: Each link's spheres in its own frame, ``(x, y, z, radius)`` before the buffer: V2 ``franka.yml`` and ORA's fingers.
SPHERES: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "panda_link0": ((0.0, 0.0, 0.085, 0.03), (-0.1, -0.0, 0.085, 0.03)),
    "panda_link1": ((0.0, -0.08, 0.0, 0.055), (0.0, -0.03, 0.0, 0.06), (0.0, 0.0, -0.12, 0.06),
                    (0.0, 0.0, -0.17, 0.06)),
    "panda_link2": ((0.0, 0.0, 0.03, 0.055), (0.0, 0.0, 0.08, 0.055), (0.0, -0.12, 0.0, 0.055),
                    (0.0, -0.17, 0.0, 0.055)),
    "panda_link3": ((0.0, 0.0, -0.06, 0.05), (0.0, 0.0, -0.1, 0.06), (0.08, 0.06, 0.0, 0.052),
                    (0.08, 0.02, 0.0, 0.052)),
    "panda_link4": ((0.0, 0.0, 0.02, 0.052), (0.0, 0.0, 0.06, 0.052), (-0.08, 0.095, 0.0, 0.055),
                    (-0.08, 0.06, 0.0, 0.052)),
    "panda_link5": ((0.0, 0.03, 0.0, 0.05), (0.0, 0.082, 0.0, 0.05), (0.0, 0.0, -0.22, 0.05),
                    (0.0, 0.052, -0.18, 0.04), (0.01, 0.08, -0.14, 0.022), (0.01, 0.085, -0.11, 0.022),
                    (0.01, 0.09, -0.08, 0.022), (0.01, 0.095, -0.05, 0.022), (-0.01, 0.08, -0.14, 0.022),
                    (-0.01, 0.085, -0.11, 0.022), (-0.01, 0.09, -0.08, 0.022), (-0.01, 0.095, -0.05, 0.022),
                    (0.0, -0.009, 0.0, 0.05)),
    "panda_link6": ((0.085, 0.035, 0.0, 0.045), (0.085, 0.0, 0.0, 0.045), (0.085, -0.015, 0.0, 0.045)),
    "panda_link7": ((0.0, 0.0, 0.07, 0.045), (0.02, 0.04, 0.08, 0.024), (0.04, 0.02, 0.08, 0.024),
                    (0.04, 0.06, 0.085, 0.02), (0.06, 0.04, 0.085, 0.02)),
    "panda_hand": tuple((0.0, y, z, r) for z, r, ys in (
        (0.01, 0.023, (-0.075, -0.045, -0.015, 0.015, 0.045, 0.075)),
        (0.03, 0.022, (-0.08, -0.045, -0.015, 0.015, 0.045, 0.08)),
        (0.045, 0.022, (-0.08, -0.045, -0.015, 0.015, 0.045, 0.08))) for y in ys),
    "panda_leftfinger": ((0.0, 0.01, 0.0365, 0.011), (0.0, 0.02, 0.0085, 0.011)),
    "panda_rightfinger": ((0.0, -0.01, 0.0365, 0.011), (0.0, -0.02, 0.0085, 0.011)),
}


#: ``panda_link0``'s collision mesh bounds in its own frame (V2's ``meshes/collision/link0.obj``: x -0.154 to 0.072, y
#: +/-0.095, z 0 to 0.14 m), rounded out to the centimetre. Its two 3 cm spheres leave most of it out, since nothing
#: moves against it but link 1, whose collisions with it the planner ignores; its depth is the robot's all the same
#: (in the 2026-11-06 qualification's world inputs, about 100 points per camera beside it and above the mount).
BASE_LINK_BOX_M = ((-0.16, 0.08), (-0.10, 0.10), (0.0, 0.15))


def sphere_set() -> dict[str, Any]:
    """The table as the planner names its sphere set (``robots.sphere_set``)."""
    return {"collision_link_names": list(COLLISION_LINKS),
            "collision_spheres": {link: [list(sphere) for sphere in SPHERES[link]]
                                  for link in COLLISION_LINKS if link in SPHERES},
            "collision_sphere_buffer": BUFFER_M, "extra_collision_spheres": {"attached_object": HELD_SLOTS}}


SPHERE_SET_SHA256 = hashlib.sha256(json.dumps(sphere_set(), sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()


def _origin(xyz: Sequence[float], rpy: Sequence[float]) -> Any:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    transform = np.eye(4)
    transform[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]
    transform[:3, 3] = xyz
    return transform


def _about_z(angle: float) -> Any:
    turn = np.eye(4)
    turn[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    return turn


def link_frames(joints: Sequence[float]) -> dict[str, Any]:
    """Each collision link's 4x4 frame in the arm's base frame (``panda_link0``) at the seven arm ``joints``."""
    if len(joints) != len(_CHAIN):
        raise ValueError(f"the Panda takes {len(_CHAIN)} arm joints, not {len(joints)}")
    frames = {"panda_link0": np.eye(4)}
    transform = np.eye(4)
    for (link, xyz, rpy), angle in zip(_CHAIN, joints):
        transform = transform @ _origin(xyz, rpy) @ _about_z(float(angle))
        frames[link] = transform
    hand = transform @ _origin((0.0, 0.0, _LINK8_M), (0.0, 0.0, 0.0)) @ _about_z(_HAND_YAW_RAD)
    frames["panda_hand"] = hand
    for link, side in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
        frames[link] = hand @ _origin((0.0, side * FINGER_LOCK_M, _FINGER_ALONG_M), (0.0, 0.0, 0.0))
    return frames


def hand(joints: Sequence[float], base: Any = None) -> Any:
    """The planner's tool frame (``panda_hand``) at ``joints``: in the arm's base frame, or in the world when ``base``
    (the base link's 4x4 world transform) is given."""
    frame = link_frames(joints)["panda_hand"]
    return frame if base is None else np.asarray(base, dtype=np.float64) @ frame


def spheres(joints: Sequence[float], base: Any = None) -> Any:
    """The robot's collision spheres at ``joints``, ``[x, y, z, radius]`` rows with the buffer in the radius: in the
    arm's base frame, or in the world when ``base`` (the base link's 4x4 world transform) is given."""
    frames = link_frames(joints)
    rows = []
    for link in COLLISION_LINKS:
        if link not in SPHERES:
            continue
        table = np.asarray(SPHERES[link], dtype=np.float64)
        centres = table[:, :3] @ frames[link][:3, :3].T + frames[link][:3, 3]
        rows.append(np.hstack([centres, table[:, 3:] + BUFFER_M]))
    result = np.vstack(rows)
    if base is None:
        return result
    world = np.asarray(base, dtype=np.float64)
    return np.hstack([result[:, :3] @ world[:3, :3].T + world[:3, 3], result[:, 3:]])


__all__ = ["BASE_LINK_BOX_M", "BUFFER_M", "COLLISION_LINKS", "FINGER_LOCK_M", "HELD_SLOTS", "ROBOT", "SPHERES", "SPHERE_SET_SHA256",
           "hand", "link_frames", "sphere_set", "spheres"]
