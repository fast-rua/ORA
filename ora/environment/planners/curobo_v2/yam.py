"""The I2RT YAM station's arm as cuRobo V2 plans it: one arm of i2rt's station URDF, for either arm.

The description is i2rt's ``yam_station_linear_4310_d405.urdf`` (i2rt fe78bae, v1.4.0), the source the station MJCF that
``backends/yam/kinematics.py`` loads is generated from, so the planner and ``StationKinematics`` read one geometry. The
worker's ``--robot-file`` is that URDF and ``robot_file`` writes the V2 file around it: ``left_base`` is the base link,
so V2 builds only the left chain (``left_joint1`` to ``left_joint6``, the fingers ``left_joint7`` and ``left_joint8``
locked open) and leaves the right arm's subtree out. The two arms are the same chain joint for joint on bases 0.61 m
apart with one orientation, so this one model plans both, as the ALOHA's does: a request is in the moving arm's base
frame and names the left joints, and the other arm stands in the world.

The tool frame ``left_tcp`` is the station's public TCP (``controllers.yam_station``): the MJCF's ``tcp_left`` site on
``left_gripper``, the centre of the two fingertip facets 144.65 mm along the flange, turned by ``TCP_AXES_IN_SITE``
(the site turns +90 degrees about z from the flange and the public axes another +90, so the TCP is the flange's axes
turned a half turn about z). The URDF has no sites, so it is an extra link here.

Collision spheres cover every link of the arm, the gripper's two fingertips at the open jaw and the wrist camera, in
each link's frame. The fit: each link's mesh is voxelised at 2 mm and closed into a solid; a candidate sphere sits on a
solid voxel and reaches no more than ``PROTRUSION_M`` past the solid's nearest outer face (``FINGER_PROTRUSION_M`` for
the fingertips, whose faces grasp); a greedy cover takes the candidate holding most of the boundary voxels still bare
until 99.5 % of them lie within 4 mm of a sphere. The buffer (``robots.SPHERE_BUFFER_M``) is added on top for the
world; self-collision checks the fit without it (``robots.OVERRIDES``). A thin
part of a mesh may stay up to 7.6 mm outside every sphere (the left fingertip's rim; 3 to 6 mm on the other links).

Self-collision is checked between every two links that can meet within the joint limits: the base, ``link1`` and
``link2`` against the wrist (``link5``, the gripper, its fingertips and camera) and against ``link3`` and ``link4``
where they are not neighbours. A pair is ignored when the two links are neighbours, ride together (the gripper, its
locked fingertips and the camera), or never come within reach of each other: over 4,000 joint vectors drawn evenly
within the limits, the two links' convex hulls never touched (``link3`` with the wrist, ``link4`` with the fingertips
and the camera, ``link5`` with the fingertips and the camera). ``link4`` and the gripper, one short joint apart, are
ignored too: their hulls meet only past about 60 degrees of joint 5 either way, hulls that overstate ``link4``'s
hollow shell, and ``StationKinematics`` does not pair them either. The base and ``link2`` are ignored as V2's Franka
ignores its base and second link: ``link2`` stands on the base's top.

The arm rests folded, ``link3`` back along ``link2``, and at rest ``link1`` and ``link2`` stand 11.5 to 25 mm from
``link3`` to ``link5`` on the meshes, closer than two fitted spheres can show it. ``FOLD_BUFFER_M`` on both gives that
gap back, so the rest pose, and the cell's measured rest about one degree past joints 2 and 3's limits (2026-11-14),
are clear of self-collision in V2's own check; two degrees past is not. Pure Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROBOT = "yam_linear_4310"
BASE_LINK = "left_base"
TOOL_FRAME = "left_tcp"
HAND_LINK = "left_gripper"
JOINT_NAMES = tuple(f"left_joint{index}" for index in range(1, 7))
FINGER_JOINTS = ("left_joint7", "left_joint8")
#: ``controllers.yam_station.FINGER_OPEN_M``: each finger's slide fully open; the planner keeps the jaw there.
FINGER_OPEN_M = 0.04695
#: ``left_gripper`` to the public TCP, V2's ``fixed_transform`` (x, y, z, qw, qx, qy, qz): the ``tcp_left`` site's
#: position, and its turn composed with ``TCP_AXES_IN_SITE``.
TCP_IN_GRIPPER = (3.88982673e-05, 9.68103407e-05, 0.144650259, 0.0, 0.0, 0.0, 1.0)
#: How far a sphere may reach past its link's surface, before the buffer: the arm's links, and the fingertips.
PROTRUSION_M = 0.012
FINGER_PROTRUSION_M = 0.006
#: The self-collision padding of ``link1`` and ``link2`` (both, so a pair of them with ``link3`` to ``link5`` gains
#: 6 mm): the room the folded rest pose leaves between them, which two fitted spheres cannot show.
FOLD_BUFFER_M = -0.006
#: V2's ``franka.yml`` values; the timing worker times every route the planner returns (``planners/timing``).
MAX_ACCELERATION = 15.0
MAX_JERK = 500.0

#: Each link's spheres in its own frame (the URDF link frame, the MJCF body frame): centre (x, y, z) and radius.
_SPHERES = {
    "base": (
        ((0.0, -0.001, 0.035), 0.0424), ((0.02, 0.083, 0.007), 0.0171), ((-0.02, 0.083, 0.007), 0.0171),
        ((0.02, -0.081, 0.007), 0.0171), ((-0.02, -0.081, 0.007), 0.0171), ((-0.016, 0.047, 0.007), 0.0171),
        ((0.02, 0.043, 0.007), 0.0171), ((-0.02, -0.045, 0.007), 0.0167), ((0.02, -0.045, 0.007), 0.0167),
        ((0.0, -0.001, 0.039), 0.0426), ((0.0, -0.001, 0.031), 0.0424), ((0.0, -0.085, 0.007), 0.0167),
        ((0.004, 0.071, 0.007), 0.0171), ((0.0, -0.053, 0.007), 0.0171), ((-0.028, 0.023, 0.007), 0.0183),
        ((0.0, 0.095, 0.003), 0.015), ((0.028, -0.017, 0.007), 0.0191), ((0.024, 0.019, 0.007), 0.0191),
        ((0.024, 0.063, 0.003), 0.015), ((-0.028, -0.017, 0.007), 0.0191), ((-0.032, 0.055, 0.003), 0.015),
        ((0.024, -0.065, 0.003), 0.015), ((-0.028, -0.081, 0.007), 0.0171), ((0.02, -0.097, 0.003), 0.015),
        ((-0.032, -0.069, 0.003), 0.015),
    ),
    "link1": (
        ((-0.044, -0.0399, -0.0185), 0.0191), ((-0.008, 0.0001, 0.0015), 0.0191),
        ((-0.044, 0.0361, -0.0185), 0.0171), ((-0.012, 0.0361, 0.0015), 0.0171), ((-0.012, -0.0319, 0.0015), 0.0167),
        ((-0.012, 0.0281, -0.0185), 0.015), ((-0.012, -0.0239, -0.0185), 0.015),
        ((-0.032, -0.0399, -0.0105), 0.0191), ((-0.032, 0.0361, -0.0105), 0.0171), ((-0.012, 0.0241, 0.0175), 0.015),
        ((-0.012, -0.0279, 0.0175), 0.015), ((-0.044, 0.0361, -0.0265), 0.0171), ((-0.012, -0.0039, -0.0185), 0.015),
        ((-0.06, 0.0321, -0.0185), 0.015), ((-0.04, 0.0361, 0.0015), 0.015),
    ),
    "link2": (
        ((0.0, -0.0358, 0.008), 0.043), ((0.0, -0.0318, 0.264), 0.0414), ((0.0, -0.0318, 0.2), 0.0391),
        ((0.0, -0.0318, 0.068), 0.0391), ((0.0, -0.0318, 0.132), 0.0363), ((0.0, -0.0318, 0.024), 0.043),
        ((-0.004, -0.0318, 0.164), 0.0351), ((0.0, -0.0358, 0.0), 0.043), ((0.012, -0.0558, 0.24), 0.0227),
        ((-0.004, -0.0318, 0.1), 0.0351), ((0.0, -0.0198, 0.22), 0.0323), ((-0.024, -0.0598, 0.236), 0.0191),
        ((0.0, -0.0358, 0.264), 0.0413), ((0.024, -0.0078, 0.244), 0.0191), ((0.0, -0.0318, 0.0), 0.0422),
        ((-0.028, -0.0118, 0.24), 0.015),
    ),
    "link3": (
        ((-0.0602, -0.0369, -0.2385), 0.041), ((-0.0602, -0.0329, -0.0585), 0.0391),
        ((-0.0602, -0.0329, -0.1785), 0.0391), ((-0.0602, -0.0329, -0.1185), 0.0363),
        ((-0.0762, -0.0729, -0.0185), 0.0171), ((-0.0762, 0.0031, -0.0185), 0.0171),
        ((-0.0322, -0.0729, 0.0095), 0.0174), ((-0.0322, 0.0031, 0.0095), 0.0174),
        ((0.0158, 0.0031, -0.0025), 0.0171), ((0.0078, -0.0729, 0.0135), 0.0171),
        ((-0.0602, -0.0369, -0.2185), 0.0408), ((-0.0082, -0.0729, -0.0185), 0.0171),
        ((-0.0162, 0.0031, -0.0185), 0.0171), ((-0.0442, -0.0729, -0.0225), 0.0171),
        ((-0.0002, 0.0031, 0.0175), 0.0171), ((-0.0482, 0.0031, -0.0225), 0.0171),
        ((0.0198, -0.0729, -0.0105), 0.0167), ((-0.0602, -0.0009, 0.0015), 0.0155),
        ((-0.0642, -0.0689, 0.0015), 0.015), ((-0.0602, -0.0289, -0.2425), 0.0391),
        ((0.0078, 0.0031, -0.0185), 0.0171), ((-0.0642, -0.0329, -0.0945), 0.0359),
        ((-0.0082, -0.0729, 0.0175), 0.0171), ((-0.0642, -0.0329, -0.1585), 0.0363),
        ((-0.0842, -0.0529, -0.0345), 0.0171), ((-0.0882, -0.0009, -0.0225), 0.015),
        ((-0.0602, -0.0449, -0.2505), 0.0351), ((0.0158, 0.0031, 0.0135), 0.0167),
        ((-0.0802, -0.0689, -0.0065), 0.0155), ((-0.0322, -0.0569, 0.0175), 0.015),
        ((-0.0802, -0.0009, -0.0065), 0.0155), ((-0.0362, -0.0729, -0.0265), 0.0174),
        ((-0.0602, -0.0289, -0.2145), 0.0391), ((0.0158, -0.0729, 0.0055), 0.0171),
        ((-0.0522, -0.0689, 0.0055), 0.0155), ((-0.0362, -0.0009, 0.0135), 0.0155),
        ((-0.0162, 0.0031, 0.0095), 0.0171), ((-0.0002, -0.0769, -0.0265), 0.015),
        ((-0.0402, 0.0031, -0.0265), 0.0174), ((-0.0562, -0.0529, -0.0425), 0.0251),
        ((-0.0482, -0.0009, 0.0095), 0.015), ((-0.0722, -0.0129, -0.0425), 0.0239),
    ),
    "link4": (
        ((-0.0672, 0.034, -0.0713), 0.0391), ((0.0008, -0.006, -0.0033), 0.0191), ((0.0008, 0.07, -0.0313), 0.0171),
        ((-0.0752, 0.034, -0.0473), 0.0291), ((0.0008, -0.002, -0.0313), 0.0171), ((0.0008, 0.034, -0.0353), 0.015),
        ((0.0008, 0.07, 0.0007), 0.015), ((-0.0272, 0.022, -0.0353), 0.015), ((-0.0272, 0.05, -0.0353), 0.015),
        ((-0.0592, 0.034, -0.0473), 0.0291), ((-0.0152, 0.07, -0.0033), 0.015), ((0.0048, 0.018, -0.0353), 0.015),
        ((0.0048, 0.05, -0.0353), 0.015), ((-0.0752, 0.034, -0.0753), 0.0351), ((-0.0112, -0.002, -0.0313), 0.0171),
        ((0.0128, 0.07, -0.0033), 0.015), ((-0.0512, 0.018, -0.0353), 0.0167), ((0.0008, -0.006, -0.0193), 0.0183),
        ((-0.0072, -0.006, 0.0087), 0.0177),
    ),
    "link5": (
        ((0.039, -0.0015, -0.0006), 0.0388), ((0.015, -0.0015, 0.0114), 0.0252), ((0.007, -0.0015, -0.0046), 0.0191),
    ),
    "gripper": (
        ((0.0, -0.0015, 0.0285), 0.0389), ((0.04, 0.0185, 0.0485), 0.0171), ((0.04, -0.0175, 0.0485), 0.0171),
        ((-0.04, 0.0185, 0.0485), 0.0171), ((-0.04, -0.0175, 0.0485), 0.0171), ((0.02, -0.0055, 0.0045), 0.0211),
        ((-0.004, 0.0065, 0.0205), 0.0331), ((-0.016, -0.0095, 0.0085), 0.022), ((-0.004, 0.0225, 0.0485), 0.0211),
        ((0.008, 0.0105, 0.0125), 0.0252), ((0.008, 0.0225, 0.0445), 0.0198), ((0.004, -0.0215, 0.0045), 0.0177),
        ((-0.032, -0.0255, 0.0485), 0.0171), ((0.008, -0.0255, 0.0485), 0.0183), ((0.036, -0.0015, 0.0485), 0.015),
    ),
    "tip_left": (
        ((0.001, -0.0573, -0.0232), 0.0158), ((0.045, -0.0573, -0.0272), 0.0151),
        ((0.005, -0.0093, -0.0072), 0.0135), ((0.005, -0.0333, -0.0072), 0.0135),
        ((0.025, -0.0533, -0.0192), 0.0131), ((0.009, -0.0493, -0.0552), 0.011), ((0.037, -0.0493, -0.0592), 0.011),
        ((0.017, -0.0493, -0.0752), 0.0091), ((-0.003, -0.0093, -0.0072), 0.0131),
        ((-0.003, -0.0333, -0.0072), 0.0131), ((0.029, -0.0493, -0.0752), 0.0091),
        ((0.045, -0.0533, -0.0472), 0.0095), ((0.005, -0.0533, -0.0472), 0.0114), ((0.025, -0.0613, -0.0152), 0.011),
        ((0.009, -0.0093, -0.0032), 0.0131), ((0.049, -0.0573, -0.0232), 0.0123),
        ((-0.007, -0.0213, -0.0072), 0.0091), ((0.009, -0.0333, -0.0032), 0.0131),
        ((0.013, -0.0573, -0.0192), 0.0144), ((0.029, -0.0653, -0.0152), 0.0091),
        ((-0.007, -0.0573, -0.0152), 0.0091), ((-0.007, -0.0013, -0.0072), 0.0091),
        ((-0.007, -0.0413, -0.0072), 0.0091), ((0.033, -0.0493, -0.0752), 0.0091),
    ),
    "tip_right": (
        ((-0.003, 0.0557, -0.0192), 0.0151), ((-0.043, 0.0557, -0.0272), 0.0151),
        ((-0.003, 0.0077, -0.0072), 0.0133), ((0.001, 0.0317, -0.0072), 0.0133), ((-0.023, 0.0557, -0.0192), 0.0135),
        ((-0.007, 0.0517, -0.0512), 0.0114), ((-0.027, 0.0477, -0.0752), 0.0095),
        ((-0.039, 0.0517, -0.0552), 0.0095), ((-0.007, 0.0317, -0.0032), 0.0123),
        ((-0.015, 0.0477, -0.0712), 0.0095), ((0.005, 0.0117, -0.0072), 0.0131), ((-0.003, 0.0557, -0.0312), 0.0133),
        ((-0.023, 0.0477, -0.0832), 0.0091), ((-0.043, 0.0517, -0.0472), 0.0114), ((-0.011, 0.0077, -0.0032), 0.011),
        ((-0.051, 0.0557, -0.0232), 0.0123), ((0.005, 0.0357, -0.0072), 0.0131), ((-0.039, 0.0477, -0.0632), 0.0091),
        ((-0.035, 0.0557, -0.0192), 0.0135), ((-0.011, 0.0357, -0.0032), 0.011), ((-0.003, 0.0597, -0.0192), 0.0151),
    ),
    "camera": (
        ((0.0101, 0.002, -0.012), 0.0251), ((-0.0139, 0.006, -0.024), 0.0224), ((0.0141, 0.042, -0.02), 0.015),
        ((-0.0139, 0.042, -0.02), 0.015), ((-0.0219, -0.002, -0.052), 0.0171), ((0.0101, -0.006, -0.012), 0.0251),
        ((0.0141, 0.022, -0.024), 0.0171), ((-0.0139, 0.05, -0.016), 0.0155), ((0.0141, 0.046, -0.016), 0.015),
        ((0.0101, 0.006, -0.012), 0.0251),
    ),
}

COLLISION_LINKS = tuple(f"left_{link}" for link in _SPHERES)
#: The wrist: the gripper, its fingertips locked open and the wrist camera ride together.
_HAND = ("gripper", "tip_left", "tip_right", "camera")
#: Each link and the links it is never checked against (see the module's account).
_IGNORE = {
    "base": ("link1", "link2"),
    "link1": ("link2",),
    "link2": ("link3",),
    "link3": ("link4", "link5", *_HAND),
    "link4": ("link5", *_HAND),
    "link5": _HAND,
    "gripper": ("tip_left", "tip_right", "camera"),
    "tip_left": ("tip_right", "camera"),
    "tip_right": ("camera",),
}
SELF_COLLISION_IGNORE = {f"left_{link}": [f"left_{other}" for other in others] for link, others in _IGNORE.items()}
SELF_COLLISION_BUFFER = {"left_link1": FOLD_BUFFER_M, "left_link2": FOLD_BUFFER_M}
#: The links the held load is not checked against: it hangs between the fingertips, under the camera.
HELD_IGNORE = ("left_link5", "left_gripper", "left_tip_left", "left_tip_right", "left_camera")
#: What a refusal calls each link; the rest is the arm.
LINK_PARTS = {"left_gripper": "hand", "left_camera": "hand", "left_tip_left": "fingers", "left_tip_right": "fingers"}


def collision_spheres() -> dict[str, list[dict[str, Any]]]:
    """Every collision link's spheres, as V2 reads them."""
    return {f"left_{link}": [{"center": list(center), "radius": radius} for center, radius in spheres]
            for link, spheres in _SPHERES.items()}


def robot_file(urdf_path: str | Path) -> dict[str, Any]:
    """The V2 robot file of one YAM arm on i2rt's station URDF at ``urdf_path``, before ORA's shared changes
    (``robots.OVERRIDES``: the buffer, the tool frame link and the held load)."""
    names = [*JOINT_NAMES, *FINGER_JOINTS]
    return {"robot_cfg": {"kinematics": {
        "format_version": 2.0, "base_link": BASE_LINK, "tool_frames": [TOOL_FRAME], "urdf_path": str(urdf_path),
        "collision_link_names": list(COLLISION_LINKS), "collision_spheres": collision_spheres(),
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": {link: list(others) for link, others in SELF_COLLISION_IGNORE.items()},
        "self_collision_buffer": dict(SELF_COLLISION_BUFFER),
        "cspace": {"joint_names": names, "default_joint_position": [0.0] * len(JOINT_NAMES) + [FINGER_OPEN_M] * 2,
                   "cspace_distance_weight": [1.0] * len(names), "null_space_weight": [1.0] * len(names),
                   "max_acceleration": [MAX_ACCELERATION] * len(names), "max_jerk": [MAX_JERK] * len(names)},
    }}, "load_dynamics": False}


__all__ = ["BASE_LINK", "COLLISION_LINKS", "FINGER_JOINTS", "FINGER_OPEN_M", "FINGER_PROTRUSION_M", "FOLD_BUFFER_M",
           "HAND_LINK", "HELD_IGNORE", "JOINT_NAMES", "LINK_PARTS", "PROTRUSION_M", "ROBOT", "SELF_COLLISION_BUFFER",
           "SELF_COLLISION_IGNORE", "TCP_IN_GRIPPER", "TOOL_FRAME", "collision_spheres", "robot_file"]
