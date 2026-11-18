"""The robots cuRobo V2 plans for: each robot's V2 file and ORA's changes to it.

``franka_panda`` is the Panda LIBERO simulates. LIBERO runs robosuite's
``PandaGripper``, whose finger collision mesh (``finger.stl``) has the same
vertices as the ``finger.obj`` of V2's Franka description, but robosuite
mounts each finger 0.0524 m from the hand body ``right_hand``
(``panda_gripper.xml``), where the Franka description mounts it 0.0584 m from
``panda_hand``. At the same joints ``right_hand`` lies 0.5 mm behind
``panda_hand`` along the approach (robosuite puts it 0.1065 m past ``link7``,
the URDF 0.107 m; the grip site sits 0.0965 m from V2's ``panda_hand``, 0.097 m
from ``right_hand``). So the simulated finger spans 0.0519 to 0.1057 m along
the approach from ``panda_hand``, its tip 2.3 mm past the 0.1034 m fingertip
point, while V2's bundled finger spheres, fitted to the URDF placement, reach
0.1124 m, 0.1164 m with the 4 mm buffer.

ORA keeps V2's fit of the finger (two 11 mm spheres per finger) and moves it
6.5 mm toward the hand, to where LIBERO's finger is. Every sphere, the
fingers' included, keeps the v0.7.8 LIBERO planner's 4 mm buffer (V2 ships 0),
so the lowest finger point lies 0.1099 m from ``panda_hand``: 4.2 mm past the
simulated tip. The rest of the robot is V2's ``franka.yml`` unchanged.

``arx_x5`` is RoboDojo's ARX X5, from the file the pinned SDK itself loads
(``robot_file``, ``Assets/Robots/x5/curobo.yml``, ``robot_cfg.kinematics`` with
``format_version`` 2.0). That file has no finger collision links, because the SDK
plans to joint goals of a wrist; a shared world needs the jaws. ORA adds both
fingers (the sphere fit of the X5 description under cuRobo's own ``robot/x5_v2``,
in ``link7``'s and ``link8``'s frames), locks the fingers open at
``joint7 = joint8 = 0.044``, the widest they go, and hangs the held-load slots
on ``link6``. Every sphere keeps the 4 mm buffer. Joints 1 to 5 run to a placeholder +/-10 rad in the URDF and in
the simulated ``ARX.usd``; the limits here are ``X5_JOINT_LIMITS_RAD``, because wider ones let IK return a
configuration a whole turn away. cuRobo clips both ends of a joint alike, so a range that is not centred on zero
keeps its wider side: joints 2 and 3 are still legal below zero, where the arm itself is not.

``aloha_agilex`` is RoboTwin's ALOHA, from the file its vendored planner loads
(``assets/embodiments/aloha-agilex/curobo_left.yml``, a v0.7.8 file), translated to
V2's fields (``_v078_kinematics``). The description's two arms are the same chain
joint for joint, so this one model plans both: a request is in the moving arm's
base frame. Its joints run to a placeholder +/-10 rad in the URDF; the limits here
are +/-``ALOHA_JOINT_LIMIT_RAD``, because wider ones let IK return a configuration a
whole turn away, and the palm's spheres are the right arm's fit. The held-load slots
hang on ``fl_link6``. ``fl_link3`` and ``fl_link5`` overlap by the sphere fit even where
the arm is clear, so each gives up ``ALOHA_FOLD_BUFFER_M``: the pair is checked from the
gap where the simulator's links really touch.

``franka_robotiq`` is RoboLab's Droid: V2's ``franka.yml`` arm with a Robotiq 2F-85 where the Franka hand was.
The arm's links keep V2's spheres; ``panda_hand`` stays the tool frame and carries the Robotiq's palm, the finger
links its two open fingers, in those links' frames (V2 locks them 0.04 m out and 0.0584 m along). The 26 spheres
cover the collision meshes of RoboLab's ``franka_robotiq_2f_85_flattened.usd`` at the open jaw: farthest-point seeds,
k-means, each sphere reaching its farthest surface sample. V2's own fitter needs a watertight mesh and these are not.

``yam_linear_4310`` is one arm of the I2RT YAM station, for either arm (``yam.py``): its robot file is i2rt's station
URDF, around which ``yam.robot_file`` writes the V2 file (base link, tool frame, ORA's sphere fit, the self-collision
pairs and the cspace), since no simulator ships one. Each arm is planned alone with the other arm in the world, as the
X5's and the ALOHA's are: the contract's goal, IK and refusal are one arm's, and one six-joint model keeps the V2
solvers' problem (and their CUDA graphs) the size the other robots' are; a simultaneous move's inter-arm clearance
stays the backend's check (``planners/README.md``). ``YAM_SPHERE_SET_SHA256`` is the set's hash with the worker's 32
held-load slots.

``collision_spheres_sha256`` names the sphere set a planner checks (every
collision link's spheres, the buffer and the held-load slots), so plan
evidence carries it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import yam

#: The bundled V2 robot file a robot starts from (``curobo/content/configs/robot/``); any other robot's file is
#: the worker's ``--robot-file``.
BUNDLED_FILES = {"franka_panda": "franka.yml", "franka_robotiq": "franka.yml"}
ROBOT_FILE_ROBOTS = ("arx_x5", "aloha_agilex", yam.ROBOT)
#: The robots whose ``--robot-file`` is a URDF: ORA writes the V2 file around it.
URDF_ROBOTS = (yam.ROBOT,)
#: The v0.7.8 LIBERO planner's sphere buffer, on every sphere.
SPHERE_BUFFER_M = 0.004
#: Where LIBERO's finger sits relative to the Franka description's, along the approach.
FINGER_SHIFT_M = 0.0524 - 0.0005 - 0.0584
_FINGER = ((0.01, 0.043), (0.02, 0.015))  # V2's (jaw-line offset, approach) finger sphere centres
_FINGER_RADIUS_M = 0.011

#: The X5's fingers open: the description's fit of ``link7`` and ``link8``, centre (x, y, z) and radius.
_X5_FINGERS = {
    "link7": (((0.030874, -0.018055, -0.004903), 0.018412), ((-0.00328, -0.006534, -0.019714), 0.022366),
              ((0.05814, -0.020917, -0.001411), 0.009154), ((-0.0006, -0.003648, 0.012437), 0.020329)),
    "link8": (((0.000291, 0.002751, -0.000116), 0.029813), ((0.033648, 0.017994, -0.002401), 0.018099),
              ((0.061318, 0.022469, -0.001401), 0.00795)),
}
X5_FINGER_OPEN_M = 0.044
#: The robots ``lock_joints`` plans with fully open fingers, whatever the jaws measure: closing them changes no plan, so
#: a refusal that names the fingers says so (``worker.py``).
FINGERS_PLANNED_OPEN = frozenset({"arx_x5", yam.ROBOT})
#: The placeholder limit of the X5's joints 1 to 5, in ``X5A.urdf`` and in the simulated ``ARX.usd``.
X5_URDF_LIMIT_RAD = 10.0
#: The half-range joints 1 to 5 plan to. Joints 1 to 4 are each the smallest that holds both the arm's rated range (the
#: ARX5 SDK's ``joint_pos_min`` and ``joint_pos_max``) and every start the simulated arm has been measured at (468,584
#: proprioception records of 3,209 RoboDojo runs); joint 5 stays under the stop the simulated arm meets, joint by joint:
#:
#: * joint 1: 3.7 from the record. Rated -3.14 to 2.618; measured -3.649 and +3.377, and runs wound by a whole turn on
#:   routes with no planner: 476 records of 21 runs reach -7.762 and +7.617. Those stay out, since a limit that held
#:   them would keep every alias. With the planner the joint stays within +/-1.69.
#: * joint 2: 3.5, rated -0.05 to 3.50. Measured -0.298 to 3.175.
#: * joint 3: 3.9 from the record. Rated -0.1 to 3.20; measured -0.044 to 3.809.
#: * joint 4: 2.4 from the record. Rated -1.6 to 1.55; measured -2.395 to 1.877.
#: * joint 5: 1.56. Rated +/-1.57. The simulated joint has no limit, yet with the robot's self-collisions on it stops
#:   near 90 degrees either way: 20 of those records pass 1.58, and arms sent past it fell behind their command from
#:   about 1.565. A plan past that ends in a tracking stop that reads as contact. A start past this limit is handled as
#:   any other, below.
#:
#: Joint 6 keeps the file's +/-3.14 rad, which the simulator shares; the arm is rated +/-2 rad, but measured starts
#: reach it. A measured start past a limit is not refused for good: ``plan_global`` hands that move to the joint route.
X5_JOINT_LIMITS_RAD = (3.7, 3.5, 3.9, 2.4, 1.56)
#: The ALOHA description's placeholder arm limit, and the limit it plans to: a full turn of range plus about one
#: radian past the seam each way, so a move may cross it but a goal has at most a narrow band of aliases. A
#: session wraps a measured start into +/-pi first (recorded runs reach -8.9 rad on the wrist), so no start is illegal.
ALOHA_URDF_LIMIT_RAD = 10.0
ALOHA_JOINT_LIMIT_RAD = 4.2
#: Half the sphere gap at which link3 and link5 stopped each other in the simulator (-7.4 mm, joint 4 at 1.76 rad);
#: 12,186 measured arm states reach -8.0 mm, the deepest of them pressed that far by the drive.
ALOHA_FOLD_BUFFER_M = -0.0037
#: The right arm's fit of link6: the file's fit for the left arm is one sphere and leaves the palm's rail out.
_ALOHA_LINK6 = (((0.022, 0.0, 0.0), 0.03), ((0.05, 0.0, 0.0), 0.01), ((0.07, 0.0, 0.0), 0.01), ((0.07, 0.03, 0.0), 0.01),
                ((0.07, 0.06, 0.0), 0.01), ((0.07, -0.03, 0.0), 0.01), ((0.07, -0.06, 0.0), 0.01))

#: The Robotiq 2F-85 open on the Franka flange: per link, centre (x, y, z) in that link's frame and radius.
_ROBOTIQ_SPHERES = {
    "panda_hand": (
        ((-0.01502, 0.01502, 0.08213), 0.03233),
        ((0.02619, 0.00054, 0.04049), 0.03324),
        ((-0.02595, -0.00042, 0.03839), 0.03275),
        ((0.01498, -0.01498, 0.08222), 0.03333),
        ((0.00046, 0.02568, 0.03757), 0.03265),
        ((-8e-05, -0.02631, 0.04038), 0.03281),
    ),
    "panda_leftfinger": (
        ((0.02799, -0.06525, 0.01445), 0.01617),
        ((0.03441, -0.07441, 0.09402), 0.01929),
        ((0.0462, -0.0862, 0.0476), 0.02122),
        ((0.02339, -0.07565, 0.06078), 0.01992),
        ((0.022, -0.04943, 0.03189), 0.02148),
        ((0.03785, -0.0874, 0.02083), 0.01909),
        ((0.04027, -0.08027, 0.0697), 0.02105),
        ((0.00964, -0.06188, 0.03206), 0.02176),
        ((0.03494, -0.06302, 0.06034), 0.02011),
        ((0.04746, -0.07783, 0.02092), 0.01946),
    ),
    "panda_rightfinger": (
        ((-0.02355, 0.06714, 0.01445), 0.01682),
        ((-0.03338, 0.07338, 0.09626), 0.01796),
        ((-0.04426, 0.08426, 0.0563), 0.0193),
        ((-0.0212, 0.07354, 0.05241), 0.01633),
        ((-0.02171, 0.0466, 0.02982), 0.01803),
        ((-0.0467, 0.0867, 0.03451), 0.01929),
        ((-0.03663, 0.07663, 0.07162), 0.01829),
        ((-0.004, 0.06489, 0.03039), 0.01829),
        ((-0.03298, 0.06095, 0.05186), 0.01685),
        ((-0.0419, 0.0819, 0.0161), 0.01828),
    ),
}

#: What a refusal calls a robot's links; any other link is the arm. The one ALOHA model plans both arms.
LINK_PARTS: dict[str, dict[str, str]] = {
    "franka_panda": {"panda_hand": "hand", "panda_leftfinger": "fingers", "panda_rightfinger": "fingers"},
    "franka_robotiq": {"panda_hand": "hand", "panda_leftfinger": "fingers", "panda_rightfinger": "fingers"},
    "arx_x5": {"link6": "hand", "link7": "fingers", "link8": "fingers"},
    "aloha_agilex": {"fl_link6": "hand", "fl_link7": "fingers", "fl_link8": "fingers"},
    yam.ROBOT: dict(yam.LINK_PARTS),
}

#: ORA's changes to each file's ``robot_cfg.kinematics``: whole links' spheres and the buffer.
#: What the RoboDojo SDK keeps of its robot files for V2 (``CuroboPlanner._build_robot_and_scene_cfg``).
_KINEMATICS_FIELDS = frozenset({
    "format_version", "base_link", "tool_frames", "collision_link_names", "collision_spheres",
    "collision_sphere_buffer", "extra_collision_spheres", "self_collision_ignore", "self_collision_buffer",
    "asset_root_path", "mesh_link_names", "lock_joints", "extra_links", "cspace", "urdf_path", "use_global_cumul",
    "grasp_contact_link_names",
})
_CSPACE_FIELDS = frozenset({
    "joint_names", "default_joint_position", "cspace_distance_weight", "null_space_weight",
    "null_space_maximum_distance", "max_acceleration", "max_jerk", "velocity_scale", "acceleration_scale",
    "jerk_scale", "position_limit_clip",
})
OVERRIDES: dict[str, dict[str, Any]] = {
    "arx_x5": {
        "collision_sphere_buffer": SPHERE_BUFFER_M,
        "collision_spheres": {link: [{"center": list(center), "radius": radius} for center, radius in spheres]
                              for link, spheres in _X5_FINGERS.items()},
        "collision_link_names": ["link7", "link8"],
        "self_collision_ignore": {"link6": ["link7", "link8"], "link7": ["link6", "link8"], "link8": ["link6"],
                                  "camera": ["link7", "link8"], "camera_base": ["link7", "link8"]},
        "lock_joints": {"joint7": X5_FINGER_OPEN_M, "joint8": X5_FINGER_OPEN_M},
        # One entry per joint left once the fingers are locked; joint 6 keeps the file's limit.
        "position_limit_clip": [X5_URDF_LIMIT_RAD - limit for limit in X5_JOINT_LIMITS_RAD] + [0.0],
        "extra_links": {"attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint", "joint_type": "FIXED",
            "link_name": "attached_object", "parent_link_name": "link6"}},
        "held_ignore": ["link5", "link6", "link7", "link8", "camera", "camera_base"],
    },
    "aloha_agilex": {
        "collision_sphere_buffer": SPHERE_BUFFER_M,
        "collision_spheres": {"fl_link6": [{"center": list(center), "radius": radius}
                                           for center, radius in _ALOHA_LINK6]},
        # The folded home pose: these links rest about a millimetre apart there, inside the solvers' margin.
        "self_collision_ignore": {"fl_link2": ["fl_link5"], "fl_link4": ["fl_link6"]},
        "self_collision_buffer": {"fl_link3": ALOHA_FOLD_BUFFER_M, "fl_link5": ALOHA_FOLD_BUFFER_M},
        "extra_links": {"attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint", "joint_type": "FIXED",
            "link_name": "attached_object", "parent_link_name": "fl_link6"}},
        "held_ignore": ["fl_link5", "fl_link6", "fl_link7", "fl_link8", "left_camera"],
    },
    "franka_robotiq": {
        "collision_sphere_buffer": SPHERE_BUFFER_M,
        "collision_spheres": {link: [{"center": list(center), "radius": radius} for center, radius in spheres]
                              for link, spheres in _ROBOTIQ_SPHERES.items()},
    },
    # The fit is the file's own (``yam.robot_file``); here the buffer, the open fingers, the tool frame and the load.
    yam.ROBOT: {
        "collision_sphere_buffer": SPHERE_BUFFER_M,
        "collision_spheres": {},
        "self_collision_ignore": {},
        # The buffer is for the world: V2 d17b54ce pads self-collision by it too (its loader takes it off a copy of
        # the padding it never reads), so each link gives it back there, on top of the fold's.
        "self_collision_buffer": {link: yam.SELF_COLLISION_BUFFER.get(link, 0.0) - SPHERE_BUFFER_M
                                  for link in yam.COLLISION_LINKS},
        "lock_joints": {joint: yam.FINGER_OPEN_M for joint in yam.FINGER_JOINTS},
        "extra_links": {
            yam.TOOL_FRAME: {"fixed_transform": list(yam.TCP_IN_GRIPPER), "joint_name": "left_tcp_joint",
                             "joint_type": "FIXED", "link_name": yam.TOOL_FRAME, "parent_link_name": yam.HAND_LINK},
            "attached_object": {"fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint",
                                "joint_type": "FIXED", "link_name": "attached_object",
                                "parent_link_name": yam.HAND_LINK}},
        "held_ignore": list(yam.HELD_IGNORE),
    },
    "franka_panda": {
        "collision_sphere_buffer": SPHERE_BUFFER_M,
        "collision_spheres": {
            link: [{"center": [0.0, side * offset, round(along + FINGER_SHIFT_M, 6)], "radius": _FINGER_RADIUS_M}
                   for offset, along in _FINGER]
            for link, side in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0))
        },
    },
}


#: ``sphere_set_sha256`` of ``yam_linear_4310`` with the worker's 32 held-load slots (``engine.HELD_SLOTS``).
YAM_SPHERE_SET_SHA256 = "9d82e6a3ec6f2dc2f1fb48446748cc34411d4c24fe0876d8089c75baf8eb2ee7"


def robot_file(robot: str, path: str | Path) -> dict[str, Any]:
    """The V2 robot file a robot starts from: the parsed YAML at ``path``, or for a URDF robot the file ORA writes."""
    if robot in URDF_ROBOTS:
        return yam.robot_file(path)
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def robot_config(robot: str, bundled: Mapping[str, Any], *, held_link: str, held_slots: int) -> dict[str, Any]:
    """``bundled`` (a parsed V2 robot file) with ORA's changes and ``held_slots`` spheres on ``held_link``."""
    data = copy.deepcopy(dict(bundled))
    if "planner" in data:  # a simulator's file: only the kinematics its V2 translation keeps
        given = data["robot_cfg"]["kinematics"]
        data = {"robot_cfg": {"kinematics": _sdk_kinematics(_v078_kinematics(given) if "ee_link" in given else given)},
                "load_dynamics": False}
    kinematics = data["robot_cfg"]["kinematics"]
    override = OVERRIDES[robot]
    spheres = kinematics["collision_spheres"]
    added = override.get("collision_link_names", ())
    for link, replacement in override["collision_spheres"].items():
        if link not in spheres and link not in added:
            raise ValueError(f"V2 {BUNDLED_FILES.get(robot, robot)} has no collision spheres for {link}")
        spheres[link] = copy.deepcopy(replacement)
    if "held_ignore" in override:  # a file with no held-load link
        kinematics["collision_link_names"] = [*kinematics["collision_link_names"], *added, held_link]
        kinematics["collision_spheres"].setdefault(held_link, [])
        ignored = {link: list(values) for link, values in (kinematics.get("self_collision_ignore") or {}).items()}
        for link, others in override["self_collision_ignore"].items():
            ignored[link] = sorted({*ignored.get(link, ()), *others})
        ignored[held_link] = list(override["held_ignore"])
        kinematics["self_collision_ignore"] = ignored
        kinematics["self_collision_buffer"] = {**(kinematics.get("self_collision_buffer") or {}),
                                               **override.get("self_collision_buffer", {}),
                                               **{link: 0.0 for link in (*added, held_link)}}
        kinematics["lock_joints"] = dict(override.get("lock_joints", kinematics.get("lock_joints") or {}))
        kinematics["extra_links"] = copy.deepcopy(override["extra_links"])
    if "position_limit_clip" in override:
        kinematics["cspace"]["position_limit_clip"] = list(override["position_limit_clip"])
    kinematics["collision_sphere_buffer"] = override["collision_sphere_buffer"]
    kinematics["extra_collision_spheres"] = {held_link: int(held_slots)}
    return data


def _v078_kinematics(kinematics: Mapping[str, Any]) -> dict[str, Any]:
    """A v0.7.8 robot file's kinematics in V2's fields: its tool frame, its spheres inline and its limits."""
    import yaml

    spheres = yaml.safe_load(Path(kinematics["collision_spheres"]).read_text(encoding="utf-8"))["collision_spheres"]
    cspace = dict(kinematics["cspace"])
    names = list(cspace["joint_names"])
    return {**kinematics, "format_version": 2.0, "tool_frames": [kinematics["ee_link"]],
            "collision_spheres": {link: spheres[link] for link in kinematics["collision_link_names"]},
            "cspace": {**cspace, "default_joint_position": list(cspace["retract_config"]),
                       "max_acceleration": [float(cspace["max_acceleration"])] * len(names),
                       "max_jerk": [float(cspace["max_jerk"])] * len(names),
                       "position_limit_clip": ALOHA_URDF_LIMIT_RAD - ALOHA_JOINT_LIMIT_RAD}}


def _sdk_kinematics(kinematics: Mapping[str, Any]) -> dict[str, Any]:
    """A RoboDojo robot file's kinematics, as its SDK hands them to V2."""
    kept = {key: copy.deepcopy(value) for key, value in kinematics.items() if key in _KINEMATICS_FIELDS}
    kept["format_version"] = 2.0
    kept["cspace"] = {key: value for key, value in kept["cspace"].items() if key in _CSPACE_FIELDS}
    return kept


def sphere_set(kinematics: Mapping[str, Any]) -> dict[str, Any]:
    """The collision geometry ``kinematics`` (a ``robot_cfg.kinematics`` mapping) gives the planner."""
    spheres = kinematics["collision_spheres"]
    return {
        "collision_link_names": list(kinematics["collision_link_names"]),
        "collision_spheres": {link: [[*map(float, sphere["center"]), float(sphere["radius"])]
                                     for sphere in spheres[link]]
                              for link in kinematics["collision_link_names"] if link in spheres},
        "collision_sphere_buffer": kinematics.get("collision_sphere_buffer", 0.0),
        "extra_collision_spheres": dict(kinematics.get("extra_collision_spheres") or {}),
    }


def sphere_set_sha256(kinematics: Mapping[str, Any]) -> str:
    text = json.dumps(sphere_set(kinematics), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["ALOHA_FOLD_BUFFER_M", "ALOHA_JOINT_LIMIT_RAD", "BUNDLED_FILES", "FINGERS_PLANNED_OPEN", "FINGER_SHIFT_M", "LINK_PARTS", "OVERRIDES", "ROBOT_FILE_ROBOTS", "SPHERE_BUFFER_M",
           "URDF_ROBOTS", "X5_FINGER_OPEN_M", "X5_JOINT_LIMITS_RAD", "X5_URDF_LIMIT_RAD", "YAM_SPHERE_SET_SHA256",
           "robot_config", "robot_file", "sphere_set", "sphere_set_sha256"]
