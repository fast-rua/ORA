"""Robot descriptions the 3D replay poses: RoboDojo's dual ARX X5 and RoboTwin's aloha-agilex.

A description tells the page where the URDF is (a URL under ``robot/<id>/``, served only from inside the
configured directory), how the robot is mounted in the episode frame, which URDF joints the logged arm joints
and fingers drive, and which link plus a fixed offset reproduces the logged pose: the page's forward-kinematics
check, which reads about 0 mm when mounts and joint mapping are right. Visual meshes too large for a browser are
listed so the page skips them instead of requesting them.

- ``x5`` (RoboDojo): one X5 URDF per arm at ``X5_MOUNTS``; the logged pose is ``ee_link`` = the URDF's ``link6``.
- ``aloha`` (RoboTwin, ``embodiment_id: aloha_agilex``): one URDF holding both arms, mounted at the embodiment
  ``config.yml``'s ``robot_pose``; its ``arm_joints_name``, ``gripper_name``, ``move_group`` and
  ``gripper_bias`` give the joint mapping, the fingers and the TCP (``move_group`` link plus ``gripper_bias``
  along its x axis, RoboTwin's ``get_*_tcp_pose``), which is the measured TCP the controller replay logs.
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET

from viewer.dashboard.config import ASSET_ROOT_ENV, ROBOTWIN_ROOT_ENV, Config
from viewer.dashboard.store import cached_file

X5_URDF = "X5A.urdf"
X5_MOUNTS = {"left": {"pos": [-0.3, -0.45, 0.765], "quat": [0.0, 0.0, 0.707213, 0.707]},
             "right": {"pos": [0.3, -0.45, 0.765], "quat": [0.0, 0.0, 0.707213, 0.707]}}
ALOHA_CONFIG = "config.yml"
#: A visual mesh larger than this is not loaded in the browser (RoboDojo's camera body, the aloha's
#: decorative cameras and boxes, hundreds of MB together); the robot's arms are all far smaller.
MAX_MESH_BYTES = 8 << 20
FLAGS = {"x5": f"--robodojo-assets or {ASSET_ROOT_ENV}", "aloha": f"--robotwin-root or {ROBOTWIN_ROOT_ENV}"}


def robot_dir(robot_id: str, cfg: Config) -> str | None:
    return {"x5": cfg.robot_dir, "aloha": cfg.aloha_dir}.get(robot_id)


def robot_file(robot_id: str, rel: str, cfg: Config) -> str | None:
    """A file of a robot description (URDF, meshes, textures), only from inside that description's directory."""
    root = robot_dir(robot_id, cfg)
    if not root or not rel or ".." in rel or rel.startswith("/") or not re.match(r"^[A-Za-z0-9._/-]+$", rel):
        return None
    p = os.path.realpath(os.path.join(root, rel))
    if not p.startswith(os.path.realpath(root).rstrip(os.sep) + os.sep) or not os.path.isfile(p):
        return None
    return p


def _visual_meshes(urdf_path: str) -> list[str]:
    def load(p: str) -> list[str]:
        root = ET.parse(p).getroot()
        return sorted({m.get("filename") for v in root.iter("visual") for m in v.iter("mesh") if m.get("filename")})
    return cached_file(urdf_path, load, "urdf-meshes") or []


def _skipped(urdf_path: str) -> list[str]:
    """Visual meshes the page must not request: missing, or larger than ``MAX_MESH_BYTES``."""
    base = os.path.dirname(urdf_path)
    out = []
    for name in _visual_meshes(urdf_path):
        p = os.path.join(base, name)
        if not os.path.isfile(p) or os.path.getsize(p) > MAX_MESH_BYTES:
            out.append(name)
    return out


def x5(cfg: Config) -> dict | None:
    root = cfg.robot_dir
    if not root or not os.path.isfile(os.path.join(root, X5_URDF)):
        return None
    joints = ["joint%d" % i for i in range(1, 7)]
    return {"id": "x5", "name": "ARX X5", "kind": "per_arm", "urdf": "robot/x5/" + X5_URDF, "mounts": X5_MOUNTS,
            "joints": {"left": joints, "right": joints},
            "fingers": {"left": ["joint7", "joint8"], "right": ["joint7", "joint8"]},
            "fk_link": {"left": "link6", "right": "link6"}, "fk_offset": [0.0, 0.0, 0.0],
            "fk_label": "URDF FK vs logged ee_link", "skip": _skipped(os.path.join(root, X5_URDF))}


def _yaml_value(text: str, key: str):
    """One top-level ``key: <flow value>`` line of a RoboTwin embodiment config (JSON-like flow sequences and
    scalars; YAML's ``1.`` read as ``1.0``)."""
    m = re.search(r"^%s:[ \t]*(.+?)[ \t]*(?:#.*)?$" % re.escape(key), text, re.M)
    if not m:
        return None
    value = re.sub(r"(?<=\d)\.(?=[\s,\]])", ".0", m.group(1)).replace("'", '"')
    try:
        return json.loads(value)
    except ValueError:
        return value.strip('"')


def aloha_config(cfg: Config) -> dict | None:
    """The fields of the aloha-agilex ``config.yml`` the page needs, or None when they are not all there."""
    root = cfg.aloha_dir
    path = os.path.join(root, ALOHA_CONFIG) if root else None
    if not path or not os.path.isfile(path):
        return None

    def load(p: str) -> dict | None:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        pose = _yaml_value(text, "robot_pose")
        arms = _yaml_value(text, "arm_joints_name")
        links = _yaml_value(text, "move_group")
        scale = _yaml_value(text, "gripper_scale")
        bias = _yaml_value(text, "gripper_bias")
        urdf = _yaml_value(text, "urdf_path")
        block = re.search(r"^gripper_name:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
        bases = re.findall(r"base:\s*\"?([A-Za-z0-9_]+)", block.group(1)) if block else []
        mimics = re.findall(r"mimic:\s*\[\[\s*\"?([A-Za-z0-9_]+)", block.group(1)) if block else []
        ok = (isinstance(pose, list) and pose and isinstance(pose[0], list) and len(pose[0]) == 7
              and isinstance(arms, list) and len(arms) == 2 and isinstance(links, list) and len(links) == 2
              and isinstance(urdf, str) and isinstance(bias, (int, float)) and len(bases) == 2)
        if not ok:
            return None
        p0 = [float(v) for v in pose[0]]
        w, x, y, z = p0[3:]
        norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        fingers = [[bases[i]] + ([mimics[i]] if i < len(mimics) else []) for i in range(2)]
        return {"urdf": os.path.normpath(urdf),
                "root": {"pos": p0[:3], "quat": [x / norm, y / norm, z / norm, w / norm]},
                "joints": {"left": list(arms[0]), "right": list(arms[1])},
                "fingers": {"left": fingers[0], "right": fingers[1]},
                "fk_link": {"left": links[0], "right": links[1]}, "tcp_offset": [float(bias), 0.0, 0.0],
                "gripper_scale": [float(v) for v in scale] if isinstance(scale, list) and len(scale) == 2 else None}
    return cached_file(path, load, "aloha-config")


def aloha(cfg: Config) -> dict | None:
    conf = aloha_config(cfg)
    if not conf:
        return None
    urdf = robot_file("aloha", conf["urdf"], cfg)
    if not urdf:
        return None
    return {"id": "aloha", "name": "aloha-agilex", "kind": "single", "urdf": "robot/aloha/" + conf["urdf"],
            "root": conf["root"], "joints": conf["joints"], "fingers": conf["fingers"], "fk_link": conf["fk_link"],
            "fk_offset": conf["tcp_offset"], "fk_label": "URDF FK vs measured TCP", "skip": _skipped(urdf)}
