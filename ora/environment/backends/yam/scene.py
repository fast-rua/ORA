"""The station model with its cell: the table, loose objects and the cameras.

Built on i2rt's station MJCF with ``mujoco.MjSpec``. The same compiled model serves the
planner (table clearance, named ``ora_table``) and, in simulation, the renderer. Every
position here is in ``yam_station_world``; the spec itself is in the station frame. The
simulated real cell (``sim_cell_adapter.py``) adds a checkerboard the person moves
(``CHECKERBOARD_BODY``, a mocap body).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers import yam_station as Y

from .kinematics import station_xml_path

# A camera's MuJoCo frame looks down -z with +y up; the OpenCV optical frame looks down
# +z with +y down. They differ by a 180 deg turn about x.
CV_FROM_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])

# Default cell of the simulated station: a table in front of the arms, its top on the
# base plane, and two blocks on it.
DEFAULT_TABLE = {"center_xy_m": [0.45, 0.0], "size_xy_m": [0.6, 0.9], "top_z_m": 0.0}
DEFAULT_OBJECTS = (
    {"name": "red_block", "size_m": [0.04, 0.04, 0.04], "position_m": [0.38, 0.12, 0.02], "rgba": [0.85, 0.15, 0.1, 1]},
    {"name": "blue_block", "size_m": [0.05, 0.03, 0.03], "position_m": [0.42, -0.14, 0.015], "rgba": [0.1, 0.3, 0.85, 1]},
)
# Fixed cameras of the simulated cell, placed like the real station's two D435i: one over
# the table looking down, one behind and above the arms looking forward and down.
DEFAULT_FIXED_CAMERAS = {
    "overhead": {"position_m": [0.45, 0.0, 1.05], "look_at_m": [0.45, 0.0, 0.0], "up_hint": [1.0, 0.0, 0.0]},
    "front": {"position_m": [-0.25, 0.0, 0.70], "look_at_m": [0.45, 0.0, 0.0], "up_hint": [0.0, 0.0, 1.0]},
}
SIM_IMAGE_SIZE = (848, 480)
CHECKERBOARD_BODY = "ora_checkerboard"
SIM_FOVY_DEG = {"wrist": 58.0, "fixed": 42.0}


def _to_station(position: Sequence[float]) -> list[float]:
    return [float(position[0]) - Y.STATION_TO_WORLD_XYZ_M[0], float(position[1]) - Y.STATION_TO_WORLD_XYZ_M[1],
            float(position[2]) - Y.STATION_TO_WORLD_XYZ_M[2]]


def _look_at(position: Sequence[float], target: Sequence[float], up_hint: Sequence[float]) -> list[float]:
    """MuJoCo ``xyaxes`` of a camera at ``position`` looking at ``target``."""

    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    x_axis = np.cross(forward, np.asarray(up_hint, float))
    if np.linalg.norm(x_axis) < 1e-6:
        raise ValueError("camera up hint is parallel to its view")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, forward)    # MuJoCo camera +y is image up
    return [*x_axis.tolist(), *y_axis.tolist()]


def build_cell_model(*, table: Mapping[str, Any] | None = DEFAULT_TABLE,
                     objects: Sequence[Mapping[str, Any]] = (),
                     fixed_cameras: Mapping[str, Mapping[str, Any]] | None = None,
                     wrist_cameras: bool = False, checkerboard: Mapping[str, Any] | None = None):
    """Compile the station model with the given cell. Returns the ``MjModel``.

    ``checkerboard`` (``square_m``, ``pattern`` of inner corners, ``margin_m``,
    ``thickness_m``) adds a plain checkerboard as the mocap body ``CHECKERBOARD_BODY``: its
    frame is the board's (first inner corner, x along the columns, y along the rows, z out of
    the printed face), placed by ``data.mocap_pos`` and ``mocap_quat`` in the station frame."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(station_xml_path()))
    world = spec.worldbody
    if table is not None:
        cx, cy = table["center_xy_m"]
        sx, sy = table["size_xy_m"]
        geom = world.add_geom()
        geom.name = "ora_table"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [sx / 2, sy / 2, 0.02]
        geom.pos = _to_station([cx, cy, float(table["top_z_m"]) - 0.02])
        geom.rgba = [0.72, 0.58, 0.42, 1.0]
    for item in objects:
        body = world.add_body()
        body.name = f"ora_object_{item['name']}"
        body.pos = _to_station(item["position_m"])
        body.add_freejoint()
        geom = body.add_geom()
        geom.name = f"ora_object_{item['name']}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [float(v) / 2 for v in item["size_m"]]
        geom.rgba = list(item["rgba"])
    if checkerboard is not None:
        _add_checkerboard(spec, checkerboard)
    if fixed_cameras or wrist_cameras:
        light = world.add_light()
        light.pos = _to_station([0.4, 0.0, 2.0])
        light.dir = [0.0, 0.0, -1.0]
        light.diffuse = [0.9, 0.9, 0.9]
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.map.znear = 0.005
        spec.visual.global_.offwidth = max(SIM_IMAGE_SIZE[0], 640)
        spec.visual.global_.offheight = max(SIM_IMAGE_SIZE[1], 480)
    for name, camera in (fixed_cameras or {}).items():
        cam = world.add_camera()
        cam.name = f"ora_{name}"
        cam.pos = _to_station(camera["position_m"])
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = _look_at(camera["position_m"], camera["look_at_m"], camera["up_hint"])
        cam.fovy = SIM_FOVY_DEG["fixed"]
    if wrist_cameras:
        for arm in Y.ARMS:
            cam = spec.body(Y.WRIST_CAMERA_BODIES[arm]).add_camera()
            cam.name = f"ora_wrist_{arm}"
            cam.quat = [0.0, 1.0, 0.0, 0.0]      # the body is the optical frame; turn it into MuJoCo's
            cam.fovy = SIM_FOVY_DEG["wrist"]
    return spec.compile()


def _add_checkerboard(spec: Any, board: Mapping[str, Any]) -> None:
    """White plate under black squares raised 0.2 mm: (columns + 1) x (rows + 1) squares
    around the inner corners, a white margin around them; matte, so no highlight hides a corner."""

    import mujoco

    square = float(board["square_m"])
    columns, rows = (int(v) for v in board["pattern"])
    margin, thickness = float(board.get("margin_m", 0.02)), float(board.get("thickness_m", 0.003))
    for name, rgba in (("ora_board_white", [0.93, 0.93, 0.93, 1.0]), ("ora_board_black", [0.04, 0.04, 0.04, 1.0])):
        material = spec.add_material()
        material.name, material.rgba = name, rgba
        material.specular, material.shininess, material.reflectance = 0.0, 0.0, 0.0
    body = spec.worldbody.add_body()
    body.name, body.mocap = CHECKERBOARD_BODY, True
    body.pos = [0.0, 0.0, -10.0]                  # out of sight until placed
    low, high = np.array([-square - margin, -square - margin]), np.array([columns * square + margin,
                                                                           rows * square + margin])
    plate = body.add_geom()
    plate.name, plate.type, plate.material = "ora_board_plate", mujoco.mjtGeom.mjGEOM_BOX, "ora_board_white"
    plate.size = [*((high - low) / 2), thickness / 2]
    plate.pos = [*((high + low) / 2), -thickness / 2]
    plate.contype = plate.conaffinity = 0
    for i in range(-1, columns):
        for j in range(-1, rows):
            if (i + j) % 2:
                continue
            geom = body.add_geom()
            geom.type, geom.material = mujoco.mjtGeom.mjGEOM_BOX, "ora_board_black"
            geom.size = [square / 2, square / 2, 0.0001]
            geom.pos = [(i + 0.5) * square, (j + 0.5) * square, 0.0001]
            geom.contype = geom.conaffinity = 0


def camera_intrinsic(fovy_deg: float, width: int, height: int) -> list[list[float]]:
    focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
