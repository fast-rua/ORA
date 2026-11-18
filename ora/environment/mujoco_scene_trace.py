"""A MuJoCo scene's bodies for the scene trace (``ora/environment/scene_trace.py``), read from the model and data.

Any scene on MuJoCo: LIBERO-Pro (``backends/libero_pro/scene_trace.py``) and RoboCasa365
(``backends/robocasa/scene_trace.py``). The caller hands in the simulator object (robosuite's ``MjSim``, or anything
with ``model``, ``data`` and a ``step`` method) and names its robots' root bodies and TCP sites. A body is a model body, the world's included; a
robot's bodies are everything under its root, a body welded to the world is ``static``, the rest are ``object``.

A body's visual geometry is its geoms in the groups robosuite renders (``VISUAL_GROUPS``; group 0 is collision) that
are not fully transparent. ``geom_pos`` and ``geom_quat`` already place a compiled mesh's centred ``mesh_vert`` in the
body's frame, and those vertices are metres after ``mesh_scale``. Primitives are built here from ``geom_size``; a
plane is a quad, which the writer's ``backdrop`` rule then leaves undrawn, and a height field or SDF has no mesh.

A geom's colour is what MuJoCo draws: its material's ``rgba`` unless the geom sets its own, multiplied by the
material's 2D texture sampled at the vertices, or by the texture's mean on a part too coarse for that, without
texture coordinates, or on a cube map. MuJoCo draws those values as they are, so they count as sRGB here.

``mj_step`` computes kinematics before it integrates, so after step ``n`` ``data.xpos`` holds the pose of ``qpos``
after ``n - 1`` steps. What the run recorded at step ``n`` reads those same arrays (a receipt's TCP, the cameras), and
a row reads them as they stand: the trace sits on the run's own record, and reading moves nothing. A receipt read after
a controller load is one step ahead of the row, since robosuite's controller runs ``mj_forward``.

Nothing here imports MuJoCo: it reads the arrays of the running model, which also keeps it testable without one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.scene_trace import (
    MIN_SAMPLED_VERTICES, Body, Mesh, SceneTraceError, SceneTraceWriter, StepTrace, linear, merge, rotation,
)

CLOCK_UNIT = "mujoco_step_since_round_reset"
#: Rows per second of simulated time: the rate the dashboard's robot frames use.
ROW_HZ = 50
#: What robosuite renders: MuJoCo's visible groups 0-2 with group 0, the collision geometry, turned off.
VISUAL_GROUPS = (1, 2)
#: ``mjtGeom``.
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
#: ``mjtTexture``'s 2D type and ``mjtTextureRole``'s RGB slot in ``mat_texid``.
TEXTURE_2D, TEXTURE_ROLE_RGB = 0, 1
#: A geom's ``rgba`` until it sets one; a geom left at this shows its material's.
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)
#: An infinite plane (a zero size) drawn this far each way.
INFINITE_PLANE_M = 5.0
#: Facets around a round primitive and rings over a sphere's half.
ROUND_SEGMENTS, DOME_RINGS = 24, 6


def _name(model: Any, address: Any) -> str:
    names = bytes(model.names)
    return names[int(address):names.index(b"\0", int(address))].decode("utf-8")


def _lathe(profile: Sequence[tuple[float, float]]) -> tuple[Any, Any]:
    """A surface of revolution about z from its (radius, z) profile."""
    import numpy as np

    angle = 2.0 * np.pi * np.arange(ROUND_SEGMENTS) / ROUND_SEGMENTS
    rings = [np.stack([radius * np.cos(angle), radius * np.sin(angle), np.full(ROUND_SEGMENTS, height)], axis=1)
             for radius, height in profile]
    faces = []
    for ring in range(len(profile) - 1):
        for step in range(ROUND_SEGMENTS):
            a, b = ring * ROUND_SEGMENTS + step, ring * ROUND_SEGMENTS + (step + 1) % ROUND_SEGMENTS
            faces += [[a, a + ROUND_SEGMENTS, b], [b, a + ROUND_SEGMENTS, b + ROUND_SEGMENTS]]
    return np.concatenate(rings), np.array(faces, dtype=np.int64)


def _dome(radius: float, top: float, sign: float) -> list[tuple[float, float]]:
    """A hemisphere's profile from its rim at ``top`` to its pole at ``top + sign * radius``."""
    import numpy as np

    return [(radius * float(np.cos(angle)), top + sign * radius * float(np.sin(angle)))
            for angle in np.linspace(0.0, np.pi / 2.0, DOME_RINGS + 1)]


def _primitive(kind: int, size: Any) -> tuple[Any, Any] | None:
    """A primitive geom's vertices and faces in the geom's frame; None for a kind with no mesh here."""
    import numpy as np

    if kind == BOX:
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        return corners * np.asarray(size[:3], dtype=np.float64), np.array(
            [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
    if kind == PLANE:
        x, y = (float(value) or INFINITE_PLANE_M for value in size[:2])
        return np.array([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]]), np.array([[0, 1, 2], [0, 2, 3]])
    radius, half = float(size[0]), float(size[1])
    if kind == CYLINDER:
        return _lathe([(0.0, half), (radius, half), (radius, -half), (0.0, -half)])
    if kind == CAPSULE:
        return _lathe(list(reversed(_dome(radius, half, 1.0))) + _dome(radius, -half, -1.0))
    if kind in (SPHERE, ELLIPSOID):
        vertices, faces = _lathe(list(reversed(_dome(1.0, 0.0, 1.0))) + _dome(1.0, 0.0, -1.0)[1:])
        scale = [radius] * 3 if kind == SPHERE else np.asarray(size[:3], dtype=np.float64)
        return vertices * scale, faces
    return None


def _mesh(model: Any, mesh: int) -> tuple[Any, Any, Any]:
    """A compiled mesh: vertices, faces, and per face corner the texture coordinate (None without any)."""
    import numpy as np

    first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    first_face, faces = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    triangles = np.asarray(model.mesh_face[first_face:first_face + faces], dtype=np.int64).reshape(-1, 3)
    start = int(model.mesh_texcoordadr[mesh])
    corners = None
    if start >= 0:
        indexes = np.asarray(model.mesh_facetexcoord[first_face:first_face + faces], dtype=np.int64).reshape(-1)
        corners = np.asarray(model.mesh_texcoord, dtype=np.float64)[start + indexes]
    return np.asarray(model.mesh_vert[first:first + count], dtype=np.float64).reshape(-1, 3), triangles, corners


def _texture(model: Any, material: int) -> tuple[Any, bool] | None:
    """A material's colour texture as rows of uint8 RGB and whether it is 2D; None without one."""
    import numpy as np

    texture = int(model.mat_texid[material][TEXTURE_ROLE_RGB])
    if texture < 0:
        return None
    height, width, channels = (int(model.tex_height[texture]), int(model.tex_width[texture]),
                               int(model.tex_nchannel[texture]))
    first = int(model.tex_adr[texture])
    pixels = np.asarray(model.tex_data[first:first + height * width * channels]).reshape(height, width, channels)
    pixels = pixels[..., :3] if channels >= 3 else np.repeat(pixels[..., :1], 3, axis=2)
    return pixels, int(model.tex_type[texture]) == TEXTURE_2D


def _sampled(pixels: Any, triangles: Any, corners: Any, count: int) -> Any:
    """Each vertex's colour: the texture at the vertex's face corners, averaged where they differ (a seam)."""
    import numpy as np

    height, width = pixels.shape[:2]
    fraction = np.mod(corners, 1.0)
    columns = np.minimum((fraction[:, 0] * width).astype(np.int64), width - 1)
    rows = np.minimum((fraction[:, 1] * height).astype(np.int64), height - 1)
    colours, vertices = pixels[rows, columns].astype(np.float64) / 255.0, triangles.reshape(-1)
    weight = np.maximum(np.bincount(vertices, minlength=count), 1)
    return np.stack([np.bincount(vertices, colours[:, channel], count) / weight for channel in range(3)], axis=1)


def _geom_rgba(model: Any, geom: int) -> Any:
    import numpy as np

    rgba = np.asarray(model.geom_rgba[geom], dtype=np.float64)
    material = int(model.geom_matid[geom])
    return np.asarray(model.mat_rgba[material], dtype=np.float64) if material >= 0 and np.allclose(rgba, DEFAULT_RGBA) \
        else rgba


def _colors(model: Any, geom: int, count: int, triangles: Any, corners: Any) -> Any:
    """One geom's vertex colours (see the module docstring)."""
    import numpy as np

    rgba = _geom_rgba(model, geom)
    material = int(model.geom_matid[geom])
    texture = _texture(model, material) if material >= 0 else None
    rgb = np.tile(rgba[:3], (count, 1))
    if texture is not None:
        pixels, flat = texture
        if flat and corners is not None and count >= MIN_SAMPLED_VERTICES:
            rgb = rgb * _sampled(pixels, triangles, corners, count)
        else:
            rgb = rgb * (pixels.reshape(-1, 3).mean(axis=0) / 255.0)
    alpha = np.full((count, 1), rgba[3])
    return np.clip(np.rint(np.concatenate([linear(rgb), alpha], axis=1) * 255.0), 0, 255).astype(np.uint8)


def _geom_mesh(model: Any, geom: int) -> Mesh | None:
    """One visual geom in its body's frame."""
    import numpy as np

    kind = int(model.geom_type[geom])
    if kind == MESH:
        vertices, faces, corners = _mesh(model, int(model.geom_dataid[geom]))
    else:
        shape = _primitive(kind, model.geom_size[geom])
        if shape is None:
            return None
        (vertices, faces), corners = shape, None
    turn, origin = rotation(model.geom_quat[geom]), np.asarray(model.geom_pos[geom], dtype=np.float64)
    return Mesh(vertices @ turn.T + origin, _colors(model, geom, len(vertices), faces, corners), faces)


def body_mesh(model: Any, body: int) -> Mesh | None:
    """A body's visual geoms as one mesh in the body's frame."""
    first = int(model.body_geomadr[body])
    parts = [_geom_mesh(model, geom) for geom in range(first, first + int(model.body_geomnum[body]))
             if int(model.geom_group[geom]) in VISUAL_GROUPS and _geom_rgba(model, geom)[3] > 0]
    return merge([part for part in parts if part is not None])


def _pose_reader(data: Any, body: int) -> Any:
    def read() -> list[float]:
        p, q = data.xpos[body], data.xquat[body]
        return [p[0], p[1], p[2], q[1], q[2], q[3], q[0]]
    return read


def scene_bodies(model: Any, data: Any, *, robot_roots: Sequence[str] = ()) -> list[Body]:
    """Every body of the model (see the module docstring); ``robot_roots`` names each robot's top body."""
    names = [_name(model, address) for address in model.name_bodyadr[:model.nbody]]
    missing = set(robot_roots) - set(names)
    if missing:
        raise SceneTraceError(f"the model has no robot root body {sorted(missing)}")
    robot = {int(model.body_rootid[names.index(root)]) for root in robot_roots}
    bodies = []
    for index, name in enumerate(names):
        welded = int(model.body_weldid[index]) == 0
        kind = "robot" if int(model.body_rootid[index]) in robot else "static" if welded else "object"
        bodies.append(Body(key=name, name=name, kind=kind, read_pose=_pose_reader(data, index),
                           mesh=body_mesh(model, index)))
    return bodies


def tcp_links(model: Any, sites: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Each arm's TCP site as its body and its fixed pose in that body's frame (xyz + quaternion xyzw)."""
    names = [_name(model, address) for address in model.name_siteadr[:model.nsite]]
    links: dict[str, dict[str, Any]] = {}
    for arm, site_name in sites.items():
        if site_name not in names:
            raise SceneTraceError(f"the model has no TCP site {site_name!r}")
        site = names.index(site_name)
        p, q = model.site_pos[site], model.site_quat[site]
        links[arm] = {"body": _name(model, model.name_bodyadr[int(model.site_bodyid[site])]),
                      "offset_xyz_xyzw": [float(value) for value in (*p, q[1], q[2], q[3], q[0])]}
    return links


def row_stride(timestep_s: float) -> int:
    """Every how many physics steps a row falls, for ``ROW_HZ``."""
    return max(1, round(1.0 / (ROW_HZ * timestep_s)))


class MuJoCoSceneTrace(StepTrace):
    """A replay's trace on any MuJoCo scene; a backend supplies the simulator and names its robots."""

    def __init__(self, directory: Path, *, run_id: str, backend: str) -> None:
        super().__init__(SceneTraceWriter(directory, run_id=run_id, backend=backend, clock_unit=CLOCK_UNIT))

    def begin_round(self, sim: Any, round_index: int, *, robot_roots: Sequence[str] = (),
                    tcp_sites: Mapping[str, str] | None = None) -> None:
        self.close()
        model, data = getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
        timestep_s = float(model.opt.timestep)
        self.writer.stride = row_stride(timestep_s)
        self.writer.tcp = tcp_links(model, tcp_sites or {})
        self.writer.begin_round(round_index, scene_bodies(model, data, robot_roots=robot_roots), timestep_s=timestep_s)
        self.watch(sim)


__all__ = ["CLOCK_UNIT", "ROW_HZ", "MuJoCoSceneTrace", "body_mesh", "row_stride", "scene_bodies", "tcp_links"]
