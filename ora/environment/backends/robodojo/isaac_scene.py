"""An Isaac Sim stage's bodies for the scene trace (``ora/environment/scene_trace.py``), read from USD and PhysX.

Nothing here is RoboDojo's: a caller names the stage root, its robots' articulation roots, its static prims and the
env origin. ``robodojo/scene_trace.py`` and ``robolab/scene_trace.py`` are its callers. Runs in the simulator's interpreter around a replay; the pose read and the USD reads change no state, and
``--check``'s comparison runs on the same re-execution and says so.

A moving body is a prim with an enabled ``UsdPhysics.RigidBodyAPI``: an object or an articulation link. Its pose comes
from one PhysX rigid-body view over all of them, read once per row: USD's transforms go stale under GPU physics, where
Fabric holds the live pose, and the view is the same tensor API Isaac Lab's assets read. A static body (the table, a
fixed collider) is a prim the caller names, posed once from USD. A deformable or a fluid has no rigid body and is not in
the trace. An object's pose is relative to the env origin, the frame the policy's calibration uses.

A body's visual geometry is every ``UsdGeom.Mesh`` under its prim the renderer draws: visible and of purpose ``default``
or ``render``, which leaves out collision meshes (hidden, or purpose ``guide``). Other gprims (a ``UsdGeom.Cube``) are
not tessellated: such a body is posed with no mesh. Vertices are in metres (Isaac stages are) in the body's frame, a
frame without scale: a pose carries none, so a body prim's own scale stays in its vertices. They come from USD's
transforms, whose stale pose cancels: mesh and body move together.

A mesh's colour is its bound material's base-colour texture sampled at its UVs, or the texture's mean on a mesh too
coarse for that, else the material's constant base colour, else grey. The material inputs read are OmniPBR's, glTF
``pbr.mdl``'s and ``UsdPreviewSurface``'s; a ``GeomSubset``'s own binding is ignored. Colours are linear, as in glTF: a
texture is sRGB, a constant is not.

The geometry is plain numpy on arrays (``part_mesh``); only the functions taking a prim touch ``pxr``, imported inside.
"""

from __future__ import annotations

import functools
import io
import re
import zipfile
from typing import Any, Callable, Sequence

from ora.environment.scene_trace import MIN_SAMPLED_VERTICES, Body, Mesh, SceneTraceError, linear, merge

#: Textures are read at most this size; a vertex colour needs no more.
TEXTURE_SIDE = 512
GREY = (0.5, 0.5, 0.5)
#: Where a shader holds its constant base colour: OmniPBR, UsdPreviewSurface, glTF ``pbr.mdl``, others.
COLOR_INPUTS = ("diffuse_color_constant", "diffuseColor", "base_color_factor", "base_color", "baseColor")
#: Where a shader holds its base-colour texture, as an asset or a connection to a texture shader.
TEXTURE_INPUTS = ("diffuse_texture", "base_color_texture", "diffuseColor", "base_color", "baseColor")
#: The asset input of a texture shader: ``UsdUVTexture``, glTF ``gltf_texture_lookup``.
TEXTURE_FILE_INPUTS = ("file", "texture")


def _numpy(value: Any) -> Any:
    import numpy as np

    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def fan_corners(counts: Any) -> Any:
    """Each polygon of ``counts`` as a triangle fan: (F, 3) positions into the mesh's face-vertex list."""
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    fans = [np.stack([starts[counts > k + 1], starts[counts > k + 1] + k, starts[counts > k + 1] + k + 1], axis=1)
            for k in range(1, int(counts.max(initial=0)) - 1)]
    return np.concatenate(fans) if fans else np.zeros((0, 3), dtype=np.int64)


def sample(image: Any, uv: Any) -> Any:
    """Linear RGB of an (H, W, 3) uint8 image at USD texture coordinates, whose origin is the bottom left."""
    import numpy as np

    height, width = image.shape[:2]
    fraction = np.mod(np.asarray(uv, dtype=np.float64), 1.0)
    columns = np.minimum((fraction[:, 0] * width).astype(np.int64), width - 1)
    rows = np.minimum(((1.0 - fraction[:, 1]) * height).astype(np.int64), height - 1)
    return linear(image[rows, columns].astype(np.float64) / 255.0)


def part_mesh(points: Any, counts: Any, indices: Any, to_body: Any, *, uv: Any = None, per_corner: bool = False,
              image: Any = None, rgb: Sequence[float] = GREY) -> Mesh:
    """One USD mesh in its body's frame. ``to_body`` is a USD 4x4 (row vectors, translation in the last row); ``uv``
    holds one row per point, or per face corner with ``per_corner``, and then each corner gets a vertex of its own."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(indices, dtype=np.int64)
    positions = (np.c_[points, np.ones(len(points))] @ np.asarray(to_body, dtype=np.float64))[:, :3]
    corners = fan_corners(counts)
    faces = indices[corners]
    sampled = image is not None and uv is not None and len(uv) == (len(indices) if per_corner else len(points))
    if sampled and len(positions) >= MIN_SAMPLED_VERTICES:
        colors = sample(image, uv)
        if per_corner:
            positions, faces = positions[indices], corners
    elif image is not None:
        colors = np.tile(linear(image.reshape(-1, 3).astype(np.float64).mean(axis=0) / 255.0), (len(positions), 1))
    else:
        colors = np.tile(np.asarray(rgb, dtype=np.float64)[:3], (len(positions), 1))
    alpha = np.ones((len(positions), 1))
    return Mesh(positions, np.clip(np.rint(np.concatenate([colors, alpha], axis=1) * 255.0), 0, 255).astype(np.uint8),
                faces)


def pose_matrix(matrix: Any) -> Any:
    """A USD 4x4 without its scale (row vectors: each of the first three rows is an axis times its scale)."""
    import numpy as np

    pose = np.array(matrix, dtype=np.float64).reshape(4, 4)
    pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=1, keepdims=True)
    return pose


def pose_xyz_xyzw(pose: Any, origin: Sequence[float]) -> list[float]:
    """A ``pose_matrix`` as xyz relative to ``origin`` and a quaternion xyzw."""
    import numpy as np

    rotation = np.asarray(pose, dtype=np.float64)[:3, :3].T     # USD's row vectors to the usual column form
    trace = float(np.trace(rotation))
    candidates = [trace, rotation[0, 0], rotation[1, 1], rotation[2, 2]]
    largest = int(np.argmax(candidates))
    if largest == 0:
        q = [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1],
             1.0 + trace]
    else:
        i, j, k = largest - 1, largest % 3, (largest + 1) % 3
        q = [0.0, 0.0, 0.0, rotation[k, j] - rotation[j, k]]
        q[i] = 1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]
        q[j] = rotation[j, i] + rotation[i, j]
        q[k] = rotation[k, i] + rotation[i, k]
    quaternion = np.asarray(q) / np.linalg.norm(q)
    position = np.asarray(pose, dtype=np.float64)[3, :3] - np.asarray(origin, dtype=np.float64)
    return [float(value) for value in (*position, *quaternion)]


class RigidPoses:
    """World poses of many rigid prims from one PhysX view (xyz, quaternion xyzw), fetched once per ``refresh``."""

    def __init__(self, view: Any, paths: Sequence[str], origin: Sequence[float]) -> None:
        self.view, self.origin, self.values = view, _numpy(origin).astype(float), None
        self.columns = {str(path): index for index, path in enumerate(view.prim_paths)}
        missing = sorted(set(paths) - set(self.columns))
        if missing:
            raise SceneTraceError(f"the rigid-body view holds no {missing}")

    def refresh(self) -> None:
        self.values = None

    def reader(self, path: str) -> Any:
        column = self.columns[path]

        def read() -> list[float]:
            if self.values is None:
                self.values = _numpy(self.view.get_transforms())
            pose = self.values[column]
            return [*(pose[:3] - self.origin), *pose[3:7]]

        return read


@functools.lru_cache(maxsize=64)
def _image(resolved: str) -> Any:
    """A texture as (H, W, 3) uint8, from a file or inside a usdz; None for one this reader cannot open."""
    import numpy as np

    try:
        from PIL import Image

        package = re.fullmatch(r"(.+\.usdz)\[(.+)\]", resolved)
        if package:
            with zipfile.ZipFile(package[1]) as archive:
                image = Image.open(io.BytesIO(archive.read(package[2].removeprefix("./"))))
                image.load()
        else:
            image = Image.open(resolved)
        image = image.convert("RGB")
        image.thumbnail((TEXTURE_SIDE, TEXTURE_SIDE))
        return np.asarray(image)
    except Exception:  # noqa: BLE001 - a texture it cannot read leaves the material's colour
        return None


def _shader_texture(shader: Any) -> str | None:
    """The base-colour texture a shader names, directly or through the texture shader connected to it."""
    from pxr import Sdf, UsdShade

    for name in TEXTURE_INPUTS:
        slot = shader.GetInput(name)
        if not slot:
            continue
        value = slot.Get()
        if isinstance(value, Sdf.AssetPath) and value.path:
            return value.resolvedPath or value.path
        if slot.HasConnectedSource():
            source = UsdShade.Shader(slot.GetConnectedSource()[0].GetPrim())
            for key in TEXTURE_FILE_INPUTS:
                file = source.GetInput(key)
                value = file.Get() if file else None
                if isinstance(value, Sdf.AssetPath) and value.path:
                    return value.resolvedPath or value.path
    return None


def _shader_color(shader: Any) -> list[float] | None:
    for name in COLOR_INPUTS:
        slot = shader.GetInput(name)
        value = slot.Get() if slot else None
        if hasattr(value, "__len__") and not isinstance(value, str) and len(value) >= 3:
            return [min(max(float(item), 0.0), 1.0) for item in list(value)[:3]]
    return None


def material_look(prim: Any) -> tuple[Any, list[float] | None]:
    """The (image, constant colour) of a mesh's bound material; either may be None."""
    from pxr import Usd, UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        return None, None
    image, color = None, None
    for node in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(node)
        if not shader:
            continue
        if image is None:
            texture = _shader_texture(shader)
            image = _image(texture) if texture else None
        color = color or _shader_color(shader)
    return image, color


def _texture_coordinates(prim: Any) -> tuple[Any, bool]:
    """A mesh's flattened UVs and whether they are per face corner; (None, False) without them."""
    from pxr import UsdGeom

    primvars = UsdGeom.PrimvarsAPI(prim)
    named = [primvars.GetPrimvar(name) for name in ("st", "uv") if primvars.HasPrimvar(name)]
    roles = [primvar for primvar in primvars.GetPrimvars() if primvar.GetTypeName().role == "TextureCoordinate"]
    for primvar in (*named, *roles):
        values = primvar.ComputeFlattened()
        if values is not None and len(values):
            return _numpy(values).reshape(-1, 2), primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying
    return None, False


def _display_color(prim: Any) -> list[float] | None:
    from pxr import UsdGeom

    primvars = UsdGeom.PrimvarsAPI(prim)
    values = primvars.GetPrimvar("displayColor").Get() if primvars.HasPrimvar("displayColor") else None
    return _numpy(values).reshape(-1, 3).mean(axis=0).tolist() if values is not None and len(values) else None


def _drawn(prim: Any) -> bool:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    return (prim.IsA(UsdGeom.Mesh) and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render))


def _traversal() -> Any:
    from pxr import Usd

    return Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)


def body_mesh(body: Any, cache: Any) -> Mesh | None:
    """Everything the renderer draws under a body's prim as one mesh in its frame (see the module docstring)."""
    import numpy as np
    from pxr import Usd, UsdGeom

    world_to_body = np.linalg.inv(pose_matrix(cache.GetLocalToWorldTransform(body)))
    parts = []
    for prim in Usd.PrimRange(body, _traversal()):
        if not _drawn(prim):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        if points is None or counts is None or not len(points) or not len(counts):
            continue
        image, color = material_look(prim)
        uv, per_corner = _texture_coordinates(prim)
        parts.append(part_mesh(_numpy(points), _numpy(counts), _numpy(mesh.GetFaceVertexIndicesAttr().Get()),
                               np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ world_to_body,
                               uv=uv, per_corner=per_corner, image=image, rgb=color or _display_color(prim) or GREY))
    return merge(parts)


def _key(path: str, root: str, *, robot: bool, seen: dict[str, int]) -> str:
    """A body's name across rounds: a robot link keeps its articulation, anything else is its own prim's name."""
    relative = path.removeprefix(root.rstrip("/") + "/")
    name = relative if robot else relative.rsplit("/", 1)[-1]
    seen[name] = seen.get(name, 0) + 1
    return name if seen[name] == 1 else f"{name}#{seen[name]}"


def scene_bodies(stage: Any, root: str, physics_view: Any, *, robots: Sequence[str] = (), statics: Sequence[str] = (),
                 origin: Sequence[float] = (0.0, 0.0, 0.0),
                 mesh_of: Callable[[Any, Any], Mesh | None] = body_mesh) -> tuple[list[Body], RigidPoses]:
    """Every body under ``root``, and the view their poses come from (``refresh`` it after each physics step).

    ``physics_view`` is the simulation's PhysX view (``SimulationManager.get_physics_sim_view()``); ``robots`` the
    articulation roots whose links are ``robot`` bodies; ``statics`` the prims that are ``static`` bodies;
    ``mesh_of(prim, cache)`` a body's mesh, for a stage drawn with more than ``UsdGeom.Mesh``.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    cache, origin = UsdGeom.XformCache(), _numpy(origin).astype(float)
    moving = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(root), _traversal())
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False]
    paths = [str(prim.GetPath()) for prim in moving]
    if not paths:
        raise SceneTraceError(f"no rigid body under {root}")
    poses = RigidPoses(physics_view.create_rigid_body_view(paths), paths, origin)
    bodies, seen = [], {}
    for prim, path in zip(moving, paths):
        robot = any(path.startswith(owner.rstrip("/") + "/") for owner in robots)
        key = _key(path, root, robot=robot, seen=seen)
        bodies.append(Body(key=key, name=key, kind="robot" if robot else "object", read_pose=poses.reader(path),
                           mesh=mesh_of(prim, cache)))
    for path in statics:
        prim = stage.GetPrimAtPath(path)
        key = _key(path, root, robot=False, seen=seen)
        pose = pose_xyz_xyzw(pose_matrix(cache.GetLocalToWorldTransform(prim)), origin)
        bodies.append(Body(key=key, name=key, kind="static", read_pose=lambda pose=pose: pose,
                           mesh=mesh_of(prim, cache)))
    return bodies, poses


__all__ = ["RigidPoses", "body_mesh", "fan_corners", "material_look", "part_mesh", "pose_matrix", "pose_xyz_xyzw",
           "sample", "scene_bodies"]
