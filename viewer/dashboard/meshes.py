"""Convert a RoboDojo object asset (object.usdz) into a small mesh for the run dashboard's 3D replay.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31). Runs under an interpreter with
``usd-core``, numpy and Pillow; the dashboard calls it as a subprocess (``--mesh-python``, default the
dashboard's own interpreter) and never imports it:

    <python with usd-core> -I meshes.py <asset_dir> <out.bin> [--max-faces 20000]

Output (little endian): uint32 vertex count V, uint32 face count F, float32 positions V*3 (metres, Z up), uint8
colours V*3, uint32 faces F*3. The visual mesh is triangulated, simplified by vertex clustering to about
``max_faces`` faces, and coloured from its texture when it has one, else from its material's constant diffuse
colour (what the simulator renders, e.g. OmniPBR ``diffuse_color_constant``), else from displayColor.

Frame: the asset's default prim (``/root``) with that prim's own transform left out. RoboDojo references the asset
onto the instance prim and Isaac Sim replaces the root's xform ops with the instance's pose and scale (XFormPrim
``_set_xform_properties``), so a translate authored on ``/root`` (factory_bolt: z -0.02625) never reaches the
simulator; transforms below the root (factory_nut ``/root/visual``: z -0.01556) do. Positions are therefore in the
frame whose pose the ground-truth capture records for the instance.
"""

from __future__ import annotations

import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade


def _texture(usdz: Path, stage, mesh_prim) -> np.ndarray | None:
    """The diffuse texture bound to this mesh, as an HxWx3 uint8 array, if any."""
    try:
        from PIL import Image
    except ImportError:
        return None
    material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]
    names = []
    if material:
        for prim in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(prim)
            if shader and shader.GetIdAttr().Get() == "UsdUVTexture":
                asset = shader.GetInput("file").Get()
                if asset is not None:
                    names.append(asset.path)
    with zipfile.ZipFile(usdz) as z:
        members = z.namelist()
        pick = None
        for n in names:
            for m in members:
                if m.endswith(n.lstrip("./").split("/")[-1]):
                    pick = m
                    break
            if pick:
                break
        if pick is None:  # no bound texture found: the single image in the archive, if there is exactly one
            images = [m for m in members if m.lower().endswith((".png", ".jpg", ".jpeg"))]
            pick = images[0] if len(images) == 1 else None
        if pick is None:
            return None
        try:
            image = Image.open(io.BytesIO(z.read(pick))).convert("RGB")
        except Exception:  # an unreadable texture (some assets ship formats PIL cannot open): plain colour instead
            return None
    image.thumbnail((512, 512))
    return np.asarray(image)


_DIFFUSE_INPUTS = ("diffuse_color_constant", "diffuseColor", "base_color", "baseColor")


def _material_color(mesh_prim) -> np.ndarray | None:
    """The constant diffuse colour (0..255 RGB) of the material bound to this mesh, if it has one."""
    material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]
    if not material:
        return None
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        for name in _DIFFUSE_INPUTS:
            inp = shader.GetInput(name)
            value = inp.Get() if inp else None
            if value is not None and len(value) >= 3:
                return np.clip(np.asarray(value, np.float64)[:3], 0, 1) * 255
    return None


def _root_inverse(stage, cache) -> np.ndarray:
    """Inverse of the default prim's own transform (see the module docstring); identity without a default prim."""
    root = stage.GetDefaultPrim()
    if not root or not root.IsA(UsdGeom.Xformable):
        return np.eye(4)
    return np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root), np.float64))


def convert(asset_dir: Path, out: Path, max_faces: int = 20000) -> None:
    usdz = asset_dir / "object.usdz"
    stage = Usd.Stage.Open(str(usdz))
    cache = UsdGeom.XformCache()
    root_inv = _root_inverse(stage, cache)
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    visual = [p for p in meshes if "/visual" in str(p.GetPath())] or [p for p in meshes if "collision" not in str(p.GetPath())] or meshes
    positions, colors, faces, offset = [], [], [], 0
    for prim in visual:
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
        if not len(pts) or not len(counts):
            continue
        # row vectors: local -> stage (prim's local-to-world), then stage -> root frame without the root's own xform
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64) @ root_inv
        world = (np.c_[pts, np.ones(len(pts))] @ matrix)[:, :3] * UsdGeom.GetStageMetersPerUnit(stage)
        # triangulate polygons as fans: corner arrays per triangle
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri_corners = []
        for k in range(1, counts.max() - 1):
            sel = counts > k + 1
            s = starts[sel]
            tri_corners.append(np.stack([s, s + k, s + k + 1], axis=1))
        corners = np.concatenate(tri_corners) if tri_corners else np.zeros((0, 3), np.int64)
        tris = idx[corners]
        # per-vertex colour: texture sampled at UV (vertex or faceVarying), else displayColor, else gray
        vcol = np.full((len(pts), 3), 180.0)
        primvars = UsdGeom.PrimvarsAPI(prim)
        texture = _texture(usdz, stage, prim)
        uv_var = primvars.GetPrimvar("st") if primvars.HasPrimvar("st") else primvars.GetPrimvar("uv")
        uv = uv_var.Get() if uv_var else None
        if texture is not None and uv is not None:
            uv = np.asarray(uv, np.float64)
            if uv_var.GetIndices():
                uv = uv[np.asarray(uv_var.GetIndices())]
            h, w = texture.shape[:2]
            u = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
            v = np.clip((1 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
            sampled = texture[v, u].astype(np.float64)
            interp = uv_var.GetInterpolation()
            if interp == UsdGeom.Tokens.faceVarying and len(sampled) == len(idx):
                acc = np.zeros((len(pts), 3)); n = np.zeros(len(pts))
                np.add.at(acc, idx, sampled); np.add.at(n, idx, 1)
                vcol = np.where(n[:, None] > 0, acc / np.maximum(n, 1)[:, None], vcol)
            elif len(sampled) == len(pts):
                vcol = sampled
        elif (mc := _material_color(prim)) is not None:
            vcol = np.tile(mc, (len(pts), 1))
        else:
            dc = primvars.GetPrimvar("displayColor").Get() if primvars.HasPrimvar("displayColor") else None
            if dc is not None and len(dc):
                vcol = np.tile(np.asarray(dc, np.float64).mean(axis=0) * 255, (len(pts), 1))
        positions.append(world)
        colors.append(vcol)
        faces.append(tris + offset)
        offset += len(pts)
    if not positions:
        raise SystemExit(f"no mesh in {usdz}")
    P, C, F = np.vstack(positions), np.vstack(colors), np.vstack(faces)
    # vertex clustering: shrink the grid until the face count is under max_faces
    extent = float(np.ptp(P, axis=0).max())
    cells = 160
    while True:
        cell = extent / cells
        keys = np.floor((P - P.min(axis=0)) / cell).astype(np.int64)
        uniq, cluster = np.unique(keys, axis=0, return_inverse=True)
        cluster = cluster.ravel()
        tri = cluster[F]
        keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        tri = np.unique(np.sort(tri[keep], axis=1), axis=0) if keep.any() else tri[:0]
        if len(tri) <= max_faces or cells <= 20:
            break
        cells = int(cells * 0.8)
    n = len(uniq)
    cnt = np.bincount(cluster, minlength=n).astype(np.float64)[:, None]
    VP = np.zeros((n, 3)); VC = np.zeros((n, 3))
    np.add.at(VP, cluster, P); np.add.at(VC, cluster, C)
    VP /= cnt; VC /= cnt
    # sorting the corners above lost the winding; restore it against the original faces' normals is costly, so the
    # viewer renders double-sided
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n, len(tri)))
        f.write(VP.astype("<f4").tobytes())
        f.write(np.clip(VC, 0, 255).astype(np.uint8).tobytes())
        f.write(tri.astype("<u4").tobytes())
    tmp.replace(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    max_faces = 20000
    if "--max-faces" in args:
        i = args.index("--max-faces")
        max_faces = int(args[i + 1])
        del args[i:i + 2]
    convert(Path(args[0]), Path(args[1]), max_faces)
