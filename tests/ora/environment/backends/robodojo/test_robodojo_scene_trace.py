"""An Isaac stage's bodies for the scene trace, and RoboDojo's use of them: stand-ins shaped like USD's and PhysX's
objects, so no simulator. What only a live stage can show (the block's texture, the pose read) is in the design note."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import isaac_scene
from ora.environment.backends.robodojo.isaac_scene import (
    RigidPoses, fan_corners, part_mesh, pose_matrix, pose_xyz_xyzw, sample, scene_bodies,
)
from ora.environment.backends.robodojo.scene_trace import CLOCK_UNIT, STRIDE, RoboDojoSceneTrace, static_prims
from ora.environment.scene_trace import Body, Mesh, SceneTraceError

IDENTITY = np.eye(4)
QUAD_POINTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)


def _translated(x: float, y: float, z: float, scale: float = 1.0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[3, :3] = (x, y, z)
    return matrix


class Prim:
    """A stand-in for ``Usd.Prim``: the world transform, schemas and a mesh's arrays are plain fields."""

    def __init__(self, path: str, *children: "Prim", kind: str = "Xform", world: np.ndarray = IDENTITY, rigid: bool = False,
                 enabled: bool | None = True, visibility: str = "inherited", purpose: str = "default",
                 points=None, counts=None, indices=None, primvars=None, material=None, shader=None) -> None:
        self.path, self.children, self.kind, self.world = path, list(children), kind, world
        self.rigid, self.enabled, self.visibility, self.purpose = rigid, enabled, visibility, purpose
        self.points, self.counts, self.indices = points, counts, indices
        self.primvars, self.material, self.shader = primvars or {}, material, shader

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def HasAPI(self, _api) -> bool:
        return self.rigid

    def IsA(self, _kind) -> bool:
        return self.kind == "Mesh"


def _mesh(path: str, *, x: float = 0.0, **fields) -> Prim:
    return Prim(path, kind="Mesh", world=_translated(x, 0, 0), points=QUAD_POINTS, counts=[4], indices=[0, 1, 2, 3], **fields)


class Stage:
    def __init__(self, *prims: Prim) -> None:
        self.prims = {prim.path: prim for top in prims for prim in top}

    def GetPrimAtPath(self, path: str) -> Prim:
        return self.prims[path]


class Attribute:
    def __init__(self, value) -> None:
        self.value = value

    def Get(self):
        return self.value


class Asset:
    def __init__(self, path: str, resolved: str = "") -> None:
        self.path, self.resolvedPath = path, resolved


class Input:
    def __init__(self, value=None, source: Prim | None = None) -> None:
        self.value, self.source = value, source

    def Get(self):
        return self.value

    def HasConnectedSource(self) -> bool:
        return self.source is not None

    def GetConnectedSource(self):
        return (SimpleNamespace(GetPrim=lambda: self.source), "out", None)


class Primvar:
    def __init__(self, values, interpolation: str) -> None:
        self.values, self.interpolation = values, interpolation

    def ComputeFlattened(self):
        return self.values

    def GetInterpolation(self) -> str:
        return self.interpolation

    def GetTypeName(self):
        return SimpleNamespace(role="TextureCoordinate")


@pytest.fixture()
def pxr(monkeypatch: pytest.MonkeyPatch):
    class Shader:
        def __init__(self, prim: Prim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return self.prim.shader is not None

        def GetInput(self, name: str):
            return self.prim.shader.get(name)

    class Material:
        def __init__(self, prim: Prim) -> None:
            self.prim = prim

        def GetPrim(self) -> Prim:
            return self.prim

    module = ModuleType("pxr")
    module.Usd = SimpleNamespace(PrimRange=lambda prim, _predicate=None: iter(prim), PrimDefaultPredicate=None,
                                 TraverseInstanceProxies=lambda predicate: predicate)
    module.Sdf = SimpleNamespace(AssetPath=Asset)
    module.UsdPhysics = SimpleNamespace(RigidBodyAPI=lambda prim: SimpleNamespace(
        GetRigidBodyEnabledAttr=lambda: Attribute(prim.enabled)))
    module.UsdShade = SimpleNamespace(
        Shader=Shader, MaterialBindingAPI=lambda prim: SimpleNamespace(
            ComputeBoundMaterial=lambda: (Material(prim.material) if prim.material else None, None)))
    module.UsdGeom = SimpleNamespace(
        Mesh=lambda prim: SimpleNamespace(GetPointsAttr=lambda: Attribute(prim.points),
                                          GetFaceVertexCountsAttr=lambda: Attribute(prim.counts),
                                          GetFaceVertexIndicesAttr=lambda: Attribute(prim.indices)),
        Imageable=lambda prim: SimpleNamespace(ComputeVisibility=lambda: prim.visibility,
                                               ComputePurpose=lambda: prim.purpose),
        Tokens=SimpleNamespace(invisible="invisible", default_="default", render="render", faceVarying="faceVarying"),
        XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda prim: prim.world),
        PrimvarsAPI=lambda prim: SimpleNamespace(HasPrimvar=lambda name: name in prim.primvars,
                                                 GetPrimvar=lambda name: prim.primvars[name],
                                                 GetPrimvars=lambda: list(prim.primvars.values())))
    monkeypatch.setitem(sys.modules, "pxr", module)
    isaac_scene._image.cache_clear()
    return module


class Tensor:
    """What a CUDA tensor allows: no ``numpy`` of its own, ``detach().cpu().numpy()`` only."""

    def __init__(self, values) -> None:
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return SimpleNamespace(numpy=lambda: np.asarray(self.values))

    def __array__(self, *_args, **_kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


class View:
    """A PhysX rigid-body view: a transform per path, counting the reads."""

    def __init__(self, paths) -> None:
        self.prim_paths, self.reads = list(paths), 0
        self.transforms = np.array([[float(i), 0, 0, 0, 0, 0, 1] for i in range(len(self.prim_paths))], np.float32)

    def get_transforms(self):
        self.reads += 1
        return self.transforms.copy()


class Simulation:
    def __init__(self) -> None:
        self.view = None

    def create_rigid_body_view(self, paths):
        self.view = View(paths)
        return self.view


def test_polygons_become_triangle_fans_and_an_empty_mesh_none() -> None:
    assert fan_corners([3, 4]).tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6]]
    assert fan_corners([]).shape == (0, 3)


def test_a_mesh_lands_in_its_body_s_frame_in_one_colour() -> None:
    mesh = part_mesh(QUAD_POINTS, [4], [0, 1, 2, 3], _translated(1, 2, 3, scale=2.0), rgb=(1.0, 0.0, 0.0))

    assert mesh.positions[2].tolist() == [3.0, 4.0, 3.0]        # (1, 1, 0) scaled by 2, then moved
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert mesh.colors.tolist() == [[255, 0, 0, 255]] * 4


def test_a_texture_is_sampled_from_the_bottom_left_and_turned_linear() -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    image[1, 0] = [255, 255, 255]       # the bottom left pixel
    image[0, 1] = [188, 188, 188]       # sRGB 188 is about linear 0.5

    colours = sample(image, [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1]])

    assert colours[0].tolist() == [1.0, 1.0, 1.0] and colours[2].tolist() == [0.0, 0.0, 0.0]
    assert colours[1][0] == pytest.approx(0.5, abs=0.01)


def test_colours_per_face_corner_give_each_corner_a_vertex_and_a_coarse_mesh_shows_the_texture_s_mean(monkeypatch) -> None:
    image = np.zeros((2, 2, 3), np.uint8)
    image[1, :] = 255
    uv = [[0.5, 0.1], [0.5, 0.9], [0.5, 0.9], [0.5, 0.1], [0.5, 0.1], [0.5, 0.9]]    # two triangles, by corner
    points = QUAD_POINTS
    indices = [0, 1, 2, 2, 3, 0]

    coarse = part_mesh(points, [3, 3], indices, IDENTITY, uv=uv, per_corner=True, image=image)
    assert len(coarse.positions) == 4 and len(set(map(tuple, coarse.colors.tolist()))) == 1
    assert coarse.colors[0, 0] == pytest.approx(255 * isaac_scene.linear(0.5), abs=1)

    monkeypatch.setattr(isaac_scene, "MIN_SAMPLED_VERTICES", 1)
    fine = part_mesh(points, [3, 3], indices, IDENTITY, uv=uv, per_corner=True, image=image)
    assert len(fine.positions) == 6 and fine.faces.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert fine.positions[3].tolist() == points[2].tolist()
    assert fine.colors[:3, 0].tolist() == [255, 0, 0]


@pytest.mark.parametrize(("rows", "expected"), [
    (np.eye(3), [0.0, 0.0, 0.0, 1.0]),
    ([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),     # a quarter turn about z
    ([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [1.0, 0.0, 0.0, 0.0]),                      # half turns: no w to divide by
    ([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0.0, 1.0, 0.0, 0.0]),
    ([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0.0, 0.0, 1.0, 0.0]),
])
def test_a_usd_matrix_gives_a_pose_without_its_scale(rows, expected) -> None:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rows, float) * np.array([[2.0], [3.0], [0.5]])
    matrix[3, :3] = (1.0, 2.0, 3.0)

    pose = pose_xyz_xyzw(pose_matrix(matrix), (1.0, 0.0, 0.0))

    assert pose[:3] == [0.0, 2.0, 3.0]
    assert pose[3:] == pytest.approx(expected) or pose[3:] == pytest.approx([-value for value in expected])


def test_poses_are_read_once_per_refresh_and_relative_to_the_env_origin() -> None:
    view = View(["/a", "/b"])
    poses = RigidPoses(view, ["/a", "/b"], (1.0, 0.0, 0.0))
    a, b = poses.reader("/a"), poses.reader("/b")

    assert a()[:3] == [-1.0, 0.0, 0.0] and b()[:3] == [0.0, 0.0, 0.0] and a()[3:] == [0.0, 0.0, 0.0, 1.0]
    assert view.reads == 1
    view.transforms[1, 0] = 5.0
    assert b()[0] == 0.0
    poses.refresh()
    assert b()[0] == 4.0 and view.reads == 2


def test_a_body_the_view_does_not_hold_is_an_error_not_a_gap() -> None:
    with pytest.raises(SceneTraceError, match="/b"):
        RigidPoses(View(["/a"]), ["/a", "/b"], (0.0, 0.0, 0.0))


def _scene(pxr):
    block = Prim("/env/rigid/block/block_9_1",
                 _mesh("/env/rigid/block/block_9_1/visual", x=0.5, material=Prim(
                     "/m", shader={"diffuse_color_constant": Input((0.0, 1.0, 0.0))})),
                 _mesh("/env/rigid/block/block_9_1/collision", visibility="invisible"),
                 world=_translated(0.5, 0, 0), rigid=True)
    link = Prim("/env/robot0/link1", _mesh("/env/robot0/link1/visuals/mesh"),
                _mesh("/env/robot0/link1/collisions/mesh", purpose="guide"), rigid=True)
    gone = Prim("/env/rigid/block/off", rigid=True, enabled=False)
    table = Prim("/env/Table/top/table_0", kind="Mesh", world=_translated(0, 0, 1, scale=2.0), points=QUAD_POINTS,
                 counts=[4], indices=[0, 1, 2, 3])
    return Stage(Prim("/env", Prim("/env/rigid", Prim("/env/rigid/block", block, gone)), Prim("/env/robot0", link),
                      Prim("/env/Table", Prim("/env/Table/top", table))))


def test_the_stage_s_rigid_prims_and_named_statics_are_the_bodies(pxr) -> None:
    simulation = Simulation()
    bodies, poses = scene_bodies(_scene(pxr), "/env", simulation, robots=["/env/robot0"],
                                 statics=["/env/Table/top/table_0"], origin=Tensor([0.0, 0.0, 0.5]))
    by_key = {body.key: body for body in bodies}

    assert list(by_key) == ["block_9_1", "robot0/link1", "table_0"]     # the disabled rigid body is no body
    assert simulation.view.prim_paths == ["/env/rigid/block/block_9_1", "/env/robot0/link1"]
    assert [by_key[key].kind for key in by_key] == ["object", "robot", "static"]
    assert isinstance(poses, RigidPoses)
    assert by_key["robot0/link1"].read_pose()[0] == 1.0                  # from the view, in its order
    assert by_key["table_0"].read_pose()[:3] == [0.0, 0.0, 0.5]          # once from USD, relative to the origin
    assert simulation.view.reads == 1


def test_a_body_s_mesh_is_what_the_renderer_draws_in_a_frame_without_scale(pxr) -> None:
    bodies, _ = scene_bodies(_scene(pxr), "/env", Simulation(), robots=["/env/robot0"],
                             statics=["/env/Table/top/table_0"])
    by_key = {body.key: body for body in bodies}

    block, link, table = by_key["block_9_1"].mesh, by_key["robot0/link1"].mesh, by_key["table_0"].mesh
    assert len(block.positions) == 4 and len(link.positions) == 4        # hidden and guide meshes are left out
    assert block.positions[0].tolist() == [0.0, 0.0, 0.0]                # the mesh sits where the body does
    assert block.colors[0].tolist() == [0, 255, 0, 255]
    assert link.colors[0].tolist() == [128, 128, 128, 255]               # no material: grey
    assert table.positions[2].tolist() == [2.0, 2.0, 0.0]                # the prim's own scale stays in the vertices
    assert by_key["table_0"].read_pose()[3:] == [0.0, 0.0, 0.0, 1.0]


def test_a_stage_without_a_rigid_body_is_refused(pxr) -> None:
    with pytest.raises(SceneTraceError, match="no rigid body"):
        scene_bodies(Stage(Prim("/env")), "/env", Simulation())


def _png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return stream.getvalue()


def _textured(shaders: list[Prim], *, uv: bool = True) -> Prim:
    material = Prim("/m", *shaders)
    return Prim("/mesh", kind="Mesh", material=material,
                primvars={"st": Primvar(np.zeros((4, 2), np.float32), "vertex")} if uv else {})


def test_a_gltf_material_s_texture_comes_from_a_usdz_through_its_connected_lookup(pxr, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    usdz = tmp_path / "object.usdz"
    with zipfile.ZipFile(usdz, "w") as archive:
        archive.writestr("SubUSDs/textures/color.png", _png((255, 0, 0)))
    lookup = Prim("/m/baseColorTex", shader={"texture": Input(Asset("./SubUSDs/textures/color.png",
                                                                    f"{usdz}[SubUSDs/textures/color.png]"))})
    surface = Prim("/m/pbr", shader={"base_color_texture": Input(source=lookup), "base_color_factor": Input((1.0, 1.0, 1.0))})

    image, color = isaac_scene.material_look(_textured([surface, lookup]))

    assert image.shape == (4, 4, 3) and image[0, 0].tolist() == [255, 0, 0] and color == [1.0, 1.0, 1.0]


def test_an_omnipbr_texture_is_a_file_and_a_preview_surface_s_a_connected_uv_texture(pxr, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    (tmp_path / "a.png").write_bytes(_png((0, 0, 255)))
    direct = Prim("/m/omni", shader={"diffuse_texture": Input(Asset("a.png", str(tmp_path / "a.png")))})
    assert isaac_scene.material_look(_textured([direct]))[0][0, 0].tolist() == [0, 0, 255]

    texture = Prim("/m/tex", shader={"file": Input(Asset(str(tmp_path / "a.png")))})
    surface = Prim("/m/surface", shader={"diffuseColor": Input((0.2, 0.2, 0.2), source=texture)})
    image, color = isaac_scene.material_look(_textured([surface, texture]))
    assert image[0, 0].tolist() == [0, 0, 255] and color == [0.2, 0.2, 0.2]


def test_a_texture_that_cannot_be_read_leaves_the_material_s_colour(pxr, tmp_path: Path) -> None:
    shader = Prim("/m/s", shader={"diffuse_texture": Input(Asset(str(tmp_path / "missing.png"))),
                                  "diffuse_color_constant": Input((0.25, 0.5, 2.0))})

    assert isaac_scene.material_look(_textured([shader])) == (None, [0.25, 0.5, 1.0])
    assert isaac_scene.material_look(Prim("/bare")) == (None, None)


def test_robodojo_s_static_prims_are_its_layout_s_instances_without_a_rigid_body(pxr) -> None:
    stage = _scene(pxr)
    kinds = {"block_9_1": "rigid", "table_0": "table", "missing": "room"}

    assert static_prims(stage, "/env", kinds) == ["/env/Table/top/table_0"]


class Session:
    def __init__(self) -> None:
        self.clock, self.reset_physics_steps = 844, 844

    def _clock(self) -> int:
        return self.clock

    def tick(self, trace: RoboDojoSceneTrace) -> None:
        self.clock += STRIDE
        trace.after_tick(self)


def _manifest_and_rows(trace: Path) -> tuple[dict, list[dict], np.ndarray]:
    manifest = json.loads((trace / "manifest.json").read_text())
    rows = [json.loads(line) for line in (trace / "steps.jsonl").read_text().splitlines()]
    return manifest, rows, np.frombuffer((trace / "poses.bin").read_bytes(), "<f4").reshape(manifest["shape"])


def test_a_trace_has_one_row_per_native_action_on_the_capture_s_clock(tmp_path: Path, monkeypatch) -> None:
    from ora.environment.backends.robodojo import scene_trace as robodojo_trace

    view = View(["/block"])
    poses = RigidPoses(view, ["/block"], (0.0, 0.0, 0.0))
    body = Body("block", "block", "object", poses.reader("/block"),
                Mesh(np.eye(3, dtype=np.float32), np.full((3, 4), 200, np.uint8), np.array([[0, 1, 2]])))
    monkeypatch.setattr(robodojo_trace, "bodies_of", lambda _session: ([body], poses))
    session, trace = Session(), RoboDojoSceneTrace(tmp_path / "trace", run_id="run")

    trace.begin_round(session, 0)
    trace.begin_action({"sequence": 0, "action_id": "a0"})
    view.transforms[0, 0] = 1.0
    session.tick(trace)
    view.transforms[0, 0] = 2.0
    session.tick(trace)
    trace.end_action()
    trace.begin_action({"sequence": 1, "action_id": "refused"})
    trace.end_action()
    manifest = trace.finish()

    _, rows, values = _manifest_and_rows(tmp_path / "trace")
    assert manifest["backend"] == "robodojo" and manifest["clock"] == {
        "unit": CLOCK_UNIT, "timestep_s": 0.004, "hz": 250.0, "stride": 10}
    assert [row["physics_step"] for row in rows] == [0, 10, 20]
    assert values[:, 0, 0].tolist() == [0.0, 1.0, 2.0]                  # each row is the state after its last step
    assert rows[2]["sim_time_s"] == pytest.approx(0.08) and rows[2]["action_id"] == "a0"
    assert manifest["spans"] == [
        {"round": 0, "action_sequence": 0, "action_id": "a0", "first_step": 0, "last_step": 20},
        {"round": 0, "action_sequence": 1, "action_id": "refused", "first_step": 20, "last_step": 20}]
