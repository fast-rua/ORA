"""A MuJoCo scene's bodies for the scene trace: stand-ins holding a compiled model's arrays (the reader imports no
MuJoCo), and one test on the real engine where it is installed, which holds the arrays' meaning."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment.mujoco_scene_trace import (
    BOX, CAPSULE, CYLINDER, ELLIPSOID, INFINITE_PLANE_M, MESH, PLANE, SPHERE, MuJoCoSceneTrace, body_mesh, row_stride,
    scene_bodies, tcp_links,
)
from ora.environment.scene_trace import SceneTraceError, rotation

QUARTER_ABOUT_Z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
GREY = (0.5, 0.5, 0.5, 1.0)


def _geom(kind: int = BOX, size: Any = (1.0, 1.0, 1.0), *, group: int = 1, pos: Any = (0.0, 0.0, 0.0),
          quat: Any = (1.0, 0.0, 0.0, 0.0), rgba: Any = (1.0, 0.0, 0.0, 1.0), material: int = -1, mesh: int = -1) -> dict:
    return {"kind": kind, "size": size, "group": group, "pos": pos, "quat": quat, "rgba": rgba, "material": material,
            "mesh": mesh}


def _model(bodies: dict[str, list[dict]], *, roots: list[int] | None = None, welds: list[int] | None = None,
           sites: dict[str, tuple[int, Any, Any]] | None = None, **arrays: Any) -> SimpleNamespace:
    """``bodies``: name -> geoms, in model order; ``sites``: name -> (body, pos, quat wxyz); other fields as given."""
    names, addresses = b"", {}
    for name in [*bodies, *(sites or {})]:
        addresses[name] = len(names)
        names += name.encode() + b"\0"
    geoms = [geom for body in bodies.values() for geom in body]
    counts = [len(body) for body in bodies.values()]
    sites = sites or {}
    fields: dict[str, Any] = {
        "nbody": len(bodies), "nsite": len(sites), "names": names, "opt": SimpleNamespace(timestep=0.002),
        "name_bodyadr": np.array([addresses[name] for name in bodies]),
        "name_siteadr": np.array([addresses[name] for name in sites], dtype=np.int64),
        "body_rootid": np.array(roots if roots is not None else range(len(bodies))),
        "body_weldid": np.array(welds if welds is not None else range(len(bodies))),
        "body_geomadr": np.cumsum([0, *counts[:-1]]), "body_geomnum": np.array(counts),
        "geom_type": np.array([geom["kind"] for geom in geoms]), "geom_group": np.array([geom["group"] for geom in geoms]),
        "geom_size": np.array([geom["size"] for geom in geoms], dtype=np.float64),
        "geom_pos": np.array([geom["pos"] for geom in geoms], dtype=np.float64),
        "geom_quat": np.array([geom["quat"] for geom in geoms], dtype=np.float64),
        "geom_rgba": np.array([geom["rgba"] for geom in geoms], dtype=np.float64),
        "geom_matid": np.array([geom["material"] for geom in geoms]),
        "geom_dataid": np.array([geom["mesh"] for geom in geoms]),
        "site_pos": np.array([site[1] for site in sites.values()], dtype=np.float64),
        "site_quat": np.array([site[2] for site in sites.values()], dtype=np.float64),
        "site_bodyid": np.array([site[0] for site in sites.values()], dtype=np.int64),
    }
    return SimpleNamespace(**{**fields, **arrays})


def _mesh_arrays(vertices: Any, faces: Any, uv: Any = None, *, face_uv: Any = None) -> dict[str, Any]:
    """One compiled mesh after a decoy, so a reader that ignores the addresses reads the decoy."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    decoy = np.zeros((2, 2))
    return {
        "mesh_vertadr": np.array([0, 3]), "mesh_vertnum": np.array([3, len(vertices)]),
        "mesh_faceadr": np.array([0, 1]), "mesh_facenum": np.array([1, len(faces)]),
        "mesh_vert": np.concatenate([np.full((3, 3), 9.0), vertices]),
        "mesh_face": np.concatenate([[[0, 1, 2]], faces]),
        "mesh_texcoordadr": np.array([-1, -1 if uv is None else 2]),
        "mesh_texcoord": decoy if uv is None else np.concatenate([decoy, np.asarray(uv, dtype=np.float64)]),
        "mesh_facetexcoord": np.concatenate([[[0, 0, 0]], faces if face_uv is None else np.asarray(face_uv)]),
    }


def _textures(pixels: Any, *, kind: int = 0) -> dict[str, Any]:
    """One material (index 1, after a plain one) with its RGB role on one texture, after a decoy texel."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = pixels.shape
    return {"mat_rgba": np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            "mat_texid": np.array([[-1, -1], [-1, 0]]), "tex_height": np.array([height]), "tex_width": np.array([width]),
            "tex_nchannel": np.array([channels]), "tex_adr": np.array([1]), "tex_type": np.array([kind]),
            "tex_data": np.concatenate([[7], pixels.reshape(-1)])}


@pytest.mark.parametrize(("kind", "size", "low", "high"), [
    (BOX, (0.1, 0.2, 0.3), (-0.1, -0.2, -0.3), (0.1, 0.2, 0.3)),
    (SPHERE, (0.25, 0, 0), (-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)),
    (ELLIPSOID, (0.1, 0.2, 0.3), (-0.1, -0.2, -0.3), (0.1, 0.2, 0.3)),
    (CYLINDER, (0.1, 0.3, 0), (-0.1, -0.1, -0.3), (0.1, 0.1, 0.3)),
    (CAPSULE, (0.1, 0.3, 0), (-0.1, -0.1, -0.4), (0.1, 0.1, 0.4)),       # the half length is the cylinder's part
    (PLANE, (2.0, 3.0, 0.125), (-2.0, -3.0, 0.0), (2.0, 3.0, 0.0)),
    (PLANE, (0.0, 0.0, 0.125), (-INFINITE_PLANE_M, -INFINITE_PLANE_M, 0.0), (INFINITE_PLANE_M, INFINITE_PLANE_M, 0.0)),
])
def test_a_primitive_is_as_large_as_its_geom_size(kind: int, size: Any, low: Any, high: Any) -> None:
    mesh = body_mesh(_model({"b": [_geom(kind, size)]}), 0)

    assert mesh.positions.min(axis=0) == pytest.approx(low, abs=1e-3)
    assert mesh.positions.max(axis=0) == pytest.approx(high, abs=1e-3)
    assert int(mesh.faces.max()) < len(mesh.positions) == len(mesh.colors)


def test_a_geom_s_pose_places_it_in_the_body_s_frame() -> None:
    mesh = body_mesh(_model({"b": [_geom(BOX, (0.1, 0.2, 0.3), pos=(1.0, 0.0, 0.0), quat=QUARTER_ABOUT_Z)]}), 0)

    assert mesh.positions.min(axis=0) == pytest.approx([0.8, -0.1, -0.3])  # turned, then moved
    assert mesh.positions.max(axis=0) == pytest.approx([1.2, 0.1, 0.3])


def test_a_compiled_mesh_is_read_at_its_addresses_and_placed_by_the_geom() -> None:
    model = _model({"b": [_geom(MESH, mesh=1, pos=(0.0, 0.0, 1.0), quat=QUARTER_ABOUT_Z)]},
                   **_mesh_arrays([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[2, 1, 0]]))
    mesh = body_mesh(model, 0)

    assert mesh.positions[0] == pytest.approx([0.0, 1.0, 1.0]) and mesh.positions[2] == pytest.approx([0.0, 0.0, 2.0])
    assert mesh.faces.tolist() == [[2, 1, 0]]


def test_a_body_s_visual_geoms_join_and_what_is_not_drawn_stays_out() -> None:
    bodies = {"parts": [_geom(group=0), _geom(group=1, rgba=(0, 0, 1, 1)), _geom(group=2), _geom(group=3),
                        _geom(rgba=(1, 1, 1, 0)), _geom(kind=1)],     # collision, visual, visual, off, invisible, hfield
              "frame": [], "collision": [_geom(group=0)]}
    model = _model(bodies)
    mesh = body_mesh(model, 0)

    assert len(mesh.positions) == 16 and len(mesh.faces) == 24
    assert mesh.colors[0].tolist() == [0, 0, 255, 255] and mesh.colors[8].tolist() == [255, 0, 0, 255]
    assert int(mesh.faces.max()) == 15                                 # the second geom's faces follow its vertices
    assert body_mesh(model, 1) is None and body_mesh(model, 2) is None


def test_a_geom_shows_its_own_colour_unless_it_left_the_default_to_its_material() -> None:
    arrays = {"mat_rgba": np.array([[0.0, 1.0, 0.0, 1.0]]), "mat_texid": np.array([[-1, -1]])}
    own = _model({"b": [_geom(rgba=(1.0, 0.0, 0.0, 1.0), material=0)]}, **arrays)
    left = _model({"b": [_geom(rgba=GREY, material=0)]}, **arrays)
    plain = _model({"b": [_geom(rgba=(0.5, 0.5, 0.5, 1.0))]})

    assert body_mesh(own, 0).colors[0].tolist() == [255, 0, 0, 255]
    assert body_mesh(left, 0).colors[0].tolist() == [0, 255, 0, 255]
    assert body_mesh(plain, 0).colors[0].tolist() == [55, 55, 55, 255]  # MuJoCo's values are sRGB, a vertex colour linear


def _sheet(count: int) -> tuple[Any, Any]:
    """``count`` vertices (a multiple of three): a triangle soup, as robosuite's meshes compile."""
    return np.zeros((count, 3)), np.arange(count).reshape(-1, 3)


def test_a_texture_is_sampled_at_each_vertex_s_coordinate() -> None:
    pixels = np.zeros((2, 2, 3), np.uint8)
    pixels[0, 1] = (255, 0, 0)            # row 0, column 1: v near 0, u near 1
    pixels[1, 0] = (0, 188, 0)            # sRGB 188 is about 0.5 linear
    vertices, faces = _sheet(258)
    uv = np.tile([[0.75, 0.25]], (258, 1))
    uv[1] = [0.25, 0.75]
    uv[2] = [1.75, -0.75]                 # outside 0..1 the texture repeats
    model = _model({"b": [_geom(MESH, mesh=1, rgba=GREY, material=1)]}, **_mesh_arrays(vertices, faces, uv),
                   **_textures(pixels))
    mesh = body_mesh(model, 0)

    assert mesh.colors[0].tolist() == [255, 0, 0, 255] and mesh.colors[2].tolist() == [255, 0, 0, 255]
    assert mesh.colors[1, 1] == pytest.approx(128, abs=2) and mesh.colors[1, 0] == 0


def test_a_vertex_on_a_seam_takes_the_mean_of_its_face_corners() -> None:
    vertices = np.zeros((258, 3))
    faces = np.arange(258).reshape(-1, 3)
    faces[1, 0] = 0                       # vertex 0 is a corner of two faces, with two texture coordinates
    pixels = np.zeros((1, 2, 3), np.uint8)
    pixels[0, 1] = 255
    uv = np.full((258, 2), 0.25)
    uv[3] = [0.75, 0.25]
    face_uv = np.arange(258).reshape(-1, 3)
    model = _model({"b": [_geom(MESH, mesh=1, rgba=GREY, material=1)]},
                   **_mesh_arrays(vertices, faces, uv, face_uv=face_uv), **_textures(pixels))

    assert body_mesh(model, 0).colors[0, 0] == pytest.approx(55, abs=2)  # half white in sRGB


@pytest.mark.parametrize("case", ["coarse", "no coordinates", "cube map"])
def test_a_texture_it_cannot_sample_shows_its_mean(case: str) -> None:
    count = 3 if case == "coarse" else 258
    vertices, faces = _sheet(count)
    pixels = np.full((4, 4, 3), 255, np.uint8)
    pixels[:2] = 0                        # half black, half white: mean sRGB 0.5
    uv = None if case == "no coordinates" else np.zeros((count, 2))
    model = _model({"b": [_geom(MESH, mesh=1, rgba=GREY, material=1)]}, **_mesh_arrays(vertices, faces, uv),
                   **_textures(pixels, kind=1 if case == "cube map" else 0))
    mesh = body_mesh(model, 0)

    assert mesh.colors[:, 0].tolist() == [mesh.colors[0, 0]] * count and mesh.colors[0, 0] == pytest.approx(55, abs=2)


def test_a_material_s_colour_tints_its_texture_and_a_primitive_shows_the_mean() -> None:
    arrays = _textures(np.full((2, 2, 3), 255, np.uint8))
    arrays["mat_rgba"][1] = (1.0, 0.0, 0.0, 1.0)
    mesh = body_mesh(_model({"b": [_geom(rgba=GREY, material=1)]}, **arrays), 0)

    assert mesh.colors[0].tolist() == [255, 0, 0, 255]


def test_robots_by_their_root_things_welded_to_the_world_are_static_and_the_rest_objects() -> None:
    names = ["world", "table", "base", "link", "hand", "cup", "cabinet", "drawer"]
    model = _model({name: [] for name in names}, roots=[0, 1, 2, 2, 2, 5, 6, 6], welds=[0, 0, 0, 3, 3, 5, 0, 7])
    data = SimpleNamespace(xpos=np.arange(24.0).reshape(8, 3), xquat=np.tile([0.5, 0.1, 0.2, 0.3], (8, 1)))
    bodies = {body.key: body for body in scene_bodies(model, data, robot_roots=["base"])}

    assert {key: body.kind for key, body in bodies.items()} == {
        "world": "static", "table": "static", "base": "robot", "link": "robot", "hand": "robot", "cup": "object",
        "cabinet": "static", "drawer": "object"}
    assert bodies["cup"].mesh is None
    assert bodies["cup"].read_pose() == pytest.approx([15.0, 16.0, 17.0, 0.1, 0.2, 0.3, 0.5])  # MuJoCo's wxyz as xyzw


def test_a_robot_root_or_a_tcp_site_the_model_lacks_fails_the_trace() -> None:
    model = _model({"world": []})

    with pytest.raises(SceneTraceError, match="robot root body"):
        scene_bodies(model, SimpleNamespace(), robot_roots=["robot0_base"])
    with pytest.raises(SceneTraceError, match="TCP site"):
        tcp_links(model, {"right": "grip"})


def test_a_tcp_is_its_site_s_body_and_fixed_pose_there() -> None:
    model = _model({"world": [], "hand": []}, sites={"grip": (1, (0.1, 0.2, 0.3), (0.5, 0.1, 0.2, 0.3))})

    assert tcp_links(model, {"right": "grip"}) == {
        "right": {"body": "hand", "offset_xyz_xyzw": pytest.approx([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5])}}


@pytest.mark.parametrize(("timestep_s", "stride"), [(0.002, 10), (0.004, 5), (0.005, 4), (0.05, 1)])
def test_the_stride_keeps_rows_near_fifty_hertz(timestep_s: float, stride: int) -> None:
    assert row_stride(timestep_s) == stride


class Sim:
    """A native simulator's stand-in: ``PhysicsStepTap`` wraps its class's ``step``."""

    def __init__(self, model: Any) -> None:
        self.model = SimpleNamespace(_model=model)     # robosuite's wrappers hold the raw objects under these names
        self.data = SimpleNamespace(_data=SimpleNamespace(xpos=np.zeros((2, 3)), xquat=np.tile([1.0, 0, 0, 0], (2, 1))))

    def step(self) -> None:
        self.data._data.xpos[1, 0] += 1.0


def test_the_trace_counts_each_simulator_s_steps_from_its_round_at_about_fifty_hertz(tmp_path: Path) -> None:
    model = _model({"world": [], "hand": [_geom()]}, roots=[0, 1], sites={"grip": (1, (0, 0, 0.1), (1, 0, 0, 0))})
    first, second = Sim(model), Sim(model)     # a LIBERO hard reset builds a new simulator for a round
    trace = MuJoCoSceneTrace(tmp_path / "t", run_id="r", backend="libero_pro")
    trace.begin_round(first, 0, robot_roots=["hand"], tcp_sites={"right": "grip"})
    trace.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(25):
        first.step()
    trace.end_action()
    trace.begin_round(second, 1, robot_roots=["hand"], tcp_sites={"right": "grip"})
    first.step()                                # the old simulator is no longer watched
    trace.begin_action({"action_sequence": 1, "action_id": "b"})
    for _ in range(12):
        second.step()
    trace.end_action()
    manifest = trace.finish()
    rows = np.frombuffer((tmp_path / "t" / "poses.bin").read_bytes(), "<f4").reshape(manifest["shape"])

    assert manifest["clock"] == {"unit": "mujoco_step_since_round_reset", "timestep_s": 0.002, "hz": 500.0, "stride": 10}
    assert [(s["round"], s["first_step"], s["last_step"]) for s in manifest["spans"]] == [(0, 0, 25), (1, 0, 12)]
    assert rows[:, 1, 0].tolist() == [0.0, 10.0, 20.0, 25.0, 0.0, 10.0, 12.0]
    assert manifest["tcp"] == {"right": {"body": "hand", "offset_xyz_xyzw": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]}}
    assert [body["kind"] for body in manifest["bodies"]] == ["static", "robot"]
    assert Sim.step.__qualname__ == "Sim.step"  # the tap is off the class again


REAL_MODEL = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <asset>
    <mesh name="wedge" vertex="0 0 0  1 0 0  0 2 0  0 0 3" face="0 2 1  0 1 3  0 3 2  1 2 3" scale="0.1 0.1 0.1"/>
    <material name="green" rgba="0 1 0 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" group="1"/>
    <geom name="wall" type="box" size="1.5 0.01 1" pos="0 1 1" group="1"/>
    <body name="cup" pos="0.3 0 0.2" quat="0.7071068 0 0 0.7071068">
      <joint type="free"/>
      <geom type="mesh" mesh="wedge" pos="0 0 0.05" group="1" material="green"/>
      <geom type="box" size="0.01 0.01 0.01" group="0" rgba="0.5 0.5 0.5 1"/>
      <geom type="capsule" size="0.01 0.02" pos="0 0.1 0" euler="90 0 0" group="1" rgba="1 0 0 1"/>
    </body>
    <body name="arm" pos="0 0 0.5">
      <joint name="spin" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="box" size="0.05 0.02 0.02" pos="0.05 0 0" group="1" mass="1"/>
      <site name="grip" pos="0.1 0 0" quat="0.7071068 0.7071068 0 0"/>
    </body>
  </worldbody>
  <actuator><motor joint="spin"/></actuator>
</mujoco>
"""


def test_on_the_real_engine_meshes_poses_and_the_clock_are_what_the_arrays_say(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(REAL_MODEL)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[0] = 0.01
    bodies = {body.key: body for body in scene_bodies(model, data, robot_roots=["arm"])}

    assert {key: body.kind for key, body in bodies.items()} == {"world": "static", "cup": "object", "arm": "robot"}
    cup = bodies["cup"]
    wedge = int(model.body_geomadr[list(bodies).index("cup")])  # the cup's first geom
    assert model.geom_type[wedge] == MESH
    vertices = np.asarray(model.mesh_vert[:4], dtype=np.float64)
    simulator = vertices @ data.geom_xmat[wedge].reshape(3, 3).T + data.geom_xpos[wedge]
    pose = np.asarray(cup.read_pose())
    drawn = np.asarray(cup.mesh.positions)[:4] @ rotation([pose[6], *pose[3:6]]).T + pose[:3]
    assert drawn == pytest.approx(simulator)   # the body frame's mesh lands where the simulator has the geom's vertices
    assert np.ptp(drawn, axis=0) == pytest.approx([0.2, 0.1, 0.3])
    assert cup.mesh.colors[0].tolist() == [0, 255, 0, 255] and cup.mesh.colors[-1].tolist() == [255, 0, 0, 255]
    assert len(cup.mesh.positions) > 4         # the capsule joined, the collision box did not
    assert bodies["world"].mesh.positions.max(axis=0) == pytest.approx([3.0, 3.0, 2.0], abs=0.02)

    class Engine:
        def __init__(self) -> None:
            self.model, self.data = model, data

        def step(self) -> None:
            mujoco.mj_step(self.model, self.data)

    sim = Engine()
    trace = MuJoCoSceneTrace(tmp_path / "t", run_id="r", backend="mujoco")
    trace.begin_round(sim, 0, robot_roots=["arm"], tcp_sites={"right": "grip"})
    trace.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(40):
        sim.step()
    trace.end_action()
    manifest = trace.finish()
    rows = np.frombuffer((tmp_path / "t" / "poses.bin").read_bytes(), "<f4").reshape(manifest["shape"])
    arm = [body["name"] for body in manifest["bodies"]].index("arm")
    link = manifest["tcp"]["right"]
    offset = np.asarray(link["offset_xyz_xyzw"])
    q = rows[-1, arm, 3:7].astype(np.float64)

    assert link["body"] == "arm" and manifest["clock"]["stride"] == 10 and manifest["shape"][0] == 5
    # the link plus the offset is the simulator's TCP while the arm turns
    assert rows[-1, arm, :3] + rotation([q[3], *q[:3]]) @ offset[:3] == pytest.approx(data.site_xpos[0], abs=1e-6)
    # mj_step's kinematics precede its integration: a row holds the qpos of the step before, as a receipt does
    assert float(data.qpos[-1]) > 0.0
    turned = 2.0 * math.atan2(float(q[2]), float(q[3]))
    before = float(data.qpos[-1] - data.qvel[-1] * model.opt.timestep)
    assert turned == pytest.approx(before, abs=1e-5) and turned != pytest.approx(float(data.qpos[-1]), abs=1e-5)
