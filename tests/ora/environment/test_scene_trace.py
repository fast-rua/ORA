"""The scene trace's format (``ora/environment/scene_trace.py``): meshes, rows, bodies across rounds. No simulator."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from ora.environment.scene_trace import (
    MAX_FACES, SCHEMA, Body, Mesh, SceneTraceError, SceneTraceWriter, encode_glb, merge, simplify, traces_root,
)


def _triangle(x: float = 0.0, *, size: float = 0.1) -> Mesh:
    return Mesh(np.array([[x, 0, 0], [x + size, 0, 0], [x, size, 0]], np.float32),
                np.full((3, 4), 200, np.uint8), np.array([[0, 1, 2]]))


def _sheet(n: int) -> Mesh:
    """A flat n x n grid, half black and half white by x: 2 n^2 faces."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel(), np.zeros((n + 1) ** 2)], axis=1)
    colors = np.zeros((len(positions), 4), np.uint8)
    colors[positions[:, 0] > 0.5] = 255
    index = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    return Mesh(positions, colors, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))


def _body(key: str, *, x: float = 0.0, kind: str = "object", mesh: Mesh | None = None) -> Body:
    return Body(key=key, name=key, kind=kind, read_pose=lambda: [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], mesh=mesh)


def _parse_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, length = struct.unpack("<III", data[:12])
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    size, kind = struct.unpack("<II", data[12:20])
    assert kind == 0x4E4F534A and size % 4 == 0
    document = json.loads(data[20:20 + size])
    size2, kind2 = struct.unpack("<II", data[20 + size:28 + size])
    assert kind2 == 0x004E4942 and size2 % 4 == 0
    return document, data[28 + size:]


def _read(trace: Path) -> tuple[dict, np.ndarray, list[dict]]:
    manifest = json.loads((trace / "manifest.json").read_text())
    poses = np.frombuffer((trace / "poses.bin").read_bytes(), dtype="<f4").reshape(manifest["shape"])
    return manifest, poses, [json.loads(line) for line in (trace / "steps.jsonl").read_text().splitlines()]


def test_a_mesh_is_a_valid_binary_gltf_holding_its_vertices_colours_and_faces() -> None:
    mesh = _triangle(0.5)
    document, binary = _parse_glb(encode_glb(mesh))

    position, color, index = document["accessors"]
    assert (position["count"], position["type"], index["count"]) == (3, "VEC3", 3)
    assert position["min"] == pytest.approx([0.5, 0.0, 0.0]) and position["max"] == pytest.approx([0.6, 0.1, 0.0])
    assert color["normalized"] is True and color["type"] == "VEC4"
    views = document["bufferViews"]
    assert all(view["byteOffset"] % 4 == 0 for view in views)
    read = np.frombuffer(binary, "<f4", 9, views[0]["byteOffset"]).reshape(3, 3)
    assert read == pytest.approx(mesh.positions)
    assert np.frombuffer(binary, "<u4", 3, views[2]["byteOffset"]).tolist() == [0, 1, 2]
    assert document["materials"][0]["doubleSided"] is True


def test_parts_merge_into_one_mesh_with_their_faces_renumbered() -> None:
    merged = merge([_triangle(), Mesh(np.zeros((0, 3)), np.zeros((0, 4), np.uint8), np.zeros((0, 3), int)), _triangle(1.0)])

    assert merged.faces.tolist() == [[0, 1, 2], [3, 4, 5]] and len(merged.positions) == 6
    assert merge([]) is None


def test_a_large_mesh_is_clustered_under_the_face_limit_and_a_small_one_is_left_alone() -> None:
    small = _sheet(4)
    assert simplify(small, 100) is small

    sheet = _sheet(120)
    out = simplify(sheet, 2000)

    assert 0 < len(out.faces) <= 2000 and out.colors.dtype == np.uint8
    assert np.ptp(out.positions, axis=0)[:2] == pytest.approx([1.0, 1.0], abs=0.05)
    assert out.faces.max() < len(out.positions)
    # the sheet faces +z: cross products keep their sign, so clustering kept the winding
    a, b, c = (out.positions[out.faces[:, k]] for k in range(3))
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()
    left, right = out.colors[out.positions[:, 0] < 0.3], out.colors[out.positions[:, 0] > 0.7]
    assert left.max() == 0 and right.min() == 255


def test_rows_sample_the_stride_and_keep_each_action_s_last_step(tmp_path: Path) -> None:
    trace = tmp_path / "t"
    writer = SceneTraceWriter(trace, run_id="r", backend="robotwin", clock_unit="sapien_step_since_round_reset")
    moved = [0.0]
    writer.begin_round(0, [Body("cup", "cup", "object", lambda: [moved[0], 0, 0, 0, 0, 0, 1], _triangle())], timestep_s=0.004)
    writer.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(7):
        moved[0] += 1.0
        writer.after_step()
    writer.end_action()
    writer.begin_action({"action_sequence": 1, "action_id": "b"})      # refused: no step, no row, a span of nothing
    writer.end_action()
    writer.begin_action({"action_sequence": 2, "action_id": "c"})
    for _ in range(3):
        moved[0] += 1.0
        writer.after_step()
    writer.end_action()
    manifest = writer.finish()
    written, poses, steps = _read(trace)

    assert written == manifest and manifest["schema"] == SCHEMA and manifest["source"] == "replay_program"
    assert manifest["clock"] == {"unit": "sapien_step_since_round_reset", "timestep_s": 0.004, "hz": 250.0, "stride": 5}
    assert [row["physics_step"] for row in steps] == [0, 5, 7, 10]
    assert [row["index"] for row in steps] == [0, 1, 2, 3]
    assert [row["action_sequence"] for row in steps] == [None, 0, 0, 2]
    assert [row["native_action"] for row in steps] == [None, 5, 7, 3]
    assert steps[3]["sim_time_s"] == pytest.approx(0.04)
    assert poses[:, 0, 0].tolist() == [0.0, 5.0, 7.0, 10.0]     # each row is the state after that many steps
    assert poses[0, 0, 3:].tolist() == [0.0, 0.0, 0.0, 1.0]     # xyzw
    assert [(s["action_sequence"], s["first_step"], s["last_step"]) for s in manifest["spans"]] == [
        (0, 0, 7), (1, 7, 7), (2, 7, 10)]


def test_a_body_keeps_its_column_across_rounds_and_is_nan_while_it_is_absent(tmp_path: Path) -> None:
    trace = tmp_path / "t"
    writer = SceneTraceWriter(trace, run_id="r", backend="robotwin", clock_unit="u", stride=1)
    writer.begin_round(0, [_body("table", x=1.0, mesh=_triangle()), _body("cup", x=2.0, mesh=_triangle())], timestep_s=0.01)
    writer.begin_round(1, [_body("table", x=3.0, mesh=_triangle()), _body("cup", x=4.0, mesh=_triangle(0.5)),
                           _body("frame", x=5.0)], timestep_s=0.01)
    manifest = writer.finish()
    _, poses, steps = _read(trace)

    assert [body["id"] for body in manifest["bodies"]] == ["table", "cup", "cup-2", "frame"]
    assert [row["round"] for row in steps] == [0, 1] and manifest["rounds"] == [0, 1]
    assert poses[1, 0, 0] == 3.0 and poses[1, 2, 0] == 4.0      # the table carries on, another cup mesh is another body
    assert np.isnan(poses[0, 2]).all() and np.isnan(poses[1, 1]).all()
    assert manifest["bodies"][3] == {"id": "frame", "name": "frame", "kind": "object", "mesh": None, "faces": 0}
    assert sorted(path.name for path in (trace / "meshes").iterdir()) == ["cup-2.glb", "cup.glb", "table.glb"]


def test_body_ids_are_safe_file_names_and_a_huge_static_body_is_the_backdrop(tmp_path: Path) -> None:
    writer = SceneTraceWriter(tmp_path / "t", run_id="r", backend="robotwin", clock_unit="u")
    writer.begin_round(0, [_body("../../etc/passwd", mesh=_triangle()), _body("056_switch/link_0", mesh=_triangle()),
                           _body("", mesh=_triangle()), _body("wall", kind="static", mesh=_triangle(size=6.0)),
                           _body("table", kind="static", mesh=_triangle(size=1.2))], timestep_s=0.01)
    bodies = writer.finish()["bodies"]

    assert [body["id"] for body in bodies] == ["etc_passwd", "056_switch_link_0", "body", "wall", "table"]
    assert [body["kind"] for body in bodies][3:] == ["backdrop", "static"]
    assert (tmp_path / "t" / "meshes" / "etc_passwd.glb").is_file()


def test_a_writer_can_ask_for_a_backdrop_long_in_two_directions(tmp_path: Path) -> None:
    long_counter = Mesh(np.array([[0, 0, 0], [3.8, 0, 0], [0, 0.6, 0.0]]), np.zeros((3, 4), np.uint8), np.array([[0, 1, 2]]))
    wall = Mesh(np.array([[0, 0, 0], [6.0, 0, 0], [0, 0, 3.0]]), np.zeros((3, 4), np.uint8), np.array([[0, 1, 2]]))
    bodies = [Body(name, name, "static", lambda: [0, 0, 0, 0, 0, 0, 1], mesh) for name, mesh in (("c", long_counter), ("w", wall))]

    for directions, kinds in ((1, ["backdrop", "backdrop"]), (2, ["static", "backdrop"])):
        writer = SceneTraceWriter(tmp_path / str(directions), run_id="r", backend="b", clock_unit="u",
                                  backdrop_directions=directions)
        writer.begin_round(0, bodies, timestep_s=0.002)
        assert [body["kind"] for body in writer.records] == kinds


def test_a_mesh_over_the_limit_is_written_simplified(tmp_path: Path) -> None:
    writer = SceneTraceWriter(tmp_path / "t", run_id="r", backend="robotwin", clock_unit="u")
    writer.begin_round(0, [_body("big", mesh=_sheet(120))], timestep_s=0.01)
    assert len(_sheet(120).faces) > MAX_FACES
    assert 0 < writer.finish()["bodies"][0]["faces"] <= MAX_FACES


def test_an_unknown_kind_or_an_action_never_begun_is_refused(tmp_path: Path) -> None:
    writer = SceneTraceWriter(tmp_path / "t", run_id="r", backend="robotwin", clock_unit="u")
    with pytest.raises(SceneTraceError, match="kind"):
        writer.begin_round(0, [_body("x", kind="ghost")], timestep_s=0.01)
    with pytest.raises(SceneTraceError, match="begin_action"):
        writer.end_action()


def test_the_default_directory_is_beside_the_run_s_runs_directory(tmp_path: Path) -> None:
    run = tmp_path / "exp" / "runs" / "run-1"

    assert traces_root(run) == (tmp_path / "exp" / "scene-trace").resolve()
    assert traces_root(run, tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()
