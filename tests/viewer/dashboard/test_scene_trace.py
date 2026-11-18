"""The 3D replay over a run's scene trace: the bundle, the layer's reason without one, ``api/scene-trace`` and its path
safety. The trace is a small one written by ORA's own writer into a synthetic RoboTwin or LIBERO-Pro run's workspace."""

from __future__ import annotations

import gzip
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ora.environment.scene_trace import Body, Mesh, SceneTraceWriter
from viewer.dashboard import scene_trace
from viewer.dashboard.config import Config
from viewer.dashboard.server import build_parser, config_from_args, make_server

RUN = "exp-r-robotwin-aaaa--20261101T000000Z"
LP = "exp-r-libero-pro-bbbb--20261101T000000Z"
RDJ = "exp-r-robodojo-dddd--20261101T000000Z"
RC = "exp-r-robocasa-cccc--20261101T000000Z"
RL = "exp-r-robolab-eeee--20261101T000000Z"
TRIANGLE = Mesh(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32), np.full((3, 4), 200, np.uint8),
                np.array([[0, 1, 2]]))
BIG = Mesh(TRIANGLE.positions * 60, TRIANGLE.colors, TRIANGLE.faces)


def _plan(run: Path, backend: str) -> None:
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"experiment": "exp-r", "task": "t", "seed": 1, "config": f"c-{backend}",
                                               "runtime": {"kind": backend}, "policy": {"model": "m"}}))


def _body(key: str, kind: str, x: float, mesh: Mesh | None = TRIANGLE) -> Body:
    return Body(key, key, kind, lambda: [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], mesh)


def _trace(directory: Path) -> Path:
    """Two rounds: a table, a cup, a robot link, the wall and a frame without a mesh; a pot only in the second."""
    writer = SceneTraceWriter(directory, run_id=RUN, backend="robotwin", clock_unit="sapien_step_since_round_reset")
    common = [_body("table", "static", 1.0), _body("cup", "object", 2.0), _body("robot/fl_link1", "robot", 3.0),
              _body("wall", "static", 4.0, BIG), _body("frame", "object", 5.0, None)]
    writer.begin_round(0, common, timestep_s=0.004)
    writer.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(7):
        writer.after_step()
    writer.end_action()
    writer.begin_round(1, [*common, _body("pot", "object", 6.0)], timestep_s=0.004)
    writer.finish()
    return directory


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _robodojo_run(run: Path) -> None:
    """Two rounds without ground truth: action 0 of two native actions, then action 1 of one in the second round."""
    _plan(run, "robodojo")
    native = run / "native"
    _write(native / "reset.json", {"physics_steps_before_policy": 800})
    _write(native / "round-resets" / "round-0001.json", {"physics_steps_before_policy": 1500})
    for sequence, ticks in enumerate(((1, 2), (1,))):
        action = f"action-{sequence}"
        _write(native / "receipts" / f"action-{sequence:04d}.json", {"action": "move_pose", "action_id": action})
        _write(native / "controller" / f"action-{sequence:04d}.json", {"action_id": action, "samples": [
            {"native_action": n, "segment": "move", "measured": {"left": {
                "arm_joint_qpos": [0.0] * 6, "gripper_finger_qpos": [0.0, 0.0], "pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1]}}}
            for n in ticks]})
    (native / "observation-clocks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"observation_sequence": 0, "after_physics_steps": 800}, {"observation_sequence": 1, "after_physics_steps": 1500})))
    for sequence in range(2):
        directory = run / "policy-observations" / f"robodojo:t:layout-0:observation-{sequence:04d}" / "head"
        directory.mkdir(parents=True)
        (directory / "rgb.png").write_bytes(b"")


def _robodojo_trace(directory: Path) -> None:
    writer = SceneTraceWriter(directory, run_id=RDJ, backend="robodojo", clock_unit="physx_step_since_round_reset",
                              stride=10)
    bodies = [_body("block_9_1", "object", 1.0), _body("table_0", "static", 2.0), _body("robot0/link6", "robot", 3.0)]
    writer.begin_round(0, bodies, timestep_s=0.004)
    writer.begin_action({"action_sequence": 0, "action_id": "action-0"})
    for _ in range(20):
        writer.after_step()
    writer.end_action()
    writer.begin_round(1, bodies, timestep_s=0.004)
    writer.begin_action({"action_sequence": 1, "action_id": "action-1"})
    for _ in range(10):
        writer.after_step()
    writer.end_action()
    writer.finish()


@pytest.fixture()
def outputs(tmp_path: Path) -> Path:
    experiment = tmp_path / "outputs" / "exp-r"
    _plan(experiment / "runs" / RUN, "robotwin")
    _plan(experiment / "runs" / LP, "libero_pro")
    _robodojo_run(experiment / "runs" / RDJ)
    _plan(experiment / "runs" / RC, "robocasa")
    _plan(experiment / "runs" / RL, "robolab")
    return tmp_path / "outputs"


def _cfg(outputs: Path, tmp_path: Path, **kw) -> Config:
    return Config.build(str(outputs), cache=str(tmp_path / "cache"), **kw)


@pytest.fixture()
def serve(outputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ROBODOJO_ASSET_ROOT", raising=False)
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    servers = []

    def start(**kwargs):
        srv = make_server(_cfg(outputs, tmp_path, **kwargs), "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        base = "http://127.0.0.1:%d/" % srv.server_address[1]

        def get(path: str, method: str = "GET"):
            request = urllib.request.Request(base + path, method=method)
            try:
                with urllib.request.urlopen(request, timeout=10) as r:
                    body = r.read()
                    return r.status, r.headers, gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
            except urllib.error.HTTPError as e:
                return e.code, e.headers, e.read()
        return get
    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _q(**kw) -> str:
    return "api/scene-trace?" + urllib.parse.urlencode(kw)


def _bundle(get, run: str = RUN, round_index: int = 0) -> dict:
    status, _, body = get("api/replay?" + urllib.parse.urlencode({"id": run, "round": round_index}))
    assert status == 200
    return json.loads(body)


def test_without_a_trace_the_objects_layer_says_which_command_makes_one(serve) -> None:
    data = _bundle(serve())

    assert data["scene_trace"] is None
    assert data["objects_note"] == f"no scene trace: export one with `ora replay {RUN} --scene`"


def test_a_robodojo_run_without_ground_truth_or_a_trace_names_the_command_and_its_slot(serve) -> None:
    data = _bundle(serve(), RDJ)

    assert data["scene_trace"] is None and data["ground_truth"] is False and data["rounds"] == [0]
    assert data["objects_note"] == f"no scene trace: export one with `ora replay {RDJ} --scene --resource-profile SLOT`"


def test_a_robodojo_trace_draws_the_objects_on_the_native_action_clock(serve, outputs: Path) -> None:
    _robodojo_trace(outputs / "exp-r" / "scene-trace" / RDJ)
    data = _bundle(serve(), RDJ)
    trace = data["scene_trace"]

    assert data["objects_note"] is None and data["ground_truth"] is False
    assert trace["rows"] == [[0, 0.0], [1, 1.0], [2, 2.0]]    # [row, native action]: a row every 10 physics steps
    assert data["clock"]["unit"] == "native action" and data["robot"] is None and data["robot_note"] is None
    assert [(b["id"], b["kind"]) for b in trace["bodies"]] == [
        ("block_9_1", "object"), ("table_0", "static"), ("robot0_link6", "robot")]    # no URDF: the trace's links


def test_where_the_x5_urdf_poses_the_robot_a_robodojo_trace_s_links_stay_undrawn(
    serve, outputs: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _robodojo_trace(outputs / "exp-r" / "scene-trace" / RDJ)
    monkeypatch.setattr("viewer.dashboard.robots.x5", lambda _cfg: {"urdf": "x5.urdf"})
    data = _bundle(serve(), RDJ)

    assert data["robot"] == {"urdf": "x5.urdf"} and data["robot_note"] is None
    assert [b["id"] for b in data["scene_trace"]["bodies"]] == ["block_9_1", "table_0"]


def test_a_robodojo_trace_splits_a_multi_round_run_by_its_spans_and_each_round_s_reset(serve, outputs: Path) -> None:
    _robodojo_trace(outputs / "exp-r" / "scene-trace" / RDJ)
    get = serve()
    first, second = _bundle(get, RDJ), _bundle(get, RDJ, 1)

    assert first["rounds"] == [0, 1] == second["rounds"]
    assert [(a["action_id"], a["n0"], a["n1"]) for a in first["actions"]] == [("action-0", 0, 2)]
    assert [(a["action_id"], a["n0"], a["n1"]) for a in second["actions"]] == [("action-1", 0, 1)]
    assert [f["n"] for f in first["frames"]] == [0, 1, 2] and [f["n"] for f in second["frames"]] == [0, 1]
    assert [o["n"] for o in first["observations"]] == [0.0] and [o["n"] for o in second["observations"]] == [0.0]
    assert [row[0] for row in second["scene_trace"]["rows"]] == [3, 4]


def test_ground_truth_keeps_its_place_over_a_robodojo_trace(serve, outputs: Path) -> None:
    _robodojo_trace(outputs / "exp-r" / "scene-trace" / RDJ)
    ticks = outputs / "exp-r" / "ground-truth" / RDJ / "ticks.jsonl"
    ticks.parent.mkdir(parents=True)
    ticks.write_text(json.dumps({"native_action": 0, "physics_clock": 800, "segment": "round_start", "action_id": None,
                                 "round_index": 0, "objects": {}}) + "\n")
    data = _bundle(serve(), RDJ)

    assert data["ground_truth"] is True and data["scene_trace"] is None and data["objects_note"] is None


def test_a_robocasa_run_without_controls_says_so_and_names_no_command(serve) -> None:
    data = _bundle(serve(), RC)

    assert data["scene_trace"] is None and "ora replay" not in data["objects_note"]
    assert "per-tick controls" in data["objects_note"]


def _robocasa_trace(directory: Path) -> Path:
    """Two control steps of 25 MuJoCo steps: the base's link moves, the counter does not."""
    writer = SceneTraceWriter(directory, run_id=RC, backend="robocasa", clock_unit="mujoco_step_since_round_reset",
                              stride=25)
    writer.tcp = {"right": {"body": "gripper0_right_eef", "offset_xyz_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}}
    writer.begin_round(0, [_body("gripper0_right_eef", "robot", 1.0), _body("counter", "static", 2.0)], timestep_s=0.002)
    writer.begin_action({"action_sequence": 0, "action_id": "action-0"})
    for _ in range(50):
        writer.after_step()
    writer.end_action()
    writer.finish()
    return directory


def test_a_robocasa_trace_runs_on_the_page_s_control_steps_and_draws_the_robot(serve, outputs: Path) -> None:
    run = outputs / "exp-r" / "runs" / RC
    (run / "native" / "controller-replay").mkdir(parents=True)
    (run / "native" / "controller-replay" / "manifest.json").write_text("{}")
    assert f"ora replay {RC} --scene" in _bundle(serve(), RC)["objects_note"]

    _robocasa_trace(outputs / "exp-r" / "scene-trace" / RC)
    data = _bundle(serve(), RC)

    assert data["scene_trace"]["rows"] == [[0, 0], [1, 1.0], [2, 2.0]]  # 25 MuJoCo steps to the control step
    assert {body["kind"] for body in data["scene_trace"]["bodies"]} == {"robot", "static"}
    assert data["objects_note"] is None and data["robot_note"] is None


def _libero_trace(directory: Path) -> Path:
    """A Panda's link with a mesh, its frame without one, and a bowl; the TCP on the frame, 0.1 m along its x."""
    writer = SceneTraceWriter(directory, run_id=LP, backend="libero_pro", clock_unit="mujoco_step_since_round_reset",
                              stride=10)
    writer.tcp = {"right": {"body": "gripper0_eef", "offset_xyz_xyzw": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}}
    writer.begin_round(0, [_body("robot0_link1", "robot", 1.0), _body("gripper0_eef", "robot", 2.0, None),
                           _body("bowl", "object", 3.0)], timestep_s=0.002)
    writer.finish()
    return directory


def test_without_a_trace_a_libero_run_says_how_to_get_its_objects_and_robot(serve) -> None:
    data = _bundle(serve(), LP)

    assert data["scene_trace"] is None and data["objects_note"] == scene_trace.missing_note(LP)
    assert data["robot"] is None and "ora replay RUN --scene" in data["robot_note"]


def test_a_libero_trace_draws_the_robot_where_no_urdf_does(serve, outputs: Path) -> None:
    _libero_trace(outputs / "exp-r" / "scene-trace" / LP)
    data = _bundle(serve(), LP)
    trace = data["scene_trace"]

    assert data["robot"] is None and data["robot_note"] is None and data["objects_note"] is None
    assert [(b["index"], b["id"], b["kind"]) for b in trace["bodies"]] == [
        (0, "robot0_link1", "robot"), (2, "bowl", "object")]
    assert trace["clock"]["hz"] == 500.0 and trace["clock"]["stride"] == 10
    # the TCP's frame has no mesh to draw but its column still gives the hand's path
    assert trace["tcp"] == {"right": {"index": 1, "offset": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}}


def _robolab_trace(directory: Path) -> None:
    """A Panda's link and the flange with the TCP on it; 3 native actions of 8 PhysX steps."""
    writer = SceneTraceWriter(directory, run_id=RL, backend="robolab", clock_unit="physx_step_since_round_start", stride=8)
    writer.tcp = {"right": {"body": "robot/base_link", "offset_xyz_xyzw": [0.131, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]}}
    writer.begin_round(0, [_body("robot/panda_link1", "robot", 1.0), _body("robot/base_link", "robot", 2.0),
                           _body("table", "object", 3.0), _body("table_fixture", "static", 4.0)], timestep_s=1 / 120)
    writer.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(24):
        writer.after_step()
    writer.end_action()
    writer.finish()


def test_a_robolab_trace_draws_the_robot_on_the_receipts_native_action_clock(serve, outputs: Path) -> None:
    _robolab_trace(outputs / "exp-r" / "scene-trace" / RL)
    data = _bundle(serve(), RL)
    trace = data["scene_trace"]

    assert data["robot"] is None and data["robot_note"] is None and data["objects_note"] is None
    assert [row[1] for row in trace["rows"]] == [0, 1, 2, 3]
    assert sorted(b["kind"] for b in trace["bodies"]) == ["object", "robot", "robot", "static"]
    assert trace["tcp"]["right"]["index"] == 1
    assert data["clock"]["unit"] == "native action"


def test_a_robolab_run_sealed_without_controls_says_it_cannot_be_replayed(serve) -> None:
    data = _bundle(serve(), RL)

    assert data["scene_trace"] is None and "cannot be replayed" in data["objects_note"]
    assert "ora replay" not in data["objects_note"]


def test_a_robolab_run_with_its_controls_names_the_command(serve, outputs: Path) -> None:
    _write(outputs / "exp-r" / "runs" / RL / "native" / "controller-replay" / "manifest.json", {})

    assert _bundle(serve(), RL)["objects_note"] == scene_trace.missing_note(RL)


def test_a_trace_of_a_run_with_a_urdf_robot_leaves_the_links_out(outputs: Path, tmp_path: Path) -> None:
    _trace(outputs / "exp-r" / "scene-trace" / RUN)
    bundle = scene_trace.bundle(str(outputs / "exp-r" / "runs" / RUN), _cfg(outputs, tmp_path), 0, robot=False)

    assert [b["id"] for b in bundle["bodies"]] == ["table", "cup", "pot"] and bundle["tcp"] is None


def test_a_trace_beside_runs_reaches_the_bundle_with_the_bodies_the_page_draws(serve, outputs: Path) -> None:
    _trace(outputs / "exp-r" / "scene-trace" / RUN)
    data = _bundle(serve())
    trace = data["scene_trace"]

    assert data["objects_note"] is None and trace["source"] == "replay_program"
    assert trace["clock"]["unit"] == "sapien_step_since_round_reset" and trace["clock"]["hz"] == 250.0
    assert trace["rows"] == [[0, 0], [1, 5], [2, 7]]         # [row, step]: only this round's, on the page's clock
    assert trace["body_count"] == 6                           # every column of poses.bin, over all rounds
    # no URDF here, so the trace's link is the robot; the wall stays out of the view; the page skips a body whose pose
    # is NaN in this round
    assert [(b["index"], b["id"], b["kind"]) for b in trace["bodies"]] == [
        (0, "table", "static"), (1, "cup", "object"), (2, "robot_fl_link1", "robot"), (5, "pot", "object")]
    assert trace["bodies"][1]["mesh"] == "meshes/cup.glb"


def test_each_round_gets_its_own_rows_and_a_round_outside_the_trace_none(serve, outputs: Path) -> None:
    _trace(outputs / "exp-r" / "scene-trace" / RUN)
    get = serve()
    later = _bundle(get, round_index=1)["scene_trace"]

    assert later["rows"] == [[3, 0]] and [b["id"] for b in later["bodies"]] == ["table", "cup", "robot_fl_link1", "pot"]
    assert _bundle(get, round_index=2)["scene_trace"] is None


def test_scene_traces_flag_moves_the_directory(serve, outputs: Path, tmp_path: Path) -> None:
    _trace(tmp_path / "elsewhere" / RUN)
    assert _bundle(serve())["scene_trace"] is None
    assert _bundle(serve(scene_traces=str(tmp_path / "elsewhere")))["scene_trace"]["body_count"] == 6
    args = build_parser().parse_args(["--outputs", str(outputs), "--scene-traces", str(tmp_path / "elsewhere")])
    assert config_from_args(args).scene_traces == os.path.realpath(tmp_path / "elsewhere")


def test_the_route_serves_the_poses_and_a_body_s_mesh(serve, outputs: Path) -> None:
    trace = _trace(outputs / "exp-r" / "scene-trace" / RUN)
    get = serve()

    status, headers, body = get(_q(id=RUN, file="poses.bin"))
    assert status == 200 and headers["Content-Type"] == "application/octet-stream"
    assert body == (trace / "poses.bin").read_bytes()
    assert np.frombuffer(body, "<f4").reshape(4, 6, 7)[1, 1, 0] == 2.0

    status, headers, body = get(_q(id=RUN, file="meshes/cup.glb"))
    assert status == 200 and headers["Content-Type"] == "model/gltf-binary" and body[:4] == b"glTF"
    status, headers, body = get(_q(id=RUN, file="meshes/cup.glb"), "HEAD")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0


@pytest.mark.parametrize("name", [
    "manifest.json", "steps.jsonl", "replay-program-check.json", "../../runs/%s/plan.json" % RUN, "../poses.bin",
    "/etc/passwd", "meshes/../poses.bin", "meshes/../../secret.glb", "meshes/.hidden.glb", "meshes/sub/cup.glb",
    "meshes/cup.obj", "meshes/absent.glb", "", "poses.bin/", "poses.bin\n",
])
def test_the_route_serves_nothing_else(serve, outputs: Path, name: str) -> None:
    root = _trace(outputs / "exp-r" / "scene-trace" / RUN)
    (root.parent / "secret.glb").write_bytes(b"secret")

    assert serve()(_q(id=RUN, file=name))[0] == 404


def test_a_link_out_of_the_trace_is_not_followed(serve, outputs: Path, tmp_path: Path) -> None:
    root = _trace(outputs / "exp-r" / "scene-trace" / RUN)
    (tmp_path / "secret.bin").write_bytes(b"secret")
    (root / "meshes" / "out.glb").symlink_to(tmp_path / "secret.bin")
    (root / "poses.bin").unlink()
    (root / "poses.bin").symlink_to(tmp_path / "secret.bin")
    get = serve()

    assert get(_q(id=RUN, file="meshes/out.glb"))[0] == 404
    assert get(_q(id=RUN, file="poses.bin"))[0] == 404


def test_an_unknown_run_or_a_run_without_a_trace_is_not_found(serve, outputs: Path) -> None:
    get = serve()

    assert get(_q(id="../exp-r", file="poses.bin"))[0] == 404
    assert get(_q(id=RUN, file="poses.bin"))[0] == 404


def test_the_route_is_read_only_and_writes_nothing(serve, outputs: Path) -> None:
    _trace(outputs / "exp-r" / "scene-trace" / RUN)
    before = {str(p): p.stat().st_mtime_ns for p in outputs.rglob("*")}
    get = serve()
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert get(_q(id=RUN, file="poses.bin"), method)[0] == 405
    assert get(_q(id=RUN, file="poses.bin"))[0] == 200
    _bundle(get)

    assert {str(p): p.stat().st_mtime_ns for p in outputs.rglob("*")} == before


@pytest.mark.parametrize("damage", ["schema", "shape", "poses", "steps"])
def test_an_incomplete_trace_counts_as_none(serve, outputs: Path, damage: str) -> None:
    root = _trace(outputs / "exp-r" / "scene-trace" / RUN)
    manifest = json.loads((root / "manifest.json").read_text())
    if damage == "schema":
        (root / "manifest.json").write_text(json.dumps({**manifest, "schema": "ora.scene-trace.v0"}))
    elif damage == "shape":
        (root / "manifest.json").write_text(json.dumps({key: v for key, v in manifest.items() if key != "shape"}))
    elif damage == "poses":
        (root / "poses.bin").write_bytes(b"\0" * 28)
    else:
        (root / "steps.jsonl").write_text("")
    data = _bundle(serve())

    assert data["scene_trace"] is None and RUN in data["objects_note"]


def test_the_trace_directory_follows_the_run_or_the_flag(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, tmp_path)
    assert scene_trace.directory("/o/exp/runs/r1/", cfg) == "/o/exp/scene-trace/r1"
    moved = _cfg(tmp_path, tmp_path, scene_traces=str(tmp_path / "d"))
    assert scene_trace.directory("/o/exp/runs/r1", moved) == os.path.join(os.path.realpath(tmp_path / "d"), "r1")
