"""The 3D replay bundle for every backend, over synthetic run directories shaped like RoboTwin (controller replay
npz and execution capture), LIBERO-Pro (execution capture, no joints), RoboLab (receipts only, two rounds) and
RoboCasa365 (a mobile base): one bundle shape, the clock, the commands, the observations at their receipt
boundaries, the agent's trace on the same clock, and the aloha-agilex robot description."""

from __future__ import annotations

import gzip
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from viewer.dashboard import agent, replay, robots
from viewer.dashboard.config import Config
from viewer.dashboard.server import make_server

RT = "exp-r-robotwin-aaaa--20261031T000000Z"
LP = "exp-r-libero-pro-bbbb--20261031T010000Z"
RL = "exp-r-robolab-cccc--20261031T020000Z"
RC = "exp-r-robocasa365-dddd--20261031T030000Z"
SY = "exp-r-synthetic-eeee--20261031T040000Z"
KEYS = {"backend", "clock", "frames", "actions", "observations", "video", "rounds", "round", "robot", "robot_note",
        "objects_note", "view", "workspace", "tcp_offset", "max_n", "instances", "ticks", "table_height"}


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    elif isinstance(data, str):
        path.write_text(data)
    else:
        path.write_text(json.dumps(data))


def _jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "".join(json.dumps(r) + "\n" for r in rows))


def _plan(run: Path, backend: str, **runtime) -> None:
    _write(run / "plan.json", {"experiment": "exp-r", "task": "t", "seed": 1, "config": f"c-{backend}",
                               "runtime": {"kind": backend, **runtime}, "policy": {"model": "m"}})


def _pose(x: float, y: float = 0.0, z: float = 0.9) -> list[float]:
    return [x, y, z, 0.0, 0.0, 0.0, 1.0]


def _observation(run: Path, name: str, after: str | None, arms: dict, calib: dict, **extra) -> None:
    d = run / "policy-observations" / name
    cam = d / "front_camera"
    cam.mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 100, 50)).save(cam / "rgb.png")
    np.save(cam / "depth-m.npy", np.full((4, 4), 0.4, np.float32))
    _write(cam / "calibration.json", {"intrinsic_cv": [[2, 0, 2], [0, 2, 2], [0, 0, 1]], **calib})
    _write(d / "observation.json", {"captured_unix_s": 1.0, "main_camera": "front_camera",
                                    "previous_execution_receipt": {"path": after} if after else None})
    _write(d / "proprioception.json", {"end_effectors": arms, **extra})


def _receipt(run: Path, i: int, action_id: str, action: str, *, status="reached", failure=None, before=None,
             after=None, arms=None, arguments=None, **extra) -> None:
    _write(run / "native" / "receipts" / f"action-{i:04d}.json", {
        "action_id": action_id, "action": action, "execution_status": status, "failure_code": failure,
        "arguments": arguments or {}, "arms": arms or {}, "finished_unix_s": 10.0 + i,
        "before": {a: {"pose_xyz_xyzw": p} for a, p in (before or {}).items()},
        "after": {a: {"pose_xyz_xyzw": p} for a, p in (after or {}).items()}, **extra})


def _events(run: Path, calls: list[tuple[str, str, str | None]]) -> None:
    rows = [{"type": "provider_response", "model_call_id": "mc-0", "observation_step": 0, "unix_s": 1.0,
             "payload": {"message": {"content": "go"}, "usage": {"input_tokens": 1, "output_tokens": 1}}}]
    for k, (tc, name, action_id) in enumerate(calls):
        rows.append({"type": "tool_call_proposed", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": tc,
                     "unix_s": 2.0 + k, "payload": {"name": name, "arguments": {}}})
        if action_id:
            rows.append({"type": "action_proposed", "model_call_id": "mc-0", "observation_step": 0,
                         "tool_call_id": tc, "payload": {"action_id": action_id}})
        rows.append({"type": "tool_result", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": tc,
                     "unix_s": 2.5 + k, "payload": {"name": name, "ok": True}})
    _jsonl(run / "agent-events.jsonl", rows)


def _capture(run: Path, unit: str, dt: float, spans: list[tuple[str, int, int]]) -> None:
    rows = [{"phase": "reset", "sim_step": 0, "sim_time_s": 0.0, "step_unit": unit, "round_index": 0,
             "images": {"front_camera": {"path": "frames/0.png"}}}]
    for aid, lo, hi in spans:
        rows.append({"phase": "action_start", "action_id": aid, "sim_step": lo, "sim_time_s": lo * dt,
                     "step_unit": unit, "round_index": 0, "images": {"front_camera": {"path": f"frames/{lo}.png"}}})
        rows.append({"phase": "action_end", "action_id": aid, "sim_step": hi, "sim_time_s": hi * dt,
                     "step_unit": unit, "round_index": 0, "images": {"front_camera": {"path": f"frames/{hi}e.png"}}})
    _jsonl(run / "native" / "execution-capture" / "timeline.jsonl", rows)
    _write(run / "native" / "execution-capture" / "manifest.json", {"config": {"fps": 10}, "status": "complete"})


def _robotwin_run(runs: Path) -> None:
    run = runs / RT
    _plan(run, "robotwin", public_environment_context={"embodiment_id": "aloha_agilex"})
    left = _pose(-0.3)
    _receipt(run, 0, "action-a:00-open", "set_gripper", status="executed", arguments={"targets": {"right": "open"}},
             arms={"right": {"targeted": True}, "left": {"targeted": False}},
             before={"left": left, "right": _pose(0.3)}, after={"left": left, "right": _pose(0.3)})
    _receipt(run, 1, "action-a:01-pregrasp", "move_pose",
             arguments={"interactions": {"right": "free_space"},
                        "targets": {"right": {"position_m": [0.2, 0.0, 0.85], "orientation_xyzw": [0, 0, 0, 1]}}},
             arms={"right": {"targeted": True}}, before={"left": left, "right": _pose(0.3)},
             after={"left": left, "right": _pose(0.2, z=0.85)}, controller_tracking={"physics_step_attempts": 5})
    _receipt(run, 2, "action-b", "move_pose", status="not_reached", failure="planner_failed",
             arguments={"interactions": {"right": "precision"}}, arms={"right": {"targeted": True}})
    replay_dir = run / "native" / "controller-replay"
    replay_dir.mkdir(parents=True)
    q = lambda n, v: np.full((n, 6), v, np.float32)   # noqa: E731
    p = lambda n, x: np.tile(np.asarray(_pose(x), np.float32), (n, 1))   # noqa: E731
    np.savez(replay_dir / "action-0000.npz",
             measured_left_arm__position=q(4, 0.0), measured_right_arm__position=q(4, 0.1),
             measured_left_tcp__pose_xyzw=p(4, -0.3), measured_right_tcp__pose_xyzw=p(4, 0.3),
             right_gripper__result=np.array([1.0, 0.66, 0.33, 0.0]), right_gripper__target=np.array(0.0),
             issued_left_arm__position=np.zeros((0,), np.float32))
    np.savez(replay_dir / "action-0001.npz",
             global__measured_left_arm__position=q(3, 0.0), global__measured_right_arm__position=q(3, 0.2),
             global__measured_left_tcp__pose_xyzw=p(3, -0.3), global__measured_right_tcp__pose_xyzw=p(3, 0.25),
             measured_left_arm__position=q(2, 0.0), measured_right_arm__position=q(2, 0.3),
             measured_left_tcp__pose_xyzw=p(2, -0.3), measured_right_tcp__pose_xyzw=p(2, 0.2))
    _jsonl(replay_dir / "events.jsonl", [
        {"type": "entry", "entry": {"action_id": "action-a:00-open", "artifact": {"path": "controller-replay/action-0000.npz"}}},
        {"type": "entry", "entry": {"action_id": "action-a:01-pregrasp", "artifact": {"path": "controller-replay/action-0001.npz"}}},
        {"type": "entry", "entry": {"action_id": "action-x", "artifact": {"path": "../../plan.json"}}},
    ])
    _capture(run, "sapien_step_since_round_reset", 0.004, [("action-a:00-open", 0, 4), ("action-a:01-pregrasp", 4, 9)])
    extrinsic = {"extrinsic_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.8]]}   # RoboTwin: world-to-camera only

    def arms(right_finger: float) -> dict:
        return {"left": {"pose_xyz_xyzw": left, "gripper_state": {"measured_primary_joint_position": 0.045}},
                "right": {"pose_xyz_xyzw": _pose(0.3), "gripper_state": {"measured_primary_joint_position": right_finger}}}
    _observation(run, "t:episode-000:observation-0000", None, arms(0.045), extrinsic,
                 joints={"left_arm": [0.0] * 6, "right_arm": [0.1] * 6})
    _observation(run, "t:episode-000:observation-0001", "receipts/action-0001.json", arms(0.025), extrinsic)
    _events(run, [("tc-1", "grasp_execute", "action-a"), ("tc-2", "observe", None), ("tc-3", "move_pose", "action-b")])


def _libero_run(runs: Path) -> None:
    run = runs / LP
    _plan(run, "libero_pro")
    _receipt(run, 0, "action-l0", "move_pose", arguments={"interactions": {"right": "free_space"}},
             arms={"right": {"targeted": True, "interaction": "free_space"}}, before={"right": _pose(-0.4)},
             after={"right": _pose(-0.3)}, steps_used=4)
    _receipt(run, 1, "action-l1", "set_gripper", status=None, arguments={"arm": "right", "gripper": "close"},
             arms={"right": {"targeted": True}}, before={"right": _pose(-0.3)}, after={"right": _pose(-0.3)},
             steps_used=2)
    _write(run / "native" / "controller-replay" / "action-0000.npz", b"not read for LIBERO-Pro")
    _capture(run, "mujoco_step_since_round_reset", 0.002, [("action-l0", 0, 100), ("action-l1", 100, 150)])
    cam = {"cam2world_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]]}
    _observation(run, "libero:task-01:seed-1:observation-0000", None, {"right": {"pose_xyz_xyzw": _pose(-0.4)}}, cam)
    _observation(run, "libero:task-01:seed-1:observation-0001", "receipts/action-0000.json",
                 {"right": {"pose_xyz_xyzw": _pose(-0.3)}}, cam)
    _events(run, [("tc-1", "move_pose", "action-l0"), ("tc-2", "set_gripper", "action-l1")])


def _robolab_run(runs: Path) -> None:
    run = runs / RL
    _plan(run, "robolab")
    _receipt(run, 0, "action-r0", "move_pose", arguments={"interactions": {"right": "precision"}},
             arms={"right": {"targeted": True, "interaction": "precision"}}, before={"right": _pose(0.5, z=0.1)},
             after={"right": _pose(0.5, z=0.05)}, steps_used=99, native_actions=99, round_index=0)
    _receipt(run, 1, "action-r1", "move_pose", status="not_reached", failure="controller_stalled",
             arms={"right": {"targeted": True, "interaction": "contact_approach"}}, before={"right": _pose(0.5, z=0.05)},
             after={"right": _pose(0.5, z=0.03)}, steps_used=45, native_actions=45, round_index=0)
    _receipt(run, 2, "action-r2", "move_pose", arms={"right": {"targeted": True, "interaction": "free_space"}},
             before={"right": _pose(0.4)}, after={"right": _pose(0.45)}, steps_used=10, round_index=1)
    cam = {"cam2world_cv": [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]]}
    _observation(run, "robolab:T:seed-1:round-0:observation-0000", None, {"right": {"pose_xyz_xyzw": _pose(0.5, z=0.1)}}, cam)
    _observation(run, "robolab:T:seed-1:round-0:observation-0001", "receipts/action-0000.json",
                 {"right": {"pose_xyz_xyzw": _pose(0.5, z=0.05)}}, cam)
    _observation(run, "robolab:T:seed-1:round-0:observation-0002", "receipts/action-0001.json",
                 {"right": {"pose_xyz_xyzw": _pose(0.5, z=0.03)}}, cam)
    _observation(run, "robolab:T:seed-1:round-1:observation-0003", "receipts/action-0001.json",   # after the reset
                 {"right": {"pose_xyz_xyzw": _pose(0.4)}}, cam)
    _events(run, [("tc-1", "move_pose", "action-r0"), ("tc-2", "move_pose", "action-r1"), ("tc-3", "move_pose", "action-r2")])


def _robocasa_run(runs: Path) -> None:
    run = runs / RC
    _plan(run, "robocasa")
    _receipt(run, 0, "action-c0", "move_base", status="not_reached", failure="blocked",
             arguments={"target": {"x_m": 1.0, "y_m": -1.0, "yaw_rad": 1.5}}, arms={"right": {"targeted": False}},
             before={"right": _pose(0.6, -1.3, 1.3)}, after={"right": _pose(0.9, -1.3, 1.3)},
             base_before={"pose_xy_yaw": [0.8, -1.3, 0.0]}, base_after={"pose_xy_yaw": [1.0, -1.2, 1.4]},
             steps_used=6, round_index=0)
    _observation(run, "robocasa:N:seed-5:round-0:observation-0000", None, {"right": {"pose_xyz_xyzw": _pose(0.6, -1.3, 1.3)}},
                 {"cam2world_cv": [[1, 0, 0, 0.6], [0, 1, 0, -1.6], [0, 0, 1, 1.5], [0, 0, 0, 1]]},
                 base={"pose_xy_yaw": [0.8, -1.3, 0.0]})
    _events(run, [("tc-1", "move_base", "action-c0")])


@pytest.fixture()
def outputs(tmp_path: Path) -> Path:
    runs = tmp_path / "outputs" / "exp-r" / "runs"
    _robotwin_run(runs)
    _libero_run(runs)
    _robolab_run(runs)
    _robocasa_run(runs)
    _plan(runs / SY, "synthetic")
    return tmp_path / "outputs"


@pytest.fixture()
def robotwin_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A RoboTwin checkout holding only an aloha-agilex embodiment: its config.yml and a URDF with one small and
    one "large" visual mesh."""
    emb = tmp_path / "RoboTwin" / "assets" / "embodiments" / "aloha-agilex"
    _write(emb / "config.yml", "\n".join([
        'urdf_path: "./urdf/aloha.urdf"', "gripper_bias: 0.12", "gripper_scale: [-0.01, 0.045]",
        'move_group: ["fl_link6","fr_link6"]',
        'arm_joints_name: [["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"], '
        '["fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"]]',
        "gripper_name:", '  - base: "fl_joint7"', '    mimic: [["fl_joint8", 1., 0.]]',
        '  - base: "fr_joint7"', '    mimic: [["fr_joint8", 1., 0.]]',
        "robot_pose: [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]]  # wxyz", ""]))
    _write(emb / "urdf" / "aloha.urdf", '<robot name="a"><link name="footprint"><visual><geometry>'
           '<mesh filename="meshes/link1.dae"/></geometry></visual><collision><geometry><mesh filename="meshes/big.STL"/>'
           '</geometry></collision></link><link name="box"><visual><geometry><mesh filename="meshes/big.dae"/></geometry>'
           '</visual></link></robot>')
    _write(emb / "urdf" / "meshes" / "link1.dae", b"x" * 10)
    _write(emb / "urdf" / "meshes" / "big.dae", b"x" * 100)
    _write(tmp_path / "RoboTwin" / "secret.txt", "outside the embodiment")
    monkeypatch.setattr(robots, "MAX_MESH_BYTES", 50)
    return tmp_path / "RoboTwin"


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

        def get(path: str):
            try:
                with urllib.request.urlopen(base + path, timeout=10) as r:
                    body = r.read()
                    return r.status, gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
            except urllib.error.HTTPError as e:
                return e.code, e.read()
        return get
    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _q(**kw) -> str:
    return urllib.parse.urlencode(kw)


def test_robotwin_bundle_runs_on_sapien_steps_with_the_aloha_posed_from_the_controller_replay(serve, robotwin_root):
    get = serve(robotwin_root=str(robotwin_root))
    status, body = get("api/replay?" + _q(id=RT))
    assert status == 200
    b = json.loads(body)
    assert KEYS <= set(b) and b["backend"] == "robotwin" and b["tcp_offset"] is None
    assert b["clock"]["unit"] == "SAPIEN step" and b["clock"]["hz"] == 250.0
    assert [(a["action_id"], a["n0"], a["n1"], a["status"]) for a in b["actions"]] == [
        ("action-a:00-open", 0, 4, "executed"), ("action-a:01-pregrasp", 4, 9, "reached"),
        ("action-b", 9, 9, "not_reached")]   # never captured: a mark where the robot stood
    assert b["actions"][1]["arms"]["right"]["interaction"] == "free_space"
    assert b["actions"][0]["arms"]["right"]["gripper"] == "open" and not any("_poses" in a for a in b["actions"])
    assert [(o["seq"], o["n"]) for o in b["observations"]] == [(0, 0), (1, 9)]
    # every 5th SAPIEN step plus each command's last, after the first observation; global__ arrays first
    assert [f["n"] for f in b["frames"]] == [0, 1, 4, 5, 9]
    assert [f["arms"]["right"]["pose"][0] for f in b["frames"]] == [0.3, 0.3, 0.3, 0.25, 0.2]
    assert b["frames"][0]["arms"]["right"]["q"] == [0.1] * 6 and b["frames"][-1]["arms"]["right"]["q"] == [0.3] * 6
    # the closing gripper's commanded opening, stopped at the opening measured at the next observation
    assert [f["arms"]["right"]["f"][0] for f in b["frames"]] == [0.045, 0.045, 0.025, 0.025, 0.025]
    assert b["video"]["frames"][-1][0] == 9 and b["max_n"] == 9
    robot = b["robot"]
    assert robot["id"] == "aloha" and robot["kind"] == "single" and robot["urdf"] == "robot/aloha/urdf/aloha.urdf"
    assert robot["root"]["pos"] == [0.0, -0.65, 0.0]
    assert robot["root"]["quat"] == pytest.approx([0.0, 0.0, 2 ** -0.5, 2 ** -0.5])
    assert robot["joints"]["right"][0] == "fr_joint1" and robot["fingers"]["left"] == ["fl_joint7", "fl_joint8"]
    assert robot["fk_link"] == {"left": "fl_link6", "right": "fr_link6"} and robot["fk_offset"] == [0.12, 0.0, 0.0]
    assert robot["skip"] == ["meshes/big.dae"]   # too large for a browser; never requested
    assert get("robot/aloha/urdf/aloha.urdf")[0] == 200
    assert get("static/robot/aloha/urdf/meshes/link1.dae")[0] == 200
    assert get("robot/aloha/../../../secret.txt")[0] == 404 and get("robot/x5/X5A.urdf")[0] == 404

    entries = json.loads(get("api/replay/agent?" + _q(id=RT))[1])["entries"]
    by_tool = {e["tool"]: e for e in entries}
    assert (by_tool["grasp_execute"]["n0"], by_tool["grasp_execute"]["n1"]) == (0, 9)   # the sequence's stages
    assert by_tool["grasp_execute"]["cmds"][0]["kind"] == "sequence"
    assert (by_tool["observe"]["n0"], by_tool["move_pose"]["n0"]) == (9, 9)

    status, cloud = get("api/cloud?" + _q(id=RT, obs="t:episode-000:observation-0001"))   # extrinsic_cv inverted
    assert status == 200 and int.from_bytes(cloud[:4], "little") > 0


def test_robotwin_without_the_root_or_joints_draws_paths_only(outputs, tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    run = outputs / "exp-r" / "runs" / RT
    b = replay.bundle(str(run), _cfg(outputs, tmp_path))
    assert b["robot"] is None and "--robotwin-root" in b["robot_note"] and b["robot_note_level"] == "notice"
    assert b["frames"][1]["arms"]["right"]["f"] == [0.045, 0.045]   # no gripper scale: the measured opening held
    for p in (run / "native" / "controller-replay").glob("*.npz"):
        p.unlink()
    b = replay.bundle(str(run), _cfg(outputs, tmp_path))
    assert b["robot"] is None and "no measured joints" in b["robot_note"]
    assert [f["n"] for f in b["frames"]] == [0, 0, 4, 4, 9, 9]   # the receipts' poses and the observations'


def test_libero_bundle_runs_on_mujoco_steps_from_the_receipts_poses(serve):
    get = serve()
    status, body = get("api/replay?" + _q(id=LP))
    assert status == 200
    b = json.loads(body)
    assert KEYS <= set(b) and b["clock"] == {"unit": "MuJoCo step", "hz": 500.0,
                                            "source": "the execution capture's simulator steps since the round reset"}
    assert [(a["action"], a["n0"], a["n1"], a["status"]) for a in b["actions"]] == [
        ("move_pose", 0, 100, "reached"), ("set_gripper", 100, 150, None)]
    assert b["actions"][1]["arms"]["right"]["gripper"] == "close"
    assert [(o["seq"], o["n"]) for o in b["observations"]] == [(0, 0), (1, 100)]
    assert [(f["n"], f["seg"]) for f in b["frames"]] == [(0, "observation"), (0, "before"), (100, "observation"),
                                                         (100, "after"), (100, "before"), (150, "after")]
    assert b["robot"] is None and "commands only" in b["robot_note"] and b["robot_note_level"] == "info"
    assert f"ora replay {LP} --scene" in b["objects_note"] and len(b["video"]["frames"]) == 5
    lo, hi = b["workspace"]
    assert lo[0] == pytest.approx(-1.0) and hi[2] == pytest.approx(1.3)
    status, cloud = get("api/cloud?" + _q(id=LP, obs="libero:task-01:seed-1:observation-0001"))
    assert status == 200 and int.from_bytes(cloud[:4], "little") > 0


def test_robolab_bundle_accumulates_receipt_steps_round_by_round(serve):
    get = serve()
    b = json.loads(get("api/replay?" + _q(id=RL))[1])
    assert KEYS <= set(b) and b["clock"]["unit"] == "native action" and b["clock"]["hz"] is None
    assert b["rounds"] == [0, 1] and b["video"]["frames"] == []
    assert [(a["action_id"], a["n0"], a["n1"], a["failure_code"]) for a in b["actions"]] == [
        ("action-r0", 0, 99, None), ("action-r1", 99, 144, "controller_stalled")]
    assert [(o["seq"], o["n"]) for o in b["observations"]] == [(0, 0), (1, 99), (2, 144)]
    assert b["robot"] is None and "end effector only" in b["robot_note"]
    second = json.loads(get("api/replay?" + _q(id=RL, round=1))[1])
    assert [(a["action_id"], a["n0"], a["n1"]) for a in second["actions"]] == [("action-r2", 0, 10)]
    assert [(o["seq"], o["n"]) for o in second["observations"]] == [(3, 0)]   # named round 1: its reset
    entries = json.loads(get("api/replay/agent?" + _q(id=RL, round=1))[1])["entries"]
    assert [(e["tool"], e["n0"], e["n1"]) for e in entries] == [("move_pose", 0, 10)]


def test_robocasa_bundle_carries_the_base(outputs, tmp_path):
    b = replay.bundle(str(outputs / "exp-r" / "runs" / RC), _cfg(outputs, tmp_path))
    act = b["actions"][0]
    assert (act["action"], act["n0"], act["n1"], act["arms"], act["base_target"]) == (
        "move_base", 0, 6, {}, [1.0, -1.0, 1.5])
    assert [f.get("base") for f in b["frames"]] == [[0.8, -1.3, 0.0], [0.8, -1.3, 0.0], [1.0, -1.2, 1.4]]
    assert b["robot"] is None and "base" in b["robot_note"]


def test_a_synthetic_run_has_no_replay(serve):
    get = serve()
    status, body = get("api/replay?" + _q(id=SY))
    err = json.loads(body)
    assert status == 422 and err["backend"] == "synthetic" and "RoboTwin" in err["error"]
    assert set(err["supported"]) == {"robodojo", "robotwin", "libero_pro", "robocasa", "robolab"}


def test_aloha_config_and_robot_files_stay_inside_the_embodiment(tmp_path, robotwin_root):
    cfg = Config.build(str(tmp_path), robotwin_root=str(robotwin_root), cache=str(tmp_path / "cache"))
    conf = robots.aloha_config(cfg)
    assert conf["gripper_scale"] == [-0.01, 0.045] and conf["fingers"]["right"] == ["fr_joint7", "fr_joint8"]
    assert robots.robot_file("aloha", "config.yml", cfg) == str(robotwin_root / "assets/embodiments/aloha-agilex/config.yml")
    assert robots.robot_file("aloha", "../../../secret.txt", cfg) is None
    assert robots.robot_file("aloha", "/etc/passwd", cfg) is None and robots.robot_file("nope", "config.yml", cfg) is None
    (robotwin_root / "assets/embodiments/aloha-agilex/escape.txt").symlink_to(robotwin_root / "secret.txt")
    assert robots.robot_file("aloha", "escape.txt", cfg) is None
    assert robots.aloha(Config.build(str(tmp_path), cache=str(tmp_path / "cache"))) is None


def test_agent_alignment_reads_tool_call_actions(outputs, tmp_path):
    """A tool-call surface links its calls to actions through action_proposed (no code-mode execution record)."""
    run = str(outputs / "exp-r" / "runs" / LP)
    entries = agent.replay_agent(run, _cfg(outputs, tmp_path))["entries"]
    assert [(e["tool"], e["n0"], e["n1"], [c["id"] for c in e["cmds"]]) for e in entries] == [
        ("move_pose", 0, 100, ["action-l0"]), ("set_gripper", 100, 150, ["action-l1"])]
