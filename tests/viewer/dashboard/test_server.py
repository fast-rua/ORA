"""The run dashboard over synthetic run directories: grouping, run detail, path safety, read-only serving."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import threading
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from viewer import app as viewer_app
from viewer.dashboard import replay, robots, runs
from viewer.dashboard.config import Config
from viewer.dashboard.server import build_parser, config_from_args, make_server

ROOT = Path(__file__).resolve().parents[3]
CODE_MODE = {"contract": "ora.code-mode-script.v1", "coding_tools": True}
SCRIPT = "plan.py"
SCRIPT_TEXT = '"""Stack the blocks."""\n\ndef approach(x):\n    """Move above x."""\n    return x\n'
SCRIPT_SHA = hashlib.sha256(SCRIPT_TEXT.encode()).hexdigest()


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


def _observation(run: Path, name: str, cameras: tuple[str, ...]) -> None:
    for cam in cameras:
        d = run / "policy-observations" / name / cam
        d.mkdir(parents=True)
        Image.new("RGB", (4, 4), (200, 100, 50)).save(d / "rgb.png")
        np.save(d / "depth-m.npy", np.full((4, 4), 0.5, np.float32))
        _write(d / "calibration.json", {"intrinsic_cv": [[2, 0, 2], [0, 2, 2], [0, 0, 1]],
                                        "cam2world_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]]})
    _write(run / "policy-observations" / name / "proprioception.json", {"end_effectors": {}})


def _plan(experiment: str, *, backend: str, task: str, seed: int, config: str | None, cohort: str | None,
          code_mode: bool, layout: int | None = None, profile: str = "surface_v1") -> dict:
    plan = {"experiment": experiment, "task": task, "seed": seed, "task_instruction": f"do {task}",
            "runtime": {"kind": backend, **({"layout_id": layout, "eval_seed": seed} if layout is not None else {})},
            "policy": {"model": "example-model", "code_mode": CODE_MODE if code_mode else None,
                       "baseline": {"id": "example_baseline"}, "tool_surface": {"profile": profile}}}
    if config:
        plan["config"] = config
    if cohort:
        plan["launch_admission"] = {"cohort_id": cohort, "cell_id": config, "mode": "single_attempt_cell"}
    return plan


def _robodojo_run(experiment_dir: Path) -> Path:
    run = experiment_dir / "runs" / "exp-a-robodojo-aaaa--20261030T000000Z"
    _write(run / "plan.json", _plan("exp-a", backend="robodojo", task="stack_blocks", seed=0, layout=0,
                                    config="coding-m-rdj-stack_blocks-l0-s0", cohort="cohort-1", code_mode=True,
                                    profile="toolkit_v5"))
    _write(run / "launch.json", {"created_unix_s": 1793000000.0, "execution_resource": {
        "host_id": "host-a", "slot_id": "slot-robodojo-g3s1", "worker_environment": {"CUDA_VISIBLE_DEVICES": "3"}}})
    _write(run / "result.json", {"state": "completed", "official_success": True, "behavioral_sample_valid": True,
                                 "model_calls": 1, "wall_duration_s": 42.0, "started_unix_s": 1793000001.0,
                                 "finished_unix_s": 1793000043.0, "verdict": {"native_actions": 5, "native_action_limit": 200}})
    _write(run / "native/agent-artifacts/result.json", {
        "ok": True, "exit_code": 0, "status": "completed", "output": "MOVED ABOVE",
        "workspace": [{"path": SCRIPT, "sha256": SCRIPT_SHA, "bytes": len(SCRIPT_TEXT)}]})
    _write(run / "native/agent-artifacts/frame.png", b"\x89PNG\r\n")
    _write(run / "native/code-workspace-objects" / SCRIPT_SHA, SCRIPT_TEXT)
    origin = {"kind": "code_mode", "parent_tool_call_id": "tc-1", "seq": 1}
    _jsonl(run / "agent-events.jsonl", [
        {"type": "provider_response", "model_call_id": "mc-0", "observation_step": 0, "unix_s": 1793000002.0,
         "payload": {"message": {"content": "", "reasoning_content": "look at the blocks"},
                     "usage": {"input_tokens": 100, "output_tokens": 10}}},
        {"type": "tool_call_proposed", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1",
         "unix_s": 1793000003.0, "payload": {"name": "bash", "arguments": {"command": "python plan.py"}}},
        {"type": "tool_call_proposed", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1#001",
         "unix_s": 1793000004.0, "payload": {"name": "move_pose", "arguments": {"arm": "left"}, "origin": origin}},
        {"type": "tool_result", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1#001",
         "unix_s": 1793000005.0, "payload": {"name": "move_pose", "ok": True, "origin": origin}},
        {"type": "action_proposed", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1",
         "payload": {"action_id": "action-1"}},
        {"type": "tool_result", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1",
         "unix_s": 1793000006.0, "artifact_ref": {"path": "native/agent-artifacts/result.json", "bytes": 10},
         "payload": {"name": "bash", "ok": True, "execution": {"actions": [{"action_id": "action-1"}],
                                                              "native_actions_used": 5}}},
        {"type": "tool_image", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "tc-1",
         "artifact_ref": {"path": "native/agent-artifacts/frame.png"}, "payload": {"image_index": 0}},
    ])
    _observation(run, "robodojo:stack_blocks:layout-0:observation-0000", ("head", "left_wrist"))
    _jsonl(run / "native/observation-clocks.jsonl", [{"observation_sequence": 0, "after_physics_steps": 100}])
    _write(run / "native/receipts/action-0000.json", {
        "action": "move_pose", "action_id": "action-1", "execution_status": "reached", "native_actions": 5,
        "arguments": {"targets": {"left": {"position_m": [0, 0, 1], "orientation_xyzw": [0, 0, 0, 1]}}},
        "arms": {"left": {"targeted": True, "interaction": "free_space"}}})
    arm = {"arm_joint_qpos": [0.0] * 6, "gripper_finger_qpos": [0.0, 0.0], "pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1]}
    _write(run / "native/controller/action-0000.json", {"action_id": "action-1", "samples": [
        {"native_action": n, "segment": "move", "measured": {"left": arm}} for n in range(1, 6)]})
    _jsonl(run / "native/execution-capture/timeline.jsonl", [
        {"sim_step": 100, "images": {"head": {"path": "frames/0000-head.png"}}, "phase": "observation"}])
    return run


def _robotwin_run(experiment_dir: Path) -> Path:
    run = experiment_dir / "runs" / "exp-a-robotwin-bbbb--20261030T010000Z"
    _write(run / "plan.json", _plan("exp-a", backend="robotwin", task="move_stapler_pad", seed=100000,
                                    config="base-m-rt-move_stapler_pad-s100000", cohort="cohort-1", code_mode=False))
    _write(run / "result.json", {"state": "completed", "official_success": False, "behavioral_sample_valid": True,
                                 "model_calls": 3, "started_unix_s": 1793003600.0, "finished_unix_s": 1793003700.0,
                                 "verdict": {}})
    _jsonl(run / "agent-events.jsonl", [
        {"type": "provider_response", "model_call_id": "mc-0", "observation_step": 0,
         "payload": {"message": {"content": "moving"}, "usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "tool_call_proposed", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "t-1",
         "payload": {"name": "move_pose", "arguments": {"arm": "right"}}},
        {"type": "tool_result", "model_call_id": "mc-0", "observation_step": 0, "tool_call_id": "t-1",
         "payload": {"name": "move_pose", "ok": False, "error": "unreachable"}},
    ])
    _observation(run, "move_stapler_pad:episode-000:observation-0000", ("front_camera", "head_camera"))
    _write(run / "native/receipts/action-0000.json", {"action": "set_gripper", "action_id": "action-9",
                                                      "execution_status": "executed"})
    return run


@pytest.fixture()
def outputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    _robodojo_run(root / "exp-a")
    _robotwin_run(root / "exp-a")
    # a second experiment without launch admission: its cohort is the config without the task and seed
    run_b = root / "exp-b" / "runs" / "exp-b-libero-pro-cccc--20261030T020000Z"
    _write(run_b / "plan.json", _plan("exp-b", backend="libero_pro", task="libero_goal_task-t05", seed=1,
                                      config="drive4-m-lp-libero_goal_task-t05-s1", cohort=None, code_mode=False))
    # an unfinished run (no result.json) with recent activity
    run_c = root / "exp-b" / "runs" / "exp-b-robolab-dddd--20261030T030000Z"
    _write(run_c / "plan.json", _plan("exp-b", backend="robolab", task="BananaInBowlTask", seed=2, config=None,
                                      cohort=None, code_mode=True))
    _jsonl(run_c / "events.jsonl", [{"type": "run_started", "time": 1793010000.0}])
    return root


@pytest.fixture()
def serve(outputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ROBODOJO_ASSET_ROOT", raising=False)
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    servers = []

    def start(**kwargs):
        cfg = Config.build(str(outputs), cache=str(tmp_path / "cache"), **kwargs)
        srv = make_server(cfg, "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        base = "http://127.0.0.1:%d/" % srv.server_address[1]

        def get(path: str, headers: dict | None = None, method: str = "GET"):
            req = urllib.request.Request(base + path, headers=headers or {}, method=method)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, dict(r.headers), r.read()
            except urllib.error.HTTPError as e:
                return e.code, dict(e.headers), e.read()
        return get

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _json(resp) -> dict:
    status, headers, body = resp
    if headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body)


def _q(**kw) -> str:
    return urllib.parse.urlencode(kw)


def test_overview_groups_runs_into_cohorts_with_condition_labels(serve):
    get = serve()
    ov = _json(get("api/overview"))
    assert ov["experiments"] == [{"name": "exp-a", "runs": 2}, {"name": "exp-b", "runs": 2}]
    cohorts = {(b["experiment"], b["name"]): b for b in ov["cohorts"]}
    assert set(cohorts) == {("exp-a", "*"), ("exp-a", "cohort-1"), ("exp-b", "*"), ("exp-b", "drive4-m-lp"),
                            ("exp-b", "exp-b")}
    mixed = cohorts[("exp-a", "cohort-1")]
    assert mixed["conditions"] == {"code": {"n": 1, "success": 1, "episodes": 1},
                                   "tools": {"n": 1, "success": 0, "episodes": 1}}
    assert mixed["backends"] == ["RoboDojo", "RoboTwin"] and mixed["columns"] == ["l0", "s100000"]
    assert cohorts[("exp-b", "exp-b")]["counts"] == {"running": 1}

    cohort = _json(get("api/cohort?" + _q(experiment="exp-a", name="cohort-1")))
    by_task = {r["task"]: r for r in cohort["items"]}
    assert by_task["stack_blocks"]["condition"] == "code" and by_task["stack_blocks"]["column"] == "l0"
    assert by_task["stack_blocks"]["node"] == "host-a" and by_task["stack_blocks"]["card"] == "3"
    assert by_task["move_stapler_pad"]["condition"] == "tools" and by_task["move_stapler_pad"]["verdict"] == "fail"
    assert get("api/cohort?" + _q(experiment="exp-a", name="nope"))[0] == 404


def test_hidden_cohorts_stay_out_until_asked_for(serve):
    get = serve(hide_cohorts="^cohort-")
    ov = _json(get("api/overview"))
    names = {(b["experiment"], b["name"]) for b in ov["cohorts"]}
    assert ("exp-a", "cohort-1") not in names and ("exp-a", "*") not in names and ("exp-b", "drive4-m-lp") in names
    assert ov["hidden"] == {"cohorts": 1, "runs": 2}
    assert ov["experiments"] == [{"name": "exp-b", "runs": 2}]
    assert all(r["cohort"] != "cohort-1" for r in ov["recent"])

    shown = _json(get("api/overview?hidden=1"))
    assert ("exp-a", "cohort-1") in {(b["experiment"], b["name"]) for b in shown["cohorts"]}
    assert shown["hidden"] == {"cohorts": 1, "runs": 2}

    # a hidden cohort's own page still opens; the all-cohorts page leaves it out
    assert get("api/cohort?" + _q(experiment="exp-a", name="cohort-1"))[0] == 200
    assert get("api/cohort?" + _q(experiment="exp-a", name="*"))[0] == 404
    assert _json(get("api/cohort?" + _q(experiment="exp-a", name="*", hidden="1")))["brief"]["total"] == 2


def test_a_bad_hide_cohorts_pattern_stops_at_start(outputs):
    args = build_parser().parse_args(["--outputs", str(outputs), "--hide-cohorts", "("])
    with pytest.raises(SystemExit, match="--hide-cohorts"):
        config_from_args(args)


def test_condition_flag_labels_runs_by_a_plan_path(serve):
    get = serve(condition="policy.tool_surface.profile", experiment="exp-a")
    ov = _json(get("api/overview"))
    assert [e["name"] for e in ov["experiments"]] == ["exp-a"]
    mixed = next(b for b in ov["cohorts"] if b["name"] == "cohort-1")
    assert set(mixed["conditions"]) == {"toolkit_v5", "surface_v1"}
    assert runs.condition_label({"policy": {}}, "policy.missing") == "none"


def test_run_detail_of_a_robodojo_code_mode_run(serve, outputs):
    get = serve()
    run_id = "exp-a-robodojo-aaaa--20261030T000000Z"
    detail = _json(get("api/run?" + _q(id=run_id), {"Accept-Encoding": "gzip"}))
    s = detail["summary"]
    assert (s["backend_name"], s["state"], s["verdict"], s["condition"], s["cohort"]) == \
        ("RoboDojo", "done", "success", "code", "cohort-1")
    assert s["tool_calls"] == 1 and s["script_calls"] == 1 and s["tokens_in"] == 100
    step = detail["steps"][0]
    assert set(step["images"]) == {"head", "left_wrist"}
    call = step["turns"][0]["calls"][0]
    assert call["code"] == "python plan.py" and call["moved"] is True
    assert [c["name"] for c in call["script_calls"]] == ["move_pose"]
    assert call["result_preview"].endswith("MOVED ABOVE")
    assert step["actions"][0]["receipt"]["status"] == "reached"

    files = _json(get("api/run/files?" + _q(id=run_id)))["files"]
    assert [(f["path"], [v["sha"] for v in f["versions"]]) for f in files] == [(SCRIPT, [SCRIPT_SHA])]
    assert files[0]["outline"]["defs"][0]["name"] == "approach"
    status, _, body = get("api/blob?" + _q(id=run_id, sha=SCRIPT_SHA))
    assert status == 200 and body.decode() == SCRIPT_TEXT


def test_run_detail_of_a_robotwin_run_and_its_replay(serve):
    get = serve()
    run_id = "exp-a-robotwin-bbbb--20261030T010000Z"
    detail = _json(get("api/run?" + _q(id=run_id)))
    assert detail["summary"]["backend_name"] == "RoboTwin" and detail["summary"]["code_mode"] is False
    assert set(detail["steps"][0]["images"]) == {"front_camera", "head_camera"}
    assert detail["steps"][0]["turns"][0]["calls"][0]["error"] == "unreachable"
    assert any(a.get("receipt", {}).get("status") == "executed" for st in detail["steps"] for a in st["actions"])
    bundle = _json(get("api/replay?" + _q(id=run_id)))
    assert bundle["backend"] == "robotwin" and bundle["robot"] is None
    assert [(a["action"], a["n0"], a["n1"]) for a in bundle["actions"]] == [("set_gripper", 0, 0)]
    assert get("api/replay/agent?" + _q(id=run_id))[0] == 200
    obs = "move_stapler_pad:episode-000:observation-0000"
    assert get("api/cloud?" + _q(id=run_id, obs=obs))[0] == 200


def test_robodojo_replay_without_ground_truth_takes_spans_from_the_controller(serve):
    get = serve()
    run_id = "exp-a-robodojo-aaaa--20261030T000000Z"
    bundle = _json(get("api/replay?" + _q(id=run_id)))
    assert bundle["ground_truth"] is False and bundle["robot"] is None and bundle["instances"] == {}
    assert bundle["clock"]["unit"] == "native action" and bundle["tcp_offset"] == replay.TCP_IN_LINK6
    assert "--robodojo-assets" in bundle["robot_note"]
    assert bundle["objects_note"] == f"no scene trace: export one with `ora replay {run_id} --scene --resource-profile SLOT`"
    assert [(a["action_id"], a["n0"], a["n1"]) for a in bundle["actions"]] == [("action-1", 0, 5)]
    assert [o["n"] for o in bundle["observations"]] == [0.0] and len(bundle["frames"]) == 6
    assert bundle["video"]["frames"][0][0] == 10.0
    agent = _json(get("api/replay/agent?" + _q(id=run_id)))
    entry = agent["entries"][0]
    assert entry["code"] == "python plan.py" and entry["cmds"][0]["n1"] == 5
    assert entry["changes"]["added"] == [SCRIPT] and agent["files"][0]["versions"][0]["entry"] == 0


def test_path_safety(serve, outputs, tmp_path):
    get = serve()
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    run = outputs / "exp-a" / "runs" / "exp-a-robotwin-bbbb--20261030T010000Z"
    (run / "escape.txt").symlink_to(secret)
    assert get("api/file?" + _q(path=str(run / "plan.json")))[0] == 200
    assert get("api/file?" + _q(path=str(secret)))[0] == 403
    assert get("api/file?" + _q(path=str(run / ".." / ".." / ".." / ".." / "secret.txt")))[0] == 403
    assert get("api/file?" + _q(path=str(run / "escape.txt")))[0] == 403
    assert get("api/file?" + _q(path="plan.json"))[0] == 403
    assert get("api/ls?" + _q(path=str(tmp_path)))[0] == 403
    assert get("api/run?" + _q(id="../exp-b"))[0] == 404
    assert get("static/..%2f..%2fserver.py")[0] == 404
    assert get("api/overview", method="POST")[0] == 405


def test_serving_never_writes_into_the_outputs(serve, outputs, tmp_path):
    def snapshot() -> dict:
        return {str(p.relative_to(outputs)): (p.stat().st_size, p.stat().st_mtime_ns) for p in outputs.rglob("*")}
    before = snapshot()
    get = serve()
    rd = "exp-a-robodojo-aaaa--20261030T000000Z"
    for path in ("", "static/replay.html", "static/base.css", "static/app.js", "api/overview", "api/runs",
                 "api/cohort?" + _q(experiment="exp-a", name="*"), "api/run?" + _q(id=rd),
                 "api/run/files?" + _q(id=rd), "api/replay?" + _q(id=rd), "api/replay/agent?" + _q(id=rd),
                 "api/run?" + _q(id="exp-b-robolab-dddd--20261030T030000Z")):
        assert get(path)[0] == 200, path
    status, _, body = get("api/cloud?" + _q(id=rd, obs="robodojo:stack_blocks:layout-0:observation-0000"))
    assert status == 200 and int.from_bytes(body[:4], "little") > 0
    assert snapshot() == before
    assert list((tmp_path / "cache" / "clouds").glob("*.bin"))


def test_mesh_and_robot_files_stay_inside_the_asset_root(tmp_path):
    assets = tmp_path / "Assets"
    _write(assets / "Robots/x5/X5A.urdf", "<robot/>")
    _write(assets / "Object/RoboDojo/Rigid/cube/00000/object.usdz", b"")
    outside = tmp_path / "elsewhere/Object/cube/object.usdz"
    _write(outside, b"")
    cfg = Config.build(str(tmp_path), asset_root=str(assets), cache=str(tmp_path / "cache"))
    assert replay.mesh(str(outside), cfg) is None
    assert replay.mesh(str(assets / "Object/RoboDojo/../../Robots/x5/X5A.urdf"), cfg) is None
    assert robots.robot_file("x5", "X5A.urdf", cfg) == str(assets / "Robots/x5/X5A.urdf")
    assert robots.robot_file("x5", "../x5/X5A.urdf", cfg) is None
    spec = robots.x5(cfg)
    assert spec["urdf"] == "robot/x5/X5A.urdf" and spec["kind"] == "per_arm" and spec["fk_link"]["left"] == "link6"
    recorded = "/somewhere/else/Assets/Object/RoboDojo/Rigid/cube/00000/object.usdz"
    assert replay.local_asset(recorded, cfg.object_root) == str(assets / "Object/RoboDojo/Rigid/cube/00000/object.usdz")


def test_ora_viewer_dispatches_the_dashboard(tmp_path, capsys):
    with pytest.raises(SystemExit) as done:
        viewer_app.main(["dashboard", "--help"])
    assert done.value.code == 0 and "--outputs" in capsys.readouterr().out
    with pytest.raises(SystemExit) as done:
        viewer_app.main(["dashboard", "--outputs", str(tmp_path / "missing")])
    assert "not a directory" in str(done.value.code)


def test_the_pages_ship_as_package_data():
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["viewer"]
    static = ROOT / "viewer/dashboard/static"
    pages = [str(p.relative_to(ROOT / "viewer")) for p in static.iterdir()]
    assert {"index.html", "replay.html", "app.js", "replay.js"} <= {Path(p).name for p in pages}
    assert all(any(fnmatch.fnmatch(p, pattern) for pattern in patterns) for p in pages)


def test_a_late_success_is_late_not_fail() -> None:
    assert runs.verdict_word({"official_success": False, "success_after_official_limit": True}) == "late"
    assert runs.verdict_word({"official_success": False}) == "fail"
    assert runs.verdict_word({"official_success": True}) == "success"
    assert runs.verdict_word({}) is None
