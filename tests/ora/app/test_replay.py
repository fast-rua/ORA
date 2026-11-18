"""`ora replay`: a finished run's replay behind `ora web`'s gateway, and its video export's plan.

No Node, browser or ffmpeg: the gateway in front of a stand-in web server, the command's
argument checks, the exporter's frame schedule and the web app's environment.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
from argparse import ArgumentTypeError
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer

import pytest
from werkzeug.test import Client

from ora.app import replay, replay_export, web as ora_web
from ora.app.main import build_parser
from ora.app.web_gateway import WebGateway, allowed

LAUNCH = "replay-launch-abcdefghijklmnop"
HOST = {"Host": "127.0.0.1:43211"}


class Upstream(ThreadingMixIn, UnixStreamServer):
    """Stand-in for the replay's web server: records requests, serves the page and its replay."""
    daemon_threads = True

    def __init__(self, path: str) -> None:
        self.seen: list[tuple[str, str]] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def address_string(self) -> str:
                return "unix"

            def log_message(self, *args) -> None:
                pass

            def _answer(handler, status: int, body: bytes) -> None:
                handler.send_response(status)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def do_GET(handler) -> None:  # noqa: N802
                self.seen.append(("GET", handler.path))
                handler._answer(200, json.dumps({"path": handler.path}).encode())

            def do_POST(handler) -> None:  # noqa: N802
                self.seen.append(("POST", handler.path))
                handler._answer(200, b"{}")

        super().__init__(path, Handler)


@pytest.fixture
def socket_dir():
    path = Path(tempfile.mkdtemp(prefix="ora-replay-test-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def upstream(socket_dir: Path):
    path = str(socket_dir / "web.sock")
    server = Upstream(path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, path
    server.shutdown()
    server.server_close()


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "ora-play-play-0123--20261020T000000Z"
    (root / "native" / "agent-artifacts").mkdir(parents=True)
    (root / "native" / "agent-artifacts" / ("b" * 64 + ".png")).write_bytes(b"\x89PNG stored")
    (root / "native" / "operator-view").mkdir()
    (root / "native" / "operator-view" / "000003.png").write_bytes(b"\x89PNG view")
    (root / "events.jsonl").write_text('{"type": "run_started"}\n')
    return root


def replay_client(run: Path, upstream_path: str) -> Client:
    client = Client(WebGateway(run_root=run, upstream=upstream_path, world_app=None, launch_token=LAUNCH, replay=True))
    assert client.get(f"/?ora_token={LAUNCH}&replay=1&speed=16", headers=HOST).headers["Location"] == "/?replay=1&speed=16"
    return client


def test_a_replay_passes_the_page_and_its_replay_only(run, upstream):
    client = replay_client(run, upstream[1])
    assert client.get("/?replay=1", headers=HOST).status_code == 200
    assert client.get("/api/replay", headers=HOST).json == {"path": "/api/replay"}
    origin = {**HOST, "Origin": "http://127.0.0.1:43211"}
    for method, path in (("POST", "/api/send"), ("POST", "/api/play"), ("POST", "/api/dialog"), ("GET", "/api/events"),
                         ("GET", "/api/snapshot"), ("GET", "/api/frames"), ("POST", "/api/replay"), ("GET", "/api/live/status")):
        response = client.open(path, method=method, headers=origin)
        assert response.status_code == 403, (method, path)
    assert [seen for seen in upstream[0].seen if seen[1].startswith("/api/")] == [("GET", "/api/replay")]
    assert not allowed("GET", "/api/replay"), "a live run's page has no replay route"
    assert allowed("GET", "/api/replay", replay=True) and not allowed("POST", "/api/send", replay=True)


def test_a_replay_serves_the_run_s_own_pictures_behind_the_cookie(run, upstream):
    assert Client(WebGateway(run_root=run, upstream=upstream[1], world_app=None, launch_token=LAUNCH, replay=True)).get(
        f"/ora/artifacts/{'b' * 64}.png", headers=HOST).status_code == 403
    client = replay_client(run, upstream[1])
    assert client.get(f"/ora/artifacts/{'b' * 64}.png", headers=HOST).data == b"\x89PNG stored"
    assert client.get("/ora/operator-view/000003.png", headers=HOST).data == b"\x89PNG view"


def test_a_live_run_or_one_without_evidence_is_not_replayed(tmp_path, socket_dir, run):
    assert replay.replay_run(str(run)) == run.resolve()
    empty = tmp_path / "empty"
    (empty / "native").mkdir(parents=True)
    with pytest.raises(replay.ReplayError, match="no Pi session and no Supervisor events"):
        replay.replay_run(str(empty))
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_dir / "rpc.sock"))
    (run / "native" / "pi-rpc.json").write_text(json.dumps({"socket": str(socket_dir / "rpc.sock"), "pid": os.getpid()}))
    try:
        with pytest.raises(replay.ReplayError, match="is live: `ora web"):
            replay.replay_run(str(run))
    finally:
        listener.close()
    with pytest.raises(ora_web.WebError, match="ora replay"):
        ora_web.live_run(str(empty))


def test_scene_exports_the_run_s_trace_and_its_exit_says_whether_the_check_passed(tmp_path, monkeypatch, run, capsys):
    calls = []
    answer = {"scene_trace": "somewhere", "passed": True, "failures": []}

    def export(root, traces, *, robodojo_replay=None):
        calls.append((root, traces))
        assert robodojo_replay is None                  # no slot was named
        return answer

    monkeypatch.setattr("ora.environment.replay_program.export_scene_trace", export)
    traces = tmp_path / "traces"
    args = build_parser().parse_args(["replay", str(run), "--scene", "--scene-traces", str(traces)])

    assert replay.run(args) == 0 and calls == [(run.resolve(), traces)]
    assert json.loads(capsys.readouterr().out)["scene_trace"] == "somewhere"
    answer.update(passed=False, scene_trace=None)
    assert replay.run(build_parser().parse_args(["replay", str(run), "--scene"])) == 1
    assert calls[-1] == (run.resolve(), None)


def test_a_run_without_recorded_controls_gets_a_marker_never_a_program(tmp_path, run, capsys):
    (run / "plan.json").write_text(json.dumps({"task": "T", "seed": 1, "runtime": {"kind": "robolab"}}))
    exported = tmp_path / "exported"

    assert replay.run(build_parser().parse_args(["replay", str(run), "--program", str(exported)])) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    assert sorted(path.name for path in exported.iterdir()) == ["unavailable.json"]
    assert replay.run(build_parser().parse_args(["replay", str(run), "--check", str(tmp_path / "out")])) == 2
    assert "never rebuilt from receipts" in capsys.readouterr().err


def test_a_robodojo_scene_runs_check_s_control_replay_with_the_trace_on_the_named_slot(tmp_path, monkeypatch, run, capsys):
    launched = []

    def export(root, traces, *, robodojo_replay):
        robodojo_replay(root, tmp_path / "motion", tmp_path / "trace")
        return {"scene_trace": "somewhere", "passed": True, "failures": []}

    monkeypatch.setattr("ora.environment.replay_program.export_scene_trace", export)
    monkeypatch.setattr("ora.orchestrator.backends.robodojo.motion.replay.main", launched.append)
    args = build_parser().parse_args(["replay", str(run), "--scene", "--resource-profile", "robodojo-gpu4-d1"])

    assert replay.run(args) == 0
    assert launched == [["--run", str(run.resolve()), "--output", str(tmp_path / "motion"),
                         "--resource-profile", "robodojo-gpu4-d1", "--mode", "metrics-after-replay",
                         "--scene-trace", str(tmp_path / "trace")]]


def test_the_page_s_options_and_the_video_s_size():
    args = build_parser().parse_args(["replay", "run-1", "--speed", "32", "--view", "detailed", "--expand-thinking",
                                                 "--lang", "zh", "--publish", "--from", "12"])
    assert replay.page_query(args, exporting=False) == {"replay": "1", "speed": "32", "motion": "1", "publish": "1",
                                                           "view": "detailed", "thinking": "expanded", "lang": "zh",
                                                           "from": "12"}
    exporting = replay.page_query(build_parser().parse_args(["replay", "run-1", "--export", "a.mp4"]), exporting=True)
    assert exporting == {"replay": "1", "speed": "16", "motion": "1", "export": "1"}
    same = build_parser().parse_args(["replay", "run-1", "--motion-speed", "same"])
    assert replay.page_query(same, exporting=True)["motion"] == "same"
    assert replay.parse_motion_speed("0.5") == 0.5
    for bad in ("0", "-1", "fast"):
        with pytest.raises(ArgumentTypeError):
            replay.parse_motion_speed(bad)
    defaults = build_parser().parse_args(["replay", "run-1"])
    assert (defaults.speed, defaults.fps, defaults.size, defaults.scale, defaults.zoom) == (16.0, 30, (1920, 1080), 2.0, 1.0)
    assert replay.parse_size("3840x2160") == (3840, 2160)
    for bad in ("1920", "1921x1080", "100x100", "axb"):
        with pytest.raises(ArgumentTypeError):
            replay.parse_size(bad)
    assert build_parser().parse_args(["replay", "run-1", "--export", "a.mp4"]).export == "a.mp4"


def test_ffmpeg_and_the_exporter_s_python_are_found_or_named(tmp_path, monkeypatch):
    tool = tmp_path / "ffmpeg"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("ORA_FFMPEG", raising=False)
    with pytest.raises(replay.ReplayError, match="no ffmpeg"):
        replay.find_ffmpeg(None)
    assert replay.find_ffmpeg(str(tool)) == str(tool.resolve())
    monkeypatch.setenv("ORA_FFMPEG", str(tool))
    assert replay.find_ffmpeg(None) == str(tool.resolve())
    monkeypatch.setenv("ORA_EXPORT_PYTHON", "/opt/py/bin/python")
    assert replay.exporter_python(None) == ["/opt/py/bin/python"]
    monkeypatch.delenv("ORA_EXPORT_PYTHON")
    monkeypatch.setattr(replay.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(replay.ReplayError, match=f"playwright=={replay.PLAYWRIGHT_VERSION}"):
        replay.exporter_python(None)
    monkeypatch.setattr(replay.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    assert replay.exporter_python(None) == ["/usr/bin/uv", "run", "--no-project", "--with",
                                                 f"playwright=={replay.PLAYWRIGHT_VERSION}", "python"]


def test_the_video_s_frames_follow_the_playback_and_its_report_names_each_stretch_s_rate():
    assert replay_export.frames_between(0.0, 110_000.0, fps=30) == (0.0, 1000 / 30, 3300)  # 110 s of video
    assert replay_export.frames_between(500.0, 500.0, fps=30)[2] == 1
    # The page's stretches: 15 s waiting at 16x, a 2 s action whose 0.2 s of simulated motion plays at 1x, 13 s waiting.
    segments = [
        {"kind": "wait", "from": 4_000, "to": 19_000, "playFrom": 0, "playTo": 937.5},
        {"kind": "motion", "from": 19_000, "to": 21_000, "playFrom": 937.5, "playTo": 1137.5, "simFrom": 0.1, "simTo": 0.3,
         "action": "action-a", "step": 0},
        {"kind": "wait", "from": 21_000, "to": 34_000, "playFrom": 1137.5, "playTo": 1950},
    ]
    report = replay_export.stretch_report(segments, 0, 1950, 16, 1)
    assert [item["rate"] for item in report] == ["16x recorded time", "1x simulated time (real time)", "16x recorded time"]
    assert report[1] == {"kind": "motion", "video_from_s": 0.938, "video_to_s": 1.137, "recorded_from_ms": 19_000,
                         "recorded_to_ms": 21_000, "rate": "1x simulated time (real time)", "sim_from_s": 0.1, "sim_to_s": 0.3,
                         "action": "action-a", "step": 0}
    settled = replay_export.stretch_report([{**segments[1], "settleFrom": 0.2, "settleTo": 0.3, "settleEnded": "at_rest"}],
                                               0, 1950, 16, 1)[0]
    assert (settled["settle_from_s"], settled["settle_to_s"], settled["settle_ended"]) == (0.2, 0.3, "at_rest")
    clipped = replay_export.stretch_report(segments, 1037.5, 1950, 16, 0.5)
    assert clipped[0]["video_from_s"] == 0 and clipped[0]["recorded_from_ms"] == 20_000 and clipped[0]["sim_from_s"] == 0.15
    assert replay_export.stretch_report(segments[:1], 0, 937.5, 16, "same")[0]["rate"] == "16x recorded time"
    parts = replay_export.stretches(3300, 12)
    assert parts[0] == (0, 275) and parts[-1][1] == 3300
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    assert replay_export.stretches(3, 8) == [(0, 1), (1, 2), (2, 3)]
    assert replay_export.layout({"width": 1920, "height": 1080, "scale": 2, "zoom": 1.25}) == (1536, 864, 2.5)
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (3840).to_bytes(4, "big") + (2160).to_bytes(4, "big")
    assert replay_export.png_size(header) == (3840, 2160) and replay_export.png_size(b"\xff\xd8") is None


def test_the_replay_web_app_reads_the_run_and_attaches_to_nothing(tmp_path, monkeypatch, run):
    seen: dict = {}

    class Started:
        def __init__(self, command, env, cwd, stdout, stderr):
            seen.update(command=command, env=env)
            Path(env["ORA_WEB_SOCKET"]).write_text("")

        def poll(self):
            return None

    monkeypatch.setattr(ora_web, "readiness", lambda: None)
    monkeypatch.setattr(subprocess, "Popen", Started)
    private = Path(tempfile.mkdtemp(prefix="ora-replay-app-", dir="/tmp"))
    try:
        ora_web.WebApp(run, private, "/usr/bin/node", replay=True, publish=True).start(timeout=5)
    finally:
        shutil.rmtree(private, ignore_errors=True)
    assert seen["env"]["ORA_WEB_REPLAY"] == "1" and seen["env"]["ORA_WEB_PUBLISH"] == "1"
    assert seen["env"]["ORA_WEB_RUN"] == str(run)
    assert seen["command"][-1].endswith("ora-pi/web/server/main.ts")
