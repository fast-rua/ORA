"""`ora web`'s gateway and setup, without Node, the page or a live run (ora_web_gateway, ora_web)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer

import pytest
from werkzeug.test import Client

from ora.app import web as ora_web
from ora.app.web_gateway import WebGateway, allowed, loopback_host, mark_svg

TOKEN = "launch-token-0123456789abcdef"


class Upstream(ThreadingMixIn, UnixStreamServer):
    """Stand-in for ORA's web server: records requests, serves an index page, an SSE stream and JSON."""
    daemon_threads = True

    def __init__(self, path: str) -> None:
        self.seen: list[tuple[str, str, dict, bytes]] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def address_string(self) -> str:
                return "unix"

            def log_message(self, *args) -> None:
                pass

            def _answer(handler, body: bytes, kind: str) -> None:
                handler.send_response(200)
                handler.send_header("Content-Type", kind)
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def do_GET(handler) -> None:  # noqa: N802
                self.seen.append(("GET", handler.path, dict(handler.headers), b""))
                if handler.path == "/":
                    handler._answer(b"<html><head><title>ORA</title></head><body></body></html>", "text/html")
                elif handler.path == "/api/events":
                    body = b"event: ready\ndata: {}\n\n"
                    handler._answer(body, "text/event-stream")
                else:
                    handler._answer(json.dumps({"path": handler.path}).encode(), "application/json")

            def do_POST(handler) -> None:  # noqa: N802
                length = int(handler.headers.get("Content-Length", 0))
                self.seen.append(("POST", handler.path, dict(handler.headers), handler.rfile.read(length)))
                handler._answer(b'{"ok": true}', "application/json")

        super().__init__(path, Handler)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run-1"
    (root / "native" / "agent-artifacts").mkdir(parents=True)
    (root / "native" / "agent-artifacts" / ("a" * 64 + ".png")).write_bytes(b"\x89PNG stored")
    return root


@pytest.fixture
def socket_dir():
    """A short directory for unix sockets, as ora web's own (/tmp): tmp_path under a long TMPDIR
    passes the 107-byte limit of a socket path."""
    path = Path(tempfile.mkdtemp(prefix="ora-web-test-", dir="/tmp"))
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


def world_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"world " + environ["PATH_INFO"].encode()]


def gateway_for(run: Path, upstream_path: str | None) -> tuple[WebGateway, Client]:
    gateway = WebGateway(run_root=run, upstream=upstream_path, world_app=world_app, launch_token=TOKEN)
    return gateway, Client(gateway)


def signed_in(client: Client) -> None:
    response = client.get(f"/?ora_token={TOKEN}&x=1", headers={"Host": "127.0.0.1:43211"})
    assert response.status_code == 303 and response.headers["Location"] == "/?x=1"
    assert "HttpOnly" in response.headers["Set-Cookie"] and "SameSite=Strict" in response.headers["Set-Cookie"]


def test_the_launch_token_works_once_and_nothing_but_health_answers_without_the_cookie(run, upstream):
    gateway, client = gateway_for(run, upstream[1])
    host = {"Host": "127.0.0.1:43211"}
    assert client.get("/ora/health", headers=host).json == {"service": "ora-web", "status": "ready"}
    for path in ("/", "/live", "/api/snapshot", "/ora/mark.svg", f"/ora/artifacts/{'a' * 64}.png"):
        assert client.get(path, headers=host).status_code == 403, path
    assert client.get("/?ora_token=wrong", headers=host).status_code == 403
    assert client.get(f"/?ora_token={TOKEN}", headers=host).status_code == 403, "a wrong try spends the token"

    gateway, client = gateway_for(run, upstream[1])
    signed_in(client)
    assert client.get("/live", headers=host).data == b"world /live"
    assert Client(gateway).get(f"/?ora_token={TOKEN}", headers=host).status_code == 403, "one use"
    assert Client(gateway).get("/", headers=host).status_code == 403, "another browser has no cookie"


def test_loopback_hosts_only_and_state_changes_from_this_origin_only(run, upstream):
    _, client = gateway_for(run, upstream[1])
    signed_in(client)
    assert client.get("/api/snapshot", headers={"Host": "attacker.example:43211"}).status_code == 403
    assert client.get("/api/snapshot", headers={"Host": "127.0.0.1:43211"}).status_code == 200
    send = "/api/send"
    assert client.post(send, json={"text": "hi"}, headers={"Host": "127.0.0.1:43211"}).status_code == 403
    assert client.post(send, json={"text": "hi"}, headers={
        "Host": "127.0.0.1:43211", "Origin": "http://evil.example"}).status_code == 403
    assert client.post(send, json={"text": "hi"}, headers={
        "Host": "127.0.0.1:43211", "Origin": "http://127.0.0.1:43211"}).status_code == 200
    method, path, headers, body = upstream[0].seen[-1]
    assert (method, path, json.loads(body)) == ("POST", send, {"text": "hi"})
    assert "Cookie" not in headers, "the session cookie stays with the gateway"


@pytest.mark.parametrize("method,path", [
    # Watercol/pi-web-ui's file, directory and working-directory routes, and shell or update ones.
    ("GET", "/api/files"), ("GET", "/api/file-content"), ("GET", "/api/file-raw"), ("GET", "/api/directories"),
    ("POST", "/api/cwd"), ("GET", "/api/update-check"), ("POST", "/api/bash"), ("POST", "/api/terminal"),
    ("POST", "/api/session"), ("GET", "/api/sessions"), ("POST", "/api/export"), ("PUT", "/api/send"),
    ("DELETE", "/api/snapshot"), ("GET", "/api/send/../files"),
])
def test_routes_outside_the_page_s_own_are_closed(run, upstream, method, path):
    assert not allowed(method, path)
    _, client = gateway_for(run, upstream[1])
    signed_in(client)
    response = client.open(path, method=method, headers={"Host": "127.0.0.1:1", "Origin": "http://127.0.0.1:1"})
    assert response.status_code == 403 and "not available through ora web" in response.json["error"]
    assert not any(seen[1] == path for seen in upstream[0].seen)


def test_the_page_its_events_images_and_mark_pass_through_unchanged(run, upstream):
    _, client = gateway_for(run, upstream[1])
    signed_in(client)
    host = {"Host": "127.0.0.1:43211"}
    page = client.get("/", headers=host)
    assert page.data == b"<html><head><title>ORA</title></head><body></body></html>", "nothing is injected"
    events = client.get("/api/events", headers=host)
    assert events.headers["Content-Type"] == "text/event-stream" and b"event: ready" in events.data
    image = client.get(f"/ora/artifacts/{'a' * 64}.png", headers=host)
    assert image.data == b"\x89PNG stored" and image.mimetype == "image/png"
    assert client.get("/ora/artifacts/..%2F..%2Fplan.json", headers=host).status_code == 404


def test_the_recorded_motion_and_the_operator_view_are_served_as_the_run_s_own_pngs(run, upstream):
    frames = run / "native" / "execution-capture" / "frames"
    frames.mkdir(parents=True)
    (frames / "00000012-camera-1.png").write_bytes(b"\x89PNG motion")
    views = run / "native" / "operator-view"
    views.mkdir(parents=True)
    (views / "000004.png").write_bytes(b"\x89PNG view")
    (views / "000005.png").symlink_to(run / "native" / "agent-artifacts" / ("a" * 64 + ".png"))
    _, client = gateway_for(run, upstream[1])
    host = {"Host": "127.0.0.1:43211"}
    assert client.get("/ora/capture/00000012-camera-1.png", headers=host).status_code == 403, "the cookie first"
    signed_in(client)
    motion = client.get("/ora/capture/00000012-camera-1.png", headers=host)
    assert motion.data == b"\x89PNG motion" and motion.mimetype == "image/png"
    assert client.get("/ora/operator-view/000004.png", headers=host).data == b"\x89PNG view"
    for path in ("/ora/operator-view/000005.png", "/ora/capture/12-camera-1.png", "/ora/capture/..%2Fmanifest.json",
                 "/ora/operator-view/000009.png", "/ora/capture/00000012-camera-1.jpg"):
        assert client.get(path, headers=host).status_code == 404, path
    assert allowed("GET", "/api/frames") and not allowed("POST", "/api/frames")
    assert client.get("/api/frames?step=4", headers=host).json == {"path": "/api/frames?step=4"}


def test_the_page_wears_ora_s_mark_drawn_from_the_terminal_s(run, upstream):
    from ora.agent.tui.screen import LOGO

    svg = mark_svg()
    assert svg.count('width="1" height="1"') == sum(row.count("#") for row in LOGO) and "#4ecdc4" in svg
    _, client = gateway_for(run, upstream[1])
    signed_in(client)
    mark = client.get("/ora/mark.svg", headers={"Host": "127.0.0.1:43211"})
    assert mark.mimetype == "image/svg+xml" and mark.data.decode() == svg


def test_loopback_names():
    assert loopback_host("127.0.0.1:5") and loopback_host("localhost") and loopback_host("[::1]:9")
    assert not loopback_host("10.0.0.1:5") and not loopback_host("") and not loopback_host("127.0.0.1.evil.example")


def test_the_page_is_served_only_as_built_from_this_checkout_s_sources(tmp_path):
    web = tmp_path / "web"
    (web / "src" / "lib").mkdir(parents=True)
    for name in ("package.json", "package-lock.json", "index.html", "vite.config.ts", "tsconfig.json"):
        (web / name).write_text(name)
    (web / "src" / "lib" / "a.ts").write_text("export const a = 1;\n")
    assert ora_web.readiness(web).startswith("the ORA web page is not built")
    (web / "dist").mkdir()
    (web / "dist" / "index.html").write_text("<!doctype html>")
    assert "built from other sources" in ora_web.readiness(web)
    (web / "dist" / ora_web.BUILD_STAMP).write_text(json.dumps({"sources_sha256": ora_web.source_digest(web)}))
    assert ora_web.readiness(web) is None
    (web / "node_modules").mkdir()  # an npm ci changes nothing the page is built from
    (web / "node_modules" / "x.js").write_text("x")
    assert ora_web.readiness(web) is None
    (web / "src" / "lib" / "a.ts").write_text("export const a = 2;\n")
    assert "built from other sources" in ora_web.readiness(web)


def test_a_run_that_is_not_live_is_refused(tmp_path, socket_dir, monkeypatch):
    root = tmp_path / "run-2"
    (root / "native").mkdir(parents=True)
    with pytest.raises(ora_web.WebError, match="not live here"):
        ora_web.live_run(str(root))
    (root / "native" / "pi-rpc.json").write_text(json.dumps({"socket": str(socket_dir / "gone.sock"), "pid": os.getpid()}))
    with pytest.raises(ora_web.WebError, match="not live here"):
        ora_web.live_run(str(root))
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_dir / "gone.sock"))
    try:
        assert ora_web.live_run(str(root)) == root
    finally:
        listener.close()


def test_the_token_file_is_read_once(tmp_path):
    path = tmp_path / "t.token"
    path.write_text(TOKEN + "\n")
    assert ora_web.read_token(str(path)) == TOKEN and not path.exists()
    path.write_text("x")
    with pytest.raises(ora_web.WebError):
        ora_web.read_token(str(path))


def test_a_live_run_is_found_by_its_rpc_endpoint_before_free_play_writes_a_session(tmp_path, socket_dir, monkeypatch):
    root = tmp_path / "ora-play-1"
    (root / "native").mkdir(parents=True)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_dir / "rpc.sock"))
    (root / "native" / "pi-rpc.json").write_text(json.dumps({"socket": str(socket_dir / "rpc.sock"), "pid": os.getpid()}))
    monkeypatch.setattr(ora_web.paths, "iter_run_roots", lambda: iter([root, tmp_path / "finished"]))
    try:
        assert not (root / "native" / "pi-session").exists()
        assert ora_web.live_run(None) == root
    finally:
        listener.close()
