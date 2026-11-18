"""The one origin a browser reaches for `ora web`: ORA's page and world view, behind a launch token.

The gateway listens on a 0600 unix socket (`ssh -L` forwards it) or on
127.0.0.1. Every request must name a loopback Host (DNS rebinding), and every
request but `/ora/health` must carry the session cookie, which the browser gets
once for the one-time launch token (`/?ora_token=...`, redirected away at
once). A state-changing request must also come from this origin (Origin).

Behind it:
  /live, /api/live/*, /static/*  ORA's read-only world view (world_view.py), bound to the run
  /ora/artifacts/<sha256>.<ext>  the run's stored agent images (the URLs of rpc/images.ts)
  /ora/capture/<frame>.png       the run's recorded motion frames (native/execution-capture/frames/,
                                 the URLs of rpc/frames.ts), operator-only
  /ora/operator-view/<step>.png  the environment's operator view (native/operator-view/), operator-only
  /ora/mark.svg                  ORA's mark (the terminal's pixel robot arm)
  /ora/cell, /ora/cell/...       a run on a real robot cell: the questions its worker waits on, the
                                 auto-approve switch, the held arms' alerts, and the terminal's
                                 /approve, /decline, /yolo, /home ... (web_cell.py); each outcome
                                 is handed to the run's agent process (/api/cell-note upstream)
  everything else                ORA's web server for the run (ora-pi/web/server, its own unix
                                 socket), but only the page and the routes it uses (ALLOWED): no
                                 files, directories, working directory, shell or updates.

A finished run's replay (`ora replay`, ``replay=True``) passes only the page and its one
route (REPLAY_ALLOWED: GET /api/replay) and has no world view and no cell routes; the run's
stored images, motion frames and operator views are served the same way.

The token and the cookie are never logged or written to the run.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import re
import secrets
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

from werkzeug.wrappers import Request, Response

from .web_cell import CellControls, CellRouteError
from .web_cell import dispatch as dispatch_cell

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
              "transfer-encoding", "upgrade", "content-length", "cookie"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
WORLD_PREFIXES = ("/live", "/api/live/", "/static/")
#: Routes of ORA's web server a browser may reach through the gateway, as (method, path pattern):
#: ora-pi/web/server/app.ts. The agent process refuses what the run's mode does not allow.
ALLOWED: tuple[tuple[str, re.Pattern[str]], ...] = tuple((method, re.compile(f"^{pattern}$")) for method, pattern in (
    ("GET", r"/api/snapshot"), ("GET", r"/api/events"), ("POST", r"/api/send"), ("POST", r"/api/dialog"),
    ("POST", r"/api/play"), ("GET", r"/api/models"), ("POST", r"/api/model"), ("GET", r"/api/thinking-levels"),
    ("POST", r"/api/thinking"), ("POST", r"/api/compact"), ("POST", r"/api/name"), ("POST", r"/api/new-session"),
    ("GET", r"/api/fork-messages"), ("POST", r"/api/fork"), ("GET", r"/api/stats"), ("GET", r"/api/frames"),
))
#: The one route of a finished run's replay (ora-pi/web/server/app.ts createReplayApp).
REPLAY_ALLOWED: tuple[tuple[str, re.Pattern[str]], ...] = (("GET", re.compile(r"^/api/replay$")),)
CELL_BODY_LIMIT = 8192
CELL_NOTE_TIMEOUT_S = 10.0
_ARTIFACT = re.compile(r"^/ora/artifacts/([0-9a-f]{64}\.(?:png|img|jpg|jpeg|webp))$")
_ARTIFACT_TYPES = {"png": "image/png", "img": "application/octet-stream", "jpg": "image/jpeg",
                   "jpeg": "image/jpeg", "webp": "image/webp"}
#: Operator-only pictures of the run, by URL prefix: the directory under the run and the file names it serves.
_RUN_PICTURES = {
    "/ora/capture/": ("native/execution-capture/frames", re.compile(r"^\d{8}-camera-\d{1,2}\.png$")),
    "/ora/operator-view/": ("native/operator-view", re.compile(r"^\d{6,9}\.png$")),
}


def loopback_host(host: str | None) -> bool:
    if not host:
        return False
    name = urlsplit("//" + host).hostname or ""
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def allowed(method: str, path: str, *, replay: bool = False) -> bool:
    if method in {"GET", "HEAD"} and not path.startswith("/api/"):
        return True  # the page (index, assets)
    routes = REPLAY_ALLOWED if replay else ALLOWED
    return any(method == allowed_method and pattern.match(path) for allowed_method, pattern in routes)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float | None = None) -> None:
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_path)
        self.sock = sock


def mark_svg() -> str:
    """ORA's mark, the terminal's pixel robot arm (tui/screen.py), teal on a dark tile."""
    from ora.agent.tui.screen import LOGO

    cells = "".join(f'<rect x="{x + 1}" y="{y + 3}" width="1" height="1"/>'
                    for y, row in enumerate(LOGO) for x, cell in enumerate(row) if cell == "#")
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 15 15" shape-rendering="crispEdges">'
            f'<rect width="15" height="15" rx="3" fill="#111417"/><g fill="#4ecdc4">{cells}</g></svg>')


class WebGateway:
    """The WSGI application in front of ORA's web server and the world view, for one run."""

    def __init__(self, *, run_root: Path, upstream: str | None, world_app: Callable | None, launch_token: str,
                 connect: Callable[[str], http.client.HTTPConnection] | None = None, replay: bool = False) -> None:
        self.run_root = Path(run_root)
        self.replay = replay
        self.upstream = upstream
        self.world_app = world_app
        self._launch_token: str | None = launch_token
        self.cookie_secret = secrets.token_urlsafe(32)
        self.cookie_name = "ora_web_" + hashlib.sha256(self.cookie_secret.encode()).hexdigest()[:12]
        self.connect = connect or (lambda path: _UnixConnection(path, timeout=None))
        self.cell = None if replay else CellControls(self.run_root, note=self._cell_note)

    # -- WSGI ------------------------------------------------------------------------------

    def __call__(self, environ: Mapping[str, Any], start_response: Callable) -> Iterable[bytes]:
        # A Response, or the world view's own WSGI app.
        return self.dispatch(Request(environ))(environ, start_response)

    def dispatch(self, request: Request) -> Any:
        if not loopback_host(request.host):
            return self._refuse(403, "the ORA web gateway answers loopback hosts only")
        path = request.path
        if path == "/ora/health":
            return Response(json.dumps({"service": "ora-web", "status": "ready"}), mimetype="application/json")
        token = request.args.get("ora_token")
        if token is not None:
            return self._exchange(request, token)
        cookie = request.cookies.get(self.cookie_name, "")
        if not cookie or not secrets.compare_digest(cookie, self.cookie_secret):
            return self._refuse(403, "open the link `ora web` (or ora-remote) printed; this browser has no ORA session")
        if request.method not in SAFE_METHODS and request.headers.get("Origin") != f"{request.scheme}://{request.host}":
            return self._refuse(403, "cross-origin request refused")
        if path == "/ora/mark.svg":
            return Response(mark_svg(), mimetype="image/svg+xml", headers={"Cache-Control": "no-store"})
        if path == "/ora/cell" or path.startswith("/ora/cell/"):
            return self._cell_route(request)
        match = _ARTIFACT.match(path)
        if match:
            return self._artifact(match.group(1))
        for prefix, (directory, name) in _RUN_PICTURES.items():
            if path.startswith(prefix) and name.match(path[len(prefix):]):
                return self._picture(directory, path[len(prefix):])
        if path.startswith("/ora/"):
            return self._refuse(404, "no such ORA resource")
        if self.world_app is not None and (path == "/live" or path.startswith(WORLD_PREFIXES[1:])):
            return self.world_app
        # Matched and forwarded as the browser sent it (the raw target, never a decoded path).
        target = request.environ.get("RAW_URI") or request.environ.get("REQUEST_URI") or quote(path)
        if self.upstream is None or not allowed(request.method, target.split("?", 1)[0], replay=self.replay):
            return self._refuse(403, f"{request.method} {path} is not available through ora web")
        return self._proxy(request, target)

    # -- Pieces ----------------------------------------------------------------------------

    @staticmethod
    def _refuse(status: int, message: str) -> Response:
        return Response(json.dumps({"error": message}), status=status, mimetype="application/json",
                        headers={"Cache-Control": "no-store"})

    def _exchange(self, request: Request, token: str) -> Response:
        expected, self._launch_token = self._launch_token, None  # one use, right or wrong
        if expected is None or not secrets.compare_digest(token, expected):
            return self._refuse(403, "this ORA web link was already used or is wrong; run the command again")
        query = urlencode([(key, value) for key, value in parse_qsl(request.query_string.decode()) if key != "ora_token"])
        response = Response(status=303, headers={"Location": request.path + (f"?{query}" if query else ""),
                                                 "Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
        response.set_cookie(self.cookie_name, self.cookie_secret, httponly=True, samesite="Strict", path="/")
        return response

    def _artifact(self, name: str) -> Response:
        path = self.run_root / "native" / "agent-artifacts" / name
        if path.is_symlink() or not path.is_file():
            return self._refuse(404, "no such artifact")
        return Response(path.read_bytes(), mimetype=_ARTIFACT_TYPES[name.rsplit(".", 1)[1]],
                        headers={"Cache-Control": "no-store"})

    def _picture(self, directory: str, name: str) -> Response:
        """A recorded motion frame or an operator view: a regular PNG file of the run, never a link."""

        folder = self.run_root / directory
        path = folder / name
        if any(part.is_symlink() for part in (folder, path)) or not path.is_file():
            return self._refuse(404, "no such picture")
        # Written once and never changed (a frame index or an observation step names it).
        return Response(path.read_bytes(), mimetype="image/png", headers={"Cache-Control": "private, max-age=3600"})

    def _cell_route(self, request: Request) -> Response:
        if self.cell is None:
            return self._refuse(404, "a recorded run takes no cell commands")
        if (request.content_length or 0) > CELL_BODY_LIMIT:
            return self._refuse(413, "the request is too large")
        body = request.get_data() if request.method == "POST" else b""
        try:
            result = dispatch_cell(self.cell, request.method, request.path, body)
        except CellRouteError as exc:
            return self._refuse(exc.status, str(exc))
        return Response(json.dumps(result), mimetype="application/json", headers={"Cache-Control": "no-store"})

    def _cell_note(self, note: dict[str, Any]) -> None:
        """A cell command's outcome for the run's agent process, through ORA's web server: best effort, on its
        own thread, so the page has its answer at once even when the agent process is slow or gone."""

        if self.upstream is None:
            return
        upstream = self.upstream

        def send() -> None:
            connection = self.connect(upstream)
            connection.timeout = CELL_NOTE_TIMEOUT_S
            try:
                connection.request("POST", "/api/cell-note", body=json.dumps(note),
                                   headers={"Content-Type": "application/json"})
                connection.getresponse().read()
            except OSError:
                pass                     # the run's agent process is gone: the command ran all the same
            finally:
                connection.close()

        threading.Thread(target=send, daemon=True, name="ora-web-cell-note").start()

    def _proxy(self, request: Request, target: str) -> Response:
        connection = self.connect(self.upstream)
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_BY_HOP}
        body = request.get_data() if request.method not in {"GET", "HEAD"} else None
        try:
            connection.request(request.method, target, body=body, headers=headers)
            upstream = connection.getresponse()
        except OSError as exc:
            connection.close()
            return self._refuse(502, f"ORA's web server is not answering ({exc})")
        headers = [(key, value) for key, value in upstream.getheaders() if key.lower() not in HOP_BY_HOP]

        def stream() -> Iterable[bytes]:
            try:
                while True:
                    chunk = upstream.read1(65536)
                    if not chunk:
                        return
                    yield chunk
            finally:
                connection.close()

        return Response(stream(), status=upstream.status, headers=headers, direct_passthrough=True)


__all__ = ["ALLOWED", "REPLAY_ALLOWED", "WebGateway", "allowed", "loopback_host", "mark_svg"]
