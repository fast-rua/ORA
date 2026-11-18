"""`ora web`: a live run in the browser, ORA's own page over the run's Pi RPC.

    ora web setup                       install and build the page (ora-pi/web: npm ci, then vite build)
    ora web [RUN] [--last] [--port N | --socket PATH] [--token-file PATH] [--no-browser]

The run's agent core serves Pi's RPC protocol (ora-pi/src/rpc/host.ts). ORA's
web server (ora-pi/web/server, Node; its page seeded from Watercol/pi-web-ui,
MIT) attaches to it through `ora-rpc-attach` as one controlling client and
serves the page: the conversation, the robot's cameras and each step's images
on one screen. In front of it the gateway (web_gateway.py) adds a one-time
launch token exchanged for a cookie, loopback Host and same-origin checks, the
routes the page uses and nothing else, the run's stored images and its world
view (world_view.py), and on a real robot cell the cell's controls
(web_cell.py).

The page is built once by `ora web setup`; the build records a digest of the
sources it was built from (dist/ora-build.json), and `ora web` refuses a page
built from other sources. Nothing in ora-pi/web is patched after install, so
`npm ci` there changes nothing `ora web` relies on; a changed page is rebuilt
by `ora web setup`.

`--socket` serves the gateway on a unix socket for `ssh -L` (ora-remote);
otherwise it listens on 127.0.0.1 and prints the link with the token. A run
that is not live is refused: no Pi of ours opens a finished run's session.
A finished run is replayed from its evidence by `ora replay` (replay.py),
on the same page and gateway with no Pi RPC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from ora.orchestrator import paths

WEB = paths.PROJECT_ROOT / "ora-pi" / "web"
SERVER = WEB / "server" / "main.ts"
DIST = WEB / "dist"
BUILD_STAMP = "ora-build.json"
ATTACH = paths.PROJECT_ROOT / "ora-pi" / "src" / "rpc" / "attach.ts"
RPC_ENDPOINT = Path("native") / "pi-rpc.json"
#: What the built page is made from (the web server itself runs from its sources).
BUILD_INPUTS = ("package.json", "package-lock.json", "index.html", "vite.config.ts", "tsconfig.json", "src")


class WebError(RuntimeError):
    """`ora web` cannot serve this run."""


# -- Setup ------------------------------------------------------------------------------

def source_digest(web: Path = WEB) -> str:
    """sha256 over the page's sources (paths and bytes), the identity of a build."""

    digest = hashlib.sha256()
    for name in BUILD_INPUTS:
        root = web / name
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else [root]
        for path in files:
            digest.update(path.relative_to(web).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() if path.is_file() else b"<missing>")
            digest.update(b"\0")
    return digest.hexdigest()


def readiness(web: Path = WEB) -> str | None:
    """Why the page cannot be served, or None."""

    dist = web / "dist"
    if not (dist / "index.html").is_file():
        return "the ORA web page is not built; run `ora web setup` (npm ci and a build in ora-pi/web)"
    try:
        stamp = json.loads((dist / BUILD_STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if stamp.get("sources_sha256") != source_digest(web):
        return "the ORA web page was built from other sources than this checkout's; run `ora web setup`"
    return None


def setup(web: Path = WEB) -> int:
    from ora.orchestrator.episode.agent_core import require_pi_runtime

    node = require_pi_runtime()
    npm = Path(node).with_name("npm")
    environment = {**os.environ, "PATH": f"{Path(node).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    for arguments in (["ci", "--no-audit", "--no-fund"], ["run", "build"]):
        command = [str(npm) if npm.exists() else "npm", *arguments, "--prefix", str(web)]
        print("ora web: " + " ".join(command), flush=True)
        if subprocess.call(command, env=environment):
            return 2
    (web / "dist" / BUILD_STAMP).write_text(json.dumps({
        "schema": "ora.web-build.v1", "sources_sha256": source_digest(web), "built_unix_s": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"ora web: the page is built ({web / 'dist'})", flush=True)
    return 0


# -- The run ----------------------------------------------------------------------------

def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, TypeError, ValueError):
        return False


def rpc_endpoint(root: Path) -> dict[str, Any] | None:
    """The run's live Pi RPC endpoint on this host, or None."""

    try:
        record = json.loads((root / RPC_ENDPOINT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("socket"), str):
        return None
    return record if _alive(record.get("pid")) and os.path.exists(record["socket"]) else None


def _live_runs() -> list[Path]:
    """Runs whose agent process serves Pi RPC on this host, newest first (free play has no session file
    before its first instruction, so the endpoint, not the session, says a run is live)."""

    live = [root for root in paths.iter_run_roots() if rpc_endpoint(root) is not None]
    return sorted(live, key=lambda root: (root / RPC_ENDPOINT).stat().st_mtime, reverse=True)


def live_run(value: str | None, *, last: bool = False) -> Path:
    from .interactive import resolve_run

    if value:
        root = resolve_run(value)
    else:
        live = _live_runs()
        if not live:
            raise WebError("no run on this host serves Pi RPC; start one with `ora play` or `ora interactive`")
        if len(live) > 1 and not last:
            raise WebError("several runs are live; name one (`ora sessions`) or pass --last")
        root = live[0]
    if rpc_endpoint(root) is None:
        raise WebError(f"{root.name} is not live here (its agent process serves no Pi RPC on this host); "
                       "replay a finished run with `ora replay`, or read it with `ora attach` or `ora sessions export`")
    return root


# -- ORA's web server -------------------------------------------------------------------

class WebApp:
    """ORA's web server for one run (ora-pi/web/server), on a 0600 unix socket in a private directory.

    ``replay`` serves a finished run's replay from its evidence instead (no Pi RPC);
    ``publish`` leaves the run's ID and absolute paths out of it.
    """

    def __init__(self, root: Path, private: Path, node: str, *, replay: bool = False, publish: bool = False) -> None:
        self.root, self.private, self.node = root, private, node
        self.replay, self.publish = replay, publish
        self.socket = str(private / "web.sock")
        self.process: subprocess.Popen | None = None

    def start(self, timeout: float = 60.0) -> "WebApp":
        problem = readiness()
        if problem:
            raise WebError(problem)
        environment = {key: value for key, value in os.environ.items()
                       if key in {"PATH", "LANG", "LC_ALL", "TZ", "NODE_EXTRA_CA_CERTS"}}
        environment.update({
            "PATH": f"{Path(self.node).parent}{os.pathsep}{environment.get('PATH', '')}",
            "HOME": str(self.private), "ORA_WEB_RUN": str(self.root), "ORA_WEB_SOCKET": self.socket,
            "ORA_WEB_STATIC": str(DIST), "ORA_WEB_ATTACH": str(ATTACH),
        })
        if self.replay:
            environment["ORA_WEB_REPLAY"] = "1"
            if self.publish:
                environment["ORA_WEB_PUBLISH"] = "1"
        log = (self.private / "web.log").open("ab")
        old = os.umask(0o077)
        try:
            # In this process's group: a hangup that ends `ora` (a dropped SSH session) ends it too.
            self.process = subprocess.Popen([self.node, str(SERVER)], env=environment, cwd=str(self.private),
                                            stdout=log, stderr=subprocess.STDOUT)
        finally:
            os.umask(old)
        deadline = time.monotonic() + timeout
        while not os.path.exists(self.socket):
            if self.process.poll() is not None or time.monotonic() > deadline:
                said = (self.private / "web.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
                raise WebError(f"ORA's web server did not start: {said[-1] if said else 'no output'}")
            time.sleep(0.1)
        os.chmod(self.socket, 0o600)
        return self

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# -- The served web view ----------------------------------------------------------------

class WebServer:
    """ORA's web server, the world view and the gateway for one live run, until closed.

    ``replay``: a finished run's replay instead (`ora replay`): no world view, no Pi RPC, the
    page and GET /api/replay only; ``publish`` leaves the run's ID and paths out of it.
    """

    def __init__(self, root: Path, *, socket_path: str | None = None, port: int = 0,
                 token: str | None = None, replay: bool = False, publish: bool = False,
                 server_factory=None) -> None:
        self.root = root
        self.replay, self.publish = replay, publish
        self.server_factory = server_factory  # the world view's server; world_view.WorldViewServer by default
        self.socket_path, self.port = socket_path, port
        self.token = token or secrets.token_urlsafe(32)
        self.private = Path(tempfile.mkdtemp(prefix="ora-web-", dir="/tmp"))
        self.app: WebApp | None = None
        self.server = None
        self.thread: threading.Thread | None = None
        self._hangup: Any = None

    def start(self) -> "WebServer":
        from werkzeug.serving import WSGIRequestHandler, make_server

        from ora.orchestrator.episode.agent_core import require_pi_runtime
        from .web_gateway import WebGateway

        class Quiet(WSGIRequestHandler):
            def log_request(self, *args: Any) -> None:  # the first request's URL holds the launch token
                pass

            def address_string(self) -> str:
                return "local"

        try:
            self.app = WebApp(self.root, self.private, require_pi_runtime(), replay=self.replay,
                              publish=self.publish).start()
            world_app = None
            if not self.replay:
                if self.server_factory is None:
                    from ora.orchestrator.world_view import WorldViewServer  # Flask loads only when a world view starts
                    factory = WorldViewServer
                else:
                    factory = self.server_factory
                world = factory()
                world.bind(self.root)
                world_app = world.app
            gateway = WebGateway(run_root=self.root, upstream=self.app.socket, world_app=world_app,
                                 launch_token=self.token, replay=self.replay)
            if self.socket_path:
                if os.path.lexists(self.socket_path):
                    os.unlink(self.socket_path)
                old = os.umask(0o077)
                try:
                    self.server = make_server("unix://" + self.socket_path, 0, gateway, threaded=True,
                                              request_handler=Quiet)
                finally:
                    os.umask(old)
            else:
                self.server = make_server("127.0.0.1", self.port, gateway, threaded=True, request_handler=Quiet)
                self.port = self.server.server_port
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="ora-web")
            self.thread.start()
            if threading.current_thread() is threading.main_thread():
                # A dropped SSH session hangs this process up: exit through close(), not around it.
                self._hangup = signal.signal(signal.SIGHUP, _exit_on_hangup)
        except BaseException:
            self.close()
            raise
        return self

    @property
    def url(self) -> str | None:
        return None if self.socket_path else f"http://127.0.0.1:{self.port}/?ora_token={self.token}"

    def close(self) -> None:
        hangup, self._hangup = self._hangup, None
        if hangup is not None and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGHUP, hangup)
        server, self.server = self.server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self.socket_path and os.path.lexists(self.socket_path):
            os.unlink(self.socket_path)
        if self.app is not None:
            self.app.close()
        shutil.rmtree(self.private, ignore_errors=True)


def _exit_on_hangup(_signum: int, _frame: Any) -> None:
    raise SystemExit(129)


def read_token(path: str | None) -> str | None:
    """A launch token handed over in a file (ora-remote), removed once read."""

    if not path:
        return None
    token = Path(path).read_text(encoding="utf-8").strip()
    Path(path).unlink()
    if not 16 <= len(token) <= 128 or not token.replace("-", "").replace("_", "").isalnum():
        raise WebError("the launch token file does not hold a token")
    return token


def from_environment(root: Path, *, wait_s: float = 180.0) -> WebServer | None:
    """The web view `ora-remote --web` asked for (ORA_WEB_SOCKET, ORA_WEB_TOKEN_FILE), beside an attached terminal.

    Started once the run's agent process serves Pi RPC; a failure is said and
    the terminal UI goes on without it.
    """

    socket_path = os.environ.get("ORA_WEB_SOCKET")
    if not socket_path:
        return None
    try:
        token = read_token(os.environ.get("ORA_WEB_TOKEN_FILE"))
        deadline = time.monotonic() + wait_s
        while rpc_endpoint(root) is None:
            if time.monotonic() > deadline:
                raise WebError("the run's agent process serves no Pi RPC yet")
            time.sleep(0.25)
        server = WebServer(root, socket_path=socket_path, token=token).start()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora web: {exc}; the terminal UI goes on without the web view", flush=True)
        return None
    print("ORA web: serving this run through the forwarded socket", flush=True)
    return server


def register(sub: Any) -> None:
    parser = sub.add_parser("web", help="show a live Pi-core run in the browser (ORA's page on the run's Pi RPC: "
                            "conversation and cameras); `ora web setup` first",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", help="run ID or directory, or `setup` (default: the live run)")
    parser.add_argument("--last", action="store_true", help="the most recent live run when several are")
    parser.add_argument("--port", type=int, default=0, help="loopback port (default: a free one)")
    parser.add_argument("--socket", help="serve on this unix socket (0600) instead, for `ssh -L` (ora-remote)")
    parser.add_argument("--token-file", help="the launch token, in a file this command removes (ora-remote)")
    parser.add_argument("--no-browser", action="store_true", help="print the link instead of opening it")


def run(args: argparse.Namespace) -> int:
    try:
        if args.run == "setup":
            return setup()
        token = read_token(args.token_file)  # removed first, whatever happens next
        root = live_run(args.run, last=args.last)
        server = WebServer(root, socket_path=args.socket, port=args.port, token=token).start()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora web: {exc}", file=sys.stderr)
        return 2
    try:
        if server.url:
            print(f"ORA web for {root.name}: {server.url}", flush=True)
            remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
            if not args.no_browser and not remote:
                webbrowser.open_new_tab(server.url)
        else:
            print(f"ORA web for {root.name} on {args.socket} (Ctrl+C stops it; the run continues)", flush=True)
        announced = False
        while True:
            time.sleep(1)
            if not announced and rpc_endpoint(root) is None:
                print("the run's agent session has closed; the page shows what it recorded. Ctrl+C closes it.",
                      flush=True)
                announced = True
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()


__all__ = ["WebApp", "WebError", "WebServer", "from_environment", "live_run", "readiness", "register", "rpc_endpoint",
           "run", "source_digest"]
