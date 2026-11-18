"""Optional read-only world observer for ORA; never owns a Runtime or simulator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_WORLD_VIEW = {
    "view": "auto", "port": 0, "browser": "auto", "bind_host": "127.0.0.1",
    "human_live_watch": True, "human_interactive": True, "json_implicit": False,
    "sealed_watch_auto": False, "connected_requires_decoded_frame": True,
}


class _GivenPort(argparse.Action):
    """``--world-port``, noting that the operator gave it: a given port, 0 included, opts in to the world view."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.world_port_given = True


def add_world_options(parser) -> None:
    parser.add_argument("--world-view", choices=("auto", "off"), default=DEFAULT_WORLD_VIEW["view"],
                        help="automatically show the exact live run's cameras (default: human terminal only)")
    parser.add_argument("--world-port", type=int, default=DEFAULT_WORLD_VIEW["port"], action=_GivenPort,
                        help="loopback world-view port; 0 chooses a free port")
    parser.set_defaults(world_port_given=False)
    parser.add_argument("--world-browser", choices=("auto", "off"), default=DEFAULT_WORLD_VIEW["browser"],
                        help="open the local world view after its server is ready")
    parser.add_argument("--world-public-url", help="browser-facing /live URL, for example a local SSH-forwarded URL")


def observer_factory(args, *, enabled: bool):
    """Keep headless defaults quiet, including direct JSON protocol consumers."""
    if not 0 <= args.world_port <= 65535:
        raise ValueError("world-port must be in [0, 65535]")
    public_url = args.world_public_url
    if public_url:
        parsed = urllib.parse.urlsplit(public_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
                or parsed.path != "/live" or parsed.query or parsed.fragment
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError("world-public-url must be an HTTP(S) /live URL with a valid port and no credentials, query or fragment")
    if args.world_view == "off" or not enabled:
        return None

    def create(*, root: Path | None, finished: bool, emit: Callable[[Mapping[str, Any]], None]):
        if finished:
            emit({"op": "world_view", "phase": "replay", "connected": False, "visible": False,
                  "url": None, "message": "Live world view is off for a finished-run transcript replay."})
            return None
        return WorldObserver(emit, root=root, port=args.world_port, browser=args.world_browser,
                             public_url=public_url)
    return create


class WorldObserver:
    """A background observer; UI polling never waits for HTTP or image decoding."""

    def __init__(self, emit: Callable[[Mapping[str, Any]], None], *, root: Path | None = None,
                 port: int = 0, browser: str = "auto", public_url: str | None = None,
                 server_factory=None, open_browser=webbrowser.open_new_tab,
                 health_check=None, environment=None) -> None:
        self.emit = emit
        self.port, self.browser, self.public_url = port, browser, public_url
        self.server_factory, self.open_browser = server_factory, open_browser
        self.health_check = health_check or self._health
        self.environment = dict(os.environ if environment is None else environment)
        self._lock = threading.Lock()
        self._root, self._generation = root, 0
        self._stop = threading.Event()
        self._snapshot: dict[str, Any] = {"op": "world_view", "phase": "starting", "connected": False,
                                          "visible": False, "url": public_url, "message": "Starting world view…"}
        self._last: str | None = None
        self._server = None
        self._browser_message = ""
        self._browser_thread = None
        self._server_close_lock = threading.Lock()
        self._server_close_claimed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="ora-world-view")
        self._thread.start()

    @staticmethod
    def _health(url: str) -> bool:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=0.5) as response:
            value = json.loads(response.read(4096))
            return response.status == 200 and value.get("service") == "ora-live-world" and value.get("status") == "ready"

    def bind(self, root: Path | None) -> None:
        with self._lock:
            self._root = root
            self._generation += 1
            self._snapshot = {"op": "world_view", "phase": "waiting", "connected": False, "visible": False,
                              "url": self.public_url or self._snapshot.get("url"),
                              "run_id": root.name if root else None, "message": "Waiting for the selected run's cameras…"}
        self.poll()

    def poll(self) -> None:
        with self._lock:
            snapshot = dict(self._snapshot)
        encoded = json.dumps(snapshot, sort_keys=True)
        if encoded != self._last:
            self._last = encoded
            self.emit(snapshot)

    def _run(self) -> None:
        server = None
        try:
            if self.server_factory is None:
                from ora.orchestrator.world_view import WorldViewServer  # Flask loads only when a world view starts
                factory = WorldViewServer
            else:
                factory = self.server_factory
            server = factory(host=DEFAULT_WORLD_VIEW["bind_host"], port=self.port).start()
            with self._lock:
                self._server = server
                root, generation = self._root, self._generation
            server.bind(root)
            remote = bool(self.environment.get("SSH_CONNECTION") or self.environment.get("SSH_TTY"))
            if self.browser == "auto" and not remote:
                self._browser_thread = threading.Thread(target=self._open_browser_when_ready,
                                                        args=(server.url, generation), daemon=True,
                                                        name="ora-world-browser")
                self._browser_thread.start()
            elif remote and not self.public_url:
                with self._lock:
                    self._browser_message = "SSH session: forward this port and open the URL on your computer."
            while not self._stop.is_set():
                with self._lock:
                    root, current = self._root, self._generation
                if current != generation:
                    server.bind(root)
                    generation = current
                state = dict(server.status())
                with self._lock:
                    if current == self._generation:
                        self._snapshot = {**state, "op": "world_view", "url": self.public_url or server.url,
                                          "browser_message": self._browser_message}
                self._stop.wait(0.2)
        except Exception as exc:
            with self._lock:
                self._snapshot = {"op": "world_view", "phase": "unavailable", "connected": False,
                                  "visible": False, "url": self.public_url,
                                  "message": f"World view unavailable: {type(exc).__name__}: {exc}"}
        finally:
            if server is not None:
                self._close_server(server)

    def _open_browser_when_ready(self, service_url: str, generation: int) -> None:
        # OS browser launchers can block. Keep them outside camera polling and shutdown.
        ready = False
        for _ in range(4):
            if self._stop.is_set():
                return
            try:
                ready = self.health_check(service_url.rsplit("/", 1)[0] + "/health")
            except (OSError, ValueError):
                pass
            if ready or self._stop.wait(0.1):
                break
        message = ""
        if ready and not self._stop.is_set():
            try:
                opened = self.open_browser(self.public_url or service_url)
            except Exception:
                opened = False
            if not opened:
                message = "Open this URL in your browser."
        elif not ready:
            message = "World-view health check failed; open the URL when ready."
        with self._lock:
            if not self._stop.is_set() and generation == self._generation:
                self._browser_message = message

    def _close_server(self, server) -> None:
        with self._server_close_lock:
            if self._server_close_claimed:
                return
            self._server_close_claimed = True
        server.close()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            server = self._server
        if server is not None:
            self._close_server(server)
        self._thread.join(timeout=1)


def world_explicit(args) -> bool:
    """The operator gave a world-view port or a browser-facing URL (``ora-remote`` passes both).

    A forwarded default port must not turn a quiet JSON consumer into an implicit
    browser or server launch, so only a port the operator gave counts, 0 included.
    """

    return bool(getattr(args, "world_port_given", False)) or args.world_public_url is not None


def watch_observer(args):
    """The world view of ``ora watch`` and of the JSON-lines backend (``python -m ora.app.backend``,
    ``ora interactive --emit jsonl``).

    On in a human terminal or under the pi-tui front end (``ORA_WORLD_UI=1``), unless
    ``--plain``; on with a world port or URL the operator gave; ``--world-view off``
    turns it off. A TTY alone does not opt a JSON consumer into browser side effects.
    """

    human = (sys.stdout.isatty() and args.emit is None) or os.environ.get("ORA_WORLD_UI") == "1"
    return observer_factory(args, enabled=(human and not getattr(args, "plain", False)) or world_explicit(args))


def terminal_observer(args):
    """The world view beside Pi's terminal UI (`ora interactive --ui pi`, `ora attach`, `ora play`).

    On in a human terminal, or when a world port or browser-facing URL is given
    (``ora-remote`` passes both); ``--world-view off`` turns it off.
    """

    args = argparse.Namespace(**{"world_view": DEFAULT_WORLD_VIEW["view"], "world_port": DEFAULT_WORLD_VIEW["port"],
                                 "world_browser": DEFAULT_WORLD_VIEW["browser"], "world_public_url": None,
                                 **vars(args)})
    explicit = args.world_public_url is not None or args.world_port != 0
    return observer_factory(args, enabled=sys.stdout.isatty() or explicit)


#: What the attached terminal shows of the world view (attach.ts relays these to the TUI).
TERMINAL_FIELDS = ("phase", "connected", "visible", "url", "message", "browser_message", "run_id")
WORLD_FD_ENV = "ORA_WORLD_FD"


class TerminalWorldFeed:
    """One run's world view while a Pi terminal is attached to it, its state sent to the attach client.

    The observer reads the run directory, so it outlives agent-process restarts
    and re-attaches. Each attach client (``ora-pi/src/tui/attach.ts``) gets a
    pipe named by ``ORA_WORLD_FD`` carrying one JSON line per change of state.
    """

    def __init__(self, factory, root: Path) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._line: bytes | None = None
        self._fd: int | None = None
        self._stop = threading.Event()
        self.observer = factory(root=root, finished=False, emit=self._update) if factory is not None else None
        self._thread = None
        if self.observer is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True, name="ora-world-terminal")
            self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(0.25):
            self.observer.poll()

    def _update(self, snapshot: Mapping[str, Any]) -> None:
        state = {key: snapshot.get(key) for key in TERMINAL_FIELDS}
        frame = snapshot.get("frame")
        state["frame_index"] = frame.get("index") if isinstance(frame, Mapping) else None
        line = (json.dumps(state, sort_keys=True) + "\n").encode()
        with self._lock:
            if line == self._line:
                return  # the observer's snapshot also carries frame ages, which tick
            self._state, self._line = state, line
            self._send(line)

    def _send(self, line: bytes) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, line)
        except BlockingIOError:
            pass  # a busy client: the next change carries the whole state again
        except OSError:
            os.close(self._fd)
            self._fd = None

    def url(self, wait_s: float = 1.5) -> str | None:
        """The browser-facing URL once the server is up (None when off or not up in time)."""

        if self.observer is None:
            return None
        deadline = time.monotonic() + wait_s
        while True:
            self.observer.poll()
            with self._lock:
                state = dict(self._state or {})
            if state.get("url") and state.get("phase") != "starting" or time.monotonic() >= deadline:
                return state.get("url")
            time.sleep(0.05)

    def call(self, argv: list[str]) -> int:
        """Run one attach client with this feed's pipe (``subprocess.call`` without a world view)."""

        if self.observer is None:
            return subprocess.call(argv)
        read, write = os.pipe()
        os.set_blocking(write, False)
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
            self._fd = write
            if self._line is not None:
                self._send(self._line)
        try:
            with subprocess.Popen(argv, pass_fds=(read,), env={**os.environ, WORLD_FD_ENV: str(read)}) as child:
                os.close(read)
                read = -1
                try:
                    return child.wait()
                except BaseException:
                    child.kill()
                    raise
        finally:
            if read >= 0:
                os.close(read)
            with self._lock:
                if self._fd == write:
                    os.close(write)
                    self._fd = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.observer is not None:
            self.observer.close()
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
