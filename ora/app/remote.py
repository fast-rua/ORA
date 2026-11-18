"""Use ORA on an SSH host from this computer: its terminal UI here, its cameras in a local browser.

    ora-remote HOST [--repo PATH] play --env synthetic|libero|robotwin|robodojo [...]
    ora-remote HOST [--repo PATH] interactive --ui pi [...]
    ora-remote HOST [--repo PATH] attach [RUN]
    ora-remote HOST [--repo PATH] watch RUN
    ora-remote HOST [--repo PATH] web [RUN]           the run in the browser (ORA's page on its Pi RPC)
    ora-remote HOST [--repo PATH] --web play|interactive|attach ...   the terminal UI and the web view
    (the older form, with `--` before the ORA command, is accepted too)

One SSH connection carries both: `ssh -tt` runs the ORA command on the host,
whose Pi terminal UI draws on this terminal (image protocol included), and
`-L` forwards one local loopback port to the world view the command serves on
the host's loopback. The browser opens once that page answers. Nothing listens
beyond loopback on either side and nothing else is forwarded; there is no web
terminal.

With `web` or `--web`, the forwarded port reaches `ora web` instead: its
gateway listens on a 0600 unix socket in the host user's runtime directory
(`ssh -L` to a socket path), and the browser opens with a launch token this
launcher generated and handed over in a 0600 file on a first, short SSH
command (one connection: both share an SSH control master), never on a
command line.

Detaching (Ctrl+D) or losing the connection leaves the run going on the host
under its detached Supervisor. Running the same `play` or `interactive`
command again attaches to that run while it is going (`--new` starts another);
once it has finished, the same command starts a new one. This launcher never
launches a task itself, retries an SSH session, or adds launch confirmation to
the supplied ORA command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import PurePosixPath
from typing import Mapping
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

_local_http = build_opener(ProxyHandler({})).open

#: ORA commands this launcher runs on the host; `play` and `interactive` resume their run.
COMMANDS = ("play", "interactive", "attach", "watch", "web")
RESUMABLE = ("play", "interactive")
_VALUE_OPTIONS = {"--repo", "--local-port", "--remote-port", "--connect-timeout"}


def available_port(requested: int = 0) -> int:
    if isinstance(requested, bool) or not 0 <= requested <= 65535:
        raise ValueError("local port must be between 0 and 65535")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", requested))
        return listener.getsockname()[1]


def session_key(host: str, repo: str | None, command: list[str]) -> str:
    """The name the host records this command's run under: the same command finds it again."""

    digest = hashlib.sha256(json.dumps([host, repo or "", command]).encode()).hexdigest()[:16]
    return f"{command[0]}-{digest}"


def terminal_overrides(environ: Mapping[str, str]) -> dict[str, str]:
    """pi-tui overrides that carry this terminal's image protocol, colour and links to the host.

    SSH forwards TERM but not TERM_PROGRAM, KITTY_WINDOW_ID and the other
    variables pi-tui recognises a terminal by, so the remote TUI would fall back
    to text. This mirrors pi-tui's detection here and passes the result as its
    explicit PI_IMAGE_PROTOCOL / PI_TRUE_COLOR / PI_HYPERLINKS settings (any of
    them already set here wins). The escape sequences then pass through SSH's
    terminal unchanged.
    """

    program = environ.get("TERM_PROGRAM", "").lower()
    term = environ.get("TERM", "").lower()
    detected: dict[str, str] = {}
    if environ.get("TMUX") or term.startswith(("tmux", "screen")):
        pass  # pi-tui leaves images off under a multiplexer; the forwarded TERM says so on the host too
    elif (program in {"herdr", "kitty", "ghostty", "wezterm", "warpterminal"} or "ghostty" in term
          or any(environ.get(name) for name in ("KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE",
                                                 "WARP_SESSION_ID", "WARP_TERMINAL_SESSION_UUID"))):
        detected = {"PI_IMAGE_PROTOCOL": "kitty", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}
    elif environ.get("ITERM_SESSION_ID") or program == "iterm.app":
        detected = {"PI_IMAGE_PROTOCOL": "iterm2", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}
    elif environ.get("WT_SESSION") or program in {"alacritty", "vscode", "zed"}:
        detected = {"PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}
    elif environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        detected = {"PI_TRUE_COLOR": "1"}
    explicit = {name: environ[name] for name in ("PI_IMAGE_PROTOCOL", "PI_TRUE_COLOR", "PI_HYPERLINKS")
                if re.fullmatch(r"[A-Za-z0-9]{1,16}", environ.get(name, ""))}
    return {**detected, **explicit, **ui_language(environ)}


def ui_language(environ: Mapping[str, str]) -> dict[str, str]:
    """ORA_UI_LANG for the host: ORA's terminal words in this terminal's language (zh or en).

    An explicit ORA_UI_LANG here wins; else a Chinese locale here (LC_ALL,
    LC_MESSAGES, LANG) asks for Chinese. SSH does not carry the locale reliably.
    """

    chosen = environ.get("ORA_UI_LANG", "").strip().lower()[:2]
    if chosen in {"zh", "en"}:
        return {"ORA_UI_LANG": chosen}
    locale = environ.get("LC_ALL") or environ.get("LC_MESSAGES") or environ.get("LANG") or ""
    return {"ORA_UI_LANG": "zh"} if locale.lower().startswith("zh") else {}


def ssh_command(host: str, repo: str | None, command: list[str], *, local_port: int,
                remote_port: int, timeout: float = 30, session: str | None = None, fresh: bool = False,
                environment: Mapping[str, str] | None = None, web: Mapping[str, str] | None = None,
                options: list[str] | None = None) -> tuple[list[str], str]:
    """The SSH invocation and the local URL (the world view, or with `web` the ORA web gateway).

    With `repo`, the host runs that checkout's ``./scripts/ora``; without it,
    the ``ora`` its login shell finds on PATH. `web` = {socket, token_file} on
    the host: the forward goes to the gateway's socket.
    """

    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host):
        raise ValueError("host must be an SSH hostname or configured alias, optionally user@host")
    if repo is not None and (not PurePosixPath(repo).is_absolute() or any(c in repo for c in "\x00\r\n")):
        raise ValueError("--repo must be an absolute path on the SSH host")
    if not command or command[0] not in COMMANDS:
        raise ValueError(f"name the ORA command to run on the host: {', '.join(COMMANDS)}")
    if any(arg.split("=", 1)[0].startswith("--world-") for arg in command):
        raise ValueError("the remote launcher owns --world-* options; use its port options instead")
    if any("\x00" in arg for arg in command):
        raise ValueError("command arguments must not contain NUL")
    if not all(isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= 65535
               for p in (local_port, remote_port)):
        raise ValueError("forwarded ports must be between 1 and 65535")
    if command[0] == "web" and web is None:
        raise ValueError("`web` needs the web forward (ora-remote prepares it)")
    url = f"http://127.0.0.1:{local_port}/" if web is not None else f"http://127.0.0.1:{local_port}/live"
    assignments = dict(environment or {})
    if session is not None and command[0] in RESUMABLE:
        assignments["ORA_REMOTE_SESSION"] = session
        if fresh:
            assignments["ORA_REMOTE_FRESH"] = "1"
    if any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) or "\x00" in value for name, value in assignments.items()):
        raise ValueError("remote environment names must be upper-case identifiers")
    if web is None:
        ora_args = [*command, "--world-view", "auto", "--world-port", str(remote_port),
                    "--world-browser", "off", "--world-public-url", url]
        forward = f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"
    elif command[0] == "web":
        ora_args = [*command, "--socket", web["socket"], "--token-file", web["token_file"], "--no-browser"]
        forward = f"127.0.0.1:{local_port}:{web['socket']}"
    else:
        # The cameras are in the web view's dock; the terminal UI serves no world view of its own.
        assignments.update(ORA_WEB_SOCKET=web["socket"], ORA_WEB_TOKEN_FILE=web["token_file"])
        ora_args = [*command, "--world-view", "off"]
        forward = f"127.0.0.1:{local_port}:{web['socket']}"
    prefix = ["env", *(f"{name}={value}" for name, value in sorted(assignments.items()))] if assignments else []
    if repo is not None:
        remote = f"cd -- {shlex.quote(repo)} && exec {shlex.join([*prefix, './scripts/ora', *ora_args])}"
    else:
        inner = f"exec {shlex.join([*prefix, 'ora', *ora_args])}"
        remote = f'exec "${{SHELL:-/bin/sh}}" -lc {shlex.quote(inner)}'
    return ["ssh", "-tt", *(options or []), "-o", "ExitOnForwardFailure=yes",
            "-o", f"ConnectTimeout={max(1, int(timeout))}", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-L", forward, "--", host, remote], url


#: Run by the first, short SSH command of a web session: a private directory, the token from stdin.
_PREPARE = ('umask 077; d="${XDG_RUNTIME_DIR:-$HOME/.cache}/ora-remote"; mkdir -p "$d" && chmod 700 "$d" '
            # Sockets and tokens a lost session left behind (12 h old) go first.
            '&& { find "$d" -maxdepth 1 \\( -name "*.sock" -o -name "*.token" \\) -mmin +720 -delete 2>/dev/null; true; } '
            '&& cat > "$d/$1.token" && printf "%s\\n" "$d"')


def prepare_web(host: str, name: str, token: str, options: list[str], *, timeout: float,
                run=subprocess.run) -> dict[str, str]:
    """Hand the launch token to the host in a 0600 file; the paths the web session uses there."""

    if not re.fullmatch(r"[a-z0-9-]{1,80}", name):
        raise ValueError("web session names are lowercase letters, digits and dashes")
    script = f"exec sh -c {shlex.quote(_PREPARE)} ora-remote {name}"
    done = run(["ssh", *options, "-o", f"ConnectTimeout={max(1, int(timeout))}", "--", host, script],
               input=token + "\n", capture_output=True, text=True, timeout=timeout + 30)
    directory = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    if done.returncode or not PurePosixPath(directory).is_absolute() or any(c in directory for c in "\x00\r\n"):
        raise OSError(f"could not prepare the web session on {host}: {done.stderr.strip() or 'no directory'}")
    return {"socket": f"{directory}/{name}.sock", "token_file": f"{directory}/{name}.token"}


def open_when_ready(url: str, stopped: threading.Event, *, timeout: float | None = None,
                    opener=webbrowser.open_new_tab, fetch=_local_http, report=None,
                    refused: threading.Event | None = None, browse: str | None = None) -> bool:
    """Confirm the forwarded observer, then request one local browser tab.

    Polls until the page answers, `stopped` is set (the SSH session ended) or
    `timeout` passes (None: no deadline; a real launch can preflight for
    minutes before its terminal attaches). `refused` is set when the host
    refuses the forward itself; that is reported at once, before the remote
    TUI owns the terminal. HTTP readiness is not proof of a displayed frame:
    the page's decode ACK determines that separate state, which ORA displays
    in the terminal.
    """
    report = report or (lambda message: print(f"ora-remote: {message}", file=sys.stderr, flush=True))
    web = not url.endswith("/live")
    health = url.rstrip("/") + "/ora/health" if web else url.removesuffix("/live") + "/health"
    service = "ora-web" if web else "ora-live-world"
    deadline = None if timeout is None else time.monotonic() + timeout
    while not stopped.is_set() and (deadline is None or time.monotonic() < deadline):
        if refused is not None and refused.is_set():
            report(FORWARD_REFUSED)
            return False
        try:
            with fetch(health, timeout=0.5) as response:
                raw = response.read(4097)
            if len(raw) > 4096:
                raise ValueError("unexpected health response")
            value = json.loads(raw)
            if value.get("service") != service or value.get("status") != "ready":
                raise ValueError("forwarded service is not ORA's")
            if stopped.is_set():
                return False
            if opener is None:
                return True
            try:
                opened = opener(browse or url)
            except (OSError, webbrowser.Error) as exc:
                report(f"Browser did not open ({exc}); open {browse or url} locally.")
                return False
            if not opened:
                report(f"Browser did not open; open {browse or url} locally.")
                return False
            return True
        except (OSError, URLError, ValueError, AttributeError):
            stopped.wait(0.25)
    if not stopped.is_set():
        report(f"World window is not reachable yet at {url}; check SSH forwarding and the remote CLI. "
               "The SSH session will not be restarted.")
    return False


#: What a refused forward means, said once and before the remote TUI draws.
FORWARD_REFUSED = ("the host refused the SSH port forward (administratively prohibited): its sshd disables "
                   "forwarding (AllowTcpForwarding or DisableForwarding), so the world view cannot reach this "
                   "computer over SSH. The terminal UI works; for the cameras, run ORA on a host that permits "
                   "forwarding or ask its administrator. ora-remote does not work around it.")
_CHANNEL_OPEN_FAILED = re.compile(r"^channel \d+: open failed: (.*)$")


def relay_ssh_diagnostics(stream, refused: threading.Event, write=None) -> None:
    """ssh's own stderr, minus the forward's per-connection failures.

    Each probe of the forwarded port before the host's world view listens is a
    `channel N: open failed: connect failed` line from ssh; drawn over the
    remote TUI they would wreck it, so they are dropped. A forward the host
    refuses (`administratively prohibited`) sets `refused` instead. Everything
    else ssh says is passed through.
    """
    write = write or (lambda text: (sys.stderr.write(text), sys.stderr.flush()))
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", "replace")
        failed = _CHANNEL_OPEN_FAILED.match(line.strip())
        if failed:
            if "administratively prohibited" in failed.group(1):
                refused.set()
            continue
        write("\r" + line)  # the remote output before it may have left the cursor mid-line


#: What the remote Pi TUI turns on in this terminal (pi-tui's ProcessTerminal) and would turn off when
#: it leaves: bracketed paste, the kitty keyboard protocol, modifyOtherKeys, a hidden cursor, a
#: progress indicator. A dropped or killed ssh never delivers that, so ora-remote does; the cursor
#: goes below the TUI first.
TERMINAL_RESET = "\x1b[999B\r\n\x1b[?2004l\x1b[<u\x1b[>4;0m\x1b[?25h\x1b]9;4;0\x07"


def saved_terminal():
    """This terminal's tty modes, to put back if ssh cannot (None without a tty or termios)."""

    try:
        import termios

        return termios.tcgetattr(sys.stdin.fileno()) if sys.stdin.isatty() else None
    except (ImportError, OSError, ValueError):
        return None


def restore_terminal(saved, *, abnormal: bool) -> None:
    """ssh leaves raw mode itself when it ends normally; after a drop or a kill, ora-remote does."""

    if saved is not None:
        try:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
        except (ImportError, OSError, ValueError):
            pass
    if abnormal and sys.stdout.isatty():
        sys.stdout.write(TERMINAL_RESET)
        sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ora-remote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="ora-remote HOST [options] {play,interactive,attach,watch} [ORA arguments ...]",
        epilog="Example: ora-remote gpu-host --repo /path/to/ORA play --env synthetic --provider env")
    parser.add_argument("host", help="SSH hostname or configured alias (normal SSH authentication/config)")
    parser.add_argument("--repo", default=os.environ.get("ORA_REMOTE_REPO") or None,
                        help="absolute checkout on the host whose ./scripts/ora runs (default: $ORA_REMOTE_REPO, "
                             "else `ora` on the host's login PATH)")
    parser.add_argument("--local-port", type=int, default=0, help="local loopback port (default: choose a free port)")
    parser.add_argument("--remote-port", type=int, help="remote loopback port (default: use chosen local port)")
    parser.add_argument("--connect-timeout", type=float, default=30,
                        help="seconds SSH waits to connect (default 30); the browser waits for the world view "
                             "as long as the session lasts")
    parser.add_argument("--no-browser", action="store_true", help="print the world-view URL instead of opening it")
    parser.add_argument("--new", action="store_true",
                        help="start a new run even if this command's previous run is still going on the host")
    parser.add_argument("--web", action="store_true",
                        help="also serve the run in the browser (ORA's page: the conversation and the cameras); "
                             "the host needs `ora web setup`")
    return parser


def split_arguments(values: list[str]) -> tuple[list[str], list[str]]:
    """Launcher options, and the ORA command from its first word (or after `--`)."""

    index = 0
    while index < len(values):
        value = values[index]
        if value == "--":
            return values[:index], values[index + 1:]
        if value in COMMANDS and index > 0:
            return values[:index], values[index:]
        index += 2 if value in _VALUE_OPTIONS else 1
    return values, []


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    own, command = split_arguments(values)
    args = parser.parse_args(own)
    if not command:
        parser.error(f"name the ORA command to run on {args.host}: {', '.join(COMMANDS)}")
    try:
        if not 0 < args.connect_timeout <= 300:
            raise ValueError("connect timeout must be finite and in (0, 300] seconds")
        local_port = available_port(args.local_port)
        key = session_key(args.host, args.repo, command)
        web = browse = None
        options: list[str] = []
        if args.web or command[0] == "web":
            if command[0] == "watch":
                raise ValueError("--web follows a live Pi-core run: play, interactive, attach or web")
            # One SSH connection for both commands (a single authentication), mux socket private here.
            control = tempfile.mkdtemp(prefix="ora-remote-")
            options = ["-o", "ControlMaster=auto", "-o", f"ControlPath={control}/cm", "-o", "ControlPersist=10"]
            token = secrets.token_urlsafe(32)
            web = prepare_web(args.host, f"{key}-{secrets.token_hex(4)}", token, options,
                              timeout=args.connect_timeout)
        ssh, url = ssh_command(args.host, args.repo, command, local_port=local_port,
                               remote_port=local_port if args.remote_port is None else args.remote_port,
                               timeout=args.connect_timeout, session=key, fresh=args.new,
                               environment=terminal_overrides(os.environ), web=web, options=options)
        if web is not None:
            browse = f"{url}?ora_token={token}"
            print(f"ORA web: {url} ({'open ' + browse + ' here' if args.no_browser else 'opens'} "
                  "once the host's web view answers; the link works once)", flush=True)
        else:
            print(f"World window: {url} ({'open it in a browser here' if args.no_browser else 'opens'} "
                  "once the host's world view answers)", flush=True)
        saved = saved_terminal()
        child = subprocess.Popen(ssh, stderr=subprocess.PIPE)
        stopped, refused = threading.Event(), threading.Event()
        threading.Thread(target=relay_ssh_diagnostics, args=(child.stderr, refused), daemon=True,
                         name="ora-ssh-stderr").start()
        browser = threading.Thread(target=open_when_ready, args=(url, stopped),
                                   kwargs={"refused": refused, "browse": browse,
                                           **({"opener": None} if args.no_browser else {})},
                                   daemon=True, name="ora-local-browser")
        browser.start()
        try:
            code = child.wait()
        except KeyboardInterrupt:
            # Terminate only this SSH UI; never signal the remote Supervisor.
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            restore_terminal(saved, abnormal=True)
            return 130
        finally:
            stopped.set()
            browser.join(timeout=0.7)
        restore_terminal(saved, abnormal=code == 255 or code < 0)
        if code == 255 or code < 0:  # ssh's own failure, or the ssh client was killed
            again = shlex.join(["ora-remote", *values])
            print(f"ora-remote: SSH to {args.host} ended (connection lost or refused). A run already started "
                  f"there keeps going under its Supervisor; `{again}` attaches to it again.", file=sys.stderr)
        return code
    except (OSError, ValueError) as exc:
        print(f"ora-remote: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
