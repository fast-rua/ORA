"""Drive `ora-remote` from this host against an SSH host, as from a laptop, and keep the evidence.

    python -m tests.support.agent_core.remote_world_demo OUT --host HOST --python REMOTE_PYTHON
        [--node-dir DIR] [--actions N]

This host plays the laptop: a tmux pane runs `scripts/ora-remote HOST
--no-browser interactive --ui pi --provider env`, and a headless Chromium
(if Python Playwright is importable here) opens the forwarded world view.
HOST plays the GPU server: the Synthetic run, its Supervisor, the Pi agent
core and the scripted loopback model (fake_model_server.py) run there. OUT
must be a directory both hosts see (shared filesystem): the remote checkout
shim, the model's port file and the run workspace live under it.

The checks follow one run through the whole flow:
  launch    ora-remote opens the forward and the remote Pi TUI; the header
            shows the world view's forwarded URL; the page serves frames
  browser   Chromium decodes a frame; the TUI says the browser shows it
  detach    Ctrl+D: the run continues on HOST
  reattach  the same command attaches to that run (kitty images forced, the
            pane's raw bytes logged: image sequences pass through SSH)
  drop      the ssh client is killed: the run continues; the same command
            attaches again
  stop      /stop ends the run, which lands and is sealed
Frames are pane captures (text and ANSI) under OUT/frames; screenshots are
OUT/terminal-*.png (rendered from the ANSI capture, with Rich) and
OUT/world-page.png; checks and the outcome go to OUT/summary.json. The
remote checkout must be committed (preflight admits a clean tree only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from tests.support.agent_core.tui_attach_demo import COLUMNS, LINES, SUMMARY, Pane, count, outcome, wait

REPO = Path(__file__).resolve().parents[3]
WORLD_WINDOW = re.compile(r"World window: (http://127\.0\.0\.1:(\d+)/live)")
_local = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def ssh(host: str, command: str, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True, **kwargs)


def get_json(url: str) -> dict[str, Any]:
    with _local(url, timeout=5) as response:
        return json.loads(response.read())


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b_[^\x1b]*\x1b\\")


def printed(out: Path, since: int = 0) -> str:
    """Everything the pane received since byte `since` (pipe-pane), without escape sequences.

    The TUI clears the screen and its scrollback when it draws, so lines printed
    before it (ora-remote's world window, the re-attach notice) are read here.
    """
    data = (out / "pane.raw.log").read_bytes()[since:].decode("utf-8", "replace")
    return ANSI.sub("", data).replace("\r", "")


def history(pane: Pane) -> str:
    """The pane with its scrollback: the TUI's header scrolls away as the transcript grows."""
    return pane.tmux("capture-pane", "-p", "-J", "-S", "-", "-t", "main:0")


def status_world_row(pane: Pane, out: Path, name: str) -> str | None:
    """Open /status, keep the frame, close it; the panel's `world view` row."""
    pane.type("/status")
    wait(lambda: "ORA run status" in pane.text(), 20, "status panel")
    text = capture(pane, out, name)
    pane.keys("Enter")
    time.sleep(0.5)
    return next((line.strip() for line in text.splitlines() if line.strip().startswith("world view")), None)


def fresh(pane: Pane) -> None:
    """A cleared pane and scrollback, so the next waits see only what the next command prints."""
    time.sleep(1)
    pane.type("clear")
    time.sleep(0.5)
    pane.tmux("clear-history", "-t", "main:0")


def run_root(workspace: Path) -> Path | None:
    roots = sorted((workspace / "outputs" / "ora-interactive" / "runs").glob("*"))
    return roots[-1] if roots else None


def capture(pane: Pane, out: Path, name: str) -> str:
    """The pane as text (Pane.frame) and with its colours (ANSI), for the screenshots."""
    text = pane.frame(name)
    ansi = pane.tmux("capture-pane", "-p", "-e", "-t", "main:0")
    (pane.frames / f"{pane.count:02d}-{name}.ansi").write_text(ansi, encoding="utf-8")
    return text


def render_terminal(ansi_path: Path, png: Path, browser: Any) -> bool:
    """An ANSI pane capture as a PNG: Rich draws it as SVG, Chromium rasterises it."""
    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        return False
    console = Console(record=True, width=COLUMNS, file=open(os.devnull, "w"), force_terminal=True,
                      color_system="truecolor")
    console.print(Text.from_ansi(ansi_path.read_text(encoding="utf-8")), soft_wrap=True)
    svg = png.with_suffix(".svg")
    svg.write_text(console.export_svg(title="ora-remote (laptop terminal)"), encoding="utf-8")
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(svg.as_uri())
    page.locator("svg").screenshot(path=str(png))
    page.close()
    return True


def remote_shim(out: Path, python: str, node_dir: str | None) -> Path:
    """The "remote checkout": its scripts/ora runs this checkout's with the demo's environment."""
    shim = out / "remote-checkout"
    (shim / "scripts").mkdir(parents=True, exist_ok=True)
    path = f"{node_dir}:" if node_dir else ""
    (shim / "scripts" / "ora").write_text(
        "#!/bin/sh\n"
        "# Demo shim: the scripted model on this host's loopback, the run workspace under OUT.\n"
        f"ORA_BASE_URL=http://127.0.0.1:$(cat {shlex.quote(str(out / 'model.port'))})/v1\n"
        f"export ORA_BASE_URL ORA_API_KEY=local-demo ORA_MODEL=fake-demo PYTHON={shlex.quote(python)}\n"
        f"export ORA_WORKSPACE_ROOT={shlex.quote(str(out / 'workspace'))} PATH={shlex.quote(path)}\"$PATH\"\n"
        f"exec {shlex.quote(str(REPO / 'scripts' / 'ora'))} \"$@\"\n", encoding="utf-8")
    (shim / "scripts" / "ora").chmod(0o755)
    return shim


def scenario(args: argparse.Namespace, out: Path, checks: dict[str, Any]) -> dict[str, Any]:
    host = args.host
    (out / "model.port").unlink(missing_ok=True)
    # `cd; nohup ... &`: only the server goes to the background, with no fd of this ssh session.
    started = ssh(host, f"cd {shlex.quote(str(REPO))} || exit 1; nohup {shlex.quote(args.python)} -m "
                        f"tests.support.agent_core.fake_model_server {shlex.quote(str(out / 'model.port'))} "
                        f"--delay {args.delay} < /dev/null > {shlex.quote(str(out / 'model.log'))} 2>&1 & echo $!")
    model_pid = int(started.stdout.strip())
    wait((out / "model.port").exists, 60, "remote model server")
    shim = remote_shim(out, args.python, args.node_dir)
    workspace = out / "workspace"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ORA_", "PI_")) and "API_KEY" not in key and key != "TMUX"}
    env.update(TERM="xterm-256color", PYTHON=sys.executable)
    pane = Pane("ora-remote-demo", out, env)
    pane.tmux("pipe-pane", "-o", "-t", "main:0", f"cat >> {shlex.quote(str(out / 'pane.raw.log'))}")
    command = (f"{REPO / 'scripts' / 'ora-remote'} {host} --repo {shim} --no-browser "
               f"interactive --ui pi --provider env --actions {args.actions}")
    browser = playwright = None
    try:
        try:
            from playwright.sync_api import sync_playwright
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch()
        except Exception as exc:  # no Playwright here: the page is still checked over HTTP
            checks["browser"] = f"skipped: {type(exc).__name__}: {exc}"

        # -- launch: the forward, the remote TUI and its world view line --------------------------------
        pane.type(f"cd {out} && {command}")
        url = wait(lambda: WORLD_WINDOW.search(pane.text()), 60, "ora-remote's world window line").group(1)
        root = wait(lambda: run_root(workspace), 180, "run directory on the remote host")
        wait(lambda: "observation 0" in pane.text(), 240, "the remote Pi TUI on this terminal")
        wait(lambda: f"world view {url}" in pane.text(), 60, "the TUI header shows the forwarded world view")
        state = wait(lambda: (lambda s: s if s.get("frame") else None)(get_json(url.replace("/live", "/api/live/status"))),
                     120, "a camera frame through the tunnel")
        checks["launch"] = {"url": url, "run": root.name, "frame_index": state["frame"]["index"],
                            "cameras": [camera["name"] for camera in state["frame"]["cameras"]],
                            "health": get_json(url.replace("/live", "/health"))}
        capture(pane, out, "attached-world-view")

        # -- browser: a decoded frame, acknowledged, shown in the TUI ------------------------------------
        if browser is not None:
            page = browser.new_page(viewport={"width": 1100, "height": 760})
            page.goto(url)
            # The page's CSP forbids eval, so wait with a locator (Playwright's own isolated world).
            page.locator("#world-connection", has_text="Original camera frame decoded").wait_for(timeout=60_000)
            wait(lambda: "browser shows frame" in pane.text(), 30, "the TUI reports the browser's frame")
            time.sleep(1.5)
            page.screenshot(path=str(out / "world-page.png"), full_page=True)
            text = capture(pane, out, "browser-shows-frame")
            checks["browser"] = {"page": page.inner_text("#world-connection"),
                                 "frame_line": page.inner_text("#world-frame"),
                                 "tui": next(line.strip() for line in text.splitlines() if "world view" in line)}
            page.close()

        # -- detach: the run goes on -------------------------------------------------------------------
        before = count(root, "provider_request")
        pane.keys("C-d")
        wait(lambda: "re-attach: run the same ora-remote command again" in pane.text(), 30, "detach")
        capture(pane, out, "detached")
        wait(lambda: count(root, "provider_request") > before, 120, "the run continues while detached")
        checks["detach"] = {"calls_at_detach": before, "calls_later": count(root, "provider_request")}

        # -- reattach with the same command; kitty images forced to show the bytes pass through ---------
        fresh(pane)
        raw_before = (out / "pane.raw.log").stat().st_size
        pane.type(f"PI_IMAGE_PROTOCOL=kitty {command}")
        wait(lambda: "is still going; attaching to it" in printed(out, raw_before), 60, "the same command attaches")
        wait(lambda: "observation" in pane.text(), 120, "re-attached TUI")
        time.sleep(3)
        capture(pane, out, "reattached-same-command")
        url2 = WORLD_WINDOW.search(printed(out, raw_before)).group(1)
        row = status_world_row(pane, out, "reattached-status-world-view")
        raw = (out / "pane.raw.log").read_bytes()[raw_before:]
        checks["reattach"] = {"same_run": run_root(workspace) == root, "kitty_image_sequences": raw.count(b"\x1b_G"),
                              "world_url": url2, "status_row": row,
                              "world_health": get_json(url2.replace("/live", "/health"))}
        pane.keys("C-d")
        wait(lambda: "re-attach: run the same ora-remote command again" in printed(out, raw_before), 30,
             "second detach")

        # -- drop: kill the ssh client (the connection is gone, as when a laptop sleeps) ----------------
        fresh(pane)
        since = (out / "pane.raw.log").stat().st_size
        pane.type(command)
        wait(lambda: "is still going; attaching to it" in printed(out, since), 60, "attached before the drop")
        wait(lambda: "observation" in pane.text(), 120, "TUI before the drop")
        port = WORLD_WINDOW.search(printed(out, since)).group(2)
        clients = subprocess.run(["pgrep", "-f", f"ssh -tt .*-L 127.0.0.1:{port}:"], capture_output=True, text=True)
        pids = [int(pid) for pid in clients.stdout.split()]
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        wait(lambda: "attaches to it again" in printed(out, since), 30, "ora-remote reports the dropped connection")
        capture(pane, out, "connection-dropped")
        before = count(root, "provider_request")
        process = json.loads((root / "process.json").read_text())
        wait(lambda: count(root, "provider_request") > before, 120, "the run continues after the drop")
        alive = ssh(host, f"kill -0 {int(process['pid'])} && echo alive").stdout.strip()
        checks["drop"] = {"killed_ssh_clients": pids, "supervisor_on_host": alive,
                          "calls_after_drop": count(root, "provider_request")}
        fresh(pane)
        since = (out / "pane.raw.log").stat().st_size
        pane.type(command)
        wait(lambda: "is still going; attaching to it" in printed(out, since) and "observation" in pane.text(),
             120, "the same command attaches after the drop")
        time.sleep(2)
        capture(pane, out, "reattached-after-drop")
        checks["reattach_after_drop"] = {"same_run": run_root(workspace) == root,
                                         "status_row": status_world_row(pane, out, "after-drop-status-world-view")}

        # -- stop: the run lands and is sealed ---------------------------------------------------------
        pane.type("/stop")
        wait(lambda: "Stop this run?" in pane.text(), 20, "stop confirmation")
        pane.keys("Enter")
        wait(lambda: SUMMARY.search(pane.text()), 300, "landing summary")
        time.sleep(1)
        capture(pane, out, "stopped-summary")
        if browser is not None:
            for path in sorted(pane.frames.glob("*.ansi")):
                render_terminal(path, out / f"terminal-{path.stem}.png", browser)
        return outcome(root)
    finally:
        try:
            capture(pane, out, "final")
        finally:
            pane.close()
            if browser is not None:
                browser.close()
            if playwright is not None:
                playwright.stop()
            ssh(host, f"kill {model_pid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--host", required=True, help="SSH host that plays the GPU server")
    parser.add_argument("--python", required=True, help="the project interpreter on that host")
    parser.add_argument("--node-dir", help="directory of Node.js 22 on that host, put first on its PATH")
    parser.add_argument("--actions", type=int, default=48)
    parser.add_argument("--delay", type=float, default=5.0, help="seconds the scripted model holds each reply")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    result: Any = None
    try:
        result = scenario(args, out, checks)
    except Exception as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "summary.json").write_text(json.dumps({"checks": checks, "outcome": result}, indent=2,
                                                     sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"checks": checks, "outcome": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
