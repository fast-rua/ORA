"""Drive `ora interactive` / `ora attach` in tmux and keep the screens as evidence.

    python -m tests.support.agent_core.tui_attach_demo OUT [--scenario main|esc|all]

Each scenario runs the real CLI path (resolve, preflight, lease, detached
Supervisor, Pi agent core with ORA_PI_TUI=1) against the scripted loopback model
in fake_model_server.py, inside a tmux pane on this host. Frames are plain-text
pane captures under OUT/<scenario>/frames; checks and run outcomes go to
OUT/<scenario>/summary.json.

  main  provider picker, launch and attach, /help, /status, /model, /login, a
        refused /tree, an unknown command, a steer delivered to the model (and
        shown as "› you: …" once delivered), Ctrl+D detach
        while the run goes on, `ora sessions`, `ora attach` (full redraw),
        /export, SIGKILL of the agent process (Supervisor restart, automatic
        re-attach), /stop during a model call, a second terminal's `ora attach`
        while the run lands (it says so, then opens the landed run read-only)
  esc   Esc during a model call, with the kitty image protocol forced for the
        attach client (its raw output is logged to show the image sequences)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
COLUMNS, LINES = 130, 42


class Pane:
    def __init__(self, socket: str, out: Path, env: dict[str, str]) -> None:
        self.socket = socket
        self.frames = out / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.tmux("kill-server", check=False)
        subprocess.run(["tmux", "-L", socket, "-f", "/dev/null", "new-session", "-d", "-s", "main",
                        "-x", str(COLUMNS), "-y", str(LINES)], env=env, check=True)
        self.tmux("set", "-g", "history-limit", "20000")

    def tmux(self, *args: str, check: bool = True) -> str:
        return subprocess.run(["tmux", "-L", self.socket, *args], capture_output=True, text=True,
                              check=check).stdout

    def window(self, name: str) -> None:
        """A second terminal on this host (tmux window `name`), as a second operator would open."""
        self.tmux("new-window", "-d", "-t", "main", "-n", name)

    def text(self, window: str = "0") -> str:
        return self.tmux("capture-pane", "-p", "-J", "-t", f"main:{window}")

    def keys(self, *keys: str, window: str = "0") -> None:
        self.tmux("send-keys", "-t", f"main:{window}", *keys)

    def type(self, text: str, enter: bool = True, window: str = "0") -> None:
        self.tmux("send-keys", "-t", f"main:{window}", "-l", text)
        if enter:
            time.sleep(0.3)
            self.keys("Enter", window=window)

    def frame(self, name: str, window: str = "0") -> str:
        self.count += 1
        text = "\n".join(line.rstrip() for line in self.text(window).splitlines()).rstrip() + "\n"
        (self.frames / f"{self.count:02d}-{name}.txt").write_text(text, encoding="utf-8")
        return text

    def close(self) -> None:
        self.tmux("kill-server", check=False)


def wait(predicate: Callable[[], Any], timeout: float, what: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.25)
    raise TimeoutError(what)


def events(root: Path) -> list[dict[str, Any]]:
    path = root / "agent-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count(root: Path, kind: str) -> int:
    return sum(event["type"] == kind for event in events(root))


#: The outcome line `ora interactive`/`ora attach` print once the run has landed.
SUMMARY = re.compile(r": state \w+ · \d+ decisions · ")


def session_closed(root: Path) -> bool:
    return any(event["type"] == "session_closed" for event in events(root))


def run_root(workspace: Path) -> Path | None:
    roots = sorted((workspace / "outputs" / "ora-interactive" / "runs").glob("*"))
    return roots[-1] if roots else None


def outcome(root: Path) -> dict[str, Any]:
    load = lambda name: json.loads((root / name).read_text()) if (root / name).is_file() else {}
    result, evaluation = load("result.json"), load("evaluation.json")
    return {
        "run": root.name, "state": result.get("state"), "termination": result.get("termination"),
        "decisions": result.get("decisions"), "model_calls": result.get("model_calls"),
        "human_intervention": result.get("human_intervention"),
        "operator_message_count": result.get("operator_message_count"),
        "evaluation": evaluation.get("status"), "reasons": evaluation.get("reasons"),
        "sealed": (root / "seal.json").is_file(),
        "pi_restarts": (root / "native" / "pi-restarts.jsonl").read_text().splitlines()
        if (root / "native" / "pi-restarts.jsonl").is_file() else [],
    }


def start_model(out: Path, delay: float) -> tuple[subprocess.Popen, int]:
    port_file = out / "model.port"
    port_file.unlink(missing_ok=True)
    model = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                              "--delay", str(delay)], cwd=REPO, stdout=(out / "model.log").open("w"),
                             stderr=subprocess.STDOUT)
    wait(port_file.exists, 30, "model server port")
    return model, int(port_file.read_text())


def environment(out: Path, port: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    # No real credential reaches the demo, from the environment or the user's ORA store: the
    # provider picker offers only the local model as ready.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ORA_") and "API_KEY" not in key and key != "TMUX"}
    env.update(ORA_CONFIG_HOME=str(out / "ora-config"),
               ORA_BASE_URL=f"http://127.0.0.1:{port}/v1", ORA_API_KEY="local-demo", ORA_MODEL="fake-demo",
               ORA_WORKSPACE_ROOT=str(out / "workspace"), PYTHON=sys.executable, TERM="xterm-256color",
               PATH=f"{Path(shutil.which('node') or 'node').parent}{os.pathsep}{os.environ['PATH']}")
    env.update(extra or {})
    return env


ORA = str(REPO / "scripts" / "ora")


def scenario_main(out: Path, checks: dict[str, Any]) -> dict[str, Any]:
    model, port = start_model(out, delay=6.0)
    pane = Pane("ora-demo-main", out, environment(out, port))
    workspace = out / "workspace"
    try:
        # The world allows more actions than the scenario spends before /stop (the fake model moves on
        # every call); with 6, a fast host reached the last action before /stop and the episode ended.
        pane.type(f"cd {out} && {ORA} interactive --actions 24")
        wait(lambda: "Model provider for this run" in pane.text(), 60, "provider picker")
        pane.frame("provider-picker")
        pane.keys("Enter")  # the ready choices come first: here the ORA_* environment
        root = wait(lambda: run_root(workspace), 120, "run directory")
        wait(lambda: "observation 0" in pane.text(), 180, "first observation in the TUI")
        wait(lambda: count(root, "provider_request") >= 1, 60, "first model call")
        time.sleep(1)
        pane.frame("attached")

        pane.type("/help")
        wait(lambda: "ORA interactive" in pane.text(), 20, "help panel")
        pane.frame("help")
        pane.keys("Enter")
        time.sleep(0.5)

        pane.type("/status")
        wait(lambda: "ORA run status" in pane.text(), 20, "status panel")
        status = pane.frame("status")
        checks["status_shows_budget"] = "budget left" in status and "model calls" in status
        pane.keys("Enter")
        time.sleep(0.5)

        pane.type("/model")
        wait(lambda: "bound by the RunPlan" in pane.text(), 20, "/model notice")
        pane.frame("model")
        # /login works in every run: credentials go to the user-level ORA store, never the run.
        pane.type("/login")
        time.sleep(3)
        login = pane.frame("login")
        checks["login_open_in_a_benchmark_run"] = ("Unknown command /login" not in login
                                                   and "Select authentication method" in login)
        pane.keys("Escape")  # closes the login selector (it has the keyboard; the run goes on)
        time.sleep(1.2)
        pane.type("/tree")
        wait(lambda: "/tree is not available here" in pane.text(), 20, "/tree refusal")
        pane.frame("tree-refused")
        pane.type("/frobnicate")
        wait(lambda: "Unknown command /frobnicate" in pane.text(), 20, "unknown command refusal")
        pane.frame("unknown-command")

        steer = "Operator: keep the gripper orientation level"
        pane.type(steer)
        wait(lambda: count(root, "operator_message") >= 1, 120, "steer delivered")
        wait(lambda: "Operator note received" in pane.text(), 120, "reply acknowledging the steer")
        delivered = pane.frame("steer-delivered")
        checks["steer_delivered"] = True
        # Queued and delivered read apart: a notice when queued, "› you: …" once the Supervisor delivered it.
        checks["steer_echoed_once_delivered"] = f"› you: {steer}" in delivered

        before = count(root, "provider_request")
        pane.keys("C-d")
        wait(lambda: "re-attach: ora attach" in pane.text(), 30, "detach")
        pane.frame("detached")
        wait(lambda: count(root, "provider_request") > before, 60, "the run continues while detached")
        checks["run_continued_detached"] = {"calls_at_detach": before, "calls_later": count(root, "provider_request")}

        pane.type(f"{ORA} sessions")
        wait(lambda: "attachable" in pane.text(), 30, "sessions listing")
        pane.frame("sessions")

        pane.type(f"{ORA} attach {root.name}")
        wait(lambda: "observation" in pane.text(), 60, "re-attached TUI")
        time.sleep(2)
        pane.frame("reattached")

        pane.type("/export session-from-tui.html")
        wait(lambda: (out / "session-from-tui.html").is_file(), 30, "/export file")
        wait(lambda: "Session exported to" in pane.text(), 10, "/export notice")
        pane.frame("export")
        checks["tui_export_bytes"] = (out / "session-from-tui.html").stat().st_size

        endpoint = json.loads((root / "native" / "pi-tui.json").read_text())
        wait(lambda: count(root, "provider_request") > count(root, "provider_response"), 60, "a call in flight")
        os.kill(int(endpoint["pid"]), signal.SIGKILL)
        wait(lambda: json.loads((root / "native" / "pi-tui.json").read_text())["pid"] != endpoint["pid"], 120,
             "restarted agent process")
        wait(lambda: "observation" in pane.text(), 120, "automatic re-attach to the restarted agent process")
        time.sleep(3)
        pane.frame("core-restarted-reattached")
        checks["core_restarted"] = {"killed_pid": endpoint["pid"],
                                    "new_pid": json.loads((root / "native" / "pi-tui.json").read_text())["pid"]}

        pane.type("/stop")
        wait(lambda: "Stop this run?" in pane.text(), 20, "stop confirmation")
        pane.frame("stop-confirm")
        pane.keys("Enter")
        # The episode is over once the agent process closes its session; the Supervisor then
        # evaluates and seals. A second terminal attaches inside that window.
        wait(lambda: session_closed(root) or (root / "result.json").is_file(), 240, "the stopped episode to end")
        pane.window("second")
        pane.type(f"cd {out} && {ORA} attach {root.name}", window="second")
        wait(lambda: "evaluating and sealing" in pane.text("second") or "· finished" in pane.text("second"),
             60, "the second terminal reports the landing")
        checks["attach_caught_landing"] = "evaluating and sealing" in pane.text("second")
        pane.frame("attach-while-landing", window="second")
        wait(lambda: SUMMARY.search(pane.text()), 240, "landing summary after /stop")
        time.sleep(1)
        pane.frame("stopped-summary")
        wait(lambda: "· finished" in pane.text("second"), 120, "read-only view of the landed run")
        pane.type("please continue", window="second")
        wait(lambda: "its session is read-only" in pane.text("second"), 20, "input refused in the view")
        pane.frame("finished-run-read-only", window="second")
        pane.keys("C-d", window="second")
        time.sleep(2)

        export = subprocess.run([ORA, "sessions", "export", root.name, "--output", str(out / "session.html")],
                                capture_output=True, text=True, cwd=out, env=environment(out, port))
        export_jsonl = subprocess.run([ORA, "sessions", "export", root.name, "--output", str(out / "session.jsonl")],
                                      capture_output=True, text=True, cwd=out, env=environment(out, port))
        listing = subprocess.run([ORA, "sessions", "--json"], capture_output=True, text=True, cwd=out,
                                 env=environment(out, port))
        (out / "sessions.json").write_text(listing.stdout)
        checks["cli_export"] = {"html": export.returncode == 0 and (out / "session.html").is_file(),
                                "jsonl": export_jsonl.returncode == 0 and (out / "session.jsonl").is_file()}
        return outcome(root)
    finally:
        pane.frame("final")
        pane.close()
        model.terminate()


def scenario_esc(out: Path, checks: dict[str, Any]) -> dict[str, Any]:
    model, port = start_model(out, delay=12.0)
    raw = out / "attach-client.raw.log"
    pane = Pane("ora-demo-esc", out, environment(out, port, {"PI_IMAGE_PROTOCOL": "kitty", "PI_TUI_WRITE_LOG": str(raw)}))
    workspace = out / "workspace"
    try:
        pane.type(f"cd {out} && {ORA} interactive --provider env --actions 3")
        root = wait(lambda: run_root(workspace), 120, "run directory")
        wait(lambda: count(root, "provider_response") >= 1 and count(root, "provider_request") >= 2, 180,
             "second model call in flight")
        time.sleep(2)
        pane.frame("call-in-flight")
        pane.keys("Escape")
        wait(lambda: count(root, "session_interrupted") >= 1, 60, "operator interrupt")
        wait(lambda: SUMMARY.search(pane.text()) or "inspect" in pane.text(), 240, "landing summary after Esc")
        time.sleep(1)
        pane.frame("after-esc")
        data = raw.read_bytes() if raw.is_file() else b""
        checks["kitty_image_sequences"] = data.count(b"\x1b_G")
        return outcome(root)
    finally:
        pane.frame("final")
        pane.close()
        model.terminate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--scenario", choices=("main", "esc", "all"), default="all")
    args = parser.parse_args()
    report: dict[str, Any] = {}
    for name, scenario in (("main", scenario_main), ("esc", scenario_esc)):
        if args.scenario not in {name, "all"}:
            continue
        out = args.out / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        checks: dict[str, Any] = {}
        try:
            report[name] = {"outcome": scenario(out, checks), "checks": checks}
        except Exception as exc:  # keep the frames and say where it stopped
            report[name] = {"error": f"{type(exc).__name__}: {exc}", "checks": checks}
        (out / "summary.json").write_text(json.dumps(report[name], indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
