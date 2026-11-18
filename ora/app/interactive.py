"""``ora interactive``, ``ora attach``, ``ora sessions`` and ``ora local``: ORA runs in Pi's terminal UI.

The agent core of a Pi-core run is a headless Node process the Supervisor
owns. Launched with ``ORA_PI_TUI=1`` it also serves Pi's InteractiveMode on a
unix socket it announces in ``<run>/native/pi-tui.json``; ``ora attach`` runs
a thin pi-tui client that shows it on this terminal (``ora-pi/src/tui/``).
Detaching leaves the run going; the Supervisor restarts a lost agent process
and the client re-attaches to the new one.

    ora interactive [--provider PROFILE|env] [--task TEXT] [--actions N]
    ora interactive --spec SPEC [--spec SPEC ...] [--confirm-launch]   the benchmark task picker
    ora attach [RUN] [--last]          (a finished run opens read-only)
    ora sessions [--json] | ora sessions export RUN [--output PATH]
    ora local [--suite S --task-id N --seed N] [--confirm-launch]     the local CPU MuJoCo task picker

``ora interactive --spec`` without ``--ui pi`` opens the task picker: the
pi-tui front end (``ora-pi/src/watch/index.mjs``) draws what its Python child,
``python -m ora.app.backend interactive ... --emit jsonl``, serves; with
``--emit jsonl`` that backend runs here. ``ora local`` prepares the local CPU
task (``local.py``) and opens ``ora interactive`` on it.

Credentials stay with ORA: the provider profile names environment variables,
the Supervisor qualifies the key and hands it to the agent process by
environment only; Pi keeps no auth.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator import paths
from ora.orchestrator.episode import agent_core

ORA_PI = paths.PROJECT_ROOT / "ora-pi"
TUI = ORA_PI / "src" / "tui"
ENDPOINT = Path("native") / "pi-tui.json"
#: attach.ts exit codes
DETACHED, NOT_REACHABLE, RUN_ENDED, CONNECTION_LOST = 0, 2, 3, 4


class PiTuiError(RuntimeError):
    """The Pi terminal UI cannot start or attach."""


# -- Node and the ora-pi package -------------------------------------------------------

def node_binary() -> str:
    """The Pi core's Node.js and locked ora-pi packages (agent_core's resolution, as the Supervisor's)."""

    try:
        return agent_core.require_pi_runtime()
    except agent_core.AgentCoreError as exc:
        raise PiTuiError(str(exc)) from exc


def pick(title: str, items: list[dict[str, str]], *, hint: str | None = None) -> str | None:
    """A pi-tui choice list on this terminal; None when cancelled."""

    node = node_binary()
    with tempfile.TemporaryDirectory(prefix="ora-pick-") as directory:
        choices, out = Path(directory) / "choices.json", Path(directory) / "choice"
        choices.write_text(json.dumps({"title": title, "hint": hint, "items": items}), encoding="utf-8")
        code = subprocess.call([node, str(TUI / "pick.ts"), str(choices), str(out)])
        return out.read_text(encoding="utf-8") if code == 0 and out.is_file() else None


# -- Provider profiles ------------------------------------------------------------------

ENV_PROVIDER = "env"
#: The selector's "add an endpoint of your own" entry (``providers.add_interactively``).
ADD_PROVIDER = "+add"
KEY_SOURCES = {"environment": "ready (key from the environment)", "stored": "ready (key stored with ORA)"}


def provider_choices() -> list[dict[str, Any]]:
    """Live-admitted provider profiles (the user's own too), adding an endpoint, and the ORA_* environment.

    A profile without its key reads "needs VARIABLE": choosing it asks for the key.
    """

    from ora.agent.providers.user_store import all_profiles, key_source

    choices: list[dict[str, Any]] = []
    for profile_id, profile in all_profiles().items():
        if profile.get("admission") == "registered_only" or profile.get("usage_scope") == "supported_coding_tools_only":
            continue
        source = key_source(profile["api_key_env"])
        choices.append({
            "value": profile_id, "label": profile_id, "ready": source != "missing",
            "description": f"{profile['model']} · {profile['protocol']}" + (" · your profile" if profile.get("user")
                                                                            else "")
                           + f" · {KEY_SOURCES.get(source, 'needs ' + profile['api_key_env'] + ': choose to enter it')}",
        })
    choices.append({"value": ADD_PROVIDER, "label": "+ add an endpoint", "ready": True,
                    "description": "an OpenAI-compatible or Anthropic endpoint of your own: base URL, model, key"})
    env_key = os.environ.get("ORA_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    choices.append({
        "value": ENV_PROVIDER, "label": "env (ORA_* variables)", "ready": bool(env_key),
        "description": (os.environ.get("ORA_MODEL") or "model from ORA_MODEL")
                       + f" · {os.environ.get('ORA_BASE_URL') or 'default endpoint'}"
                       + f" · {'ready' if env_key else 'needs ORA_API_KEY'}",
    })
    return choices


def choose_provider(requested: str | None) -> str:
    if requested:
        return requested
    if not _terminal():
        raise PiTuiError("choose a provider with --provider PROFILE (see `ora providers`) or --provider env")
    from ora.agent.providers.user_store import apply_stored_credentials

    chosen = _choose_provider()
    apply_stored_credentials()  # a key entered here serves this launch too
    return chosen


def _choose_provider() -> str:
    """The selector: a profile missing its key asks for it; "+ add an endpoint" asks for one."""

    from . import providers
    from ora.agent.providers.user_store import key_source

    while True:
        # Ready profiles and the environment first, then adding an endpoint, then profiles needing a key.
        choices = provider_choices()
        ordered = ([item for item in choices if item["ready"] and item["value"] != ADD_PROVIDER]
                   + [item for item in choices if item["value"] == ADD_PROVIDER]
                   + [item for item in choices if not item["ready"]])
        chosen = pick("Model provider for this run (credentials stay with ORA)", [
            {key: item[key] for key in ("value", "label", "description")} for item in ordered
        ])
        if chosen is None:
            raise PiTuiError("no provider chosen")
        if chosen == ADD_PROVIDER:
            added = providers.add_interactively(lambda title, items: pick(title, items))
            if added is not None:
                return added
            continue
        variable, _ = providers.key_variable(chosen)
        if chosen == ENV_PROVIDER or key_source(variable) != "missing":
            return chosen
        # The key is entered here, stored outside the run, and the provider probed at once.
        if providers.configure_key(chosen) or input("Use it anyway? [y/N]: ").strip().lower() in {"y", "yes"}:
            return chosen


# -- The interactive Synthetic RunSpec --------------------------------------------------

DEFAULT_TASK = (
    "Read the robot manual, then use robot.submit_action to move the left end effector, one action at a time, "
    "to position_m [0, 0, 1.02], then [0.05, 0, 1.05], then back to [0, 0, 1.02], always with orientation_xyzw "
    "[0, 0, 0, 1]. Look at the camera image after each action. The episode ends after the last accepted action."
)


def interactive_spec(*, provider: str, task: str | None, actions: int) -> dict[str, Any]:
    """The quickstart's Synthetic episode (``spec.quickstart.synthetic_spec``), interactive and longer.

    `provider` is a provider-profile id, or ``env`` for the quickstart's
    ORA_PROVIDER/ORA_BASE_URL/ORA_MODEL/ORA_API_KEY. Operator steering is
    plan-bound (``operator_control: file_inbox_v1`` under the
    ``ora-interactive`` experiment), so a steered run is marked human-assisted.
    """

    from ora.orchestrator.spec.quickstart import environment_policy, synthetic_spec

    if actions < 1 or actions > 50:
        raise PiTuiError("--actions must be between 1 and 50")
    if provider == ENV_PROVIDER:
        policy = environment_policy()
    else:
        from ora.orchestrator.spec.inputs import provider_policy

        policy = provider_policy(provider)
        os.environ.setdefault(policy["base_url_env"], policy["endpoint"])  # as `ora run` does
    spec = synthetic_spec(policy)
    spec.update({
        "experiment": "ora-interactive", "config": "synthetic-pi-interactive", "task": "move-interactive",
        "output_workspace": "outputs/ora-interactive", "task_instruction": task or DEFAULT_TASK,
        "budgets": {"max_robot_decisions": actions, "max_model_calls": 8 * actions + 4,
                    "max_sim_steps": actions, "wall_time_s": 3600},
        "operator_control": {"mode": "file_inbox_v1"},
    })
    spec["runtime"] = {**spec["runtime"], "terminal_actions": actions}
    return spec


def command_interactive(args: argparse.Namespace) -> int:
    """Launch one run on the Pi core and show it in Pi's terminal UI."""

    node = node_binary()
    if args.spec:
        if not args.confirm_launch:
            raise PiTuiError("a prepared benchmark RunSpec launches only with --confirm-launch")
        from .task_catalog import load_tasks

        tasks = load_tasks(args.spec)
        task = tasks[0]
        if len(tasks) > 1:
            if not _terminal():
                raise PiTuiError("choose among several --spec in a terminal, or pass one")
            chosen = pick("Benchmark task", [item.item() for item in tasks])
            if chosen is None:
                raise PiTuiError("no task chosen")
            task = next(item for item in tasks if item.value == chosen)
        # The task picker's composition: an ora-interactive plan, steerable, on a fresh admission cell.
        spec = task.interactive_spec(uuid.uuid4().hex, [])
    else:
        spec = interactive_spec(provider=choose_provider(args.provider), task=args.task, actions=args.actions)
    from .world import terminal_observer

    return launch(spec, node=node, no_attach=args.no_attach, world=terminal_observer(args))


def launch(spec: Mapping[str, Any], *, node: str, no_attach: bool, world=None) -> int:
    """Resolve, preflight and launch one run on the Pi core with its terminal UI, then attach here."""

    from ora.orchestrator.launch import _preflight, _run_root, _spawn
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    plan = resolve_run_spec(spec)
    if plan["policy"]["kind"] != "native_api":
        raise PiTuiError("the Pi terminal UI drives a native_api policy")
    try:
        agent_core.selected(plan["policy"])
    except agent_core.AgentCoreError as exc:
        raise PiTuiError(str(exc)) from None
    print(f"ORA · {plan['experiment']} · {plan['task']} · {plan['policy'].get('model')}", flush=True)
    preflight = _preflight(plan, compact=True)
    if not preflight["passed"]:
        return 2
    os.environ["ORA_PI_TUI"] = "1"  # the agent core serves its terminal UI (ora-pi/src/tui/host.ts)
    run_id = _spawn(plan, preflight, yes=True, report=lambda line: None)
    root = _run_root(run_id)
    print(f"run {run_id}", flush=True)
    if no_attach:
        print(f"attach: ora attach {run_id}")
        return 0
    return attach_loop(root, node, world=world)


# -- Attaching --------------------------------------------------------------------------

def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _alive(pid: Any) -> bool:
    try:
        # The Supervisor this process launched is its child: reap it once it exits.
        if os.waitpid(int(pid), os.WNOHANG)[0] == int(pid):
            return False
    except (ChildProcessError, TypeError, ValueError):
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def endpoint(root: Path) -> dict[str, Any] | None:
    """The live TUI endpoint of a run's agent process on this host, or None."""

    record = _load(root / ENDPOINT)
    if not record or not isinstance(record.get("socket"), str):
        return None
    if not _alive(record.get("pid")) or not os.path.exists(record["socket"]):
        return None
    return record


def supervisor_running(root: Path) -> bool:
    """The run's Supervisor process is alive on this host (it may be landing the run)."""

    process = _load(root / "process.json")
    return bool(process) and _alive(process.get("pid"))


def agent_session_closed(root: Path) -> bool:
    """The agent process recorded `session_closed`: the episode is over, the run is landing."""

    path = root / "agent-events.jsonl"
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            return b'"session_closed"' in stream.read()
    except OSError:
        return False


#: Where a run is, as `ora attach` sees it from this host.
LIVE, STARTING, LANDING, FINISHED, LOST = "live", "starting", "landing", "finished", "lost"


def run_phase(root: Path, *, not_pid: Any = None) -> str:
    """live: a terminal endpoint answers; starting: the Supervisor runs but its agent process
    is (re)starting; landing: the episode is over and the Supervisor evaluates and seals;
    finished: landed; lost: the Supervisor is gone without a result."""

    found = endpoint(root)
    if found is not None and found.get("pid") != not_pid:
        return LIVE
    if supervisor_running(root):
        return LANDING if (root / "result.json").is_file() or agent_session_closed(root) else STARTING
    return FINISHED if (root / "result.json").is_file() else LOST


def run_state(root: Path) -> str:
    status = _load(root / "status.json") or {}
    return str(status.get("state") or ("finished" if (root / "result.json").is_file() else "starting"))


def wait_for_landing(root: Path, *, announce: bool = True, limit_s: float = 3600.0, every_s: float = 30.0) -> bool:
    """Say that the run is landing, and wait (saying so again now and then) until it has.

    ``announce=False`` when the attached terminal has just printed the same news.
    """

    if announce:
        print("the episode has ended; ORA is evaluating and sealing the run…", flush=True)
    started = time.monotonic()
    said = started
    while supervisor_running(root):
        now = time.monotonic()
        if now - started > limit_s:
            print(f"still landing after {int(now - started)} s; inspect {root / 'supervisor.log'}", flush=True)
            return False
        if now - said >= every_s:
            print(f"still evaluating and sealing ({int(now - started)} s)…", flush=True)
            said = now
        time.sleep(0.25)
    return True


def summary(root: Path) -> int:
    """Print a landed run's outcome."""

    result = _load(root / "result.json")
    if result is None:
        print(f"{root.name}: {run_state(root)}; inspect {root / 'supervisor.log'}")
        return 1
    evaluation = _load(root / "evaluation.json") or {}
    parts = [
        f"state {result.get('state')}", f"{result.get('decisions', 0)} decisions",
        f"{result.get('model_calls', 0)} model calls",
        f"official_success {json.dumps(result.get('official_success'))}",
        f"evaluation {evaluation.get('status', 'pending')}",
        "sealed" if (root / "seal.json").is_file() else "not sealed",
    ]
    if result.get("human_intervention"):
        parts.append(f"human-assisted ({result.get('operator_message_count', 0)} operator messages)")
    print(f"{root.name}: " + " · ".join(parts), flush=True)
    return 0 if result.get("state") == "completed" else 1


def _terminal() -> bool:
    return sys.stdin.isatty()


def view_finished(root: Path, node: str) -> int:
    """A landed run: its Pi session read-only in a terminal, else its outcome."""

    if any((root / "native" / "pi-session").glob("*.jsonl")) and _terminal():
        return subprocess.call([node, str(TUI / "view.ts"), str(root)])
    summary(root)
    print(f"export its session with `ora sessions export {root.name}`")
    return 0


def attach_loop(root: Path, node: str | None = None, *, open_view: bool = False, world=None) -> int:
    """Show the run's Pi TUI here until it is detached or the run lands.

    Every wait says what it waits for. A run that ends while attached prints
    its outcome once landed; with ``open_view`` (an attach that found the run
    already over) the landed run opens read-only instead. ``world``
    (``world.terminal_observer``) serves the run's cameras on a loopback
    browser page while this terminal is attached; the TUI shows its address.
    """

    from .remote_session import record
    from .world import TerminalWorldFeed

    node = node or node_binary()
    record(root)
    feed = TerminalWorldFeed(world if run_phase(root) != FINISHED else None, root)
    web = None
    try:
        url = feed.url()
        if url:
            print(f"world view: {url}", flush=True)
        if run_phase(root) != FINISHED:
            from .web import from_environment
            web = from_environment(root)  # `ora-remote --web`: the browser view beside this terminal
        return _attach(root, node, feed, open_view=open_view)
    finally:
        feed.close()
        if web is not None:
            web.close()


def _attach(root: Path, node: str, feed, *, open_view: bool) -> int:
    from .remote_session import key

    previous_pid = None
    starting_since = None
    while True:
        phase = run_phase(root, not_pid=previous_pid)
        if phase == LIVE:
            starting_since = None
            found = endpoint(root)
            code = feed.call([node, str(TUI / "attach.ts"), str(found["socket"])])
            if code == DETACHED:
                print("re-attach: run the same ora-remote command again" if key()
                      else f"re-attach: ora attach {root.name}")
                return 0
            if code == RUN_ENDED:  # the terminal printed the termination and that the run is landing
                wait_for_landing(root, announce=False)
                return summary(root)
            # The agent process went away (or was unreachable): the Supervisor restarts it.
            previous_pid = found.get("pid")
            if run_phase(root, not_pid=previous_pid) == STARTING:
                print("the agent process restarted; re-attaching", flush=True)
        elif phase == STARTING:
            if starting_since is None:
                starting_since = time.monotonic()
                print("waiting for the run's agent process…", flush=True)
            elif time.monotonic() - starting_since > 180:
                print(f"no agent process answered in 180 s; inspect {root / 'supervisor.log'}", flush=True)
                return 1
            time.sleep(0.2)
        elif phase == LANDING:
            if not wait_for_landing(root):
                return 1
            return view_finished(root, node) if open_view else summary(root)
        elif phase == FINISHED:
            return view_finished(root, node) if open_view else summary(root)
        else:
            raise PiTuiError(f"{root.name} has no running Supervisor here and no result ({run_state(root)}); "
                             "an action-free incomplete run can be retried with `ora retry`")


def _runs_with_sessions() -> list[Path]:
    roots = [root for root in paths.iter_run_roots() if any((root / "native" / "pi-session").glob("*.jsonl"))]
    return sorted(roots, key=lambda root: (root / "native" / "pi-session").stat().st_mtime, reverse=True)


def resolve_run(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_dir() else paths.resolve_run_root(value)


def command_attach(args: argparse.Namespace) -> int:
    node = node_binary()
    if args.run_id:
        root = resolve_run(args.run_id)
    else:
        live = [root for root in _runs_with_sessions() if endpoint(root) is not None]
        if not live:
            raise PiTuiError("no run on this host serves a Pi terminal; start one with `ora interactive`")
        if args.last or len(live) == 1 or not _terminal():
            root = live[0]
        else:
            chosen = pick("Attach to a running ORA run", [
                {"value": str(root), "label": root.name, "description": _describe(root)} for root in live
            ])
            if chosen is None:
                return 1
            root = Path(chosen)
    # A run that has ended (or ends while this command waits) opens read-only once landed.
    from .world import terminal_observer

    return attach_loop(root, node, open_view=True, world=terminal_observer(args))


# -- Sessions ---------------------------------------------------------------------------

def _describe(root: Path) -> str:
    plan = _load(root / "plan.json") or {}
    result = _load(root / "result.json") or _load(root / "status.json") or {}
    model = (plan.get("policy") or {}).get("model", "?")
    return (f"{plan.get('experiment', '?')} · {model} · {run_state(root)}"
            f" · {result.get('decisions', 0)} decisions · {result.get('model_calls', 0)} calls")


def session_rows() -> list[dict[str, Any]]:
    rows = []
    for root in _runs_with_sessions():
        plan = _load(root / "plan.json") or {}
        result = _load(root / "result.json") or {}
        status = _load(root / "status.json") or {}
        files = sorted((root / "native" / "pi-session").glob("*.jsonl"))
        rows.append({
            "run_id": root.name, "run_root": str(root), "experiment": plan.get("experiment"),
            "task": plan.get("task"), "model": (plan.get("policy") or {}).get("model"),
            "state": run_state(root), "attachable": endpoint(root) is not None,
            "decisions": result.get("decisions", status.get("decisions")),
            "model_calls": result.get("model_calls", status.get("model_calls")),
            "human_intervention": result.get("human_intervention"),
            "sealed": (root / "seal.json").is_file(), "pi_session": str(files[-1]),
            "updated_unix_s": files[-1].stat().st_mtime,
        })
    return rows


def command_sessions(args: argparse.Namespace) -> int:
    if args.action == "export":
        if not args.run_id:
            raise PiTuiError("usage: ora sessions export RUN [--output PATH.html|PATH.jsonl]")
        root = resolve_run(args.run_id)
        files = sorted((root / "native" / "pi-session").glob("*.jsonl"))
        if not files:
            raise PiTuiError(f"{root.name} has no Pi session (a Native-core run?)")
        output = args.output or f"{root.name}.html"
        completed = subprocess.run([node_binary(), str(TUI / "export.ts"), str(files[-1]), output],
                                   capture_output=True, text=True, check=False)
        if completed.returncode:
            raise PiTuiError(completed.stderr.strip() or "export failed")
        print(completed.stdout.strip())
        return 0
    rows = session_rows()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no Pi-core runs yet; start one with `ora interactive`")
        return 0
    for row in rows:
        flags = ["attachable" if row["attachable"] else "", "sealed" if row["sealed"] else "",
                 "human-assisted" if row["human_intervention"] else ""]
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["updated_unix_s"]))
        print(f"{row['run_id']}\t{stamp}\t{row['state']}\t{row['model']}\t"
              f"{row['decisions'] or 0} decisions\t{row['model_calls'] or 0} calls\t"
              + " ".join(flag for flag in flags if flag))
    return 0


COMMANDS = ("interactive", "attach", "sessions", "local")
#: The ``ora local`` / ``ora interactive`` help lines.
HELP = {
    "local": "open the local CPU MuJoCo task picker (LIBERO on this computer)",
    "interactive": "run ORA in Pi's terminal UI (Synthetic, or a prepared RunSpec) and steer it while it runs",
    "attach": "show a running Pi-core run's terminal UI here (detach leaves it running)",
    "sessions": "list runs with Pi sessions, or export one as HTML or JSONL",
}


def add_interactive_arguments(parser: argparse.ArgumentParser) -> None:
    """``ora interactive``'s options (``python -m ora.app.backend interactive`` takes them too)."""

    from .world import add_world_options

    parser.add_argument("--spec", type=Path, action="append",
                        help="prepared Native benchmark RunSpec; repeat to populate the task picker "
                             "(without --spec: a Synthetic episode in Pi's terminal UI)")
    parser.add_argument("--ui", choices=("pi", "watch"),
                        help="pi: the run's own Pi terminal UI on the Pi agent core (the default without "
                             "--spec); watch: the task picker and transcript (the default with --spec)")
    parser.add_argument("--provider", metavar="PROFILE",
                        help="provider profile (`ora providers`), or env for ORA_BASE_URL/ORA_MODEL/"
                             "ORA_API_KEY; asked in a terminal when omitted")
    parser.add_argument("--task", metavar="TEXT", help="Synthetic task instruction (default: three moves)")
    parser.add_argument("--actions", type=int, default=3, help="Synthetic robot decisions before the episode ends")
    parser.add_argument("--no-attach", action="store_true",
                        help="launch and print the `ora attach` command instead of attaching")
    parser.add_argument("--confirm-launch", action="store_true",
                        help="authorize launching a selected task after its normal preflight passes")
    parser.add_argument("--emit", choices=("jsonl",), help="JSON-lines RPC backend instead of the Pi UI")
    parser.add_argument("--poll", type=float, default=0.2, help="event polling interval in seconds")
    add_world_options(parser)


def add_attach_options(parser: argparse.ArgumentParser) -> None:
    from .world import add_world_options

    parser.add_argument("run_id", nargs="?", help="run ID or run directory (default: choose a running one)")
    parser.add_argument("--last", action="store_true", help="the most recently active running run")
    add_world_options(parser)


def add_sessions_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", nargs="?", choices=("list", "export"), default="list")
    parser.add_argument("run_id", nargs="?", help="with export: run ID or run directory")
    parser.add_argument("--output", "-o", help="with export: .html (Pi's viewer) or .jsonl (the raw Pi session)")
    parser.add_argument("--json", action="store_true")


def register(sub: Any) -> None:
    from .local import add_local_options
    from .world import add_world_options

    add_interactive_arguments(sub.add_parser("interactive", help=HELP["interactive"]))
    add_attach_options(sub.add_parser("attach", help=HELP["attach"]))
    add_sessions_options(sub.add_parser("sessions", help=HELP["sessions"]))
    local = sub.add_parser("local", help=HELP["local"])
    add_local_options(local)
    add_world_options(local)


def front_end_interactive(args: argparse.Namespace) -> None:
    """Replace this process with the pi-tui task picker (``ora-pi/src/watch/index.mjs``).

    The front end starts ``python -m ora.app.backend interactive ... --emit jsonl``
    with this interpreter and draws what it serves.
    """

    from ora.agent.tui import cli as watch_cli

    front_end = watch_cli._pi_front_end()
    if front_end is None:
        raise RuntimeError(f"interactive UI needs Node.js {watch_cli.MIN_NODE_TEXT} or newer "
                           f"({watch_cli.node_diagnosis()}) and npm ci --prefix {watch_cli.FRONT_END}; "
                           "put ~/.local/bin first on PATH in the shell or tmux pane that starts ORA, "
                           "or set ORA_NODE; --emit jsonl exposes the RPC backend")
    node, script = front_end
    passed = ["--interactive", "--python", sys.executable, "--poll", str(args.poll)]
    for path in args.spec:
        passed += ["--spec", str(path.expanduser().resolve())]
    if args.confirm_launch:
        passed.append("--confirm-launch")
    passed += ["--world-view", args.world_view, "--world-port", str(args.world_port),
               "--world-browser", args.world_browser]
    if args.world_public_url:
        passed += ["--world-public-url", args.world_public_url]
    environment = dict(os.environ)
    root = str(watch_cli.FRONT_END.parent)
    environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"]
                                       if environment.get("PYTHONPATH") else "")
    os.execve(node, [node, str(script), *passed], environment)


def run(args: argparse.Namespace) -> int:
    if args.command == "local":
        from .local import interactive_arguments
        from .main import main
        from .world import world_explicit

        # The local route prepares inputs only. The interactive backend keeps
        # manual task selection and every launch gate.
        return main(interactive_arguments(args, world_explicit=world_explicit(args)))
    try:
        if args.command == "attach":
            return command_attach(args)
        if args.command == "sessions":
            return command_sessions(args)
        if not args.spec or args.ui == "pi":
            if args.emit is not None:
                raise ValueError("--emit jsonl drives the task picker; pass --spec")
            return command_interactive(args)
    except PiTuiError as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2
    from .run import bind_prepared_specs
    from .world import watch_observer

    try:
        bind_prepared_specs(list(args.spec))  # specs ora run --prepare-only wrote: their .env, marker and services
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora interactive: {exc}", file=sys.stderr)
        return 2
    world = watch_observer(args)  # checks the world options before the front end gets them
    if args.emit is None:
        front_end_interactive(args)
    from .backend import serve

    return serve(args.spec, confirm_launch=args.confirm_launch, poll=args.poll, observer_factory=world)


__all__ = [
    "COMMANDS", "add_attach_options", "add_interactive_arguments", "add_sessions_options", "attach_loop",
    "command_attach", "command_interactive", "command_sessions", "endpoint", "interactive_spec",
    "provider_choices", "register", "run", "session_rows",
]
