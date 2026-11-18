"""A run's replay program: its agent actions as code that reproduces the run exactly.

Every sealed run carries ``replay-program/``, written just before the seal inventory
(``orchestrator.evaluation.evaluate.seal``) from the evidence the run already holds;
writing it starts no simulator.

* ``program.json`` (``ora.replay-program.v1``): one step per agent action, in order,
  with the receipt pose each step must end at and the verdict the run reached.
* ``program.py``: the same steps as Python, one line per action. Running it, or
  ``ora replay RUN --check OUT``, re-executes the steps on the run's simulator and
  checks every step's end pose bit for bit and the verdict.

Step kinds. RoboTwin, LIBERO-Pro, RoboCasa365 and RoboLab (``native/controller-replay``):
``controls`` runs the action's recorded controls, ``hold`` steps the scene with no new target
(a RoboTwin ``wait``), ``skip`` is an action that moved nothing (a refused plan). RoboDojo
(``native/controller``): ``ticks`` feeds the action's recorded native ticks, ``skip`` has none.
A record the replay cannot reproduce is an ``unreplayable`` step, the program's ``status`` says
why and checking it is refused. A run that records no controls has no program.

The steps execute through the Viewer replay's per-action code (``environment.replay_worker``)
for RoboTwin and LIBERO-Pro, the run's own session and control step for RoboCasa365 (which also
expects the base's end pose and a hash of the whole scene), the run's own environment worker
for RoboLab (``replay_program_worker.py``), and the qualification control replay (``ora qualify
robodojo motion-replay``) for RoboDojo. A program edited away from its recording runs without
receipt expectations (not RoboDojo), so it is a starting point for rewriting a run as code.

``export_scene_trace`` (``ora replay RUN --scene``) runs the same check on every backend in
``SCENE_BACKENDS`` and keeps the scene the re-execution passed through
(``environment/scene_trace.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_SCHEMA = "ora.replay-program.v1"
CHECK_SCHEMA = "ora.replay-program-check.v1"
UNAVAILABLE_SCHEMA = "ora.replay-program-unavailable.v1"
PROGRAM_DIRECTORY = "replay-program"
PROGRAM_JSON = "program.json"
PROGRAM_PY = "program.py"
UNAVAILABLE_JSON = "unavailable.json"
KINDS = ("controls", "hold", "skip", "ticks", "unreplayable")
#: The plan ``runtime.kind`` values whose runs carry a replay program.
PROGRAM_BACKENDS = frozenset({"robotwin", "libero_pro", "robodojo", "robocasa", "robolab"})
#: The backends ``export_scene_trace`` reads a scene from.
SCENE_BACKENDS = ("robotwin", "libero_pro", "robodojo", "robocasa", "robolab")
#: The backends whose replay program reads ``native/controller-replay``.
CONTROLLER_REPLAY_BACKENDS = frozenset({"robotwin", "libero_pro", "robocasa", "robolab"})
#: What a replay's verdict must repeat besides ``official_success``, by backend.
VERDICT_KEYS = {"robotwin": ("first_success_action",), "libero_pro": ("first_termination_action",),
                "robolab": ("first_termination_action",),
                "robocasa": ("first_success_action", "first_success_control_step")}
CHECK_TIMEOUT_S = 4 * 60 * 60
#: The failure of a check whose re-execution ends in another verdict than the run's.
VERDICT_DIFFERS = "the replay's verdict differs from the run's"


class ReplayProgramError(RuntimeError):
    """A replay program cannot be written, loaded or checked."""


class ReplayProgramUnavailable(ReplayProgramError):
    """The run holds no per-tick controls: receipts are not rebuilt into a program, so it can never replay."""


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipts(native: Path) -> dict[int, Mapping[str, Any]]:
    """Receipts by action sequence (``receipts/action-NNNN.json``).

    The sequence, not the ``action_id``, names one executed decision: a LIBERO-Pro
    move that ran as a joint motion and then an OSC refinement keeps one
    ``action_id`` across its two receipts.
    """
    receipts: dict[int, Mapping[str, Any]] = {}
    for path in sorted((native / "receipts").glob("action-*.json")):
        try:
            sequence = int(path.stem.split("-")[1])
            receipt = _load(path)
        except (OSError, ValueError, IndexError):
            continue
        if isinstance(receipt, Mapping):
            receipts[sequence] = receipt
    return receipts


def _expected_poses(receipt: Mapping[str, Any] | None) -> dict[str, list[float]]:
    after = (receipt or {}).get("after")
    if not isinstance(after, Mapping):
        return {}
    poses = {}
    for arm in ("left", "right"):
        pose = (after.get(arm) or {}).get("pose_xyz_xyzw") if isinstance(after.get(arm), Mapping) else None
        if isinstance(pose, list) and len(pose) == 7:
            poses[arm] = [float(value) for value in pose]
    base = (receipt or {}).get("base_after")
    base = base.get("pose_xy_yaw") if isinstance(base, Mapping) else None
    if isinstance(base, list) and len(base) == 3:  # a mobile base's end pose (x, y, yaw)
        poses["base"] = [float(value) for value in base]
    return poses


def _action_name(entry: Mapping[str, Any], receipt: Mapping[str, Any] | None) -> str | None:
    action = entry.get("action")
    if isinstance(action, Mapping):
        action = action.get("action")
    if not isinstance(action, str) and isinstance(receipt, Mapping):
        action = receipt.get("action")
    return action if isinstance(action, str) else None


def _controller_replay_steps(
    run_root: Path, backend: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    from .controller_replay import load_controller_replay_manifest
    from . import replay_worker as worker

    native = run_root / "native"
    manifest_path = native / "controller-replay" / "manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    receipts = _receipts(native)
    if backend == "robolab" and len(manifest["entries"]) != len(receipts):  # a failed recording ends the record
        raise ReplayProgramUnavailable(f"the record holds {len(manifest['entries'])} of {len(receipts)} actions")
    steps, problems = [], []
    for entry in manifest["entries"]:
        receipt = receipts.get(int(entry["action_sequence"]))
        if receipt is not None and receipt.get("action_id") != entry.get("action_id"):
            receipt = None  # a receipt of another action: no expectation rather than a wrong one
        execution = entry.get("execution") if isinstance(entry.get("execution"), Mapping) else {}
        ran = entry.get("physics_steps")
        ran = ran if isinstance(ran, int) and not isinstance(ran, bool) else None
        step: dict[str, Any] = {
            "action_sequence": int(entry["action_sequence"]),
            "action_id": entry.get("action_id"),
            "action": _action_name(entry, receipt),
            "round_index": int(entry.get("round_index", 0)),
            "controller": entry.get("controller"),
            "status": execution.get("status"),
            "physics_steps": ran,
        }
        try:
            if backend == "robotwin":
                arrays = worker._artifact_arrays(manifest_path, entry)
                prefix = worker._robotwin_fault_prefix_steps(entry, arrays)
                hold = 0 if (entry.get("replayable") or prefix) else worker._robotwin_hold_steps(entry)
                if entry.get("replayable") or prefix:
                    step["kind"] = "controls"
                    if prefix:
                        abort = ((entry.get("metadata") or {}).get("controller_tracking") or {}).get("abort") or {}
                        step["note"] = f"the prefix that ran before {abort.get('code')} stopped it"
                elif hold:
                    step["kind"] = "hold"
                elif not ran:
                    step["kind"] = "skip"
                else:
                    raise worker.ReplayWorkerError(f"ran {ran} physics steps its record cannot replay")
            else:
                if entry.get("replayable"):
                    step["kind"] = "controls"
                elif not ran:
                    step["kind"] = "skip"
                else:
                    raise worker.ReplayWorkerError(f"ran {ran} control steps its record cannot replay")
        except worker.ReplayWorkerError as exc:
            step["kind"] = "unreplayable"
            step["note"] = str(exc)
            problems.append(f"action {step['action_sequence']}: {exc}")
        step["expect"] = _expected_poses(receipt)
        state = (entry.get("metadata") or {}).get("state_sha256") or {}
        step["expect"].update({f"state_{moment}": state[moment] for moment in ("before", "after") if state.get(moment)})
        steps.append(step)
    verdict = _load(native / "official-verdict.json") if (native / "official-verdict.json").is_file() else {}
    source = {"path": "native/controller-replay/manifest.json", "sha256": _sha256(manifest_path)}
    return steps, {"expect": _expected_verdict(backend, verdict), "record": source}, problems


def _expected_verdict(backend: str, verdict: Mapping[str, Any]) -> dict[str, Any]:
    if backend == "robotwin":
        terminal = verdict.get("official_terminal") if isinstance(verdict.get("official_terminal"), Mapping) else {}
        return {"official_success": verdict.get("official_success"), "first_success_action": terminal.get("action_sequence")}
    if backend == "robocasa":
        first = verdict.get("first_success") if isinstance(verdict.get("first_success"), Mapping) else {}
        return {"official_success": verdict.get("official_success"), "first_success_action": first.get("action_sequence"),
                "first_success_control_step": first.get("control_step"), "round_index": verdict.get("round_index")}
    name = "first_success" if backend == "robolab" else "first_termination"
    first = verdict.get(name) if isinstance(verdict.get(name), Mapping) else {}
    return {"official_success": verdict.get("official_success"), "first_termination_action": first.get("action_sequence")}


def _robodojo_steps(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    native = run_root / "native"
    traces = sorted((native / "controller").glob("action-*.json"))
    sequences = [int(path.stem.split("-")[1]) for path in traces]
    if sequences != list(range(len(sequences))):
        raise ReplayProgramError("RoboDojo controller traces are not contiguous from action-0000")
    steps = []
    for sequence in sequences:
        trace = _load(native / "controller" / f"action-{sequence:04d}.json")
        receipt_path = native / "receipts" / f"action-{sequence:04d}.json"
        receipt = _load(receipt_path) if receipt_path.is_file() else {}
        ticks = len(trace.get("samples") or [])
        steps.append({
            "action_sequence": sequence, "action_id": receipt.get("action_id"), "action": receipt.get("action"),
            "round_index": int(receipt.get("round_index", 0)), "status": receipt.get("execution_status"),
            "kind": "ticks" if ticks else "skip", "ticks": ticks,
        })
    verdict = _load(native / "official-verdict.json") if (native / "official-verdict.json").is_file() else {}
    expect = {"terminal": verdict.get("terminal"), "official_success": verdict.get("official_success"),
              "native_actions": verdict.get("native_actions")}
    return steps, {"expect": expect, "record": {"path": "native/controller"}}, []


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


#: The code-mode script tools: ``bash``, and in a sealed run the retired ``run_python``.
SCRIPT_TOOLS = ("run_python", "bash")


def _script_groups(run_root: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Mark the steps a code-mode script issued (plan ``policy.code_mode``); the scripts' sources.

    Every action a script causes is recorded as an ``action_proposed`` of the script's own call
    (one ``ora.tool-execution.v1`` parent); that call's ``tool_call_proposed`` holds the source the
    model wrote, a ``run_python`` script or a ``bash`` command (its programs are in the workspace
    snapshots).
    """

    events = _jsonl(run_root / "agent-events.jsonl")
    sources = {row.get("tool_call_id"): (lambda arguments: arguments.get("code") or arguments.get("command"))(
                   (row.get("payload") or {}).get("arguments") or {})
               for row in events
               if row.get("type") == "tool_call_proposed" and (row.get("payload") or {}).get("name") in SCRIPT_TOOLS
               and not (row.get("payload") or {}).get("origin")}
    if not sources:
        return {}
    issued = {(row.get("payload") or {}).get("action_id"): row.get("tool_call_id") for row in events
              if row.get("type") == "action_proposed" and row.get("tool_call_id") in sources}
    scripts: dict[str, Any] = {}
    for step in steps:
        action_id = str(step.get("action_id") or "")
        # A skill sequence's stages are recorded as <action_id>:NN-<stage> under the one action the script proposed.
        script = issued.get(action_id) or issued.get(action_id.split(":", 1)[0])
        if script is None:
            continue
        step["script"] = str(script)
        scripts.setdefault(str(script), {"code": sources.get(script)})
    return scripts


def build_program(run_root: Path) -> dict[str, Any]:
    """The run's replay program from its sealed-evidence layout; starts no simulator."""
    run_root = Path(run_root).resolve()
    plan = _load(run_root / "plan.json")
    native = run_root / "native"
    if (native / "controller-replay" / "manifest.json").is_file():
        backend = _load(native / "controller-replay" / "manifest.json").get("backend")
        if backend not in CONTROLLER_REPLAY_BACKENDS:
            raise ReplayProgramError(f"no replay program for controller-replay backend {backend!r}")
        steps, extra, problems = _controller_replay_steps(run_root, backend)
    elif (native / "controller").is_dir() and any((native / "controller").glob("action-*.json")):
        backend = "robodojo"
        steps, extra, problems = _robodojo_steps(run_root)
    else:
        raise ReplayProgramUnavailable("the run records no controller replay, only receipts, and a replay program "
                                       "is never rebuilt from receipts")
    scripts = _script_groups(run_root, steps)
    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    return {
        "schema": PROGRAM_SCHEMA,
        "run_id": run_root.name,
        "backend": backend,
        "task": plan.get("task"),
        "seed": plan.get("seed"),
        "source_commit": source.get("commit"),
        "status": "replayable" if not problems else "unreplayable",
        "problems": problems,
        **extra,
        **({"scripts": scripts} if scripts else {}),
        "steps": steps,
    }


def _comment(step: Mapping[str, Any], unit: str) -> str:
    parts = [str(step.get("action") or "action")]
    if step.get("controller"):
        parts.append(str(step["controller"]))
    if step["kind"] == "ticks":
        parts.append(f"{step['ticks']} native ticks")
    elif step.get("physics_steps") is not None and step["kind"] != "skip":
        parts.append(f"{step['physics_steps']} {unit}")
    if step["kind"] == "skip" and step.get("status"):
        parts.append(f"{step['status']}, no physics")
    if step.get("note"):
        parts.append(str(step["note"]))
    return ", ".join(parts)


def render_program(program: Mapping[str, Any]) -> str:
    """``program.py``: the steps as code, one line per agent action."""
    lines = [
        f'"""Replay program of run {program["run_id"]}.',
        "",
        f"{program['backend']} {program.get('task')}, seed {program.get('seed')}, "
        f"recorded at commit {program.get('source_commit')}.",
        "Written from the run's evidence when it was sealed; program.json beside this file holds",
        "the same steps with the receipt pose each step must reproduce and the run's verdict.",
        "Each line below is one agent action, in the order the run took them.",
        "",
        f"    ora replay {program['run_id']} --check OUT    # re-execute on the simulator and check",
        '"""',
        "",
        "from ora.environment.replay_program import Program",
        "",
        "program = Program(__file__)",
    ]
    width = 0
    calls = []
    scripts = program.get("scripts") or {}
    shown: set[str] = set()
    for step in program["steps"]:
        script = step.get("script")
        if script is not None and script not in shown:  # the run_python script these steps came from
            shown.add(script)
            code = (scripts.get(script) or {}).get("code") or "(source not recorded)"
            calls.append((f"# script {script}: the code the model ran; the steps below are its actions", None))
            calls.extend((f"#     {line}".rstrip(), None) for line in str(code).splitlines())
        sequence = step["action_sequence"]
        if step["kind"] == "hold":
            call = f"program.hold({sequence}, steps={step['physics_steps']})"
        elif step["kind"] == "ticks":
            call = f"program.ticks({sequence})"
        else:
            call = f"program.{step['kind']}({sequence})"
        calls.append((call, _comment(step, "physics steps" if program["backend"] == "robotwin" else "control steps")))
        width = max(width, len(call))
    lines += [call if comment is None else f"{call.ljust(width)}  # {comment}" for call, comment in calls]
    lines += ["program.end()", ""]
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_program(run_root: Path, destination: Path | None = None) -> dict[str, Any]:
    """Write ``program.json`` and ``program.py``; default destination ``RUN/replay-program``."""
    run_root = Path(run_root).resolve()
    destination = Path(destination).resolve() if destination is not None else run_root / PROGRAM_DIRECTORY
    if destination != run_root / PROGRAM_DIRECTORY and destination.is_relative_to(run_root):
        raise ReplayProgramError("an exported replay program stays outside the run")
    if destination == run_root / PROGRAM_DIRECTORY and (run_root / "seal.json").exists():
        raise ReplayProgramError("the run is sealed: export its program with --program DIR outside the run")
    program = build_program(run_root)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_text(destination / PROGRAM_JSON, json.dumps(program, indent=1, sort_keys=True) + "\n")
    _atomic_text(destination / PROGRAM_PY, render_program(program))
    return {"path": str(destination), "status": program["status"], "steps": len(program["steps"]),
            "problems": program["problems"]}


def write_unavailable(directory: Path, run_id: str, reason: str) -> dict[str, Any]:
    """The marker of a run without a program: ``unavailable.json`` where ``program.json`` would be."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"schema": UNAVAILABLE_SCHEMA, "run_id": run_id, "reason": reason}
    _atomic_text(directory / UNAVAILABLE_JSON, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return {"path": str(directory), "status": "unavailable", "reason": reason}


def write_program_at_seal(run_root: Path) -> dict[str, Any]:
    """Seal-time export, fail-open: a run whose program cannot be written records why.

    Runs of other backends (Synthetic, scene-only play) have no recorded controls and
    get nothing.
    """
    directory = Path(run_root) / PROGRAM_DIRECTORY
    try:
        runtime = _load(Path(run_root) / "plan.json").get("runtime") or {}
    except (OSError, ValueError):
        runtime = {}
    if not isinstance(runtime, Mapping) or runtime.get("kind") not in PROGRAM_BACKENDS:
        return {"path": None, "status": "not_applicable"}
    try:
        return write_program(run_root)
    except Exception as exc:  # noqa: BLE001 - the reason is sealed instead of blocking the seal
        return write_unavailable(directory, Path(run_root).name, f"{type(exc).__name__}: {exc}")


class Program:
    """What ``program.py`` builds: its steps, then ``end()`` checks them when run as a script."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.directory = Path(path).resolve().parent
        self.recorded = _load(self.directory / PROGRAM_JSON)
        if self.recorded.get("schema") != PROGRAM_SCHEMA:
            raise ReplayProgramError(f"{self.directory / PROGRAM_JSON} is not {PROGRAM_SCHEMA}")
        self.steps: list[dict[str, Any]] = []

    def _add(self, kind: str, sequence: int, **fields: Any) -> None:
        self.steps.append({"action_sequence": int(sequence), "kind": kind, **fields})

    def controls(self, sequence: int) -> None:
        """Run action ``sequence``'s recorded controls."""
        self._add("controls", sequence)

    def hold(self, sequence: int, *, steps: int) -> None:
        """Step the scene ``steps`` times holding every drive target (a RoboTwin wait)."""
        self._add("hold", sequence, physics_steps=int(steps))

    def skip(self, sequence: int) -> None:
        """An action that moved nothing."""
        self._add("skip", sequence)

    def ticks(self, sequence: int) -> None:
        """Feed action ``sequence``'s recorded RoboDojo native ticks."""
        self._add("ticks", sequence)

    def unreplayable(self, sequence: int) -> None:
        """An action whose record cannot be reproduced; checking refuses the program."""
        self._add("unreplayable", sequence)

    def end(self) -> None:
        """Check the steps when ``program.py`` runs as a script; loading it only collects them."""
        if sys._getframe(1).f_globals.get("__name__") != "__main__":
            return
        import argparse

        parser = argparse.ArgumentParser(description=f"re-execute {self.recorded['run_id']} and check it")
        parser.add_argument("--check", type=Path, required=True, metavar="OUT", help="report directory")
        parser.add_argument("--run", type=Path, help="the run directory (default: the program's own run)")
        arguments = parser.parse_args()
        run_root = arguments.run or self.directory.parent
        report = check(run_root, arguments.check, program_dir=self.directory)
        print(json.dumps({"report": report["path"], "passed": report["passed"]}))
        raise SystemExit(0 if report["passed"] else 1)


def load_program(directory: Path) -> dict[str, Any]:
    """Execute ``program.py`` to its step list; ``recorded`` tells whether it still equals ``program.json``."""
    directory = Path(directory).resolve()
    recorded = _load(directory / PROGRAM_JSON)
    namespace = runpy.run_path(str(directory / PROGRAM_PY), run_name="ora_replay_program")
    built = namespace.get("program")
    if not isinstance(built, Program):
        raise ReplayProgramError(f"{directory / PROGRAM_PY} defines no Program named program")
    plain = [{key: step[key] for key in ("action_sequence", "kind", "physics_steps") if key in step}
             for step in built.steps]
    recorded_plain = [{key: step[key] for key in ("action_sequence", "kind")}
                      | ({"physics_steps": step["physics_steps"]} if step["kind"] == "hold" else {})
                      for step in recorded["steps"]]
    unchanged = plain == recorded_plain
    by_sequence = {step["action_sequence"]: step for step in recorded["steps"]}
    steps = []
    for step in built.steps:
        full = dict(by_sequence.get(step["action_sequence"], {}))
        full.update(step)
        if not unchanged:
            full.pop("expect", None)
        steps.append(full)
    return {**recorded, "steps": steps, "recorded": unchanged}


def _allowed_dotenv_values(names: set[str]) -> dict[str, str]:
    """Only the named variables from the checkout's ``.env`` (paths the plan binds), never anything else."""
    result: dict[str, str] = {}
    candidates = [PROJECT_ROOT / ".env"]
    common = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
                            check=False, capture_output=True, text=True)
    if common.returncode == 0:
        common_path = Path(common.stdout.strip())
        if not common_path.is_absolute():
            common_path = (PROJECT_ROOT / common_path).resolve()
        candidates.append(common_path.parent / ".env")
    for path in reversed(list(dict.fromkeys(candidates))):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() not in names:
                continue
            try:
                parts = shlex.split(value.strip(), comments=True, posix=True)
            except ValueError:
                continue
            if len(parts) == 1:
                result[name.strip()] = parts[0]
    return result


def worker_environment(plan: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """The plan's simulator interpreter and the environment it runs in."""
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("python_env"), str):
        raise ReplayProgramError("run plan omits its simulator Python binding")
    names = {runtime["python_env"]} | {
        str(runtime[field]) for field in ("root_env", "libero_pro_root_env", "data_root_env")
        if isinstance(runtime.get(field), str) and runtime.get(field)}
    environment = os.environ.copy()
    for name, value in _allowed_dotenv_values(names).items():
        environment.setdefault(name, value)
    python = environment.get(runtime["python_env"])
    if not python:
        raise ReplayProgramError(f"checking a replay program needs {runtime['python_env']}")
    binding = runtime.get("resource_binding")
    worker = binding.get("worker_environment") if isinstance(binding, Mapping) else None
    for name, value in (worker or {}).items():
        # A device the caller was given (a router lane) wins over the recording's.
        if isinstance(name, str) and isinstance(value, str):
            environment.setdefault(name, value)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    return python, environment


def _compare(program: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    failures, mismatched = [], []
    for step in measured.get("steps", []):
        for arm, comparison in (step.get("arms") or {}).items():
            if not comparison.get("bit_identical"):
                mismatched.append({"action_sequence": step["action_sequence"], "arm": arm,
                                   "position_mm": comparison.get("position_mm")})
    if mismatched:
        failures.append(f"{len(mismatched)} action end poses differ from their receipts")
    expect = program.get("expect") or {}
    keys = VERDICT_KEYS[program["backend"]]
    verdict = {"expected": {key: expect.get(key) for key in ("official_success", *keys)},
               "replayed": {key: measured.get(key) for key in ("official_success", *keys)}}
    if program.get("recorded"):
        if bool(expect.get("official_success")) != bool(measured.get("official_success")) or any(
                expect.get(key) is not None and expect.get(key) != measured.get(key) for key in keys):
            failures.append(VERDICT_DIFFERS)
    return {"pose_mismatches": mismatched, "verdict": verdict, "failures": failures}


def prepare_check(run_root: Path, output: Path, *, program_dir: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    """Validate the paths and load the program to check (written into ``OUT/program`` when the run has none)."""
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output == run_root or output.is_relative_to(run_root):
        raise ReplayProgramError("the check report stays outside the run")
    if output.exists() and any(output.iterdir()):
        raise ReplayProgramError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(program_dir) if program_dir is not None else run_root / PROGRAM_DIRECTORY
    if not (directory / PROGRAM_PY).is_file():
        directory = output / "program"
        write_program(run_root, directory)
    program = load_program(directory)
    program["directory"] = str(directory)
    if program["status"] != "replayable" or any(step["kind"] == "unreplayable" for step in program["steps"]):
        raise ReplayProgramError(f"the program is not replayable: {program.get('problems')}")
    if program["backend"] == "robodojo" and not program["recorded"]:
        raise ReplayProgramError("a RoboDojo program replays only as recorded (the control replay runs every tick)")
    return run_root, output, program


def finish_check(run_root: Path, output: Path, program: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``replay-program-check.json`` from a backend's outcome (its ``failures`` decide)."""
    report = {"schema": CHECK_SCHEMA, "run_id": Path(run_root).name, "backend": program["backend"],
              "program": program.get("directory"), "recorded_program": program["recorded"], **outcome}
    report["passed"] = not report.get("failures")
    report["path"] = str(Path(output) / "replay-program-check.json")
    _atomic_text(Path(output) / "replay-program-check.json", json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def check(run_root: Path, output: Path, *, program_dir: Path | None = None,
          timeout_s: float = CHECK_TIMEOUT_S) -> dict[str, Any]:
    """Re-execute a run's replay program on its simulator and check it (not RoboDojo).

    A RoboDojo program is checked by ``ora replay RUN --check OUT --resource-profile P``,
    which runs the qualification control replay and then ``robodojo_outcome``.
    """
    run_root, output, program = prepare_check(run_root, output, program_dir=program_dir)
    if program["backend"] == "robodojo":
        raise ReplayProgramError("check a RoboDojo program with `ora replay RUN --check OUT --resource-profile P`")
    return finish_check(run_root, output, program, controller_replay_outcome(run_root, output, program, timeout_s))


def controller_replay_outcome(run_root: Path, output: Path, program: Mapping[str, Any],
                              timeout_s: float = CHECK_TIMEOUT_S, *,
                              scene_output: Path | None = None) -> dict[str, Any]:
    """Run the program in the plan's simulator interpreter and compare it with the run.

    ``scene_output`` also writes the re-execution's scene trace there.
    """
    from .replay_worker import PROGRAM_JOB_SCHEMA

    plan = _load(run_root / "plan.json")
    job = output / "replay-program-job.json"
    worker_output = output / "worker"
    _atomic_text(job, json.dumps({"schema": PROGRAM_JOB_SCHEMA, "run_root": str(run_root),
                                  "output_root": str(worker_output), "project_root": str(PROJECT_ROOT),
                                  "program": program,
                                  **({"scene_trace": {"directory": str(scene_output)}} if scene_output else {})},
                                 indent=1, sort_keys=True) + "\n")
    if program["backend"] == "robolab":
        from .robolab_replay import replay_in_environment_worker

        returncode = replay_in_environment_worker(plan, job, worker_output, output / "worker.log", timeout_s)
    else:
        python, environment = worker_environment(plan)
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            returncode = subprocess.run([python, "-m", "ora.environment.replay_worker", "--job", str(job)],
                                        cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                        check=False, timeout=timeout_s).returncode
    result_path = worker_output / "replay-program-result.json"
    if returncode or not result_path.is_file():
        return {"failures": [f"the replay worker exited {returncode}; see worker.log"],
                "worker_returncode": returncode}
    measured = _load(result_path)
    return {"worker_returncode": 0, "steps": len(measured.get("steps", [])), **_compare(program, measured)}


def _capture_spans(run_root: Path) -> dict[tuple[int, int], tuple[int, int]]:
    """Each action's first and last simulator step in the run's execution capture, by (round, action sequence).

    A LIBERO-Pro move's OSC refinement runs as a nested action inside the move's frame, so a move that took both
    phases gets no ``action_end`` of its own: its joint phase ends where the refinement starts, and the outer frame's
    closing row names no action. A step past an action's start belongs to the action open at that moment.
    """
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    current: tuple[int, int] | None = None
    for row in _jsonl(run_root / "native" / "execution-capture" / "timeline.jsonl"):
        sequence, step, phase = row.get("action_sequence"), row.get("sim_step"), row.get("phase")
        if not isinstance(step, int) or phase not in ("action_start", "action_end"):
            continue
        key = (int(row.get("round_index") or 0), sequence) if isinstance(sequence, int) else current
        if key is None:
            continue
        if phase == "action_start" and current not in (None, key) and current[0] == key[0]:
            spans[current] = (spans[current][0], step)
        first, last = spans.get(key, (step, step))
        spans[key] = (first, step) if phase == "action_end" else (first, last)
        current = key if phase == "action_start" else current
    return spans


def _clock_failures(run_root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """The trace is drawn on the execution capture's clock: every action must span the same simulator steps."""
    if manifest.get("backend") == "robolab":  # it has no capture
        from .robolab_replay import receipt_clock_failures

        return receipt_clock_failures(_receipts(run_root / "native"), manifest)
    spans = _capture_spans(run_root)
    differing = [span["action_sequence"] for span in manifest["spans"]
                 if spans.get((span["round"], span["action_sequence"]), (span["first_step"], span["last_step"]))
                 != (span["first_step"], span["last_step"])]
    return [f"the scene trace's steps differ from the execution capture's at actions {differing}"] if differing else []


def _robodojo_scene_replay(run_root: Path, output: Path, program: Mapping[str, Any], trace: Path,
                           replay: Callable[[Path, Path, Path], Any]) -> tuple[dict[str, Any], Any]:
    """``--check``'s RoboDojo control replay with the worker writing ``trace``: its outcome and the trace's manifest."""
    motion = output / "motion-replay"
    replay(run_root, motion, trace)
    worker = motion / "pass-a" / "replay-worker.json"
    return dict(robodojo_outcome(program, motion)), _load(worker).get("scene_trace") if worker.is_file() else None


def _for_scene(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """A check's outcome as a scene trace reads it: the bits decide whose scene this is, so a verdict that reads
    differently is reported (``verdict_differs``) and is not a failure."""
    failures = outcome.get("failures", ())
    return {**outcome, "failures": [item for item in failures if item != VERDICT_DIFFERS],
            "verdict_differs": VERDICT_DIFFERS in failures}


def export_scene_trace(run_root: Path, traces: Path | None = None, *, timeout_s: float = CHECK_TIMEOUT_S,
                       robodojo_replay: Callable[[Path, Path, Path], Any] | None = None) -> dict[str, Any]:
    """Re-execute a run's recorded program (``SCENE_BACKENDS``); keep its scene trace if the check passes.

    The check is ``--check``'s, on the same re-execution; a run that diverges gets no trace, and its report stays
    in a work directory beside the traces. A different verdict alone is no divergence: the trace is kept and the
    result says ``verdict_differs``. Publishing is one rename, so a trace directory is always complete.
    ``robodojo_replay(run, output, trace)`` launches a RoboDojo run's control replay: the launcher is the
    orchestrator's, which this layer may not import, so ``ora replay`` passes it.
    """
    from .scene_trace import CHECK_REPORT, MANIFEST, traces_root

    run_root = Path(run_root).resolve()
    root = traces_root(run_root, traces)
    final = root / run_root.name
    if root == run_root or root.is_relative_to(run_root):
        raise ReplayProgramError("the scene trace stays outside the run")
    if final.exists():
        raise ReplayProgramError(f"{final} already holds this run's scene trace; remove it to export again")
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{run_root.name}.", dir=root))
    try:
        run_root, output, program = prepare_check(run_root, work / "check")
        if program["backend"] not in SCENE_BACKENDS:
            raise ReplayProgramError(f"a scene trace is written for {', '.join(SCENE_BACKENDS)} runs; "
                                     f"this run is {program['backend']}")
        if not program["recorded"]:
            raise ReplayProgramError("program.py differs from its recording: its scene is not this run's")
        if program["backend"] == "robodojo" and robodojo_replay is None:
            raise ReplayProgramError("a RoboDojo scene trace needs --resource-profile (a qualified RoboDojo slot)")
        trace = work / "trace"
        if program["backend"] == "robodojo":
            outcome, manifest = _robodojo_scene_replay(run_root, output, program, trace, robodojo_replay)
            outcome = _for_scene(outcome)
        else:
            outcome = _for_scene(controller_replay_outcome(run_root, output, program, timeout_s, scene_output=trace))
            measured = _load(output / "worker" / "replay-program-result.json") if not outcome.get("failures") else {}
            manifest = measured.get("scene_trace")
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if not outcome.get("failures"):
        outcome["failures"] = _clock_failures(run_root, manifest) if manifest else ["the worker wrote no scene trace"]
    report = finish_check(run_root, output, program, outcome)
    if not report["passed"]:
        shutil.rmtree(trace, ignore_errors=True)
        return {"scene_trace": None, "passed": False, "failures": report["failures"], "report": report["path"]}
    shutil.copy2(report["path"], trace / CHECK_REPORT)
    os.replace(trace, final)
    shutil.rmtree(work, ignore_errors=True)
    return {"scene_trace": str(final), "passed": True, "failures": [], "verdict_differs": outcome["verdict_differs"],
            "verdict": outcome.get("verdict"), "report": str(final / CHECK_REPORT), "rows": manifest["shape"][0], "bodies": manifest["shape"][1],
            "bytes": sum(path.stat().st_size for path in final.rglob("*") if path.is_file()),
            "manifest": str(final / MANIFEST)}


def robodojo_outcome(program: Mapping[str, Any], motion_output: Path) -> dict[str, Any]:
    """Compare a RoboDojo control replay (``ora qualify robodojo motion-replay`` output) with the run."""
    report_path = Path(motion_output) / "robodojo-motion-replay-report.json"
    if not report_path.is_file():
        return {"failures": ["the RoboDojo control replay wrote no report"]}
    replay = _load(report_path)
    failures = [f"control replay: {item}" for item in replay.get("failures", [])]
    worker_path = Path(motion_output) / "pass-a" / "replay-worker.json"
    rounds = (_load(worker_path).get("replay") or {}).get("rounds") if worker_path.is_file() else None
    expect = program.get("expect") or {}
    verdict: dict[str, Any] = {"expected": expect, "replayed": None}
    if rounds:
        last = rounds[-1]
        official = last.get("official_terminal") or {}
        terminal = bool(last.get("terminal_after_round"))
        verdict["replayed"] = {"terminal": terminal, "official_success": official.get("official_success"),
                               "native_actions": official.get("native_actions")}
        if terminal != bool(expect.get("terminal")) or (terminal and (
                official.get("official_success") != expect.get("official_success")
                or official.get("native_actions") != expect.get("native_actions"))):
            failures.append(VERDICT_DIFFERS)
    else:
        failures.append("the RoboDojo control replay recorded no rounds")
    divergence = ((replay.get("passes") or {}).get("pass-a") or {}).get("divergence_vs_original")
    return {"motion_replay_report": str(report_path), "divergence": divergence, "verdict": verdict,
            "failures": failures}


__all__ = [
    "CHECK_SCHEMA", "PROGRAM_BACKENDS", "PROGRAM_DIRECTORY", "PROGRAM_SCHEMA", "Program", "ReplayProgramError",
    "ReplayProgramUnavailable", "SCENE_BACKENDS",
    "build_program", "check", "controller_replay_outcome", "export_scene_trace", "finish_check", "load_program",
    "prepare_check",
    "render_program", "robodojo_outcome", "worker_environment", "write_program", "write_program_at_seal",
    "write_unavailable",
]
