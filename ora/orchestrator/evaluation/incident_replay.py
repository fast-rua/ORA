"""``ora replay RUN --incident N``: one bad action of a sealed run, again on this checkout.

The tool replays a run up to the action that went wrong and runs that action again, with no
model. Two branches start from the same replayed state:

* ``recorded`` runs the action's recorded controls, so its report shows what happened then,
  including the private scene poses a receipt never held;
* ``current`` submits the action's recorded target to this checkout's motion stack.

**RoboTwin and LIBERO-Pro** run in ``environment/incident_replay_worker.py``. The session is the
one a live episode builds, from the runtime of ``--stack-run``: a run sealed on this checkout,
such as a merge smoke, because a run's own plan pins the motion stack it ran on. The episode's
identity (task, seed, scene and simulator revisions) stays the source run's, and differing
simulator revisions are refused: that prefix could not be reproduced.

**RoboDojo** goes through ``ora toolkit replay --live --execute-action``, which restores the
recorded native ticks and calls the tool at the run's parent call. It needs a toolkit run
(``native/toolkit-io``) and has no ``recorded`` branch here; the sealed receipt stands for it.
The tool runs on what a fresh run of this checkout uses (:func:`robodojo_stack`): the default
baseline's tool surface, planner and timing, whatever the source run's plan pinned. The report's
``stack`` names them, and its branch's ``motion`` says what the worker really used.

Nothing here reaches a policy. The report may hold private scene poses, so it stays under the
caller's output, outside the run. It is evidence for a developer, not a task result: a
``current`` branch that was not refused says what one move did from one state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.incident_replay_worker import BRANCHES, JOB_SCHEMA, scene_changes
from ora.environment.replay_program import PROJECT_ROOT, build_program, worker_environment

REPORT_SCHEMA = "ora.incident-replay.v1"
REPORT_NAME = "incident-replay.json"
BRANCH_TIMEOUT_S = 60 * 60
#: Runtime fields that name the episode rather than the motion stack, kept from the source run.
EPISODE_FIELDS = {
    "libero_pro": ("suite", "task_id", "robustness_case", "expected_task_language", "workspace_bounds_m",
                   "workspace_profile", "public_environment_context", "environment_driver",
                   "environment_prompt_profile"),
    "robotwin": ("task", "task_config", "workspace_bounds_m", "public_environment_context",
                 "environment_prompt_profile", "task_instruction_source"),
}
#: Simulator revisions the prefix depends on: a stack run on another revision cannot reproduce it.
REVISION_FIELDS = {
    "libero_pro": ("required_libero_pro_revision", "required_data_revision"),
    "robotwin": ("required_revision",),
}
#: The runtime fields of RoboDojo's motion stack, taken from the default baseline.
ROBODOJO_STACK_FIELDS = ("motion_planner", "motion_timing")
#: Tool arguments of older surfaces that today's RoboDojo tools refuse; the move is the same without them.
RETIRED_ARGUMENTS = frozenset({"native_action_budget"})
#: Receipt fields a report shows side by side.
RECEIPT_FIELDS = ("action", "execution_status", "failure_code", "failure_stage", "not_reached_reason",
                  "motion_strategy", "state_changed", "plan_rejections", "after")


class IncidentReplayError(RuntimeError):
    """An incident cannot be replayed from this run."""


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _receipts(run_root: Path, before: int) -> list[dict[str, Any]]:
    return [_load(path) for path in sorted((run_root / "native/receipts").glob("action-*.json"))
            if int(path.stem.split("-")[1]) < before]


def _previous_record(run_root: Path, sequence: int) -> dict[str, Any] | None:
    """The record a session keeps of the action before ``sequence``: path, hash and receipt."""

    if sequence == 0:
        return None
    relative = f"receipts/action-{sequence - 1:04d}.json"
    path = run_root / "native" / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "receipt": _load(path)}


def _receipt(run_root: Path, sequence: int) -> dict[str, Any]:
    path = run_root / f"native/receipts/action-{sequence:04d}.json"
    if not path.is_file():
        raise IncidentReplayError(f"{run_root.name} has no receipt for action {sequence}")
    return _load(path)


def incident_action(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The action to submit again: what the receipt says was asked, or why it cannot be asked twice."""

    action, arguments = receipt.get("action"), receipt.get("arguments")
    if action == "recover_motion":
        raise IncidentReplayError("recover_motion returns to recorded poses the replay does not restore")
    if not isinstance(arguments, Mapping) or "targets" not in arguments:
        raise IncidentReplayError(f"action {action!r} has no target to submit again (a plan_ref names a dead probe)")
    return {"action_id": str(receipt["action_id"]).split(":", 1)[0], "action": action, "arguments": dict(arguments)}


def stack_runtime(source: Mapping[str, Any], stack: Mapping[str, Any]) -> dict[str, Any]:
    """The stack run's runtime holding the source run's episode; refuses another simulator revision."""

    kind = source["runtime"].get("kind")
    if kind not in EPISODE_FIELDS:
        raise IncidentReplayError(f"no incident replay for backend {kind!r}")
    if stack["runtime"].get("kind") != kind:
        raise IncidentReplayError(f"--stack-run is a {stack['runtime'].get('kind')} run, not {kind}")
    for field in REVISION_FIELDS[kind]:
        if source["runtime"].get(field) != stack["runtime"].get(field):
            raise IncidentReplayError(f"runtime.{field} differs from the stack run's: the prefix cannot be reproduced")
    runtime = dict(stack["runtime"])
    for field in EPISODE_FIELDS[kind]:
        if field in source["runtime"]:
            runtime[field] = source["runtime"][field]
        else:
            runtime.pop(field, None)
    return {**runtime, "execution_capture": {"enabled": False}, "operator_execution_video": False}


def _worker_config(*, source: Mapping[str, Any], stack: Mapping[str, Any], stack_root: Path,
                   output: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    """What the session factory hands a live worker for this episode on the stack run's motion stack."""

    from ora.environment.factory import build_environment_session
    from ora.orchestrator.backends.registry import backend

    runtime = stack_runtime(source, stack)
    bind = backend(str(runtime["kind"])).supervisor_binding
    native = output / "native"
    native.mkdir(parents=True)
    shutil.copyfile(stack_root / "preflight.json", output / "preflight.json")  # the worker binds its timing from it
    built = build_environment_session(
        runtime_config=runtime, output_root=native, task=str(source["task"]), seed=int(source["seed"]),
        max_sim_steps=int(source["budgets"]["max_sim_steps"]), action_codec=str(stack["policy"]["action_codec"]),
        qualified_runtime_binding=None if bind is None else bind(stack_root, runtime),
    )
    config = {**built.delegate.worker_config, "shared_root": str(output), "operator_world_video": False}
    python, environment = worker_environment({**source, "runtime": runtime})
    return config, python, environment


def _summary(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if receipt is None else {key: receipt[key] for key in RECEIPT_FIELDS if key in receipt}


def _prefix_report(result: Mapping[str, Any]) -> dict[str, Any] | None:
    prefix = result.get("prefix")
    if not isinstance(prefix, Mapping):
        return None
    arms = [(row["action_sequence"], arm, comparison) for row in prefix["steps"]
            for arm, comparison in row["arms"].items()]
    different = [{"action_sequence": sequence, "arm": arm, "position_mm": comparison["position_mm"]}
                 for sequence, arm, comparison in arms if not comparison["bit_identical"]]
    return {"actions": len(prefix["steps"]), "physics_steps": prefix["physics_steps"], "poses_checked": len(arms),
            "bit_identical": not different, "different": different}


def _branch(run_root: Path, output: Path, branch: str, *, job: Mapping[str, Any], python: str,
            environment: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    job_path = output / "incident-job.json"
    job_path.write_text(json.dumps({**job, "branch": branch}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [python, "-m", "ora.environment.incident_replay_worker", "--job", str(job_path)], cwd=PROJECT_ROOT,
            env=dict(environment), stdout=log, stderr=subprocess.STDOUT, check=False, timeout=timeout_s)
    path = output / "incident-result.json"
    if not path.is_file():
        return {"error": f"the worker exited {completed.returncode} with no result; see {output / 'worker.log'}"}
    result = _load(path)
    return {key: result[key] for key in ("error", "ran", "arms", "refused", "elapsed_s", "synchronised", "scene",
                                         "scene_error") if result.get(key) is not None} | {
        "prefix": _prefix_report(result), "receipt": _summary(result.get("receipt")),
        "result": str(path)}


def _source_program(run_root: Path, sequence: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    program = build_program(run_root)
    if program["status"] != "replayable":
        raise IncidentReplayError(f"the run's program is not replayable: {program['problems']}")
    steps = program["steps"]
    index = next((position for position, step in enumerate(steps) if step["action_sequence"] == sequence), None)
    if index is None:
        raise IncidentReplayError(f"{run_root.name} records no action {sequence}")
    return steps[:index], steps[index]


def _simulator_incident(run_root: Path, output: Path, sequence: int, *, stack_root: Path | None,
                        branches: Sequence[str], timeout_s: float) -> dict[str, Any]:
    if stack_root is None:
        raise IncidentReplayError("a RoboTwin or LIBERO-Pro incident needs --stack-run, a run sealed on this checkout")
    source, stack = _load(run_root / "plan.json"), _load(stack_root / "plan.json")
    prefix, incident = _source_program(run_root, sequence)
    original = _receipt(run_root, sequence)
    job = {"schema": JOB_SCHEMA, "run_root": str(run_root), "action_sequence": sequence, "prefix": prefix,
           "incident": incident, "action": incident_action(original), "receipts": _receipts(run_root, sequence),
           "previous_receipt": _previous_record(run_root, sequence)}
    report: dict[str, Any] = {
        "stack_run": stack_root.name, "stack_commit": (stack.get("source") or {}).get("commit"),
        "original": _summary(original), "branches": {},
    }
    for branch in branches:
        folder = output / branch
        config, python, environment = _worker_config(source=source, stack=stack, stack_root=stack_root, output=folder)
        report["branches"][branch] = _branch(
            run_root, folder, branch, job={**job, "worker_config": config, "output_root": str(folder)},
            python=python, environment=environment, timeout_s=timeout_s)
    return report


def _parent_call(run_root: Path, action_id: str) -> tuple[str, str, dict[str, Any]]:
    """The tool call that proposed ``action_id``: its id, tool name and arguments."""

    calls: dict[str, Mapping[str, Any]] = {}
    parent = None
    for line in (run_root / "agent-events.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        payload = row.get("payload") or {}
        if row.get("type") == "tool_call_proposed":
            calls[str(row.get("tool_call_id"))] = payload
        elif row.get("type") == "action_proposed" and payload.get("action_id") == action_id:
            parent = str(row.get("tool_call_id"))
            break
    if parent is None or parent not in calls:
        raise IncidentReplayError(f"no tool call proposed {action_id}")
    arguments = {key: value for key, value in (calls[parent].get("arguments") or {}).items()
                 if key not in RETIRED_ARGUMENTS}
    return parent, str(calls[parent].get("name")), arguments


def robodojo_stack() -> dict[str, Any]:
    """What a fresh RoboDojo run of this checkout moves with, from the default baseline."""

    from ora.agent.composition.surface.toolkit_surface import declared_required_grants
    from ora.orchestrator.spec.inputs import default_instance

    catalog, instance = default_instance("robodojo", PROJECT_ROOT)
    runtime, surface = instance["runtime"], instance["tool_surface"]
    return {"baseline": catalog["id"], "tool_surface": surface, "toolkit_grants": declared_required_grants(surface),
            "runtime": {field: runtime[field] for field in ROBODOJO_STACK_FIELDS if field in runtime}}


def _implementation(configured: Any) -> str | None:
    return configured.get("implementation") if isinstance(configured, Mapping) else None


def _stack_report(stack: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """The stack the action was given and the one its run was sealed on, side by side."""

    runtime = source["runtime"]
    return {"baseline": stack["baseline"], "tool_surface": stack["tool_surface"],
            "motion_planner": _implementation(stack["runtime"].get("motion_planner")),
            "motion_timing": _implementation(stack["runtime"].get("motion_timing")),
            "source": {"tool_surface": ((source.get("policy") or {}).get("tool_surface") or {}).get("profile"),
                       "motion_planner": _implementation(runtime.get("motion_planner")),
                       "motion_timing": _implementation(runtime.get("motion_timing"))}}


def _robodojo_preview(run_root: Path, output: Path, call: str, tool: str, arguments: Mapping[str, Any],
                      resource_profile: str, stack: Mapping[str, Any]) -> dict[str, Any]:
    """What a dry run of the action shows on the state it ran from, its picture beside the report; nothing moves."""
    import base64

    from ora.agent.toolkit.interface import dry_run
    from ora.orchestrator.toolkit.replay import experiment

    with dry_run.dry():
        result = experiment(source=run_root, at=call, tool=tool, arguments=arguments, output=output / "preview",
                            live=True, resource_profile=resource_profile, execute_action=False, motion_stack=stack)
    if result.get("status") == "error":
        raise IncidentReplayError(f"the preview did not run: {result['error']['message']}")
    record = dict(result.get("result") or {})
    images = record.pop("images_base64", None) or []
    if images:
        (output / "preview.png").write_bytes(base64.b64decode(images[0]))
    return {**record, **({"png": "preview.png"} if images else {})}


def _robodojo_incident(run_root: Path, output: Path, sequence: int, *, resource_profile: str | None,
                       preview: bool = False) -> dict[str, Any]:
    from ora.orchestrator.toolkit.replay import experiment

    if not resource_profile:
        raise IncidentReplayError("a RoboDojo incident needs --resource-profile (a qualified RoboDojo slot)")
    if not (run_root / "native/toolkit-io").is_dir():
        raise IncidentReplayError("a RoboDojo incident replays only from a toolkit run (native/toolkit-io)")
    original = _receipt(run_root, sequence)
    action_id = str(original.get("action_id")).split(":", 1)[0]
    call, tool, arguments = _parent_call(run_root, action_id)
    stack = robodojo_stack()
    if preview:
        return {"original": _summary(original), "parent_call": call, "tool": tool, "arguments": arguments,
                "stack": _stack_report(stack, _load(run_root / "plan.json")), "branches": {},
                "preview": _robodojo_preview(run_root, output, call, tool, arguments, resource_profile, stack)}
    result = experiment(source=run_root, at=call, tool=tool, arguments=arguments, output=output / "current",
                        live=True, resource_profile=resource_profile, execute_action=True, motion_stack=stack)
    return {"original": _summary(original), "parent_call": call, "tool": tool, "arguments": arguments,
            "stack": _stack_report(stack, _load(run_root / "plan.json")),
            "branches": {"current": _robodojo_branch(result, output / "current")}}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [] if not path.is_file() else [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _motion_used(output: Path) -> dict[str, Any]:
    """What the worker moved with, from its own records: ``planner`` is null when none started."""

    root = output / "live-worker/environment/motion-planner"
    binding = root / "binding.json"
    surface = output / "surface-compatibility.json"
    return {"tool_surface": (_load(surface).get("current") or {}).get("profile_id") if surface.is_file() else None,
            "planner": _load(binding).get("implementation") if binding.is_file() else None,
            "plans": [{key: row.get(key) for key in ("arm", "status", "reason", "error") if key in row}
                      for row in _rows(root / "plans.jsonl")],
            "timing": sorted({str((row.get("timing") or {}).get("implementation"))
                              for row in _rows(root / "timing.jsonl")})}


def _robodojo_branch(result: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """What one ``ora toolkit replay`` result says: restored state, the tool's outcome, the action's receipts."""

    def poses(objects: Any) -> dict[str, list[float]] | None:
        return None if not isinstance(objects, Mapping) else {
            name: [*pose["position_m"], *pose["orientation"]] for name, pose in objects.items()}

    verification = output / "verification.json"
    execution = result.get("execution") if isinstance(result.get("execution"), Mapping) else {}
    prefix = _load(verification) if verification.is_file() else None
    state = execution.get("state") if isinstance(execution.get("state"), Mapping) else {}
    return {
        "status": result.get("status"), "error": result.get("error"),
        "prefix": None if prefix is None else {"bit_identical": prefix.get("passed"),
                                               "physics": prefix.get("physics_prefix")},
        "receipts": [_summary(item.get("receipt", item)) for item in execution.get("receipts") or []],
        "check_task": execution.get("check_task"), "result": str(output / "result.json"),
        "motion": _motion_used(output),
        "scene": scene_changes(poses(execution.get("objects_before")), poses(state.get("objects"))),
    }


def _commit() -> str | None:
    done = subprocess.run(["git", "-C", os.fspath(PROJECT_ROOT), "rev-parse", "HEAD"], check=False,
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def replay_incident(run_root: Path, output: Path, sequence: int, *, stack_run: Path | None = None,
                    branches: Sequence[str] = BRANCHES, resource_profile: str | None = None,
                    timeout_s: float = BRANCH_TIMEOUT_S, preview: bool = False) -> dict[str, Any]:
    """Replay ``run_root`` up to action ``sequence`` and write ``OUT/incident-replay.json``.

    ``preview`` (RoboDojo) asks for the action's dry run at that state instead of running it."""

    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output == run_root or output.is_relative_to(run_root):
        raise IncidentReplayError("the report stays outside the run")
    if output.exists() and any(output.iterdir()):
        raise IncidentReplayError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    kind = _load(run_root / "plan.json")["runtime"].get("kind")
    stack_root = None if stack_run is None else Path(stack_run).resolve()
    if kind == "robodojo":
        body = _robodojo_incident(run_root, output, sequence, resource_profile=resource_profile, preview=preview)
    else:
        body = _simulator_incident(run_root, output, sequence, stack_root=stack_root, branches=branches,
                                   timeout_s=timeout_s)
    report = {"schema": REPORT_SCHEMA, "run_id": run_root.name, "backend": kind, "action_sequence": sequence,
              "checkout": os.fspath(PROJECT_ROOT), "checkout_commit": _commit(), **body}
    report["path"] = str(output / REPORT_NAME)
    (output / REPORT_NAME).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["BRANCHES", "EPISODE_FIELDS", "IncidentReplayError", "REPORT_NAME", "REPORT_SCHEMA", "incident_action",
           "replay_incident", "robodojo_stack", "stack_runtime"]
