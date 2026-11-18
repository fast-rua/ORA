"""Seed-0 task-contract witnesses through the real RoboLab worker; fills the reviews' witness checks.

    ora qualify robolab contract-witness --task T [--task T ...] --gpu N --evidence-root DIR
    ora qualify robolab contract-witness --pending [--shard I/N] --gpu N --evidence-root DIR

For each task, one Isaac Sim worker (``factory.build_robolab_worker_session``, no planner, no
policy, no provider) starts on the card, resets seed 0 and runs the witness plan of the task's
review in its ``start`` (``environment/backends/robolab/contract_witness.py``), then closes as a
run's worker does. Tasks run one after another: one Kit process per card. The record lands at
``DIR/outputs/ora-backends/task-contract-witness/robolab/<Task>-s0-witness.json``; the review in
this checkout gets that path and sha256 as evidence and its ``success_witness``,
``counterexamples`` and ``public_observability`` checks (and ``assets_and_frames`` for the upright
family) from the cases. A task solved at reset, or one whose worker fails, is recorded as blocked.
Nothing here promotes a review: ``ora dev robolab-tasks promote`` does, once all nine checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.orchestrator.paths import PROJECT_ROOT

from .task_catalog import REGISTRY, _json_text, _load_json

JOB_SCHEMA = "ora.robolab-witness-job.v1"
WITNESS_DIRECTORY = "outputs/ora-backends/task-contract-witness/robolab"
EVIDENCE_ROLE = "private_success_witness_and_counterexamples"


def review_path(task: str, *, root: Path = PROJECT_ROOT) -> str:
    registry = _load_json(root / REGISTRY)
    entry = registry.get("pending_review", {}).get(task) or registry["tasks"].get(task)
    if entry is None:
        raise SystemExit(f"{task} has no review in {REGISTRY}")
    return entry["semantic_review"]


def worker_runtime(gpu: int, job: Path) -> dict[str, Any]:
    """A run's worker inputs without the planner: the witness moves nothing through the motion stack."""
    return {"root_env": "ROBOLAB_ROOT", "python_env": "ROBOLAB_PYTHON",
            "graphics_runtime_env": "ROBOLAB_GRAPHICS_RUNTIME", "renderer_gpu": gpu,
            "startup_timeout_s": 900, "request_timeout_s": 3600, "close_timeout_s": 120,
            "contract_witness_job": str(job)}


def run_task(task: str, *, gpu: int, evidence_root: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    from ora.environment.factory import build_robolab_worker_session

    review = _load_json(root / review_path(task, root=root))
    plan = review.get("witness_plan")
    if not isinstance(plan, Mapping):
        raise SystemExit(f"{task}'s review has no witness_plan")
    directory = evidence_root / WITNESS_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{task}-s0-witness.json"
    job = directory / f"{task}-s0-job.json"
    job.write_text(json.dumps({"schema": JOB_SCHEMA, "task": task, "seed": int(plan.get("seed", 0)),
                               "plan": plan, "output": str(output)}, indent=1) + "\n", encoding="utf-8")
    if output.exists():
        output.rename(output.with_suffix(f".replaced-{int(time.time())}.json"))
    session = build_robolab_worker_session(runtime=worker_runtime(gpu, job), output_root=directory / f"{task}-s0-session",
                                           task=task, seed=int(plan.get("seed", 0)), action_codec="motion_action_v3")
    error = None
    try:
        session.start()
    except Exception as exc:  # noqa: BLE001 - a worker that cannot run the plan is a recorded outcome
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            error = error or f"close: {type(exc).__name__}: {exc}"
    if not output.is_file():
        return {"task": task, "error": error or "the worker wrote no witness record"}
    return {"task": task, "path": output, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "record": _load_json(output), "error": error}


def fill_review(review: dict[str, Any], result: Mapping[str, Any], *, evidence_root: Path) -> dict[str, Any]:
    """The review with the witness's evidence and the checks it settles."""
    review = dict(review)
    checks = {key: dict(value) for key, value in review["checks"].items()}
    witness_result: dict[str, Any] = {"run_unix_s": time.time()}
    if "record" not in result:
        reason = result.get("error") or "no record"
        blocked = "solved_at_reset" if "already solved at reset" in reason else "witness_worker_failed"
        witness_result["blocked"] = [{"code": blocked, "detail": reason[:600]}]
        for key in ("success_witness", "counterexamples"):
            checks[key] = {"status": "failed", "basis": f"Witness did not run: {reason[:300]}"}
        review.update(checks=checks, witness_result=witness_result)
        return _block(review, witness_result)
    record, path = result["record"], Path(result["path"])
    try:  # the record lies under the root as named, even where a directory below it is a symbolic link
        relative = path.relative_to(evidence_root).as_posix()
    except ValueError:
        relative = path.resolve().relative_to(evidence_root.resolve()).as_posix()
    cases = record["cases"]
    positives = [case for case in cases if case["expected"]]
    negatives = [case for case in cases if not case["expected"]]
    reset = next((case for case in cases if case["case"] == "reset_state"), None)
    if reset is not None and reset.get("official_success"):
        witness_result["blocked"] = [{"code": "solved_at_reset", "detail": "the reset state already satisfies "
                                      "the success termination"}]

    def summary(items: Sequence[Mapping[str, Any]]) -> str:
        return "; ".join(f"{case['case']} -> {case.get('official_success')}" + (
            f" ({case['error']})" if case.get("error") else "") for case in items)

    sha = result["sha256"]
    checks["success_witness"] = {
        "status": "passed" if positives and all(case["matches"] for case in positives) else "failed",
        "basis": f"Witness (seed 0, sha256 {sha[:12]}): {summary(positives)}; each case from a fresh reset, objects "
                 f"placed by direct state edits with the robot held."}
    checks["counterexamples"] = {
        "status": "passed" if negatives and all(case["matches"] for case in negatives) else "failed",
        "basis": f"Witness (seed 0, sha256 {sha[:12]}): {summary(negatives)}."}
    if review.get("family") == "upright":
        checks["assets_and_frames"] = {
            "status": checks["success_witness"]["status"],
            "basis": "Upright is the object's local +z against world up; the witness's upright pose (identity "
                     "orientation) " + ("gives the success termination." if checks["success_witness"]["status"]
                                        == "passed" else "does not give success: the asset's up axis needs review.")}
    visibility = record.get("visibility_at_reset") or {}
    named = [name for name, item in visibility.items()]
    hidden = sorted(name for name, item in visibility.items() if not item.get("visible"))
    if visibility and not hidden:
        checks["public_observability"] = {"status": "passed", "basis": (
            f"At reset every named object ({len(named)}) projects into the over-shoulder or wrist camera with a "
            f"depth reading that does not hide it (witness sha256 {sha[:12]}).")}
    else:
        checks["public_observability"] = {"status": "pending", "basis": (
            "Visibility probe at reset: " + (f"not seen in either public camera: {hidden}" if hidden else "no probe")
            + ". A reviewer decides from the reset images whether the clauses stay observable.")}
    executed = [{"case": f"{case['case']}: {'false' if not case.get('official_success') else 'true'}.",
                 "expected": f"witness case {case['case']}"} for case in negatives if case["matches"]]
    kept = [item for item in review.get("counterexamples", []) if item.get("expected") == "source-derived"]
    witness_result.update(all_cases_match=record["all_cases_match"], path=relative, sha256=sha,
                          plan_sha256=record.get("plan_sha256"))
    review.update(checks=checks, counterexamples=executed + kept, witness_result=witness_result,
                  evidence=[{"role": EVIDENCE_ROLE, "path": relative, "sha256": sha}])
    return _block(review, witness_result)


#: Blockers a witness run records; a later witness of the same task replaces them.
WITNESS_BLOCKERS = ("witness_worker_failed", "solved_at_reset")


def _block(review: dict[str, Any], witness_result: Mapping[str, Any]) -> dict[str, Any]:
    """This run's witness blockers replace an earlier run's in the review's compatibility, as regeneration
    would merge them; a source blocker (from the task audit) stays."""
    blocked = list(witness_result.get("blocked") or [])
    compatibility = dict(review.get("compatibility") or {"notes": [], "blocked": []})
    kept = [item for item in compatibility.get("blocked", []) if item.get("code") not in WITNESS_BLOCKERS]
    if not blocked and len(kept) == len(compatibility.get("blocked", [])):
        return review
    compatibility["blocked"] = kept + blocked
    if compatibility["blocked"]:
        compatibility["status"] = "blocked"
    else:
        compatibility["status"] = "ready_with_note" if any(  # task_review.compatibility's rule
            note.get("code") != "reset_relation" for note in compatibility.get("notes", [])) else "ready"
    review["compatibility"] = compatibility
    return review


def _shard(tasks: list[str], spec: str | None) -> list[str]:
    if not spec:
        return tasks
    index, count = (int(part) for part in spec.split("/"))
    if not 0 <= index < count:
        raise SystemExit("--shard is I/N with 0 <= I < N")
    return tasks[index::count]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ora qualify robolab contract-witness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", default=[], help="a task to witness (repeatable)")
    parser.add_argument("--pending", action="store_true", help="every task in the registry's pending_review")
    parser.add_argument("--shard", help="I/N: every N-th task from the I-th, to split --pending across cards")
    parser.add_argument("--gpu", type=int, default=None,
                        help="the renderer card (default: the single CUDA_VISIBLE_DEVICES entry)")
    parser.add_argument("--evidence-root", type=Path, required=True,
                        help="the root the reviews' evidence paths are relative to (records go to "
                             f"DIR/{WITNESS_DIRECTORY})")
    parser.add_argument("--no-update-reviews", action="store_true", help="write the records only")
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    tasks = list(args.task)
    if args.pending:
        tasks += sorted(_load_json(root / REGISTRY).get("pending_review", {}))
    tasks = _shard(list(dict.fromkeys(tasks)), args.shard)
    if not tasks:
        parser.error("name --task or --pending")
    gpu = args.gpu
    if gpu is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit():
            parser.error("--gpu is required unless CUDA_VISIBLE_DEVICES names exactly one card")
        gpu = int(visible)
    evidence_root = args.evidence_root.expanduser().resolve()
    failures = 0
    for task in tasks:
        started = time.time()
        result = run_task(task, gpu=gpu, evidence_root=evidence_root, root=root)
        if not args.no_update_reviews:
            path = root / review_path(task, root=root)
            path.write_text(_json_text(fill_review(_load_json(path), result, evidence_root=evidence_root)),
                            encoding="utf-8")
        record = result.get("record") or {}
        ok = bool(record.get("all_cases_match")) and not result.get("error")
        failures += not ok
        print(json.dumps({"task": task, "all_cases_match": record.get("all_cases_match"), "error": result.get("error"),
                          "sha256": result.get("sha256"), "seconds": round(time.time() - started, 1),
                          "cases": [(case["case"], case.get("official_success"), case["matches"])
                                    for case in record.get("cases", [])]}), flush=True)
    return 1 if failures else 0


__all__ = ["JOB_SCHEMA", "fill_review", "main", "run_task", "worker_runtime"]
