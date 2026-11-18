"""Play-scene audit: qualify ORA's own RoboTwin free-play scenes on this host (``robotwin_play_scene_audit``).

    ./scripts/ora play audit --env robotwin --scene-set tabletop-v1 \\
        [--scene NAME ...] [--seeds N | --seed-list S ...] [--host HOST] [--resource-profile SLOT]

``ora play`` offers a scene of a scene set (``ora/environment/backends/robotwin/
play_scenes``) only at a seed a local audit admits, as it offers an official
RoboTwin task only at a seed a local seed manifest admits. The scenes have no
task, expert or predicate, so the seed scan's oracle has nothing to run; the
audit asks what free play needs instead. For each seed it builds the scene
in the RoboTwin worker interpreter on the host's RoboTwin slot (one process
per scene, one scene at a time) and records that it builds, settles (objects
at rest, on the table, inside the arms' workspace, not interpenetrating) and
renders every camera (``backends/robotwin/play_scene_audit.py``).

Each scene's record (``ora.robotwin-play-scene-audit.v1``) is written to
``outputs/ora-play/qualification/play-scene-audits/<UTC>-<ns>/<set>/<scene>.json``
with the seed evidence and camera images beside it. It binds the pinned
RoboTwin revision (the checkout must be at it and clean), the scene set's
source binding (the sha256 of the ORA files that define its scenes, which
must be committed) and the scene configuration; ``audit_sha256`` covers the
rest. ``play_scene_admission`` is the one admission rule, applied by
``play.scenes`` when it lists scenes and by launch preflight when it re-admits
the plan's seeds: a changed scene module, another revision or a tampered
record admits nothing. No expert runs, no robot action is published and no
verdict exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.environment.benchmarks.robotwin_play_scenes import (
    PlaySceneSetError, scene_definition, scene_definitions, scene_sets, source_binding, source_files,
)

from ora.orchestrator.paths import PROJECT_ROOT, resolve_outputs_root, resolve_workspace_root
from ora.orchestrator.evaluation.evidence_io import digest

AUDIT_SCHEMA = "ora.robotwin-play-scene-audit.v1"
KIND = "robotwin_play_scene_audit"
DIRECTORY = "qualification/play-scene-audits"
DEFAULT_SEEDS = 3
#: A scene's worker process builds every seed in turn; RoboTwin's stability check takes ~10 s a seed.
WORKER_TIMEOUT_S_PER_SEED = 300


def audit_root(*, root: Path = PROJECT_ROOT) -> Path:
    """Where this host's play-scene audits live: ``<outputs root>/ora-play/qualification/play-scene-audits``."""
    return resolve_outputs_root(resolve_workspace_root(root), project_root=root) / "ora-play" / DIRECTORY


def play_scene_admission(record: Mapping[str, Any], *, scene_set: str | None, scene: str, seed: int,
                         task_config: str | None, revision: str | None,
                         root: Path = PROJECT_ROOT) -> tuple[bool, dict[str, Any]]:
    """Whether an audit record admits ``seed`` of ``scene``, and the evidence for the verdict.

    The record must be an intact audit of this scene of this set, at this
    scene configuration and revision, made from the scene sources as they are
    now; the seed must be accepted and its own evidence passed.
    """
    body = {key: value for key, value in record.items() if key != "audit_sha256"}
    digest_valid = digest(body) == record.get("audit_sha256")
    try:
        current = source_binding(str(scene_set), root=root)["sha256"]
    except (PlaySceneSetError, OSError):
        current = None
    seeds = record.get("seeds") if isinstance(record.get("seeds"), Mapping) else {}
    evidence = seeds.get(str(seed)) if isinstance(seeds.get(str(seed)), Mapping) else {}
    recorded = (record.get("scene_source") or {}).get("sha256") if isinstance(record.get("scene_source"),
                                                                              Mapping) else None
    accepted = isinstance(record.get("accepted_seeds"), list) and seed in record["accepted_seeds"]
    passed = (
        record.get("schema") == AUDIT_SCHEMA and record.get("kind") == KIND
        and scene_set is not None and record.get("scene_set") == scene_set and record.get("scene") == scene
        and record.get("task_config") == task_config and record.get("robotwin_revision") == revision
        and digest_valid and current is not None and recorded == current
        and accepted and evidence.get("passed") is True
    )
    return passed, {
        "schema": record.get("schema"), "kind": record.get("kind"), "scene_set": record.get("scene_set"),
        "expected_scene_set": scene_set, "scene": record.get("scene"), "task_config": record.get("task_config"),
        "expected_task_config": task_config, "robotwin_revision": record.get("robotwin_revision"),
        "expected_revision": revision, "audit_sha256_valid": digest_valid,
        "scene_source_sha256": recorded, "current_scene_source_sha256": current, "accepted": accepted,
        "seed": {key: evidence.get(key) for key in ("passed", "failures", "object_count")},
    }


def build_record(*, scene_set: str, scene: str, task_config: str, revision: str, summary: Mapping[str, Any],
                 seed_records: Mapping[int, Mapping[str, Any]], source_commit: str, execution: Mapping[str, Any],
                 created_unix_s: float | None = None) -> dict[str, Any]:
    """One scene's audit record from its worker summary and per-seed records, sealed by ``audit_sha256``."""
    seeds = {}
    for seed, evidence in sorted(seed_records.items()):
        kept = {key: value for key, value in evidence.items() if key not in {"traceback_tail", "schema"}}
        # Camera paths are relative to the record's directory: <scene>/seed-<n>/cameras/<camera>.png.
        kept["cameras"] = {name: {**image, "path": f"{scene}/seed-{seed}/cameras/{image['path']}"}
                           for name, image in (evidence.get("cameras") or {}).items()}
        seeds[str(seed)] = kept
    record = {
        "schema": AUDIT_SCHEMA, "kind": KIND, "scene_set": scene_set, "scene": scene, "task_config": task_config,
        "robotwin_revision": revision, "scene_source": dict(summary["scene_source"]),
        "robotwin_sources": list(summary.get("robotwin_sources") or []), "criteria": dict(summary["criteria"]),
        "ora_source": {"commit": source_commit}, "execution": dict(execution),
        "seeds": seeds,
        "accepted_seeds": sorted(seed for seed, evidence in seed_records.items() if evidence.get("passed") is True),
        "expert_called": False, "robot_actions_published": 0, "verdict": None,
        "created_unix_s": round(time.time() if created_unix_s is None else created_unix_s, 3),
    }
    record["audit_sha256"] = digest(record)
    return record


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=60).strip()


def require_sources(scene_set: str, *, robotwin_root: Path, revision: str, root: Path = PROJECT_ROOT) -> str:
    """The ORA commit, after checking the RoboTwin checkout and the committed scene sources."""
    observed = _git(robotwin_root, "rev-parse", "HEAD")
    if observed != revision:
        raise ValueError(f"RoboTwin checkout {robotwin_root} is at {observed}, not the pinned {revision}")
    if _git(robotwin_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"RoboTwin checkout {robotwin_root} has modified tracked files")
    relative = [path.relative_to(root).as_posix() for path in source_files(scene_set)]
    if _git(root, "status", "--porcelain", "--", *relative):
        raise ValueError("commit the scene set's sources before auditing them: " + ", ".join(relative))
    return _git(root, "rev-parse", "HEAD")


def _run_worker(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path, timeout: float) -> int:
    with log.open("xb") as stream:
        try:
            return subprocess.run(list(command), cwd=cwd, env=dict(env), stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            return -9


def audit(scene_set: str, scenes: Sequence[str], seeds: Sequence[int], *, robotwin_root: Path,
          robotwin_python: Path, host: str | None = None, resource_profile: str | None = None,
          root: Path = PROJECT_ROOT, destination: Path | None = None,
          runner: Callable[..., int] = _run_worker, lease: Any = None) -> dict[str, Any]:
    """Audit ``scenes`` of ``scene_set`` at ``seeds``; write and return one record per scene."""
    from ora.orchestrator.spec.benchmark import execution_host, robotwin_revision
    from ora.orchestrator.preflight.resources import default_slot
    from ora.orchestrator.preflight.resources import resolve_execution_resource

    definitions = [scene_definition(scene_set, name) for name in scenes]
    revision = robotwin_revision(root)
    commit = require_sources(scene_set, robotwin_root=robotwin_root, revision=revision, root=root)
    required_host = execution_host(host)
    profile = resource_profile or default_slot("robotwin", required_host)
    binding = resolve_execution_resource(profile, runtime_kind="robotwin", required_host=required_host)
    destination = destination or audit_root(root=root) / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, **binding["worker_environment"], "PYTHONPATH": str(root),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    execution = {"required_host": required_host, "resource_profile": profile, "slot_id": binding.get("slot_id"),
                 "worker_environment": dict(binding["worker_environment"])}
    release = (lease or _claim_lease)(binding, scene_set)
    records: dict[str, Any] = {}
    try:
        for definition in definitions:  # one RoboTwin process at a time
            output = destination / scene_set / definition.name
            output.mkdir(parents=True)
            command = [str(robotwin_python), "-m", "ora.environment.backends.robotwin.play_scene_audit",
                       "--robotwin-root", str(robotwin_root), "--output-root", str(output),
                       "--scene-set", scene_set, "--scene", definition.name,
                       "--task-config", definition.task_config, "--seeds", *map(str, seeds)]
            code = runner(command, cwd=robotwin_root, env=environment, log=output / "worker.log",
                          timeout=WORKER_TIMEOUT_S_PER_SEED * len(seeds))
            summary_path = output / "summary.json"
            if code != 0 or not summary_path.is_file():
                records[definition.name] = {"error": f"worker exited {code}", "log": str(output / "worker.log")}
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed_records = {seed: json.loads((output / f"seed-{seed}" / "record.json").read_text(encoding="utf-8"))
                            for seed in seeds}
            record = build_record(scene_set=scene_set, scene=definition.name, task_config=definition.task_config,
                                  revision=revision, summary=summary, seed_records=seed_records,
                                  source_commit=commit, execution=execution)
            path = destination / scene_set / f"{definition.name}.json"
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            records[definition.name] = {"record": str(path), "accepted_seeds": record["accepted_seeds"],
                                        "rejected": {seed: evidence["failures"]
                                                     for seed, evidence in record["seeds"].items()
                                                     if not evidence.get("passed")}}
    finally:
        release()
    return {"destination": str(destination), "scene_set": scene_set, "records": records}


def _claim_lease(binding: Mapping[str, Any], scene_set: str) -> Callable[[], None]:
    """Hold the slot's resource lease while the audit runs (as a seed scan does)."""
    from ora.orchestrator.backends.robotwin import seed_lanes as robotwin_seed_lanes
    from ora.orchestrator.leases import claim_resource_lease, release_resource_lease

    owner = f"robotwin-play-audit-{binding.get('slot_id')}-{os.getpid()}-{time.time_ns()}"
    plan = {"plan_sha256": digest({"kind": KIND, "scene_set": scene_set, "owner": owner}),
            "runtime": {"kind": "robotwin", "resource_binding": dict(binding)}}
    claimed = robotwin_seed_lanes.claim_with_probe_retry(claim_resource_lease, plan, owner, os.getpid())
    return lambda: release_resource_lease(owner, claim_token=claimed["claim_token"])


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set (this host's RoboTwin checkout and interpreter)")
    return Path(value).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ORA's RoboTwin free-play scenes on this host.")
    parser.add_argument("--scene-set", required=True, choices=scene_sets())
    parser.add_argument("--scene", action="append", help="a scene of the set (repeat; default: every scene)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="audit seeds 0..N-1 (default 3)")
    seeds.add_argument("--seed-list", type=int, nargs="+", help="audit exactly these seeds")
    parser.add_argument("--host", help="registered execution host (default: ORA_EXECUTION_HOST)")
    parser.add_argument("--resource-profile", help="the host's RoboTwin slot (default: its first)")
    args = parser.parse_args(argv)
    try:
        names = args.scene or [definition.name for definition in scene_definitions(args.scene_set)]
        chosen = args.seed_list or list(range(args.seeds))
        if not chosen or any(seed < 0 for seed in chosen) or not 1 <= len(chosen) <= 100:
            raise ValueError("audit 1 to 100 non-negative seeds")
        report = audit(args.scene_set, names, chosen, robotwin_root=_environment_path("ROBOTWIN_ROOT"),
                       robotwin_python=_environment_path("ROBOTWIN_PYTHON"), host=args.host,
                       resource_profile=args.resource_profile)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"robotwin_play_audit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if all(item.get("accepted_seeds") for item in report["records"].values()) else 1


__all__ = ["AUDIT_SCHEMA", "DIRECTORY", "KIND", "audit", "audit_root", "build_record", "main",
           "play_scene_admission", "require_sources"]
