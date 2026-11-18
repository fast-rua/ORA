"""Play-scene audit: qualify ORA's own RoboDojo free-play scenes on this host (``robodojo_play_scene_audit``).

    uv run --locked python -m ora.orchestrator.robodojo_play_audit [--scene-set isaac-tabletop-v1] \\
        [--scene NAME ...] [--seeds N | --seed-list S ...] [--host HOST] [--resource-profile SLOT]

``ora play --env robodojo`` offers a scene of a scene set
(``ora/environment/backends/robodojo/play_scenes``) only at a seed a local
audit admits, as ``ora play --env robotwin --scene-set`` does with
``backends.robotwin.play_audit``. A scene has no task, expert or predicate, so the
audit asks what free play needs. For each scene it starts the live play
session in the RoboDojo interpreter on the host's RoboDojo slot (one process
per scene and pass, one at a time, under the slot's resource lease), loads
each seed through the session's own scene reset, and records that the scene
builds, settles, holds still over a stability window, lies in the arms'
reach envelope, keeps clear of the grippers' start footprints, can be
approached by either arm from at least one of align_plan's declared sides
(align_plan's own poses and default standoff, read-only IK) and renders
every camera; a CUDA scene gets a second pass on CPU physics, where PhysX
reports contacts (no interpenetration, no robot contact). The checks are
``backends/robodojo/play_scenes/audit.py``'s.

Each scene's record (``ora.robodojo-play-scene-audit.v2``) is written to
``outputs/ora-play/qualification/play-scene-audits/<UTC>-<ns>/<set>/<scene>.json``,
the worker reports, generated layouts and camera images beside it. It binds
the pinned RoboDojo revisions (the checkouts must be at them and clean), the
play kernel at the audited commit (the benchmark worker kernel plus the play
code: ``robodojo_play_kernel``), the scene's own source binding (its canonical
index entry, its YAML and the generator: ``robodojo_play_scenes.source_binding``;
the kernel and these sources must be committed) and each seed's generated
layout; ``audit_sha256`` covers the rest. ``play_scene_admission`` is the one
admission rule, applied by ``play.scenes`` when it lists scenes and by launch
preflight when it re-admits the plan's seeds: another play kernel, another
source binding of this scene, another revision, a changed layout or a
tampered record admits nothing. Another scene's edit changes none of them.
No robot action is published and no verdict exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from ora.environment.benchmarks.robodojo import PUBLIC_CAMERAS, REVISIONS, REVISION_FIELDS
from ora.environment.benchmarks.robodojo_play_scenes import (
    DEFAULT_SCENE_SET, PLAY_EXECUTION_CAPTURE, PLAY_NATIVE_ACTION_LIMIT, PlaySceneSetError, SceneDefinition, index_file, layout_sha256,
    scene_definition, scene_definitions, scene_language, scene_sets, source_binding,
)

from .paths import PROJECT_ROOT, resolve_outputs_root, resolve_workspace_root
from .evaluation.evidence_io import digest

AUDIT_SCHEMA = "ora.robodojo-play-scene-audit.v2"
KIND = "robodojo_play_scene_audit"
DIRECTORY = "qualification/play-scene-audits"
DEFAULT_SEEDS = 5
WORKER_MODULE = "ora.environment.backends.robodojo.play_scenes.audit"
#: One pass: Kit start and the first scene (~2 min), then each further seed's relaunch, settle and checks. The
#: reach check asks the shared planner's IK 1,080 poses a seed for a nine-object scene, 11.3 min on a recorded
#: desk seed (the SDK's IK took 49 s), so a seed has 20 min.
PASS_TIMEOUT_S = 420
PASS_TIMEOUT_S_PER_SEED = 1200


def audit_root(*, root: Path = PROJECT_ROOT) -> Path:
    """Where this host's play-scene audits live: ``<outputs root>/ora-play/qualification/play-scene-audits``."""
    return resolve_outputs_root(resolve_workspace_root(root), project_root=root) / "ora-play" / DIRECTORY


def passes_for(definition: SceneDefinition) -> tuple[str, ...]:
    """``session`` on the scene's device; a CUDA scene adds ``contacts`` on CPU physics."""
    return ("session",) if definition.physics_device == "cpu" else ("session", "contacts")


def generated_layouts(definition: SceneDefinition, seeds: Sequence[int], asset_root: Path) -> dict[str, str]:
    """``{seed: layout sha256}`` of the scene's generated layouts (pure Python, on the host)."""
    from ora.environment.benchmarks import robodojo_play_layouts as generator

    spec = generator.load_scene(definition.path)
    return {str(seed): layout_sha256(generator.generate(spec, scene=definition.name, seed=int(seed),
                                                        asset_root=asset_root)["layout"])
            for seed in seeds}


# --- admission -----------------------------------------------------------------------------------

def play_scene_admission(record: Mapping[str, Any], *, scene_set: str | None, scene: str, seed: int,
                         play_kernel_sha256: str | None, layout_sha256: str | None = None,
                         root: Path = PROJECT_ROOT) -> tuple[bool, dict[str, Any]]:
    """Whether an audit record admits ``seed`` of ``scene``, and the evidence for the verdict.

    The record must be an intact audit of this scene of this set, at the
    pinned revisions and on the scene's physics device, measured on the play
    kernel ``play_kernel_sha256`` (the current tree's when listing, the plan
    commit's at launch) and on this scene's current source binding (read from
    ``root``; launch preflight runs on the plan's clean commit); the seed must
    be accepted, its evidence passed and, when given, its layout the one the
    plan binds.
    """
    body = {key: value for key, value in record.items() if key != "audit_sha256"}
    digest_valid = digest(body) == record.get("audit_sha256")
    try:
        definition = scene_definition(str(scene_set), scene, root=root)
        current_source = source_binding(definition.scene_set, scene, root=root)["sha256"]
    except (PlaySceneSetError, OSError):
        definition, current_source = None, None
    seeds = record.get("seeds") if isinstance(record.get("seeds"), Mapping) else {}
    evidence = seeds.get(str(seed)) if isinstance(seeds.get(str(seed)), Mapping) else {}
    kernel = record.get("play_kernel") if isinstance(record.get("play_kernel"), Mapping) else {}
    recorded_source = (record.get("scene_source") or {}).get("sha256") if isinstance(
        record.get("scene_source"), Mapping) else None
    accepted = isinstance(record.get("accepted_seeds"), list) and seed in record["accepted_seeds"]
    passed = (
        record.get("schema") == AUDIT_SCHEMA and record.get("kind") == KIND and definition is not None
        and record.get("scene_set") == scene_set and record.get("scene") == scene
        and record.get("physics_device") == definition.physics_device
        and record.get("robodojo_revisions") == REVISIONS and digest_valid
        and play_kernel_sha256 is not None and kernel.get("sha256") == play_kernel_sha256
        and current_source is not None and recorded_source == current_source
        and accepted and evidence.get("passed") is True
        and isinstance(evidence.get("layout_sha256"), str)
        and (layout_sha256 is None or evidence.get("layout_sha256") == layout_sha256)
    )
    return passed, {
        "schema": record.get("schema"), "kind": record.get("kind"), "scene_set": record.get("scene_set"),
        "expected_scene_set": scene_set, "scene": record.get("scene"),
        "physics_device": record.get("physics_device"), "robodojo_revisions_current": record.get(
            "robodojo_revisions") == REVISIONS,
        "audit_sha256_valid": digest_valid, "play_kernel_sha256": kernel.get("sha256"),
        "current_play_kernel_sha256": play_kernel_sha256, "scene_source_sha256": recorded_source,
        "current_scene_source_sha256": current_source, "accepted": accepted,
        "seed": {key: evidence.get(key) for key in ("passed", "failures", "object_count", "layout_sha256")},
        "expected_layout_sha256": layout_sha256,
    }


# --- the record ----------------------------------------------------------------------------------

def merge_seed(seed: int, reports: Mapping[str, Mapping[str, Any]], expected_layout: str | None,
               passes: Sequence[str], destination: Path, scene_root: Path) -> dict[str, Any]:
    """One seed's evidence from every pass: passed only when each pass audited it and passed."""
    evidence: dict[str, Any] = {"passes": {}, "failures": []}
    layouts = set()
    for name in passes:
        report = reports.get(name) or {}
        record = (report.get("seeds") or {}).get(str(seed))
        if not isinstance(record, Mapping):
            evidence["failures"].append(f"{name}:not_audited" + (f" ({report['error']})" if report.get("error") else ""))
            continue
        layouts.add(record.get("layout_sha256"))
        evidence["passes"][name] = {
            "passed": record.get("passed") is True, "failures": list(record.get("failures") or []),
            "physics_device": record.get("physics_device"), "object_count": record.get("object_count"),
            "contact_points_sampled": record.get("contact_points_sampled"), "seconds": record.get("seconds"),
        }
        if name == "session" and isinstance(record.get("cameras"), Mapping):
            evidence["cameras"] = {
                camera: {**image, "path": (scene_root / f"pass-{name}" / f"seed-{seed}" / "observations"
                                           / image["path"]).relative_to(destination).as_posix()}
                for camera, image in (record["cameras"].get("cameras") or {}).items()}
        if name == "session":
            evidence["object_count"] = record.get("object_count")
        evidence["failures"] += [f"{name}:{failure}" for failure in record.get("failures") or []]
        if record.get("passed") is not True and not record.get("failures"):
            evidence["failures"].append(f"{name}:not_passed")
    layout = layouts.pop() if len(layouts) == 1 else None
    if layouts or (expected_layout is not None and layout != expected_layout):
        evidence["failures"].append("layout_differs_between_passes_or_from_the_generator")
        layout = None
    evidence["layout_sha256"] = layout
    evidence["passed"] = not evidence["failures"] and layout is not None
    return evidence


def build_record(*, definition: SceneDefinition, seeds: Mapping[int, Mapping[str, Any]],
                 reports: Mapping[str, Mapping[str, Any]], play_kernel: Mapping[str, Any], source_commit: str,
                 execution: Mapping[str, Any], criteria: Mapping[str, Any],
                 created_unix_s: float | None = None) -> dict[str, Any]:
    """One scene's audit record from its merged seed evidence, sealed by ``audit_sha256``."""
    session = reports.get("session") or {}
    record = {
        "schema": AUDIT_SCHEMA, "kind": KIND, "scene_set": definition.scene_set, "scene": definition.name,
        "physics_device": definition.physics_device, "robodojo_revisions": dict(REVISIONS),
        "scene_source": source_binding(definition.scene_set, definition.name), "play_kernel": dict(play_kernel),
        "criteria": dict(criteria), "passes": list(passes_for(definition)),
        "ora_source": {"commit": source_commit}, "execution": dict(execution),
        "graphics_compatibility": session.get("graphics_compatibility"),
        "device_binding": session.get("device_binding"),
        "seeds": {str(seed): dict(evidence) for seed, evidence in sorted(seeds.items())},
        "accepted_seeds": sorted(seed for seed, evidence in seeds.items() if evidence.get("passed") is True),
        "expert_called": False, "robot_actions_published": 0, "model_calls": 0, "verdict": None,
        "created_unix_s": round(time.time() if created_unix_s is None else created_unix_s, 3),
    }
    record["audit_sha256"] = digest(record)
    return record


# --- the launch ----------------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=120).strip()


def require_sources(*, robodojo_root: Path, asset_root: Path, root: Path = PROJECT_ROOT,
                    scene_sources: Sequence[str] = ()) -> dict[str, Any]:
    """The audited commit and its play kernel, after checking the RoboDojo checkouts and committed sources.

    ``scene_sources`` are the audited scenes' own files (their set's index and
    YAMLs): they must be committed too, since each record binds them.
    """
    from .robodojo_kernel import WORKER_KERNEL_PATHS
    from .robodojo_play_kernel import PLAY_KERNEL_PATHS, play_kernel

    roots = {"robodojo": robodojo_root, "isaaclab": robodojo_root / "third_party/IsaacLab",
             "curobo": robodojo_root / "third_party/curobo", "xpolicylab": robodojo_root / "XPolicyLab",
             "assets": asset_root}
    for name, path in roots.items():
        observed = _git(path, "rev-parse", "HEAD")
        if observed != REVISIONS[name]:
            raise ValueError(f"RoboDojo {name} checkout {path} is at {observed}, not the pinned {REVISIONS[name]}")
        if _git(path, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"RoboDojo {name} checkout {path} has modified tracked files")
    paths = [*WORKER_KERNEL_PATHS, *PLAY_KERNEL_PATHS, *scene_sources]
    if _git(root, "status", "--porcelain", "--", *paths):
        raise ValueError("commit the play kernel (benchmark worker kernel and play code) and the audited scenes' "
                         "index and YAMLs before auditing them")
    commit = _git(root, "rev-parse", "HEAD")
    return {"commit": commit, "play_kernel": play_kernel(commit, project_root=root)}


def audit_runtime(definition: SceneDefinition, *, resource_profile: str, host: str, seeds: Sequence[int],
                  asset_root: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The worker runtime a play plan of this scene resolves to, on the slot's GPU, capture off."""
    from ora.agent.composition import resolve_tool_surface
    from ora.environment.workspace_profiles import resolve_workspace_profile

    from .spec.defaults import expand_native_defaults
    from .preflight.resources import resolve_execution_resource

    binding = resolve_execution_resource(resource_profile, runtime_kind="robodojo", required_host=host)
    gpu = str(binding["worker_environment"]["CUDA_VISIBLE_DEVICES"])
    runtime = spec_runtime(definition, gpu=gpu, host=host)
    expanded = expand_native_defaults({"policy": {"kind": "native_api", "supports_images": True},
                                       "runtime": runtime, "play": {}}, root=root)
    runtime = expanded["runtime"]
    runtime["workspace_bounds_m"] = resolve_workspace_profile(runtime.pop("workspace_profile"))["bounds_m"]
    surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
    runtime["public_geometry_descriptor"] = copy.deepcopy(surface.view_config["public_geometry_descriptor"])
    runtime["resource_binding"] = binding
    runtime["execution_capture"] = {"enabled": False}
    runtime["play_layouts"] = generated_layouts(definition, seeds, asset_root)
    runtime["play_scene_source_sha256"] = source_binding(definition.scene_set, definition.name, root=root)["sha256"]
    return runtime


def spec_runtime(definition: SceneDefinition, *, gpu: str, host: str) -> dict[str, Any]:
    """A play scene's RunSpec runtime (before default expansion): the benchmark's pins, the scene's identity."""
    return {
        "kind": "robodojo", "transport": "local_rpc",
        "root_env": "ROBODOJO_ROOT", "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{REVISION_FIELDS[name]: revision for name, revision in REVISIONS.items()},
        "required_host": host, "require_gpu": True,
        "task_variant": definition.name, "eval_seed": 0, "num_envs": 1, "env_config": "arx_x5",
        "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": PLAY_NATIVE_ACTION_LIMIT, "native_actions_per_command": 20,
        "physics_device": definition.physics_device, "renderer_gpu": int(gpu), "cuda_visible_devices": str(gpu),
        "main_camera": "head", "policy_cameras": list(PUBLIC_CAMERAS),
        "startup_timeout_s": 600, "request_timeout_s": 600, "close_timeout_s": 30,
        "expected_task_language": scene_language(definition), "play_scene_set": definition.scene_set,
        "execution_capture": copy.deepcopy(PLAY_EXECUTION_CAPTURE),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pass(definition: SceneDefinition, pass_name: str, seeds: Sequence[int], runtime: Mapping[str, Any], *,
             output: Path, commit: str) -> int:
    """One worker process for one scene and pass, under the slot's lease; returns its exit code."""
    from ora.environment.robodojo_play_factory import robodojo_play_worker_config
    from ora.environment.rpc import RPC_SCHEMA
    from .preflight import host_storage
    from .backends.robodojo.motion.replay import _leased_spawn

    worker_runtime = dict(runtime)
    if pass_name == "contacts":
        worker_runtime.update(physics_device="cpu", audit_physics_device="cpu")
    output.mkdir(parents=True, exist_ok=False)
    config, python, environment = robodojo_play_worker_config(
        runtime=worker_runtime, output_root=output / "environment", task=definition.name, seed=int(seeds[0]),
        max_sim_steps=PLAY_NATIVE_ACTION_LIMIT * 10, action_codec="motion_action_v3")
    config.update({"schema": RPC_SCHEMA, "session_id": f"env-{uuid.uuid4().hex}", "shared_root": str(output)})
    _write_json(output / "worker-config.json", config)
    command = [str(python), "-m", WORKER_MODULE, "--config", str(output / "worker-config.json"),
               "--output", str(output), "--pass", pass_name, "--seeds", *map(str, seeds)]
    audit_id = f"robodojo-play-audit-{definition.name}-{pass_name}-{uuid.uuid4().hex[:8]}"
    staged = host_storage.stage_launch({"runtime": worker_runtime}, audit_id)
    lease_plan = {"schema": "ora.robodojo-infrastructure-plan.v1", "runtime": dict(runtime),
                  "harness_commit": commit, "episodes": 1, "task_solving": False,
                  "diagnostic": f"robodojo play-scene audit of {definition.scene_set}/{definition.name} ({pass_name})"}
    lease_plan["plan_sha256"] = hashlib.sha256(json.dumps(lease_plan, sort_keys=True).encode()).hexdigest()
    launch = {"audit_id": audit_id, "pass": pass_name, "seeds": list(seeds), "command": command, **staged}
    _write_json(output / "launch.json", launch)
    try:
        code, seconds = _leased_spawn(lease_plan, SimpleNamespace(worker_environment=environment), command, output,
                                      PASS_TIMEOUT_S + PASS_TIMEOUT_S_PER_SEED * len(seeds),
                                      host_storage.scratch_environment(staged))
    finally:
        host_storage.release_run_scratch(output)
    _write_json(output / "launch.json", {**launch, "returncode": code, "seconds": seconds})
    return code


def audit(scene_set: str, scenes: Sequence[str], seeds: Sequence[int], *, host: str | None = None,
          resource_profile: str | None = None, root: Path = PROJECT_ROOT, destination: Path | None = None,
          runner: Callable[..., int] = run_pass, sources: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit ``scenes`` of ``scene_set`` at ``seeds``; write and return one record per scene."""
    from ora.environment.benchmarks.robodojo_play_scenes import (
        AUDIT_CRITERIA as CRITERIA, AUDIT_WORKER_REPORT_NAME as WORKER_REPORT_NAME,
    )
    from .spec.benchmark import execution_host
    from .preflight.resources import default_slot

    definitions = [scene_definition(scene_set, name) for name in scenes]
    asset_root = _environment_path("ROBODOJO_ASSET_ROOT")
    scene_sources = sorted({index_file(scene_set), *(definition.scene_file for definition in definitions)})
    identity = dict(sources) if sources is not None else require_sources(
        robodojo_root=_environment_path("ROBODOJO_ROOT"), asset_root=asset_root, root=root,
        scene_sources=scene_sources)
    required_host = execution_host(host)
    profile = resource_profile or default_slot("robodojo", required_host)
    destination = destination or audit_root(root=root) / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    for definition in definitions:  # one Isaac process at a time
        runtime = audit_runtime(definition, resource_profile=profile, host=required_host, seeds=seeds,
                                asset_root=asset_root, root=root)
        scene_root = destination / scene_set / definition.name
        reports: dict[str, Any] = {}
        codes: dict[str, int] = {}
        for pass_name in passes_for(definition):
            output = scene_root / f"pass-{pass_name}"
            codes[pass_name] = runner(definition, pass_name, list(seeds), runtime, output=output,
                                      commit=identity["commit"])
            try:
                reports[pass_name] = json.loads((output / WORKER_REPORT_NAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                reports[pass_name] = {"error": f"worker exited {codes[pass_name]} without a report", "seeds": {}}
        evidence = {int(seed): merge_seed(int(seed), reports, runtime["play_layouts"].get(str(seed)),
                                          passes_for(definition), destination / scene_set, scene_root)
                    for seed in seeds}
        execution = {"required_host": required_host, "resource_profile": profile,
                     "slot_id": runtime["resource_binding"].get("slot_id"),
                     "worker_environment": dict(runtime["resource_binding"]["worker_environment"]),
                     "returncodes": codes,
                     "worker_reports": {name: {"path": f"{definition.name}/pass-{name}/{WORKER_REPORT_NAME}",
                                               "sha256": hashlib.sha256((scene_root / f"pass-{name}" / WORKER_REPORT_NAME)
                                                                        .read_bytes()).hexdigest()}
                                        for name in reports if (scene_root / f"pass-{name}" / WORKER_REPORT_NAME).is_file()}}
        record = build_record(definition=definition, seeds=evidence, reports=reports,
                              play_kernel=identity["play_kernel"], source_commit=identity["commit"],
                              execution=execution, criteria=CRITERIA)
        path = destination / scene_set / f"{definition.name}.json"
        _write_json(path, record)
        records[definition.name] = {"record": str(path), "accepted_seeds": record["accepted_seeds"],
                                    "scene_source_sha256": record["scene_source"]["sha256"],
                                    "rejected": {seed: item["failures"] for seed, item in record["seeds"].items()
                                                 if not item.get("passed")}}
    return {"destination": str(destination), "scene_set": scene_set, "play_kernel": identity["play_kernel"]["sha256"],
            "records": records}


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set (this host's pinned RoboDojo checkout, assets and interpreter)")
    return Path(value).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ora.orchestrator.robodojo_play_audit",
                                     description="Audit ORA's RoboDojo free-play scenes on this host.")
    parser.add_argument("--scene-set", default=DEFAULT_SCENE_SET, choices=scene_sets())
    parser.add_argument("--scene", action="append", help="a scene of the set (repeat; default: every scene)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="audit seeds 0..N-1 (default 5)")
    seeds.add_argument("--seed-list", type=int, nargs="+", help="audit exactly these seeds")
    parser.add_argument("--host", help="registered execution host (default: ORA_EXECUTION_HOST)")
    parser.add_argument("--resource-profile", help="the host's RoboDojo slot (default: its first)")
    args = parser.parse_args(argv)
    try:
        names = args.scene or [definition.name for definition in scene_definitions(args.scene_set)]
        chosen = args.seed_list or list(range(args.seeds))
        if not chosen or any(seed < 0 for seed in chosen) or not 1 <= len(chosen) <= 32 or len(set(chosen)) != len(chosen):
            raise ValueError("audit 1 to 32 distinct non-negative seeds")
        report = audit(args.scene_set, names, chosen, host=args.host, resource_profile=args.resource_profile)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"robodojo_play_audit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if all(item.get("accepted_seeds") for item in report["records"].values()) else 1


__all__ = ["AUDIT_SCHEMA", "DIRECTORY", "KIND", "audit", "audit_root", "audit_runtime", "build_record",
           "generated_layouts", "main", "merge_seed", "passes_for", "play_scene_admission", "require_sources",
           "run_pass", "spec_runtime"]


if __name__ == "__main__":
    raise SystemExit(main())
