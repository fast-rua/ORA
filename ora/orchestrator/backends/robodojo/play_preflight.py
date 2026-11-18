"""Host-safe launch preflight of a RoboDojo free-play plan (``runtime.play_scene_set``).

``backends.robodojo.preflight`` dispatches a RoboDojo plan that names one of ORA's play
scene sets here. A play scene has no infrastructure qualification: its seeds
are admitted by a play-scene audit (``robodojo_play_audit``, checked by the
``seed_qualification`` preflight check against the play kernel at the plan's
commit). What stays the benchmark's: the host marker, the shared motion base
(``preflight.shared_motion_checks``: the timing worker the session binds from
this record, and the planner's interpreter variables), the interpreter and
roots, the pinned and clean RoboDojo checkouts, the worker's package stack,
the selected GPU's free memory and the NVIDIA driver with its Vulkan
compatibility repair, whose fingerprint must equal the one the audit's worker
recorded. Added for play: every scene the session may load regenerates, from
the committed scene sources and the pinned assets, to the layout the plan binds.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

#: The free memory a play worker needs on its card when the slot names no floor (the benchmark's default).
UNPROFILED_MINIMUM_FREE_MEMORY_MIB = 20000
NVIDIA_SMI_TIMEOUT_S = 60


def _audit_record(plan: Mapping[str, Any], workspace_root: Path, project_root: Path) -> Mapping[str, Any] | None:
    manifest = (plan.get("qualification") or {}).get("manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("path"), str):
        return None
    relative = Path(manifest["path"])
    path = (workspace_root / relative if relative.parts[:2] == ("outputs", plan["experiment"])
            else project_root / relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def play_runtime_checks(plan: Mapping[str, Any], runner: Callable[..., subprocess.CompletedProcess[str]], *,
                        project_root: Path, check: Callable[..., dict[str, Any]],
                        git: Callable[..., subprocess.CompletedProcess[str]],
                        workspace_root: Path | None = None) -> list[dict[str, Any]]:
    from ora.environment.benchmarks.robodojo import REVISION_FIELDS
    from ora.environment.benchmarks.robodojo_play_scenes import PlaySceneSetError, scene_definition

    from ora.orchestrator.backends.robodojo.preflight import shared_motion_checks
    from ora.orchestrator.paths import resolve_workspace_root
    from ora.orchestrator.robodojo_play_audit import generated_layouts
    from ora.orchestrator.robodojo_qualification import (
        GRAPHICS_READBACK_TIMEOUT_S, inspect_graphics_compatibility, validate_runtime_driver,
    )

    runtime = plan["runtime"]
    checks = [check(
        "execution_host",
        bool(runtime.get("required_host")) and os.environ.get("ORA_EXECUTION_HOST") == runtime["required_host"],
        {"expected": runtime.get("required_host"), "observed": os.environ.get("ORA_EXECUTION_HOST")},
    )]
    checks.extend(shared_motion_checks(runtime, check))
    try:
        definition = scene_definition(str(runtime.get("play_scene_set")), str(plan["task"]))
        checks.append(check("robodojo_play_scene", runtime.get("physics_device") == definition.physics_device, {
            "scene_set": definition.scene_set, "scene": definition.name, "physics_device": definition.physics_device,
            "plan_physics_device": runtime.get("physics_device")}))
    except PlaySceneSetError as exc:
        return [*checks, check("robodojo_play_scene", False, str(exc))]
    roots: dict[str, Path] = {}
    for field in ("root_env", "asset_root_env", "python_env"):
        raw = os.environ.get(runtime[field])
        path = Path(os.path.abspath(os.path.expanduser(raw))) if raw else None
        ready = bool(path and (path.is_file() if field == "python_env" else path.is_dir()))
        checks.append(check(f"robodojo_{field}", ready, {"environment_variable": runtime[field],
                                                         "path": str(path) if path else None}))
        if ready and path:
            roots[field] = path
    if len(roots) != 3:
        return checks
    root, assets, python = (roots[key] for key in ("root_env", "asset_root_env", "python_env"))
    source_roots = {"robodojo": root, "isaaclab": root / "third_party/IsaacLab", "curobo": root / "third_party/curobo",
                    "xpolicylab": root / "XPolicyLab", "assets": assets}
    for name, source_root in source_roots.items():
        revision = git(source_root, "rev-parse", "HEAD")
        dirty = git(source_root, "status", "--porcelain", "--untracked-files=no")
        checks.append(check(f"robodojo_{name}_source", (
            revision.returncode == 0 and revision.stdout.strip() == runtime[REVISION_FIELDS[name]]
            and dirty.returncode == 0 and not dirty.stdout.strip()
        ), {"expected_revision": runtime[REVISION_FIELDS[name]], "observed_revision": revision.stdout.strip(),
            "clean": dirty.returncode == 0 and not dirty.stdout.strip()}))
    layouts = runtime.get("play_layouts") or {}
    try:
        regenerated = generated_layouts(definition, [int(seed) for seed in layouts], assets)
        checks.append(check("robodojo_play_layouts", bool(layouts) and regenerated == dict(layouts),
                            {"planned": dict(layouts), "regenerated": regenerated}))
    except (OSError, ValueError) as exc:
        checks.append(check("robodojo_play_layouts", False, str(exc)))
    observed: dict[str, Any] | None = None
    try:
        process = runner(
            [str(python), "-c", (
                "import json,sys,hashlib; from pathlib import Path; from importlib.metadata import version; "
                "print(json.dumps({'python_executable':sys.executable, "
                "'python_binary_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),"
                "'versions':{k:version(k) for k in ('isaacsim','torch','numpy','warp-lang')}}))"
            )], cwd=root, env=os.environ.copy(), text=True, capture_output=True,
            timeout=GRAPHICS_READBACK_TIMEOUT_S, check=False,
        )
        observed = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else None
        versions = observed.get("versions", {}) if isinstance(observed, dict) else {}
        checks.append(check("robodojo_python_stack", (
            versions.get("isaacsim") == "5.1.0.0" and versions.get("torch") == "2.7.0+cu128"
            and versions.get("numpy") == "1.26.0"
        ), observed or {"returncode": process.returncode}))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        checks.append(check("robodojo_python_stack", False, str(exc)))
    record = _audit_record(plan, workspace_root or resolve_workspace_root(project_root), project_root)
    audited_graphics = (record or {}).get("graphics_compatibility")
    try:
        query = ["nvidia-smi", "--query-gpu=index,uuid,memory.free,driver_version", "--format=csv,noheader,nounits"]
        try:
            gpu = runner(query, text=True, capture_output=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            gpu = runner(query, text=True, capture_output=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
        rows = [row.strip().split(", ") for row in gpu.stdout.splitlines()]
        selected = [row for row in rows if row[0] == runtime["cuda_visible_devices"]]
        floor = int((runtime.get("resource_binding") or {}).get(
            "minimum_free_memory_mib", UNPROFILED_MINIMUM_FREE_MEMORY_MIB))
        checks.append(check("robodojo_gpu", gpu.returncode == 0 and len(selected) == 1
                            and int(selected[0][2]) >= floor,
                            {"selected_device": selected, "minimum_free_memory_mib": floor,
                             "audited_device_binding": (record or {}).get("device_binding")}))
        try:
            driver = selected[0][3] if gpu.returncode == 0 and len(selected) == 1 else ""
            graphics = inspect_graphics_compatibility(python, runner=runner) if driver.startswith("595.") else None
            validate_runtime_driver(driver, graphics_compatibility=graphics)
            if record is None:
                raise ValueError("the plan's play-scene audit record is unreadable; no graphics comparison")
            if graphics != audited_graphics:
                raise ValueError("Vulkan compatibility fingerprint differs from the play-scene audit's")
            checks.append(check("robodojo_driver", True, {"version": driver, "graphics_compatibility": graphics}))
        except (ValueError, IndexError) as exc:
            checks.append(check("robodojo_driver", False, str(exc)))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        checks.append(check("robodojo_gpu", False, str(exc)))
    return checks


__all__ = ["play_runtime_checks"]
