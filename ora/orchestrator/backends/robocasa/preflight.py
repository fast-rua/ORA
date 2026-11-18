"""Fresh admission checks for a RoboCasa365 runtime (no simulator episode)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.controllers.motion_planner import planner_client
from ora.environment.controllers.robocasa_omron import MOTION_PLANNER_PREFLIGHT_CHECK
from ora.orchestrator.backends.registry import PreflightContext
from ora.orchestrator.preflight.checks import check as _check, execution_host_check, motion_timing_check

_PROBE = r"""
import json, sys
from ora.environment.backends.robocasa.construction import source_manifest
print(json.dumps(source_manifest()))
"""


def preflight_runtime(plan: Mapping[str, Any], context: PreflightContext) -> list[dict[str, Any]]:
    return robocasa_runtime(plan, project_root=context.project_root, runner=context.runner)


def motion_planner_check(planner: Mapping[str, Any], *,
                         environment_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Start the shared planner once; the worker binds the interpreter and root this records (``shared_motion``)."""

    python, root = os.environ.get(str(planner["python_env"])), os.environ.get(str(planner["root_env"]))
    if not python or not root:
        return _check(MOTION_PLANNER_PREFLIGHT_CHECK, False, {"missing": [
            name for name in (planner["python_env"], planner["root_env"]) if not os.environ.get(str(name))]})
    python = os.path.abspath(os.path.expanduser(python))
    try:
        with tempfile.TemporaryDirectory(prefix="ora-planner-preflight-") as directory:
            client = planner_client(planner, python_executable=Path(python), root=Path(root),
                                    output_root=Path(directory), environment_overrides=environment_overrides)
            try:
                binding = client.start()
            finally:
                client.close()
    except Exception as exc:  # noqa: BLE001 - any failure to start is this check's detail
        return _check(MOTION_PLANNER_PREFLIGHT_CHECK, False, str(exc))
    return _check(MOTION_PLANNER_PREFLIGHT_CHECK, True, {
        "python": python, "root": str(Path(root).expanduser().resolve()), "revision": binding["revision"],
        "python_sha256": binding["python_sha256"], "device": binding["device"], "robot": binding["robot"]})


def robocasa_runtime(
    plan: Mapping[str, Any], *, project_root: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]]:
    """Interpreter, package versions and pinned checkouts, probed in the worker interpreter."""

    runtime = plan["runtime"]
    checks = [execution_host_check(runtime, os.environ.get("ORA_EXECUTION_HOST"))]
    timing = runtime.get("motion_timing")
    if isinstance(timing, Mapping):
        # The worker reads this check's binding from the run's record (motion_timing.bind_preflight).
        binding = runtime.get("resource_binding")
        checks.append(motion_timing_check(timing, environment_overrides=(
            binding.get("worker_environment") if isinstance(binding, Mapping) else None)))
    planner = runtime.get("motion_planner")
    if isinstance(planner, Mapping):
        binding = runtime.get("resource_binding")
        checks.append(motion_planner_check(planner, environment_overrides=(
            binding.get("worker_environment") if isinstance(binding, Mapping) else None)))
    python_env = str(runtime.get("python_env") or "ROBOCASA_PYTHON")
    raw = os.environ.get(python_env)
    python = Path(os.path.abspath(os.path.expanduser(raw))) if raw else None
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {
        "env": python_env, "configured": bool(raw), "executable": str(python) if python else None,
    }))
    if not executable:
        return checks
    environment = {key: value for key, value in os.environ.items()
                   if key in {"PATH", "HOME", "LANG", "TMPDIR"}}
    environment.update({"PYTHONPATH": str(project_root), "PYTHONNOUSERSITE": "1",
                        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    result = runner([str(python), "-c", _PROBE], capture_output=True, text=True,
                    timeout=300, env=environment, check=False)
    manifest = None
    if result.returncode == 0:
        try:
            manifest = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            manifest = None
    checks.append(_check("robocasa_source", manifest is not None, manifest or {
        "returncode": result.returncode, "stderr": result.stderr.strip()[-800:],
    }))
    return checks
