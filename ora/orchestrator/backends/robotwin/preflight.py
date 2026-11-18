"""RoboTwin launch admission on the host: the control binding, the pinned checkout and the worker interpreter."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.backends.registry import Check, PreflightContext
from ora.orchestrator.preflight.checks import execution_host_check, motion_timing_check


def robotwin_control_binding(
    plan: Mapping[str, Any], *, project_root: Path, check: Any,
) -> dict[str, Any]:
    from ora.environment.factory import (
        EnvironmentFactoryError, validate_robotwin_control_binding,
    )

    runtime = plan["runtime"]
    try:
        binding = validate_robotwin_control_binding(
            runtime, action_codec=plan.get("policy", {}).get("action_codec"),
            project_root=project_root,
        )
    except (EnvironmentFactoryError, ValueError, OSError, KeyError, TypeError) as exc:
        return check("robotwin_control_binding", False, str(exc))
    return check("robotwin_control_binding", True, {
        "required": binding is not None,
        "profile": runtime.get("control_profile"),
        "robotwin_revision": runtime.get("required_revision"),
    })


def shared_planner_check(planner: Mapping[str, Any], runtime: Mapping[str, Any], check: Any) -> dict[str, Any]:
    """The planner a run names; the passing detail is what its worker binds (``backends/robotwin/shared_motion.py``).

    The worker's session fields are fixed and its environment is allowlisted, so the
    interpreter and source travel in this record. The client's constructor checks the
    interpreter and that the source is clean at the required revision; the worker starts
    it, because a planner holds GPU memory.
    """
    from ora.contracts.robotwin_reach import ROBOT_FILE, SHARED_PLANNER_CHECK
    from ora.environment.controllers.motion_planner import planner_client

    try:
        python, root = (os.environ.get(str(planner[field])) for field in ("python_env", "root_env"))
        checkout = os.environ.get(str(runtime["root_env"]))
        if not python or not root or not checkout:
            raise RuntimeError(f"{planner['python_env']}, {planner['root_env']} and {runtime['root_env']} are required")
        with tempfile.TemporaryDirectory(prefix="ora-planner-check-") as scratch:
            client = planner_client(planner, python_executable=Path(python), root=Path(root),
                                    output_root=Path(scratch), robot_file=Path(checkout) / ROBOT_FILE)
        return check(SHARED_PLANNER_CHECK, True, {
            **{key: value for key, value in planner.items() if key not in ("python_env", "root_env")},
            "python": str(client.python_executable), "python_sha256": client.binding["python_sha256"],
            "root": str(client.root),
        })
    except Exception as exc:
        return check(SHARED_PLANNER_CHECK, False, str(exc))


_PROBE_SCRIPT = """
import importlib, json, platform, sys
names = ['ora.environment.worker', 'numpy', 'PIL', 'yaml', 'sapien', 'torch', 'curobo', sys.argv[1]]
modules = {name: importlib.import_module(name) for name in names}
print(json.dumps({
    'executable': sys.executable,
    'python': platform.python_version(),
    'modules': {name: getattr(module, '__file__', None) for name, module in modules.items()},
}, sort_keys=True))
"""


def preflight_runtime(plan: Mapping[str, Any], context: PreflightContext) -> list[Check]:
    """The control binding, host, pinned clean checkout, worker interpreter and its imports, and the GPU."""

    _check, _git = context.check, context.git
    runtime = plan["runtime"]
    checks = [robotwin_control_binding(plan, project_root=context.project_root, check=_check)]
    checks.append(execution_host_check(runtime, os.environ.get("ORA_EXECUTION_HOST")))
    root_env = runtime["root_env"]
    raw_root = os.environ.get(root_env)
    root = Path(raw_root).expanduser().resolve() if raw_root else None
    checks.append(_check("robotwin_root", bool(root and (root / "envs").is_dir()), {
        "env": root_env, "configured": bool(raw_root),
    }))
    if root and (root / ".git").exists():
        revision = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
        checks.append(_check("robotwin_revision", revision.returncode == 0 and revision.stdout.strip() == runtime["required_revision"], {
            "expected": runtime["required_revision"], "observed": revision.stdout.strip(),
        }))
        checks.append(_check("robotwin_clean", dirty.returncode == 0 and not dirty.stdout.strip(), dirty.stdout.strip() or "clean"))
    python_env = runtime["python_env"]
    raw_python = os.environ.get(python_env)
    python = (
        Path(os.path.abspath(os.path.expanduser(raw_python))) if raw_python else None
    )
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {
        "env": python_env,
        "configured": bool(raw_python),
        "executable": str(python) if python else None,
    }))
    if executable and root:
        probe_environment = os.environ.copy()
        python_paths = [str(context.project_root), str(root), str(root / "envs/curobo/src")]
        if probe_environment.get("PYTHONPATH"):
            python_paths.append(probe_environment["PYTHONPATH"])
        probe_environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        try:
            probe = subprocess.run(
                [str(python), "-c", _PROBE_SCRIPT, task_module(runtime, plan["task"])],
                cwd=root,
                env=probe_environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            observed = None
            if probe.returncode == 0:
                try:
                    observed = json.loads(probe.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    pass
            checks.append(_check(
                "environment_runtime_imports",
                probe.returncode == 0 and isinstance(observed, dict),
                observed or {
                    "returncode": probe.returncode,
                    "stderr_tail": probe.stderr.strip()[-2000:],
                    "stdout_tail": probe.stdout.strip()[-1000:],
                },
            ))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(_check("environment_runtime_imports", False, str(exc)))
    timing = runtime.get("motion_timing")
    if isinstance(timing, Mapping):
        # The worker reads this check's binding from the run's record (motion_timing.bind_preflight).
        binding = runtime.get("resource_binding")
        checks.append(motion_timing_check(timing, environment_overrides=(
            binding.get("worker_environment") if isinstance(binding, Mapping) else None)))
    planner = runtime.get("motion_planner")
    if isinstance(planner, Mapping):
        checks.append(shared_planner_check(planner, runtime, _check))
    if runtime.get("require_gpu", True):
        nvidia = shutil.which("nvidia-smi")
        result = subprocess.run([nvidia, "-L"], capture_output=True, text=True, timeout=30, check=False) if nvidia else None
        checks.append(_check("gpu", bool(result and result.returncode == 0 and result.stdout.strip()), result.stdout.strip() if result else "nvidia-smi unavailable"))
    return checks


def task_module(runtime: Mapping[str, Any], task: str) -> str:
    """The module the RoboTwin worker loads the task from: ``envs.{task}``, or ORA's play scene module."""
    if runtime.get("play_scene_set") is None:
        return f"envs.{task}"
    from ora.environment.benchmarks.robotwin_play_scenes import PlaySceneSetError, scene_definition
    try:
        return scene_definition(str(runtime["play_scene_set"]), task).module
    except PlaySceneSetError:  # the import probe then fails, naming the unknown scene
        return f"ora.environment.backends.robotwin.play_scenes.unknown_scene_{task}"


__all__ = ["preflight_runtime", "robotwin_control_binding", "task_module"]
