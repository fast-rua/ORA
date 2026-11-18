"""Fresh admission checks for a RoboLab runtime (no Kit start)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.benchmarks.robolab import PACKAGES, REVISION

from ora.orchestrator.backends.registry import PreflightContext
from ora.orchestrator.preflight.checks import check as _check, execution_host_check, motion_timing_check

_PROBE = ("import importlib.metadata as m, json; "
          "print(json.dumps({p: m.version(p) for p in ('isaacsim', 'isaaclab', 'warp-lang', 'robolab')}))")


def _planner_check(planner: Mapping[str, Any]) -> dict[str, Any]:
    """The planner's interpreter and source: the passing detail is what the worker starts it from."""

    python, root = (os.environ.get(str(planner[name])) for name in ("python_env", "root_env"))
    missing = [str(planner[name]) for name, value in (("python_env", python), ("root_env", root)) if not value]
    if missing:
        return _check("shared_motion_planner", False, {"missing": missing})
    return _check("shared_motion_planner", True, {
        "python_executable": os.path.abspath(os.path.expanduser(str(python))),
        "root": str(Path(str(root)).expanduser().resolve())})


def task_contract_check(plan: Mapping[str, Any]) -> dict[str, Any]:
    """A RoboLab task launches only with its reviewed contract delivered, or deliberately replaced.

    All 120 tasks are admitted, but 117 contracts were generated and wait for their
    seed-0 witness (the registry's ``pending_review``); the resolver withholds them
    as ``contract_review_pending``. Such a cell composes, and this check refuses it.
    """

    from ora.agent.composition.task_contract.delivery import DELIBERATE_WITHHOLDING

    disclosure = (plan.get("policy") or {}).get("task_contract")
    if not isinstance(disclosure, Mapping):
        return _check("robolab_task_contract", False, {"reason": "plan_records_no_task_contract_disclosure"})
    passed = disclosure.get("delivered") is True or disclosure.get("reason") in DELIBERATE_WITHHOLDING
    detail = {key: disclosure.get(key) for key in ("task", "delivered", "reason", "semantic_review_status")}
    if disclosure.get("semantic_review") and not passed:
        detail["semantic_review"] = disclosure["semantic_review"]
    return _check("robolab_task_contract", passed, detail)


def preflight_runtime(plan: Mapping[str, Any], context: PreflightContext) -> list[dict[str, Any]]:
    return [*robolab_runtime(plan, runner=context.runner), task_contract_check(plan)]


def robolab_runtime(
    plan: Mapping[str, Any], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]]:
    """Interpreter packages, the clean pinned checkout and the graphics runtime."""

    import json

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
        checks.append(_planner_check(planner))
    raw_python = os.environ.get(str(runtime.get("python_env")))
    python = Path(os.path.abspath(os.path.expanduser(raw_python))) if raw_python else None
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {"env": runtime.get("python_env"),
                                                            "executable": str(python) if python else None}))
    if executable:
        result = runner([str(python), "-c", _PROBE], capture_output=True, text=True, timeout=180, check=False)
        try:
            packages = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            packages = {"error": result.stderr.strip()[-600:]}
        checks.append(_check("robolab_packages", all(packages.get(name) == version for name, version in PACKAGES.items()),
                             {"observed": packages, "expected": PACKAGES}))
    raw_root = os.environ.get(str(runtime.get("root_env")))
    root = Path(raw_root).expanduser().resolve() if raw_root else None
    if root is not None and (root / ".git").exists():
        head = runner(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        dirty = runner(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False)
        checks.append(_check("robolab_source", head.stdout.strip() == REVISION and not dirty.stdout.strip(),
                             {"expected": REVISION, "observed": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}))
    else:
        checks.append(_check("robolab_source", False, {"env": runtime.get("root_env"), "configured": bool(raw_root)}))
    raw_graphics = os.environ.get(str(runtime.get("graphics_runtime_env")))
    graphics = Path(raw_graphics).expanduser() if raw_graphics else None
    checks.append(_check("graphics_runtime", bool(graphics and (graphics / "graphics-compat" / "installation.json").is_file()),
                         {"env": runtime.get("graphics_runtime_env"), "configured": bool(raw_graphics)}))
    return checks
