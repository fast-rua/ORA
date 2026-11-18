"""Fresh RoboCasa365 RunSpecs from the default baseline plus cell, provider and resource inputs.

One cell is one episode of one admitted task (``benchmarks/robocasa.TASKS``)
on one seed of the target split. Each seed samples its own kitchen, so the
cell's official instruction is the scene's own ``ep_meta['lang']``, read here
by a short probe under the RoboCasa interpreter and bound as
``runtime.expected_task_language``; the worker re-checks it at reset. The
RunSpec names no ToolSurface or prompt bundle, so the default catalog's
``robocasa`` instance composes the policy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robocasa import TASKS, validate_runtime_extension
from ora.orchestrator.spec.benchmark import _assemble, _placement, compose
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.spec.inputs import default_instance

MILESTONE = "ROBOCASA365"
_LANGUAGE_PROBE = r"""
import json, sys
from ora.environment.backends.robocasa.construction import build_environment
environment = build_environment(task=sys.argv[1], seed=int(sys.argv[2]))
environment.reset()
print(json.dumps({"lang": environment.get_ep_meta().get("lang")}))
environment.close()
"""


def episode_language(task: str, seed: int, *, python_env: str = "ROBOCASA_PYTHON") -> str:
    """The official instruction of one RoboCasa cell, read from its scene."""

    python = os.environ.get(python_env)
    if not python:
        raise ValueError(f"missing {python_env}")
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "LANG", "TMPDIR"}}
    environment.update({
        "PYTHONPATH": str(PROJECT_ROOT), "PYTHONNOUSERSITE": "1",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        "LIBGL_ALWAYS_SOFTWARE": "1", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"),
    })
    result = subprocess.run([python, "-c", _LANGUAGE_PROBE, task, str(seed)], capture_output=True,
                            text=True, timeout=600, env=environment, check=False)
    if result.returncode != 0:
        raise ValueError(f"RoboCasa language probe failed: {result.stderr.strip()[-600:]}")
    language = json.loads(result.stdout.strip().splitlines()[-1]).get("lang")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("RoboCasa scene has no task language")
    return language.strip()


def build_robocasa_spec(task: str, *, seed: int, experiment: str, resource_profile: str,
                        host: str | None = None, config: str | None = None, instruction: str | None = None,
                        provider_profile: str | None = None, provider_catalog: Path | None = None,
                        budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
                        cohort_id: str | None = None, milestone: str | None = None,
                        evidence_tier: str = "behavioral", baseline: str | None = None,
                        root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One RoboCasa365 cell (task, seed) on the default baseline."""

    default_instance("robocasa", root)
    validate_runtime_extension({"task": task, "seed": seed})
    instruction = instruction or episode_language(task, seed)
    runtime = {**_placement("robocasa", host=host, resource_profile=resource_profile),
               "task": task, "seed": seed, "expected_task_language": instruction}
    return _assemble(
        "robocasa", experiment=experiment, config=config or f"rc-{task}-s{seed}", task=task,
        instruction=instruction, seed=seed, runtime=runtime, qualification={"required": False},
        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
        service_report=service_report, cohort_id=cohort_id, milestone=milestone or MILESTONE,
        evidence_tier=evidence_tier, root=root, baseline=baseline)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    language = commands.add_parser("language", help="print the official instruction of one cell")
    language.add_argument("task", choices=sorted(TASKS))
    language.add_argument("--seed", type=int, required=True)
    write = commands.add_parser("compose", help="write one cell's RunSpec")
    write.add_argument("task", choices=sorted(TASKS))
    write.add_argument("--seed", type=int, required=True)
    write.add_argument("--experiment", required=True)
    write.add_argument("--resource-profile", required=True)
    write.add_argument("--host")
    write.add_argument("--config")
    write.add_argument("--provider-profile")
    write.add_argument("--provider-catalog", type=Path)
    write.add_argument("--service-report", type=Path)
    write.add_argument("--evidence-tier", default="behavioral")
    write.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "language":
        print(episode_language(args.task, args.seed))
        return 0
    spec = build_robocasa_spec(
        args.task, seed=args.seed, experiment=args.experiment, resource_profile=args.resource_profile,
        host=args.host, config=args.config, provider_profile=args.provider_profile,
        provider_catalog=args.provider_catalog,
        service_report=args.service_report, evidence_tier=args.evidence_tier,
    )
    print(compose(spec, output=args.output))
    return 0
