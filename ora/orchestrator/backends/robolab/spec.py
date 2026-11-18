"""Fresh RoboLab RunSpecs from the default baseline plus cell, provider and resource inputs.

One cell is one episode of one admitted task (``benchmarks/robolab.TASKS``)
on one seed. The instruction is the task's upstream default instruction. The
Kit renderer device is the resource slot's CUDA device, so the RunSpec's
``renderer_gpu`` always names the leased card. ``official_limit_multiple``
left unset runs on past the official limit (the runtime's default); 1 stops
the episode at the official limit, as RoboDojo's composer allows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robolab import TASKS, validate_runtime_extension
from ora.orchestrator.spec.benchmark import _assemble, _placement, compose
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.preflight.resources import resolve_execution_resource
from ora.orchestrator.spec.inputs import default_instance

MILESTONE = "ROBOLAB"


def build_robolab_spec(task: str, *, seed: int, experiment: str, resource_profile: str,
                       host: str | None = None, config: str | None = None,
                       provider_profile: str | None = None, provider_catalog: Path | None = None,
                       budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
                       cohort_id: str | None = None, milestone: str | None = None,
                       evidence_tier: str = "behavioral", baseline: str | None = None,
                       official_limit_multiple: int | None = None,
                       root: Path = PROJECT_ROOT) -> dict[str, Any]:
    default_instance("robolab", root)
    validate_runtime_extension({"task": task, "seed": seed})
    placement = _placement("robolab", host=host, resource_profile=resource_profile)
    binding = resolve_execution_resource(resource_profile, runtime_kind="robolab",
                                         required_host=placement["required_host"])
    device = binding["worker_environment"].get("CUDA_VISIBLE_DEVICES")
    if device is None or not str(device).isdigit():
        raise ValueError("a RoboLab slot must bind exactly one CUDA device")
    instruction = TASKS[task]["instruction"]
    runtime = {**placement, "task": task, "seed": seed, "expected_task_language": instruction,
               "renderer_gpu": int(device)}
    if official_limit_multiple is not None:
        runtime["official_limit_multiple"] = official_limit_multiple
    return _assemble(
        "robolab", experiment=experiment, config=config or f"rl-{task}-s{seed}", task=task,
        instruction=instruction, seed=seed, runtime=runtime, qualification={"required": False},
        provider_profile=provider_profile, provider_catalog=provider_catalog, budgets=budgets,
        service_report=service_report, cohort_id=cohort_id, milestone=milestone or MILESTONE,
        evidence_tier=evidence_tier, root=root, baseline=baseline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resource-profile", required=True)
    parser.add_argument("--host")
    parser.add_argument("--config")
    parser.add_argument("--provider-profile")
    parser.add_argument("--provider-catalog", type=Path)
    parser.add_argument("--service-report", type=Path)
    parser.add_argument("--evidence-tier", default="behavioral")
    parser.add_argument("--official-limit-multiple", type=int,
                        help="run on to this multiple of the official native-action limit; "
                             "1 stops at the official limit (default: the runtime's, no limit binds)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    spec = build_robolab_spec(args.task, seed=args.seed, experiment=args.experiment,
                              resource_profile=args.resource_profile, host=args.host, config=args.config,
                              provider_profile=args.provider_profile, provider_catalog=args.provider_catalog,
                              service_report=args.service_report, evidence_tier=args.evidence_tier,
                              official_limit_multiple=args.official_limit_multiple)
    print(compose(spec, output=args.output))
    return 0
