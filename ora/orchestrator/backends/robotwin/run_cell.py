"""``ora run --env robotwin``: a seed the task's seed manifest admits, scanned once when none is reusable.

Reuse needs a seed manifest that admits a seed of the task under the scene
configuration (``spec.benchmark.admitted_seed_manifest``, launch preflight's
rule); otherwise ``evaluation_seeds prepare`` and ``qualify`` (``ora qualify
robotwin``) scan that task alone and stop at the first accepted seed. The scan
runs as the subprocess an operator would type, with a progress line and its log under
``outputs/ora-interactive/logs``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.orchestrator.progress import log_path, newest, run_with_progress

SLOT = "robotwin-gpu0"


def slot(gpu: int | None) -> str:
    """The single-gpu host's RoboTwin slot on GPU ``gpu`` (``ora run --gpu``; None is GPU 0)."""
    return f"robotwin-gpu{gpu or 0}"
#: The ``.env`` keys the cell runs on, beside the perception services'.
REQUIRED_ENV = ("ROBOTWIN_ROOT", "ROBOTWIN_PYTHON", "CUROBO_ROOT", "CUROBO_PYTHON", "CUROBO_V2_ROOT",
                "CUROBO_V2_PYTHON", "ORA_TIMING_PYTHON")
EXAMPLE = "click_bell"

Confirm = Callable[[str], None]


def admitted_seed(path: Path, *, task: str, task_config: str, seed: int | None) -> int | None:
    """``seed``, or the first accepted seed, when the manifest at ``path`` admits it; else None."""
    from ora.orchestrator.spec.benchmark import admitted_seed_manifest

    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for candidate in [seed] if seed is not None else list(manifest.get("accepted_seeds") or []):
        try:
            admitted_seed_manifest(path, task=task, seed=candidate, task_config=task_config)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return candidate
    return None


def find_seed_manifest(task: str, *, task_config: str, seed: int | None,
                       workspace: Path | None = None) -> tuple[Path, int] | None:
    """The newest manifest admitting ``seed`` (or its first accepted seed) for ``task``."""
    from ora.orchestrator.backends.robotwin import evaluation_seeds

    for path in newest((workspace or evaluation_seeds.WORKSPACE).glob(f"qualification/*/seed-manifests/{task}.json")):
        admitted = admitted_seed(path, task=task, task_config=task_config, seed=seed)
        if admitted is not None:
            return path, admitted
    return None


def _scan_manifest(workspace: Path, checksum: str, task: str) -> Path:
    for prepared in newest(workspace.glob("qualification/*/prepared-plan.json")):
        try:
            if json.loads(prepared.read_text()).get("plan_sha256") == checksum:
                return prepared.parent / "seed-manifests" / f"{task}.json"
        except (OSError, ValueError):
            continue
    raise RuntimeError("the seed scan finished without its qualification record")


def robotwin_seed(task: str, *, task_config: str, seed: int | None, candidates: int, requalify: bool,
                  confirm: Confirm, environ: Mapping[str, str], stream=None, gpu: int | None = None) -> tuple[Path, int]:
    """A seed manifest and admitted seed for ``task``, scanning once when none is reusable."""
    from ora.orchestrator.backends.registry import backend
    from ora.orchestrator.backends.robotwin import evaluation_seeds

    output = stream or sys.stderr
    if not requalify:
        found = find_seed_manifest(task, task_config=task_config, seed=seed)
        if found is not None:
            print(f"RoboTwin {task}: reusing seed {found[1]} from {found[0]}", file=output, flush=True)
            return found
    plan = evaluation_seeds.prepare(
        robotwin_root=Path(environ["ROBOTWIN_ROOT"]), python=Path(environ["ROBOTWIN_PYTHON"]), tasks=[task],
        slots=[slot(gpu)], candidate_count=candidates, accepted_per_task=1, task_config=task_config)
    workspace = evaluation_seeds.WORKSPACE
    path = workspace / "preparations" / f"{time.time_ns()}" / "plan.json"
    evaluation_seeds.save(path, plan)
    minutes = plan["qualification"]["estimated_wall_minutes"]
    confirm(f"RoboTwin {task}: one-time expert seed scan on this machine (up to {candidates} seeds from the "
            f"official start seed, stops at the first the expert solves). Upper-bound estimate {minutes:.1f} min; "
            "click_bell took 2.5 min on an RTX PRO 6000.")
    checksum = plan["plan_sha256"]
    scan = backend("robotwin").qualify_tools["qualify"]
    run_with_progress(scan.command(sys.executable, "--plan", str(path), "--confirm", f"QUALIFY-ROBOTWIN:{checksum}"),
                      label=f"RoboTwin {task} seed scan", estimate=f"estimate <= {minutes:.0f} min",
                      log=log_path(f"robotwin-{task}-seed-scan"), stream=output,
                      # 2: the scan finished but did not pass (no accepted seed); explained below.
                      accepted_codes=(0, 2))
    manifest = _scan_manifest(workspace, checksum, task)
    admitted = admitted_seed(manifest, task=task, task_config=task_config, seed=seed)
    if admitted is None:
        accepted = json.loads(manifest.read_text()).get("accepted_seeds") if manifest.is_file() else []
        if seed is not None:
            raise RuntimeError(f"seed {seed} is not admitted (accepted: {accepted}); omit --seed")
        raise RuntimeError(f"the scan accepted no seed of {task} among {candidates} candidates; "
                           "rerun with a larger --candidates")
    return manifest, admitted


class RoboTwinRunCell:
    """``registry.RunCell`` of RoboTwin."""

    slot = SLOT
    required_env = REQUIRED_ENV
    example = EXAMPLE

    def compose(self, args: argparse.Namespace, *, common: Mapping[str, Any], host: str,
                services: Callable[[], Path], confirm: Confirm, stream: Any) -> Path:
        from ora.orchestrator.spec.benchmark import build_robotwin_spec, compose as write

        manifest, seed = robotwin_seed(
            args.task, task_config=args.task_config, seed=args.seed, candidates=args.candidates,
            requalify=args.requalify, confirm=confirm, environ=os.environ, stream=stream,
            gpu=getattr(args, "gpu", None))
        spec = build_robotwin_spec(args.task, seed=seed, seed_manifest=manifest, task_config=args.task_config,
                                   resource_profile=slot(getattr(args, "gpu", None)), host=host, service_report=services(), **common)
        return write(spec, seed_manifest=manifest)


RUN_CELL = RoboTwinRunCell()

__all__ = ["EXAMPLE", "REQUIRED_ENV", "RUN_CELL", "SLOT", "RoboTwinRunCell", "admitted_seed",
           "find_seed_manifest", "robotwin_seed"]
