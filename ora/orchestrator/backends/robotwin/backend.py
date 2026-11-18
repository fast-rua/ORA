"""The RoboTwin backend as the orchestrator core sees it (``registry.BACKENDS["robotwin"]``)."""

from __future__ import annotations

from ora.orchestrator.backends.registry import ANY_SCHEMA, BackendCapabilities, DoctorProbe, OrchestratorBackend, Tool
from ora.orchestrator.backends.robotwin import (
    admission, play, play_audit, preflight, qualification_exception, run_cell, runtime, seed_manifest,
)
from ora.orchestrator.spec.runtime import ENVIRONMENT_SIGNAL_POLICIES

SPEC = "ora.orchestrator.spec.benchmark:robotwin_main"
#: Product code starts its ``qualify`` as a subprocess too (``run_cell``).
SEEDS = "ora.orchestrator.backends.robotwin.evaluation_seeds:main"

BACKEND = OrchestratorBackend(
    kind="robotwin",
    resolve_runtime=runtime.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    capabilities=BackendCapabilities(
        persistent_session=True, boolean_verdict_required=True,
        signal_policies=frozenset(ENVIRONMENT_SIGNAL_POLICIES),
        baseline_instance=True, baseline_requires_seed_manifest=True,
    ),
    manifest_admitters={
        seed_manifest.SEED_MANIFEST_SCHEMA: admission.admit_seed_manifest,
        play_audit.AUDIT_SCHEMA: admission.admit_play_audit,
        # An older per-task seed manifest names no schema.
        ANY_SCHEMA: admission.admit_seed_manifest,
    },
    expand_template=runtime.expand_template,
    qualification_exception=qualification_exception,
    seed_manifest=seed_manifest,
    play=play.PLAY,
    play_audit=play_audit,
    run_cell=run_cell.RUN_CELL,
    spec_tools={
        "tasks": Tool(SPEC, "list the RoboTwin roster and its public instructions", ("tasks",)),
        "spec": Tool(SPEC, "print one task's RunSpec", ("spec",)),
        "compose": Tool(SPEC, "write one task's RunSpec and import the seed manifest that admits it", ("compose",)),
    },
    qualify_tools={
        "configs": Tool(SEEDS, "print the evaluation protocol and the task roster", ("configs",)),
        "prepare": Tool(SEEDS, "plan an expert seed scan of explicit tasks on explicit slots", ("prepare",)),
        "qualify": Tool(SEEDS, "run a prepared seed scan (--plan, --confirm) and write its seed manifests",
                        ("qualify",)),
        "merge": Tool(SEEDS, "merge a task's scan segments into one first-N seed manifest", ("merge",)),
        "seal": Tool(SEEDS, "write interim seed manifests from scans still running", ("seal",)),
    },
    doctor=DoctorProbe(python_env="ROBOTWIN_PYTHON", module="sapien", directory_envs=("ROBOTWIN_ROOT",)),
    # The shared planner's interpreter and source are the RoboDojo runtime's, as on LIBERO.
    gpu_setup=("agent-core", "perception", "robotwin-python", "robotwin-assets", "robodojo", "motion-timing"),
)

__all__ = ["BACKEND"]
