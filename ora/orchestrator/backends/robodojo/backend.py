"""The RoboDojo backend as the orchestrator core sees it (``registry.BACKENDS["robodojo"]``)."""

from __future__ import annotations

from . import toolkit_live

from ora.environment.benchmarks.robodojo import PHYSICS_STEPS_PER_NATIVE_ACTION
from ora.orchestrator import robodojo_play_audit
from ora.orchestrator.backends.registry import BackendCapabilities, OrchestratorBackend, Tool
from ora.orchestrator.backends.robodojo import (
    admission, doctor, evaluation, physics_attestation, play, preflight, round_scene, run_cell, runtime, supervision,
)
from ora.orchestrator.backends.robodojo.embodiment import DESCRIPTOR_PATH, PROJECT_ROOT

#: What a Kit runtime still writes outside TMPDIR, as far as the source shows (host_storage).
KIT_STORAGE_REASON = (
    "the bootstrap binds TMPDIR and the OptiX, XDG, CUDA and GL caches under its "
    "runtime directory, but Kit's crash reporter keeps local dumps with no dump "
    "directory set here (Kit default, not verified off /), and nothing that runs "
    "before the bootstrap is redirected")

SPEC = "ora.orchestrator.backends.robodojo.spec:main"


def _compose_task(task, binding):
    """``ora task run``'s RunSpec composer, imported on use like the spec tools."""
    from ora.orchestrator.backends.robodojo.spec import compose_task

    return compose_task(task, binding)


#: Product code starts ``card`` and ``task`` as subprocesses too (``run_cell``).
LANE = "ora.orchestrator.backends.robodojo.lane_qualification:main"

BACKEND = OrchestratorBackend(
    kind="robodojo",
    resolve_runtime=runtime.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    capabilities=BackendCapabilities(
        persistent_session=True, boolean_verdict_required=True, native_requires_images=True,
        baseline_instance=True, physics_steps_per_native_action=PHYSICS_STEPS_PER_NATIVE_ACTION,
        kit_storage_reason=KIT_STORAGE_REASON,
        # The ARX descriptor the embodiment activation compiles and binds.
        pinned_descriptors=(DESCRIPTOR_PATH.relative_to(PROJECT_ROOT).as_posix(),),
    ),
    validate_session=runtime.validate_session,
    provider_defaults=runtime.provider_defaults,
    revision_label=runtime.revision_label,
    manifest_admitters={robodojo_play_audit.AUDIT_SCHEMA: admission.admit_play_audit},
    supervisor_binding=supervision.supervisor_binding,
    round_scene=round_scene,
    official_complete=supervision.official_complete,
    evidence_errors=evaluation.evidence_errors,
    evaluation_extras=physics_attestation.evaluation_extras,
    seal_extras=physics_attestation.seal_extras,
    play=play.PLAY,
    play_audit=robodojo_play_audit,
    run_cell=run_cell.RUN_CELL,
    compose_task=_compose_task,
    spec_tools={
        "tasks": Tool(SPEC, "list the contracted tasks with their physics device and limits", ("tasks",)),
        "spec": Tool(SPEC, "print one task's RunSpec", ("spec",)),
        "compose": Tool(SPEC, "write one task's RunSpec", ("compose",)),
    },
    qualify_tools={
        "motion": Tool("ora.orchestrator.backends.robodojo.qualification:main",
                       "scripted sparse-motion qualification with host-owned execution state"),
        "card": Tool(LANE, "the lane's card record: two full-length episodes (35-56 min)", ("card",)),
        "task": Tool(LANE, "the lane's round-0 task record on a card record", ("task",)),
        "smoke": Tool(LANE, "the bounded scene smoke of one lane", ("smoke",)),
        "revalidate": Tool(LANE, "the lane gate, offline", ("revalidate",)),
        "physics-probe": Tool("ora.orchestrator.backends.robodojo.physics_probe:main",
                              "the physics scene probe of the CPU-physics lane"),
        "motion-replay": Tool("ora.orchestrator.backends.robodojo.motion.replay:main",
                              "control replay of a sealed run (layer L0)"),
        "motion-metrics": Tool("ora.orchestrator.backends.robodojo.motion.metrics:main",
                               "pinned motion-quality metrics of sealed runs (layer L0)"),
    },
    toolkit_live=toolkit_live.open_live,
    doctor=doctor.DOCTOR,
    # code-mode: the sandbox interpreter of the default policy (robodojo_toolkit_v7 runs code mode).
    gpu_setup=("agent-core", "perception", "robodojo", "motion-timing", "code-mode"),
)

__all__ = ["BACKEND", "KIT_STORAGE_REASON"]
