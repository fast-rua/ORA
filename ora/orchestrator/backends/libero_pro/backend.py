"""The LIBERO-Pro backend as the orchestrator core sees it (``registry.BACKENDS["libero_pro"]``, alias ``libero``)."""

from __future__ import annotations

from ora.orchestrator.backends.libero_pro import (
    configuration, evaluation, play, preflight, run_cell, supervision,
)
from ora.orchestrator.backends.registry import BackendCapabilities, DoctorProbe, OrchestratorBackend, Tool

SPEC = "ora.orchestrator.spec.benchmark:libero_main"

BACKEND = OrchestratorBackend(
    kind="libero_pro",
    resolve_runtime=configuration.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    capabilities=BackendCapabilities(
        persistent_session=True, native_requires_images=True, baseline_instance=True,
        # LIBERO's max_sim_steps counts control ticks, one per native action, and is the
        # step limit an episode is evaluated under (the per-suite protocol limit).
        step_cap_is_native_limit=True,
    ),
    revision_label=configuration.revision_label,
    supervisor_binding=supervision.supervisor_binding,
    round_record_errors=evaluation.round_record_errors,
    play=play.PLAY,
    run_cell=run_cell.RUN_CELL,
    # One original LIBERO base suite or LIBERO-Pro core suite cell, chosen by its suite.
    spec_tools={
        "tasks": Tool(SPEC, "list the LIBERO base suites' and LIBERO-Pro core suites' task languages", ("tasks",)),
        "spec": Tool(SPEC, "print one suite task's RunSpec", ("spec",)),
        "compose": Tool(SPEC, "write one suite task's RunSpec", ("compose",)),
    },
    qualify_tools={"motion": Tool("ora.orchestrator.backends.libero_pro.qualification:main",
                                  "the scripted sparse-motion qualification record")},
    # The probe finds the package in the checkout's own source tree first.
    doctor=DoctorProbe(python_env="LIBERO_PRO_PYTHON", module="libero",
                       directory_envs=("LIBERO_PRO_ROOT", "LIBERO_PRO_DATA_ROOT"),
                       source_env=("LIBERO_PRO_ROOT", "libero")),
    # LIBERO's planner, cuRobo V2, runs under the RoboDojo runtime's interpreter
    # (CUROBO_V2_*). Every global route is timed by the small motion-timing
    # interpreter (ORA_TIMING_PYTHON).
    gpu_setup=("agent-core", "perception", "libero", "robodojo", "motion-timing"),
)

__all__ = ["BACKEND"]
