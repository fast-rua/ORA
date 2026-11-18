"""The RoboLab backend as the orchestrator core sees it (``registry.BACKENDS["robolab"]``)."""

from __future__ import annotations

from ora.orchestrator.backends.registry import BackendCapabilities, OrchestratorBackend, Tool
from ora.orchestrator.backends.robolab import preflight, runtime

#: What a Kit runtime still writes outside TMPDIR, as far as the source shows (host_storage).
KIT_STORAGE_REASON = (
    "the bridge binds the OptiX, XDG, CUDA and GL caches under the run's runtime "
    "directory, but Kit's crash reporter keeps local dumps with no dump directory "
    "set here, and nothing that runs before the bridge is redirected")

SPEC = "ora.orchestrator.backends.robolab.spec:main"
CATALOG = "ora.orchestrator.backends.robolab.task_catalog:main"

BACKEND = OrchestratorBackend(
    kind="robolab",
    resolve_runtime=runtime.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    capabilities=BackendCapabilities(
        boolean_verdict_required=True, native_requires_images=True, baseline_instance=True,
        kit_storage_reason=KIT_STORAGE_REASON,
    ),
    revision_label=runtime.revision_label,
    spec_tools={"compose": Tool(SPEC, "write one cell's RunSpec"),
                "tasks": Tool(CATALOG, "list the roster with each task's contract and compatibility state",
                              ("tasks",))},
    qualify_tools={"motion": Tool("ora.orchestrator.backends.robolab.qualification:main",
                                  "the scripted sparse-motion qualification record"),
                   "contract-witness": Tool("ora.orchestrator.backends.robolab.contract_witness:main",
                                            "seed-0 task-contract witnesses; fills the reviews' witness checks")},
)

__all__ = ["BACKEND", "KIT_STORAGE_REASON"]
