"""The RoboCasa365 backend as the orchestrator core sees it (``registry.BACKENDS["robocasa"]``)."""

from __future__ import annotations

from ora.orchestrator.backends.registry import BackendCapabilities, OrchestratorBackend, Tool
from ora.orchestrator.backends.robocasa import preflight, runtime

SPEC = "ora.orchestrator.backends.robocasa.spec:main"

BACKEND = OrchestratorBackend(
    kind="robocasa",
    resolve_runtime=runtime.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    capabilities=BackendCapabilities(
        boolean_verdict_required=True, native_requires_images=True, baseline_instance=True,
    ),
    revision_label=runtime.revision_label,
    spec_tools={
        "language": Tool(SPEC, "print the official instruction of one cell", ("language",)),
        "compose": Tool(SPEC, "write one cell's RunSpec", ("compose",)),
    },
    qualify_tools={"motion": Tool("ora.orchestrator.backends.robocasa.qualification:main",
                                  "the scripted sparse-motion qualification record")},
)

__all__ = ["BACKEND"]
