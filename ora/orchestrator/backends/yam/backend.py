"""The YAM station backend as the orchestrator core sees it (``registry.BACKENDS["yam"]``)."""

from __future__ import annotations

from ora.orchestrator.backends.registry import BackendCapabilities, OrchestratorBackend, Tool
from ora.orchestrator.backends.yam import play, preflight, runtime

BACKEND = OrchestratorBackend(
    kind="yam",
    resolve_runtime=runtime.resolve_runtime,
    preflight_runtime=preflight.preflight_runtime,
    # No evaluator on a real cell: the verdict is null and the operator judges the run.
    capabilities=BackendCapabilities(boolean_verdict_required=False, native_requires_images=True),
    revision_label=runtime.revision_label,
    qualify_tools={"motion": Tool("ora.orchestrator.backends.yam.qualification:main",
                                  "the scripted sparse-motion qualification record (simulated station)")},
    # Free play on the cell as it stands (ora play --env yam).
    play=play.PLAY,
)

__all__ = ["BACKEND"]
