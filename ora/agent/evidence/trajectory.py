"""The agent trace's evidence API, re-exported for the orchestrator.

The names are defined in ``canonical`` (canonical bytes, hashes and the
content-addressed artifact store), ``agent_events`` (the event log),
``trace_privacy`` (the forbidden-fragment check), ``trace_verify``
(``verify_agent_trace``) and ``sft_export`` (``export_sft``). The orchestrator's
evaluation and ``ora export`` import them from here; code in ``ora.agent``
imports the defining module.
"""

from __future__ import annotations

from .agent_events import AGENT_EVENT_SCHEMA, VISIBILITY, read_agent_events
from .canonical import (
    ArtifactStore,
    TrajectoryError,
    canonical_bytes,
    file_sha256,
    image_data_url,
    sha256_bytes,
)
from .sft_export import SFT_EXPORT_SCHEMA, export_sft
from .trace_privacy import FORBIDDEN_FRAGMENTS
from .trace_verify import ACTION_DISCARD_REASONS, verify_agent_trace

__all__ = [
    "ACTION_DISCARD_REASONS",
    "AGENT_EVENT_SCHEMA",
    "ArtifactStore",
    "FORBIDDEN_FRAGMENTS",
    "SFT_EXPORT_SCHEMA",
    "TrajectoryError",
    "VISIBILITY",
    "canonical_bytes",
    "export_sft",
    "file_sha256",
    "image_data_url",
    "read_agent_events",
    "sha256_bytes",
    "verify_agent_trace",
]
