"""Session-scoped, policy-readable references for recoverable Context payloads.

The names are defined in ``markers`` (the recoverable envelope, pointer and
marker), ``read_tool`` (the ``artifact_read`` definition and result schemas),
``catalog`` (the session's ``ContextArtifactCatalog``) and ``tool_runtime``
(the ToolRuntime that serves ``artifact_read``). This module re-exports them
for the orchestrator; code in ``ora.agent`` imports the defining module.
"""

from __future__ import annotations

from .artifact_openers import ARTIFACT_READ_NAMES, artifact_read_name, recoverable_artifact_open
from .catalog import CONTEXT_ARTIFACT_INDEX_SCHEMA, ContextArtifactCatalog
from .markers import (
    CONTEXT_ARTIFACT_REF_SCHEMA,
    CONTEXT_TOOLS,
    LOSSLESS_CONTEXT_RETENTION,
    RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
    RECOVERABLE_ARTIFACT_POINTER_SCHEMA,
    RECOVERABLE_ARTIFACT_TAG,
    context_artifact_reader,
    decode_recoverable_artifact_marker,
    encode_recoverable_artifact_marker,
    recoverable_artifact_envelope,
    recoverable_artifact_pointer,
    validate_recoverable_artifact_envelope,
    validate_recoverable_artifact_pointer,
)
from .read_tool import (
    ARTIFACT_READ_RESULT_SCHEMA,
    ARTIFACT_READ_TOOL,
    CURRENT_OBSERVE_RESULT_SCHEMA,
    HISTORICAL_OBSERVE_RESULT_SCHEMA,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from .tool_runtime import RecoverableArtifactToolRuntime, build_recoverable_artifact_runtime

__all__ = [
    "ARTIFACT_READ_NAMES", "ARTIFACT_READ_RESULT_SCHEMA", "ARTIFACT_READ_TOOL",
    "CONTEXT_ARTIFACT_INDEX_SCHEMA", "CONTEXT_ARTIFACT_REF_SCHEMA", "CONTEXT_TOOLS",
    "CURRENT_OBSERVE_RESULT_SCHEMA", "HISTORICAL_OBSERVE_RESULT_SCHEMA",
    "RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA", "RECOVERABLE_ARTIFACT_POINTER_SCHEMA",
    "RECOVERABLE_ARTIFACT_TAG",
    "ContextArtifactCatalog", "LOSSLESS_CONTEXT_RETENTION",
    "RecoverableArtifactToolRuntime", "artifact_read_name", "build_recoverable_artifact_runtime",
    "context_artifact_reader",
    "decode_recoverable_artifact_marker", "encode_recoverable_artifact_marker",
    "recoverable_artifact_envelope", "recoverable_artifact_open", "validate_recoverable_artifact_envelope",
    "recoverable_artifact_pointer", "validate_recoverable_artifact_pointer",
    "recoverable_artifact_definitions", "recoverable_artifact_result_schemas",
]
