"""A read-only view of the Context Artifact index that the TypeScript core writes.

The TypeScript context policy owns ``native/context-artifacts/index.json``.
Python robot tools that reopen recoverable artifacts (``artifact_read``,
``observe(ref)``) still execute in Python, so they resolve logical refs
through this view, which re-reads the index on every lookup. The RPC is
synchronous, so the file is current whenever a tool runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ora.agent.memory.catalog import ContextArtifactCatalog, CONTEXT_ARTIFACT_INDEX_SCHEMA


@dataclass
class ReadOnlyContextCatalog(ContextArtifactCatalog):
    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.index_path = self.run_root / "native/context-artifacts/index.json"
        self._entries = {}

    def _reload(self) -> None:
        if not self.index_path.is_file():
            self._entries = {}
            return
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != CONTEXT_ARTIFACT_INDEX_SCHEMA
            or value.get("policy_session_id") != self.policy_session_id
            or not isinstance(value.get("artifacts"), Mapping)
        ):
            raise ValueError("Context Artifact index does not belong to this PolicySession")
        self._entries = dict(value["artifacts"])

    def _write_index(self) -> None:
        raise RuntimeError("the Context Artifact index is owned by the TypeScript core")

    def register_storage_ref(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("the Context Artifact index is owned by the TypeScript core")

    def resolve(self, logical_ref: str) -> tuple[dict[str, Any], bytes]:
        self._reload()
        return super().resolve(logical_ref)

    def public_refs_for_round(self, round_index: int | None) -> list[dict[str, Any]]:
        self._reload()
        return super().public_refs_for_round(round_index)
