"""Artifacts calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Artifacts:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def tool_context(self) -> dict:
        return self._backend.tool_context()

    def tool_artifact(self, *, kind: str, key: str, cache: bool = False) -> dict | None:
        return self._backend.tool_artifact(kind=kind, key=key, cache=cache)

    def publish_tool_artifact(self, *, kind: str, key: str, files: dict[str, bytes], metadata: dict) -> dict:
        return self._backend.publish_tool_artifact(kind=kind, key=key, files=files, metadata=metadata)

    def read_blob(self, ref: dict) -> bytes:
        return self._backend.read_blob(ref=ref)

    def save_array(self, value: object) -> dict:
        return self._backend.save_array(value=value)
