"""Compute calls forward to the injected owner; no mutable data is retained."""
from dataclasses import dataclass
from typing import Any
from ..types import Response


@dataclass(frozen=True, slots=True)
class Compute:
    """An immutable capability; all effects and storage belong to its backend."""
    _backend: Any

    def evaluate_numeric(self, *, source: str, bindings: dict, outputs: list | None = None) -> dict:
        return self._backend.evaluate_numeric(source=source, bindings=bindings, outputs=outputs)
