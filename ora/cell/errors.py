"""The real-cell kit's one error type: a refusal with its reason, safe to show the operator."""

from __future__ import annotations


class CellError(RuntimeError):
    """The cell refuses: occupied, unsafe, misconfigured, or a device does not answer."""
