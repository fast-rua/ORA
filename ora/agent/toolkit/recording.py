"""Mutable per-parent recording progress, separate from the data API."""
from dataclasses import dataclass


@dataclass
class InvocationRecording:
    ordinal: int = 0
    failed: bool = False
