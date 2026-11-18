"""Snapshot-only missing-key semantics; legacy/live reads retain StateError."""
from dataclasses import dataclass

from ...state.keys import StateAccessDenied, StateError
from ..types import NotRecorded
from .live import Privileged as LivePrivileged


@dataclass(frozen=True, slots=True)
class Privileged(LivePrivileged):
    def read(self, key: str):
        """Read exported privilege; never query live or substitute a later value."""
        try:
            return LivePrivileged.read(self, key)
        except StateAccessDenied:
            raise
        except StateError as error:
            raise NotRecorded(f"未记录: {key}") from error
