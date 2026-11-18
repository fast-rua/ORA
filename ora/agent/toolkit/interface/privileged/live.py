"""RSI-only state reads. No simulator query, key discovery or mutation."""
from dataclasses import dataclass
from typing import Any
from ..types import AccessDenied


@dataclass(frozen=True, slots=True)
class Privileged:
    _access: Any

    def read(self, key: str) -> Any:
        """Live/replay: read a privileged value already published at this boundary."""
        if not key.startswith('privileged.'):
            raise AccessDenied('privileged capability reads only privileged.*')
        return self._access.read(key)
