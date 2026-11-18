"""Declare model attachments to the context owner; retention belongs there."""
from dataclasses import dataclass
from typing import Any
from ..types import AccessDenied


@dataclass(frozen=True, slots=True)
class Context:
    _owner: Any = None

    def declare(self, attachment: dict) -> Any:
        """Declare an attachment; the context layer decides delivery and retention."""
        if self._owner is None:
            raise AccessDenied('context attachment capability is unavailable')
        return self._owner(attachment)
