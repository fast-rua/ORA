"""Legacy v1 proposal scope; synchronous execution never consults this guard."""
from dataclasses import dataclass
from .interface.types import AccessDenied


@dataclass
class CallScope:
    proposed: bool = False

    def admit_proposal(self, *, commit: bool = True) -> None:
        # Preserve v1 rejection bytes; v2 has no submit_action capability.
        if self.proposed:
            raise AccessDenied("one action proposal per parent tool invocation")
        if commit:
            self.proposed = True
