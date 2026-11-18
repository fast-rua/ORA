"""Historical call cursor and diagnostic output owned by a replay session."""
from dataclasses import dataclass, field


@dataclass
class ReplaySession:
    cursor: int = 0
    proposed_actions: list = field(default_factory=list)
    proposed_controls: list = field(default_factory=list)
    missing_requests: list = field(default_factory=list)
