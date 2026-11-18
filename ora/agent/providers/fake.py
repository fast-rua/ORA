"""A deterministic scripted provider for tests and offline runs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError


@dataclass
class FakeProvider:
    """Return scripted assistant messages without network access."""

    responses: Sequence[Mapping[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    closed: bool = False
    protocol: str = "fake"

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        if self.closed:
            raise PolicySessionError("FakeProvider is closed")
        if self.index >= len(self.responses):
            raise PolicySessionError("FakeProvider response script is exhausted")
        self.requests.append(copy.deepcopy(dict(request)))
        scripted = copy.deepcopy(dict(self.responses[self.index]))
        self.index += 1
        message = scripted.get("message", scripted)
        if not isinstance(message, Mapping):
            raise PolicySessionError("FakeProvider response message must be an object")
        usage = scripted.get("usage") if "message" in scripted else None
        finish_reason = scripted.get("finish_reason")
        if finish_reason is None:
            finish_reason = (
                "tool_calls"
                if isinstance(message.get("tool_calls"), list)
                and message["tool_calls"]
                else "stop"
            )
        return {
            "message": dict(message),
            "usage": usage,
            "finish_reason": finish_reason,
            "raw_events": [{"fake": True, "index": self.index - 1}],
            **({
                "output_rejection": copy.deepcopy(scripted["output_rejection"]),
            } if isinstance(scripted.get("output_rejection"), Mapping) else {}),
            "model_call_id": model_call_id,
            "started_unix_s": time.time(),
            "finished_unix_s": time.time(),
            "ttft_s": 0.0,
            "attempts": 1,
        }

    def close(self) -> None:
        self.closed = True
