"""Canonical sizing of a message list is exact and reuses the messages an edit shares.

The cases that counted how often the Python Context pipeline re-rendered its
L1-L4 stages, and that its passes copy only the messages they edit, went with
that pipeline; its projections are the recorded Context fixtures the TypeScript
pipeline replays.
"""
from __future__ import annotations

import copy
import random
from typing import Any


from tests.support.agent.context_request_size import CanonicalSizer
from ora.agent.evidence.canonical import canonical_bytes


def test_message_list_sizing_is_exact_and_reuses_shared_messages() -> None:
    rng = random.Random(7)
    sizer = CanonicalSizer()

    def message(index: int) -> dict[str, Any]:
        return {
            "role": rng.choice(["user", "tool", "assistant"]),
            "content": [{"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + "A" * rng.randrange(1024, 4096),
            }}, {"type": "text", "text": "ü" * rng.randrange(5)}],
            "_source_event_ids": [f"agent-event-{index:06d}"],
            "nested": {"_source_event_ids": [f"nested-{index}"], "v": [1, None, 2.5]},
        }

    history = [message(index) for index in range(40)]
    for _ in range(5):
        edited = list(history)
        for index in rng.sample(range(len(edited)), 5):
            replacement = dict(edited[index])
            replacement["content"] = "replaced"
            edited[index] = replacement
        for value in (history, edited, [], [history[0]]):
            size, ids = sizer.measure(value)
            assert size == len(canonical_bytes(value))
            expected = {
                item for element in value for item in (
                    *element.get("_source_event_ids", []),
                    *element.get("nested", {}).get("_source_event_ids", []),
                )
            }
            assert ids == expected
        sizer.rotate()
