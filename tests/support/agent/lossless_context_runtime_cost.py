"""Test helpers for lossless context runtime cost, shared by several test modules.

They were defined in
``tests/ora/agent/context/lossless/test_lossless_context_runtime_cost.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


def _image_bytes(label: str, size: int) -> bytes:
    seed = hashlib.sha256(label.encode()).digest()
    blocks = [
        hashlib.sha256(seed + index.to_bytes(8, "big")).digest()
        for index in range(size // 32 + 1)
    ]
    return b"".join(blocks)[:size]


def _url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _observation(step: int, session_round: int, size: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps({
                "current_actual": {
                    "step": step, "joints": [0.1 * index for index in range(20)],
                    "note": "measured " * 20,
                },
                "observation_text": f"observation {step}",
            })},
            {"type": "image_url", "image_url": {"url": _url(_image_bytes(f"obs-{step}-a", size))}},
            {"type": "image_url", "image_url": {"url": _url(_image_bytes(f"obs-{step}-b", size))}},
        ],
        "_observation": {"step": step, "receipt": (
            {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
            if step else None
        )},
        "_session_round": session_round,
        "_source_event_ids": [f"agent-event-{step * 10:06d}"],
    }
