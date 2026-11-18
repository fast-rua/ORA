"""Test helpers for lossless context runtime, shared by several test modules.

They were defined in ``tests/ora/agent/memory/test_lossless_context_runtime.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import base64
import json
from typing import Any


def _image(label: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(
            label.encode("utf-8")
        ).decode("ascii")},
    }


def _protected() -> list[dict[str, Any]]:
    return [
        {
            "role": "system", "content": "system",
            "_protected_context": True, "_context_source": "harness_system",
        },
        {
            "role": "user", "content": "embodiment",
            "_protected_context": True,
            "_context_source": "project_instructions",
        },
        {
            "role": "user", "content": "pick object",
            "_protected_context": True, "_context_source": "episode_task",
        },
    ]


def _messages(
    completed: int, *, large_decisions: set[int] | None = None,
) -> list[dict[str, Any]]:
    messages = _protected()
    large = large_decisions or set()
    for step in range(completed + 1):
        if step == 2:
            messages.append({
                "role": "user",
                "content": (
                    '<official_round_result>\n{"instruction":"reflect",'
                    '"passed":false,"round":1}\n</official_round_result>'
                ),
                "_protected_context": True,
                "_context_source": "official_round_result_1",
                "_source_event_ids": ["agent-event-900000"],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps({
                    "current_actual": {"step": step},
                    "observation_text": f"round observation {step}",
                })},
                _image(f"image-{step}"),
            ],
            "_observation": {
                "step": step,
                "receipt": (
                    {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
                    if step else None
                ),
            },
            "_session_round": 1 if step < 2 else 2,
            "_source_event_ids": [f"agent-event-{step * 5:06d}"],
        })
        if step == completed:
            continue
        read_id = f"observe-{step}"
        mutation_id = f"mutation-{step}"
        messages.extend([
            {
                "role": "assistant",
                "content": (
                    f"decision and reflection {step}" + " durable" * 20_000
                    if step in large else f"decision and reflection {step}"
                ),
                "tool_calls": [{
                    "id": read_id, "type": "function",
                    "function": {"name": "observe", "arguments": "{}"},
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 1:06d}"],
            },
            {
                "role": "tool", "tool_call_id": read_id, "name": "observe",
                "content": json.dumps({
                    "status": "actual", "views": ["agentview", "wrist"],
                    "diagnostic": "same read-only payload",
                }),
                "_source_event_ids": [f"agent-event-{step * 5 + 2:06d}"],
            },
            {
                "role": "assistant", "content": f"action intent {step}",
                "tool_calls": [{
                    "id": mutation_id, "type": "function",
                    "function": {
                        "name": "move_pose",
                        "arguments": json.dumps({"step": step}),
                    },
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 3:06d}"],
            },
            {
                "role": "tool", "tool_call_id": mutation_id,
                "name": "move_pose",
                "content": json.dumps({"status": "submitted", "step": step}),
                "_source_event_ids": [f"agent-event-{step * 5 + 4:06d}"],
            },
        ])
    return messages
