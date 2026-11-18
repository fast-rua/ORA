"""Test helpers for lossless context stride, shared by several test modules.

They were defined in ``tests/ora/agent/context/lossless/test_lossless_context_stride.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ora.agent.context.lossless.management import (
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V6_SCHEMA,
)


CAMERAS = ("head", "left_wrist", "right_wrist")
V4 = {
    "schema": CONTEXT_MANAGEMENT_V4_SCHEMA,
    "causal_exact_tail_action_steps": 3,
    "inline_media_tail_action_steps": 2,
    "eager_large_result_bytes": 50_000,
    "eager_preview_bytes": 2_048,
    "eager_exact_deduplication": True,
    # Every request carries more than two images, so L4 is always triggered.
    "trigger_inline_images": 2,
    "trigger_estimated_tokens": 48_000,
    "trigger_provider_input_ratio": 0.70,
    "episode_checkpoint_trigger_tokens": 150_000,
    "episode_checkpoint_trigger_ratio": 0.60,
    "emergency_fallback_ratio": 0.80,
    "emergency_target_ratio": 0.65,
    "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600,
}


def _v5(stride: int, **overrides: Any) -> dict[str, Any]:
    return {
        **V4, "schema": CONTEXT_MANAGEMENT_V5_SCHEMA,
        "tail_aging_stride_action_steps": stride, **overrides,
    }


def _v6(stride: int, **overrides: Any) -> dict[str, Any]:
    """``_v5`` as v6, for runs on the agent core (it refuses v4, v5 and v7): an absolute L5 watermark."""

    fields = {key: value for key, value in V4.items() if key != "episode_checkpoint_trigger_ratio"}
    return {**fields, "schema": CONTEXT_MANAGEMENT_V6_SCHEMA, "tail_aging_stride_action_steps": stride, **overrides}


CHECKPOINT = {
    "task_progress": ["The object has not been confirmed as transported."],
    "confirmed_facts": ["The retained actual view is authoritative."],
    "invalidated_beliefs": ["An older alignment belief may be stale."],
    "failed_attempts": [{
        "failure": "The earlier attempt did not establish transport.",
        "evidence": "The retained evidence did not confirm attachment.",
        "correction": "Re-ground before the next grasp attempt.",
    }],
    "current_constraints": ["Use qualitative state only."],
    "do_not_repeat": ["Do not infer attachment from closure."],
    "re_ground_next": ["Inspect the current actual view."],
}


def _image(label: str) -> dict[str, Any]:
    payload = base64.b64encode(label.encode("utf-8")).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + payload}}


def _observation(step: int, session_round: int) -> dict[str, Any]:
    body = {
        "current_actual": {
            "step": step, "actual_sequence": step,
            "observation_id": f"observation-{step:04d}",
            # The bulky measured body L4 replaces with a recoverable pointer.
            "camera_geometry": {
                camera: {"intrinsic_cv": [[600.0 + step, 0.0, 320.0]] * 24}
                for camera in CAMERAS
            },
        },
        "comparison": {"available": step > 0},
        "terminal_lifecycle": False,
    }
    content: list[dict[str, Any]] = [{
        "type": "text", "text": json.dumps(body, sort_keys=True),
    }]
    for camera in CAMERAS:
        content.extend([
            {"type": "text", "text": f"CURRENT ACTUAL · {camera}"},
            _image(f"{camera}-step-{step}"),
        ])
    return {
        "role": "user", "content": content,
        "_observation": {
            "step": step,
            "receipt": (
                {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
                if step else None
            ),
        },
        "_session_round": session_round,
        "_source_event_ids": [f"agent-event-{step * 10:06d}"],
    }


def _read_exchange(step: int) -> list[dict[str, Any]]:
    call_id = f"observe-{step}"
    return [{
        "role": "assistant", "content": f"inspect before acting {step}",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "observe", "arguments": "{}"}}],
        "_source_event_ids": [f"agent-event-{step * 10 + 1:06d}"],
    }, {
        "role": "tool", "tool_call_id": call_id, "name": "observe",
        "content": json.dumps({
            "status": "actual", "views": list(CAMERAS),
            "detail": f"read-only body {step} " + "x" * 4_000,
        }, sort_keys=True),
        "_source_event_ids": [f"agent-event-{step * 10 + 2:06d}"],
    }]


def _mutation_exchange(step: int) -> list[dict[str, Any]]:
    call_id = f"mutation-{step}"
    return [{
        "role": "assistant", "content": f"act {step}",
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "move_pose", "arguments": json.dumps({"step": step}),
        }}],
        "_source_event_ids": [f"agent-event-{step * 10 + 3:06d}"],
    }, {
        "role": "tool", "tool_call_id": call_id, "name": "move_pose",
        "content": json.dumps({"status": "submitted", "step": step}),
        "_source_event_ids": [f"agent-event-{step * 10 + 4:06d}"],
    }]


def _episode(
    completed: int, *, second_round_from: int | None = None,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Return every request snapshot of an append-only synthetic episode.

    Each ActionStep is two Provider calls, as a measured RoboDojo cadence
    (1.85 calls per ActionStep): one on the fresh observation and one after a
    read-only tool result. The mutation then closes the step.
    """

    history: list[dict[str, Any]] = [
        {"role": "system", "content": "system", "_protected_context": True,
         "_context_source": "harness_system"},
        {"role": "user", "content": "embodiment", "_protected_context": True,
         "_context_source": "project_instructions"},
        {"role": "user", "content": "task", "_protected_context": True,
         "_context_source": "episode_task"},
    ]
    snapshots: list[tuple[int, list[dict[str, Any]]]] = []
    for step in range(completed + 1):
        session_round = (
            2 if second_round_from is not None and step >= second_round_from
            else 1
        )
        if step == second_round_from:
            history.append({
                "role": "user",
                "content": (
                    '<official_round_result>\n{"instruction":"reflect",'
                    '"passed":false,"round":1}\n</official_round_result>'
                ),
                "_protected_context": True,
                "_context_source": "official_round_result_1",
                "_source_event_ids": ["agent-event-900000"],
            })
        history.append(_observation(step, session_round))
        snapshots.append((step, list(history)))
        history.extend(_read_exchange(step))
        snapshots.append((step, list(history)))
        if step < completed:
            history.extend(_mutation_exchange(step))
    return snapshots
