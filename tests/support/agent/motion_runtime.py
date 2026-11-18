"""Test helpers for motion runtime, shared by several test modules.

They were defined in ``tests/ora/environment/controllers/test_motion_runtime.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from ora.contracts import MOTION_ACTION_SCHEMA


def _action(
    name: str, arguments: dict[str, object], *, sequence: int = 0,
) -> dict[str, object]:
    return {
        "schema": MOTION_ACTION_SCHEMA,
        "observation_id": f"observation-{sequence}",
        "action_id": f"action-{sequence}",
        "action": name,
        "arguments": arguments,
        "bound_observation_sha256": "bound",
    }


def _pose() -> dict[str, list[float]]:
    return {
        "position_m": [0.4, 0.0, 0.5],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
