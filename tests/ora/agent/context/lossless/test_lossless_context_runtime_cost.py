"""Canonical request sizing is exact and reads no image twice.

The cost cases of this module bound the Python Context pipeline to the
golden digests of its pre-fix implementation; ORA removed that pipeline with
the Native core, and the TypeScript pipeline replays the recorded projections
(``tests/data/context_differential/cost_golden`` and ``captured_cost_*``).
The observation helper here also builds the histories of
``test_lossless_render_reuse.py``.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import pytest

from ora.agent.evidence.canonical import canonical_bytes
from tests.support.agent.lossless_context_runtime_cost import _image_bytes, _url

TOOLS = [{"type": "function", "function": {
    "name": name, "description": name, "parameters": {"type": "object"},
}} for name in ("move_pose", "observe", "grasp_plan")]


# Namespace migration audit: request contents differ only in schema names,
# their content hashes and derived byte/token counts. Projection transitions
# and checkpoint choices match the pre-migration recording.


def _legacy_ids(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        raw = value.get("_source_event_ids")
        if isinstance(raw, list):
            found.update(item for item in raw if isinstance(item, str))
        for child in value.values():
            found.update(_legacy_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_legacy_ids(child))
    return sorted(found)


def test_canonical_sizer_matches_serialization_exactly() -> None:
    from tests.support.agent.context_request_size import CanonicalSizer

    data_url = _url(_image_bytes("sizer", 5_000))
    odd_url = "data:text/plain," + 'quote " back \\ tab \t é 中  ' * 200
    value = [
        {"role": "user", "_source_event_ids": ["a", 3, "b"], "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": odd_url},
            {"nested": ({"_source_event_ids": ["hidden-in-tuple"]}, data_url)},
        ]},
        {"f": 2.5, "k": None, "b": True, "same": data_url, "short": "data:x"},
        {"_source_event_ids": "not-a-list", "child": [{"_source_event_ids": ["c"]}]},
    ]
    sizer = CanonicalSizer()
    size, ids = sizer.measure(value)
    assert size == len(canonical_bytes(value))
    assert sorted(ids) == _legacy_ids(value)
    assert "hidden-in-tuple" not in ids
    sizer.rotate()
    assert sizer.measure(value)[0] == size
    with pytest.raises(ValueError):
        sizer.measure({"nan": float("nan"), "image": data_url})

        fresh.project(messages=messages, tools=TOOLS, observation_step=2)
