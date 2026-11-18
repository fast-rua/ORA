"""Context management v5 stride aging: its validation and the round identity it needs.

v5 keeps the v4 tails as minimums and ages older ActionSteps only in batches
of ``tail_aging_stride_action_steps``, so between batches every Provider
request strictly extends the previous one. The cases that drove one synthetic
episode through the Python Context pipeline and counted its rewrites went
with that pipeline when ORA removed the Native core; the recorded Context
fixtures (``tests/data/context_differential/stride_*`` and ``captured_stride_*``)
hold what it produced, and the TypeScript pipeline replays them. The episode
helpers here also build the round-slot histories of
``test_round_static_facts_context.py``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from ora.agent.context.lossless.management import (
    CONTEXT_MANAGEMENT_V4_SCHEMA,
    CONTEXT_MANAGEMENT_V5_SCHEMA,
    ContextManagementConfigError,
    validate_context_management,
    validate_context_management_v4,
    validate_context_management_v5,
)
from tests.support.agent.lossless_context_stride import V4, _v5


# --- prefix stability ------------------------------------------------------


# --- limits -----------------------------------------------------------------


# --- evidence ---------------------------------------------------------------


def test_lossless_v5_sessions_carry_an_explicit_round_identity() -> None:
    from ora.orchestrator.episode.coordinator import _session_round_budget_fields

    for management in (V4, _v5(4)):
        assert _session_round_budget_fields(
            {"context_management": management},
            round_index=0, round_limit=1, multi_round=False,
        ) == {"session_round": 1, "session_rounds": 1}


# --- validation -------------------------------------------------------------


def test_v5_validates_the_stride_as_a_positive_integer() -> None:
    assert validate_context_management_v5(_v5(4))[
        "tail_aging_stride_action_steps"
    ] == 4
    assert validate_context_management(_v5(1)) == validate_context_management_v5(_v5(1))
    for invalid in (0, -1, True, 1.5, "4", None):
        with pytest.raises(ContextManagementConfigError, match="stride"):
            validate_context_management_v5(_v5(4) | {
                "tail_aging_stride_action_steps": invalid,
            })
    missing = _v5(4)
    del missing["tail_aging_stride_action_steps"]
    with pytest.raises(ContextManagementConfigError, match="missing"):
        validate_context_management_v5(missing)
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management_v5({**_v5(4), "schema": "unknown"})


def test_v4_rejects_the_stride_and_stays_frozen() -> None:
    assert validate_context_management_v4(V4) == validate_context_management(V4)
    with pytest.raises(ContextManagementConfigError, match="extra"):
        validate_context_management_v4(V4 | {"tail_aging_stride_action_steps": 4})
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management_v4({**V4, "schema": CONTEXT_MANAGEMENT_V5_SCHEMA})
    with pytest.raises(ContextManagementConfigError, match="schema"):
        validate_context_management({**V4, "schema": "unknown"})
