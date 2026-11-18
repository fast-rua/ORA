"""The extended budget's contract: a multiple, and verdict fields kept apart from the official result."""

from __future__ import annotations

import pytest

from ora.contracts import official_limit
from ora.orchestrator.episode.round_aggregates import run_round_aggregates
from ora.orchestrator.reports.aggregate import _attempt_counts


def test_an_absent_multiple_is_one_so_sealed_plans_keep_their_stop() -> None:
    assert official_limit.official_limit_multiple({}) == 1


@pytest.mark.parametrize("value", [0, -1, 1001, 2.0, True, "3", None])
def test_a_multiple_outside_the_range_is_refused(value) -> None:
    with pytest.raises(ValueError, match="official_limit_multiple"):
        official_limit.official_limit_multiple({"official_limit_multiple": value})


def test_the_multiple_the_tool_budgets_end_is_the_top_of_the_range() -> None:
    top = official_limit.TOOL_BUDGET_LIMIT_MULTIPLE
    assert official_limit.official_limit_multiple({"official_limit_multiple": top}) == top == 1000


def test_a_verdict_at_multiple_one_is_unchanged() -> None:
    assert official_limit.verdict_fields(
        official_limit=200, multiple=1, snapshot=None, success_native_actions=None,
    ) == {}
    assert official_limit.record_fields({"official_success": True}) == {}
    assert official_limit.record_fields(None) == {}


def test_an_extended_verdict_names_the_limit_the_snapshot_and_the_late_success() -> None:
    snapshot = official_limit.limit_snapshot(200)
    fields = official_limit.verdict_fields(
        official_limit=200, multiple=3, snapshot=snapshot, success_native_actions=412,
    )
    assert fields == {
        "official_native_action_limit": 200, "official_limit_multiple": 3,
        "extended_native_action_limit": 600,
        "official_limit_snapshot": {"official_success": False, "native_actions": 200},
        "success_after_official_limit": True, "success_after_official_limit_native_actions": 412,
    }
    assert set(fields) == set(official_limit.OFFICIAL_LIMIT_FIELDS)
    assert official_limit.record_fields({**fields, "official_success": False}) == fields
    assert "official_success" not in fields


def test_a_session_counts_late_successes_apart_from_its_official_ones() -> None:
    rounds = [{"official_success": True, "success_after_official_limit": False},
              {"official_success": False, "success_after_official_limit": True}]
    fields = run_round_aggregates(rounds, multi_round=True)
    assert (fields["round_success_count"], fields["round_success_after_official_limit_count"]) == (1, 1)
    plain = run_round_aggregates([{"official_success": True}], multi_round=True)
    assert "round_success_after_official_limit_count" not in plain


def test_the_aggregate_never_adds_a_late_success_to_the_official_rate() -> None:
    attempts = [{"ending": "success_latch", "official_success": True, "valid_attempt": True,
                 "success_after_official_limit": False},
                {"ending": "success_after_official_limit", "official_success": False, "valid_attempt": True,
                 "success_after_official_limit": True}]
    counts = _attempt_counts([{"validity": "valid", "attempts": attempts}])
    assert (counts["successes"], counts["success_rate"]) == (1, 0.5)
    assert counts["successes_after_official_limit"] == 1


def test_a_success_inside_the_limit_has_no_snapshot_and_no_late_success() -> None:
    fields = official_limit.verdict_fields(
        official_limit=200, multiple=3, snapshot=None, success_native_actions=None,
    )
    assert fields["official_limit_snapshot"] is None
    assert fields["success_after_official_limit"] is False
    assert fields["success_after_official_limit_native_actions"] is None
