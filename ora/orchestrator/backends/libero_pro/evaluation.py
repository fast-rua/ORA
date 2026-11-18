"""What evaluation checks of a LIBERO-Pro run beyond the shared evidence: each round's verdict."""

from __future__ import annotations

from typing import Any, Mapping


def round_record_errors(plan: Mapping[str, Any], record: Mapping[str, Any], index: int) -> list[str]:
    """A round's official verdict is null exactly when infrastructure ended it, else Boolean."""

    if (
        (
            record.get("termination_authority") == "infrastructure"
            and record.get("official_success") is not None
        )
        or (
            record.get("termination_authority") != "infrastructure"
            and not isinstance(record.get("official_success"), bool)
        )
    ):
        return [f"multi_round_libero_verdict_invalid:{index}"]
    return []


__all__ = ["round_record_errors"]
