"""Synthetic episodes on the Pi agent core reproduce their goldens byte for byte.

Each test runs one scripted FakeProvider episode on the Pi core, selected
explicitly (``ORA_AGENT_CORE=pi``), checks that it ran there, and compares,
with the remaining wall-time budget (a clock reading) pinned, every canonical
provider request and every agent event with the golden of the same scenario
(``tests/support/agent_core/golden.py``). The goldens were recorded from the
Python Native core before ORA removed it, when both cores agreed byte for byte.
They cover each context composition and retention: native_slots_v3 with
action_step_lossless_v4 or rewrite_compact_v1, and native_slots_v4 with
action_step_lossless_v4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes, golden, hint_check
from tests.support.agent_core.episodes import tool_reply

pytestmark = pytest.mark.usefixtures("pi_core")


#: Every (composition, retention) a native_api policy runs.
CONTEXTS = [
    ("native_slots_v3", "action_step_lossless_v4"),
    ("native_slots_v4", "action_step_lossless_v4"),
    ("native_slots_v3", "rewrite_compact_v1"),
]


def run_pinned(core: str, out: Path, **options: Any) -> dict[str, Any]:
    with episodes.pinned_wall_time():
        run = episodes.run_core(core, out, **options)
    assert run["core"] == core
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    return run


def mark_script() -> list[dict[str, Any]]:
    """Two aggregated ``mark_point`` batches; the middle call of the second is out of range."""

    def mark(call_id: str, x: int, y: int, point: str) -> tuple[str, str, dict[str, Any]]:
        return call_id, "mark_point", {"point_id": point, "view": "wrist", "x": x, "y": y}

    return [
        tool_reply(("obs-0", "observe", {"views": ["agentview"]})),
        tool_reply(mark("mark-a", 2, 1, "P0"), mark("mark-b", 1, 1, "P1")),
        tool_reply(mark("mark-c", 1, 2, "P2"), mark("mark-d", 99999, 1, "P3"), mark("mark-e", 2, 2, "P4")),
        tool_reply(("m-0", "close_gripper", {})),
        tool_reply(("f-0", "finish_episode", {
            "outcome": "failure", "reason": "mark batch probe", "attempts": ["scripted"]})),
    ]


def episode(core: str, out: Path, composition: str, retention: str) -> dict[str, Any]:
    """Six actions of the default script: the golden ``synthetic-<composition>-<retention>``."""

    run = run_pinned(core, out, actions=6, composition=composition, retention=retention)
    return golden.summarize(run["requests"], run["events"], state=run["result"]["state"])


def mark_batches(core: str, out: Path, composition: str, retention: str) -> dict[str, Any]:
    """``mark_script``: the golden ``mark-batches-<composition>-<retention>``."""

    run = run_pinned(core, out, responses=mark_script(), actions=1, composition=composition, retention=retention)
    combined = [event["tool_call_id"] for event in run["events"]
                if event["type"] == "tool_call_proposed" and event["payload"].get("visual_feedback") == "combined"]
    images = [event["tool_call_id"] for event in run["events"] if event["type"] == "tool_image"]
    return golden.summarize(run["requests"], run["events"], state=run["result"]["state"],
                            combined=combined, images=images)


@pytest.mark.parametrize("composition,retention", CONTEXTS)
def test_synthetic_episode_requests_equal_the_golden(tmp_path: Path, composition: str, retention: str) -> None:
    summary = episode("pi", tmp_path, composition, retention)
    assert len(summary["requests"]) == 13
    assert golden.check(f"synthetic-{composition}-{retention}", summary) == []


def test_pi_core_applies_python_retention_hints_exactly(tmp_path: Path) -> None:
    dump = tmp_path / "hints.jsonl"
    with episodes.environment(ORA_PI_HINT_DUMP=str(dump)):
        run = episodes.run_core("pi", tmp_path, actions=3)
    assert run["core"] == "pi" and run["evaluation"]["status"] == "valid"
    report = hint_check.check(dump)
    assert report["mismatch_count"] == 0, report
    assert report["records"] > 0, report


@pytest.mark.parametrize("composition,retention", CONTEXTS)
def test_aggregated_mark_point_batches_equal_the_golden(tmp_path: Path, composition: str, retention: str) -> None:
    summary = mark_batches("pi", tmp_path, composition, retention)
    assert summary["extra"]["combined"] == ["mark-a", "mark-b", "mark-c", "mark-d", "mark-e"]
    # The composite image lands on the last successful call of each batch.
    assert summary["extra"]["images"] == ["obs-0", "mark-b", "mark-e"]
    assert golden.check(f"mark-batches-{composition}-{retention}", summary) == []
