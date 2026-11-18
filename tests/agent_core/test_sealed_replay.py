"""Offline replay of the sealed acceptance runs on the Pi agent core (opt-in).

The LIBERO-Pro, RoboTwin and RoboDojo runs of 2026-10-13, sealed by the Python
Native core, live outside the repository; ``ORA_TEST_SEALED_<ENV>`` names each
one (``tests/support/agent_core/sealed.py``) and an unset one is skipped with
the reason. Each fixture is replayed through the real ``PiPlanner`` (``node
ora-pi/src/main.ts`` with the TypeScript ``ReplayProvider``) with recorded tool
results and provider responses:

* every wire and canonical request is byte-identical to the archive;
* the trace and the agent artifacts equal the sealed original apart from the
  environment receipts, which embed the run root;
* the trace verifies (``verify_agent_trace`` and the lossless-context evidence
  check), and every turn and recorded tool call is consumed;
* the Pi core's retention hints reproduce Python's delivery exactly.

The usage-inflated LIBERO-Pro replay, which compacts with L6, has no archive to
compare with; the development repository's sealed-run goldens hold it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes, hint_check, sealed
from tests.support.agent_core.replay import run_native
from tests.support.agent_core.replay.runtime import Fixture

pytestmark = pytest.mark.usefixtures("pi_core")

#: Events whose payload embeds the replay's own run root (the environment
#: receipt and the observation published after it); they differ from the
#: sealed original only by that path.
RUN_ROOT_EVENTS = frozenset({"environment_receipt", "fresh_observation_published"})


@pytest.fixture(scope="module", params=list(sealed.ENVIRONMENTS))
def faithful(request: pytest.FixtureRequest, sealed_cache: Path,
             tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    try:
        fixture_dir = sealed.replay_fixture(request.param, sealed_cache)
    except sealed.Unavailable as exc:
        pytest.skip(str(exc))
    out = tmp_path_factory.mktemp(f"replay-{request.param}")
    hints = out / "pi-hints.jsonl"
    with episodes.environment(ORA_PI_HINT_DUMP=str(hints)):
        report = run_native.replay(fixture_dir, out / "replay", core="pi")
    return {"report": report, "hints": hints, "fixture": Fixture.load(fixture_dir)}


def test_the_pi_core_reproduces_every_archived_request(faithful: dict[str, Any]) -> None:
    recorded = len(faithful["fixture"].provider_calls)
    assert recorded > 0
    report = faithful["report"]
    assert report["core"] == "pi"
    requests = report["requests"]
    assert requests["mismatches"] == [], requests["mismatches"][:3]
    assert requests["provider_calls_recorded"] == requests["provider_requests_replayed"] == recorded
    assert requests["wire_requests_byte_identical"] == requests["canonical_requests_byte_identical"] == recorded


def test_the_pi_replay_equals_the_sealed_trace(faithful: dict[str, Any]) -> None:
    events = faithful["report"]["events"]
    assert events["replayed_events"] == events["original_events"]
    assert [item for item in events["mismatches"] if item["type"] not in RUN_ROOT_EVENTS] == []
    for item in events["mismatches"]:
        assert all(problem.startswith(("payload ", "artifact_ref ")) for problem in item["problems"]), item
    # The same receipts, stored under the replay's run root, are the only other agent artifacts.
    artifacts = faithful["report"]["artifacts"]
    assert [item for item in artifacts["missing"] if item["kind"] != "environment_receipt"] == []
    assert len(artifacts["extra"]) == len(artifacts["missing"])
    assert artifacts["context_artifact_index_identical"] is True


def test_the_trace_verifies_and_consumes_the_recording(faithful: dict[str, Any]) -> None:
    report = faithful["report"]
    checks = report["trace_checks"]
    assert checks["agent_trace"]["status"] == "valid", checks["agent_trace"]["errors"]
    assert checks["lossless_context"]["reasons"] == []
    assert report["turns"]["matching"] == report["turns"]["total"] > 0
    assert report["tool_runtime"]["unconsumed_recorded_calls"] == []
    assert report["provider_transport"]["responses_served"] == report["provider_transport"]["responses_recorded"]
    assert report["compactor_calls"] == []


def test_pi_core_applies_python_retention_hints_exactly(faithful: dict[str, Any]) -> None:
    report = hint_check.check(faithful["hints"])
    assert report["mismatch_count"] == 0, report["mismatches"]
    assert report["records"] > 0
