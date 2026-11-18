"""The Pi core dies inside a call Python makes while serving it: a clear fault, never a resume.

An operator steer reaches the core through a call Python makes while serving
the core's own ``operator.poll`` (the Supervisor delivers it before a model
call). ``ORA_PI_CRASH_AT=after_operator_message:1`` SIGKILLs the core after it
recorded the message and before it answered. Whether the message reached the
conversation is unknown to Python, so the act is not resumed: the run ends as
an infrastructure fault that names the nested call, the core is restarted only
for the Supervisor's closing records, and the agent trace stays consistent.
(Before, the bridge restarted the core inside the RPC handler and then again
for the act.)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tests.support.agent_core import episodes

pytestmark = pytest.mark.usefixtures("pi_core")


def test_a_core_lost_inside_a_nested_call_ends_the_run_with_a_named_fault(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ora.agent.evidence.trace_verify import verify_agent_trace
    from ora.orchestrator import paths
    from ora.orchestrator.episode.coordinator import run_episode
    from ora.orchestrator.evaluation.evaluate import evaluate
    from ora.orchestrator.episode.operator_control import submit_command
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from ora.agent.providers.fake import FakeProvider
    from tests.support.orchestrator.native_synthetic_spec import synthetic_spec

    # The operator inbox lives under the state root, keyed by run id: keep it in this test.
    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setenv("ORA_PI_CRASH_AT", "after_operator_message:1")
    monkeypatch.setenv("ORA_PI_FAKE_DELAY_MS", "300")  # the steer lands before the second model call
    spec = synthetic_spec(seed=0)
    spec["experiment"] = "ora-interactive"
    spec["output_workspace"] = "outputs/ora-interactive"
    spec["operator_control"] = {"mode": "file_inbox_v1"}
    spec["runtime"]["terminal_actions"] = 9
    spec["budgets"] = {"max_robot_decisions": 9, "max_model_calls": 18, "max_sim_steps": 9, "wall_time_s": 3600}
    spec["policy"].update(context_retention="action_step_lossless_v4", summary_policy="rollout_checkpoint_v1",
                          context_management=episodes.management())
    with episodes.environment(ORA_AGENT_CORE="pi"):
        plan = resolve_run_spec(spec)
        root = tmp_path / "run"
        root.mkdir()
        (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        submitted = threading.Event()

        def steer() -> None:  # once the first request is in the trace, queue one steer
            events = root / "agent-events.jsonl"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if events.exists() and '"type": "provider_request"' in events.read_text(encoding="utf-8"):
                    submit_command(root, {"id": "steer-1", "type": "steer", "message": "Say which camera you use."})
                    submitted.set()
                    return
                time.sleep(0.02)

        threading.Thread(target=steer, daemon=True).start()
        result = run_episode(run_root=root, plan=plan, run_id="nested-call-loss",
                             provider_override=FakeProvider(responses=episodes.action_script(4)))
    assert submitted.is_set() and (root / "native" / "pi-crash-fired").exists()
    assert episodes.ran_core(root) == "pi"
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    fault = json.loads((root / "native" / "runtime-fault.json").read_text())
    assert result["state"] == "incomplete" and result["termination_authority"] == "infrastructure"
    assert "inside a nested operator_message call" in fault["message"] and "not resumed" in fault["message"]
    assert not [event for event in events if event["type"] == "agent_core_resumed"]
    restarts = [json.loads(line) for line in (root / "native" / "pi-restarts.jsonl").read_text().splitlines()]
    assert [record["method"] for record in restarts if "restart" in record] == ["close"]
    assert verify_agent_trace(root)["status"] == "valid"
    assert events[-1]["type"] == "session_closed"
    assert evaluate(root)["reasons"] == ["trusted_fault"]
