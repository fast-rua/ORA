"""Run Python's OpenAICompatibleProvider against a scripted loopback server.

Usage: python tests/support/agent_core/provider/recovery_live.py IN.json OUT.json

IN.json: {"base": "http://127.0.0.1:PORT", "api_key": ..., "scenarios": [{"name",
"policy", "provider": {...}, "request", "model_call_id"}]}. Every scenario
posts to {base}/{name}/py/v1 through the real transport (the subprocess worker
for the deadline policies, httpx.Client for legacy_v1). A scenario may shorten
first_token_deadline_v1's stall limit (``stall_s``, ``recovery.STALL_S``) and set
a run deadline ``run_deadline_s`` seconds after its start. Backoff sleeps
are recorded and skipped; with ``virtual_clock`` each one still moves the
monotonic clock on by its length (``SleepRecorder.advance``), so the
no-content window counts it. The deadline workers are started before each
scenario's incident window opens (common.PrestartedWorkers); each outcome
reports how many an attempt had to start inside its window
(``workers_started_in_window``). Nothing else is patched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PrestartedWorkers, SleepRecorder, canonical_text, describe_error, normalize_timing  # noqa: E402
from ora.agent.providers import openai_compatible, recovery  # noqa: E402

#: Ready workers before each deadline scenario: more than any scenario's attempts.
WORKERS = 12


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    recorder = SleepRecorder()
    recovery.time = recorder
    openai_compatible.time = recorder
    workers = PrestartedWorkers.for_recovery()
    recovery.subprocess = workers
    stall_s = recovery.STALL_S
    results = []
    try:
        for scenario in spec["scenarios"]:
            recorder.sleeps, recorder.offset = [], 0.0
            recorder.advance = scenario.get("virtual_clock", False) is True
            records = []
            workers.prepare(WORKERS if scenario["policy"] in recovery.RECOVERY_POLICIES else 0)
            recovery.STALL_S = scenario.get("stall_s", stall_s)
            run_deadline = scenario.get("run_deadline_s")
            provider = openai_compatible.OpenAICompatibleProvider(
                base_url=f"{scenario.get('base', spec['base'])}/{scenario['name']}/py/v1", api_key=spec["api_key"],
                recovery_policy=scenario["policy"], attempt_observer=lambda record: records.append(dict(record)),
                **scenario.get("provider", {}),
                request_deadline_monotonic=None if run_deadline is None else time.monotonic() + run_deadline,
            )
            outcome = {"name": scenario["name"]}
            try:
                response = provider.complete(request=scenario["request"], model_call_id=scenario["model_call_id"])
                outcome["response"] = canonical_text(normalize_timing(response))
                outcome["ttft_negative"] = response["ttft_s"] is not None and response["ttft_s"] < 0
            except Exception as exc:  # noqa: BLE001 - every outcome is compared
                outcome["error"] = describe_error(exc)
            finally:
                provider.close()
            outcome["records"] = [canonical_text(normalize_timing(record)) for record in records]
            outcome["sleeps"] = [{"value": value, "float": isinstance(value, float)} for value in recorder.sleeps]
            outcome["workers_started_in_window"] = workers.started_in_window
            results.append(outcome)
    finally:
        workers.close()
    Path(sys.argv[2]).write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scenarios": len(results)}))


if __name__ == "__main__":
    main()
