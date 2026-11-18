"""Run Python's AnthropicMessagesProvider against a scripted loopback server.

Usage: python tests/support/agent_core/provider/anthropic_live.py IN.json OUT.json

IN.json: {"base": "http://127.0.0.1:PORT", "api_key": ..., "standby": [...],
"scenarios": [{"name", "policy", "provider": {...}, "request", "model_call_id"}]}.
Every scenario posts to {base}/{name}/py/v1/messages through the real transport
(the subprocess worker for same_credential_deadline_v1, httpx.Client for
legacy_v1). Backoff sleeps are recorded and skipped. The deadline workers are
started before each scenario's incident window opens (common.PrestartedWorkers);
each outcome reports how many an attempt had to start inside its window
(``workers_started_in_window``). Nothing else is patched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PrestartedWorkers, SleepRecorder, canonical_text, describe_error, normalize_timing  # noqa: E402
from ora.agent.providers import anthropic_messages, recovery  # noqa: E402

#: Ready workers before each deadline scenario: more than any scenario's attempts.
WORKERS = 6


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    recorder = SleepRecorder()
    recovery.time = recorder
    anthropic_messages.time = recorder
    workers = PrestartedWorkers.for_recovery()
    recovery.subprocess = workers
    results = []
    try:
        for scenario in spec["scenarios"]:
            recorder.sleeps = []
            records = []
            options = dict(scenario.get("provider", {}))
            if options.pop("two_keys", False):
                options["standby_api_keys"] = tuple(spec["standby"])
            workers.prepare(WORKERS if scenario["policy"] in recovery.RECOVERY_POLICIES else 0)
            provider = anthropic_messages.AnthropicMessagesProvider(
                base_url=f"{scenario.get('base', spec['base'])}/{scenario['name']}/py/v1", api_key=spec["api_key"],
                recovery_policy=scenario["policy"], attempt_observer=lambda record: records.append(dict(record)),
                **options,
            )
            outcome = {"name": scenario["name"]}
            try:
                response = provider.complete(request=scenario["request"], model_call_id=scenario["model_call_id"])
                outcome["response"] = canonical_text(normalize_timing(response))
            except Exception as exc:  # noqa: BLE001 - every outcome is compared
                outcome["error"] = describe_error(exc)
            finally:
                provider.close()
            outcome["records"] = [canonical_text(normalize_timing(record)) for record in records]
            outcome["sleeps"] = [{"value": value, "float": isinstance(value, float)} for value in recorder.sleeps]
            outcome["rotations"] = [{k: v for k, v in item.items() if k != "unix_s"}
                                    for item in provider.credential_rotations()]
            outcome["workers_started_in_window"] = workers.started_in_window
            results.append(outcome)
    finally:
        workers.close()
    Path(sys.argv[2]).write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scenarios": len(results)}))


if __name__ == "__main__":
    main()
