"""Python side of the sealed-run provider differential (read-only evidence).

Usage: python tests/support/agent_core/provider/sealed_run.py RUN_DIR EXTRACT_DIR OUT.json

For every provider_request/provider_response pair of a sealed native run:
* rebuild the execution request: the canonical request (continuations
  stripped) plus each assistant message's `_provider_continuation_state`,
  found by its canonical_message_sha256 among the run's provider_response
  messages (the k-th message with one sha takes the k-th response with it:
  replies that differ only in their reasoning share a canonical sha);
* `prepare_request` it with the run's provider options and compare the sha256
  of its canonical bytes with the sealed `native_provider_wire_request`;
* feed the sealed `native_provider_wire_response` events through complete()'s
  parse phase and compare the message/usage/finish_reason with the sealed
  provider_response payload.
Archived request artifacts are read from EXTRACT_DIR (a scratch extraction of
native/provider-requests.tar.zst); the run directory is never written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import canonical_text, python_complete  # noqa: E402
from ora.agent.evidence.canonical import canonical_bytes  # noqa: E402
from ora.agent.providers import openai_compatible  # noqa: E402
from ora.agent.providers.continuation import canonical_assistant_sha256  # noqa: E402


def load(run: Path, extract: Path, ref: dict) -> tuple[bytes, object]:
    path = run / ref["path"]
    if not path.exists():
        path = extract / ref["path"]
    raw = path.read_bytes()
    return raw, json.loads(raw)


def main() -> None:
    run, extract, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    policy = json.loads((run / "plan.json").read_text())["policy"]
    options = {"reasoning_field": policy.get("reasoning_field") or "reasoning_content"}
    events = [json.loads(line) for line in (run / "agent-events.jsonl").read_text().splitlines() if line]
    requests = {e["model_call_id"]: e for e in events if e["type"] == "provider_request"}
    responses = {e["model_call_id"]: e for e in events if e["type"] == "provider_response"}
    states: dict[str, list] = {}
    for event in responses.values():
        state = event["payload"]["message"].get("_provider_continuation_state")
        if state:
            states.setdefault(state["canonical_message_sha256"], []).append(state)
    calls = []
    for call_id, request_event in requests.items():
        _, canonical = load(run, extract, request_event["payload"]["canonical_request_ref"])
        wire_raw, _ = load(run, extract, request_event["artifact_ref"])
        execution = copy.deepcopy(canonical)
        attached = 0
        seen: dict[str, int] = {}
        for message in execution["messages"]:
            if message.get("role") == "assistant":
                key = canonical_assistant_sha256(message)
                occurrence = seen[key] = seen.get(key, -1) + 1
                candidates = states.get(key, [])
                state = candidates[occurrence] if occurrence < len(candidates) else None
                if state is not None:
                    message["_provider_continuation_state"] = copy.deepcopy(state)
                    attached += 1
        provider = openai_compatible.OpenAICompatibleProvider(
            base_url="https://replay.invalid/v1", api_key="sk-replay-key", client=object(), **options)
        body = provider.prepare_request(execution)
        wire_sha = hashlib.sha256(canonical_bytes(body)).hexdigest()
        response_event = responses[call_id]
        raw_events_bytes, raw_events = load(run, extract, response_event["artifact_ref"])
        outcome = python_complete(execution, raw_events, model_call_id=call_id, **options)
        payload = response_event["payload"]
        accepted = json.loads(outcome["response"]) if "response" in outcome else None
        calls.append({
            "model_call_id": call_id,
            "canonical_ref": request_event["payload"]["canonical_request_ref"]["path"],
            "wire_ref": request_event["artifact_ref"]["path"],
            "response_ref": response_event["artifact_ref"]["path"],
            "continuations_attached": attached,
            "wire_sha256_sealed": request_event["artifact_ref"]["sha256"],
            "wire_sha256_file": hashlib.sha256(wire_raw).hexdigest(),
            "wire_sha256_python": wire_sha,
            "response_events_sha256_ok": hashlib.sha256(raw_events_bytes).hexdigest()
                == response_event["artifact_ref"]["sha256"],
            "python": outcome,
            "sealed_message_match": accepted is not None
                and canonical_text(accepted["message"]) == canonical_text(payload["message"]),
            "sealed_usage_match": accepted is not None
                and canonical_text(accepted["usage"]) == canonical_text(payload["usage"]),
            "sealed_finish_match": accepted is not None and accepted["finish_reason"] == payload["finish_reason"],
        })
    out.write_text(json.dumps({"reasoning_field": options["reasoning_field"], "calls": calls},
                              ensure_ascii=False), encoding="utf-8")
    summary = {
        "calls": len(calls),
        "python_wire_matches_sealed": sum(c["wire_sha256_python"] == c["wire_sha256_sealed"] for c in calls),
        "sealed_messages_reproduced": sum(c["sealed_message_match"] for c in calls),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
