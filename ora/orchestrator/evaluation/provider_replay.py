"""Replay frozen OpenAI provider responses; compare full requests without a real model.

The serve and compare operations never start a simulator or submit a job. Source artifacts, including compressed
requests, are read-only. The old-tool treatment, which derived a RunSpec for the tape, was retired on 2026-11-01 with
its replay baseline ``native_full_explicit_motion_direct_replay_v1``; sealed replays keep their recorded baseline.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ora.agent.evidence.request_archive import verified_bytes
from .request_comparison import compare_requests


def events(root: Path, kind: str) -> list[dict]:
    return [row for line in (root / "agent-events.jsonl").read_text().splitlines()
            if (row := json.loads(line)).get("type") == kind]


def artifact(root: Path, row: dict) -> bytes:
    data = verified_bytes(root, row["artifact_ref"])
    if data is None:
        raise ValueError(f"missing/corrupt {row['type']} artifact: {row['artifact_ref']['path']}")
    return data


def policy_configuration(root: Path) -> dict | None:
    """Expose the recorded policy configuration, including a replay-only baseline identity."""
    path = root / "plan.json"
    if not path.is_file():
        return None
    plan = json.loads(path.read_text())
    policy = plan["policy"]
    return {"baseline": policy.get("baseline"), "require_live_marker": policy.get("require_live_marker"),
            "tool_surface": policy.get("tool_surface"),
            "implementation_commit": plan.get("source", {}).get("commit")}


def compare_runs(source: Path, candidate: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("comparison output must be outside both runs")
    output.mkdir(parents=True, exist_ok=False)
    old, new = events(source, "provider_request"), events(candidate, "provider_request")
    rows = []
    for ordinal, (a, b) in enumerate(zip(old, new)):
        result = compare_requests(artifact(source, a), artifact(candidate, b))
        result.update(ordinal=ordinal, source_model_call_id=a["model_call_id"],
                      candidate_model_call_id=b["model_call_id"],
                      observation_steps=[a.get("observation_step"), b.get("observation_step")])
        path = f"request-{ordinal:04d}.json"
        (output / path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        rows.append({"ordinal": ordinal, "report": path, "byte_equal": result["byte_equal"],
                     "json_equal": result["json_equal"], "difference_count": len(result["differences"]),
                     "image_counts": result["image_counts"]})
    report = {"schema": "ora.provider-replay-comparison.v1", "source_run": str(source.resolve()),
              "candidate_run": str(candidate.resolve()),
              "policy_configuration": {"source": policy_configuration(source),
                                       "candidate": policy_configuration(candidate)}, "counts": [len(old), len(new)],
              "byte_equal": len(old) == len(new) and all(row["byte_equal"] for row in rows),
              "requests": rows, "unmatched_source_calls": [row["model_call_id"] for row in old[len(new):]],
              "unmatched_candidate_calls": [row["model_call_id"] for row in new[len(old):]],
              "claim_boundary": "No normalization. Image metrics do not waive byte, count, order, label or reference differences."}
    from ora.agent.toolkit.interface.protocol import audit
    from ora.contracts.toolkit_protocol import identity_warning
    report["protocol_compatibility"] = audit(source)
    report["candidate_protocol_compatibility"] = audit(candidate)
    report["warnings"] = [*report["protocol_compatibility"]["warnings"],
                          *report["candidate_protocol_compatibility"]["warnings"]]
    old_identity = (report["policy_configuration"]["source"] or {}).get("implementation_commit")
    new_identity = (report["policy_configuration"]["candidate"] or {}).get("implementation_commit")
    if old_identity is None or old_identity != new_identity:
        report["warnings"].append(identity_warning("harness_commit", old_identity, new_identity))
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class Tape:
    """Consume a recorded response exactly once per HTTP request; never invent a reply."""
    def __init__(self, source: Path, output: Path):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("replay output must be outside the source run")
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.index = output, 0
        responses = events(source, "provider_response")
        if not responses or any(row.get("payload", {}).get("accepted") is not True for row in responses):
            raise ValueError("replay tape requires accepted provider responses only")
        self.responses = [(row, json.loads(artifact(source, row))) for row in responses]

    def reply(self, request: bytes) -> tuple[bytes, str]:
        value = json.loads(request)
        if self.index >= len(self.responses):
            raise ValueError("recorded provider response tape exhausted")
        row, response = self.responses[self.index]
        streaming = isinstance(response, list)
        if bool(value.get("stream")) != streaming:
            raise ValueError("request stream mode differs from the recorded response")
        number = self.index
        # Persist consumption before replying. A disconnect never reuses the response.
        (self.output / f"request-{number:04d}.json").write_bytes(request)
        with (self.output / "served.jsonl").open("a") as stream:
            stream.write(json.dumps({"ordinal": number, "source_model_call_id": row["model_call_id"],
                                     "response_ref": row["artifact_ref"]}) + "\n")
        self.index += 1
        if streaming:
            return ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in response)
                    + "data: [DONE]\n\n").encode(), "text/event-stream"
        return json.dumps(response).encode(), "application/json"


def serve(source: Path, output: Path, listen: str, port: int, ready_file: Path | None = None) -> None:
    tape = Tape(source, output)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                body, content_type = tape.reply(self.rfile.read(int(self.headers["Content-Length"])))
            except (ValueError, KeyError) as error:
                self.send_error(409, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer((listen, port), Handler) as server:
        readiness = {"endpoint": f"http://{listen}:{server.server_port}/v1", "responses": len(tape.responses),
                     "output": str(output.resolve())}
        if ready_file is not None:
            temporary = ready_file.with_suffix(".tmp")
            with temporary.open("x") as stream:
                stream.write(json.dumps(readiness) + "\n")
            temporary.replace(ready_file)
        print(json.dumps(readiness), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def run(args) -> int:
    if args.provider_replay_command == "serve":
        serve(args.source, args.output, args.listen, args.port, args.ready_file)
        return 0
    report = compare_runs(args.source, args.candidate, args.output)
    print(json.dumps({"counts": report["counts"], "byte_equal": report["byte_equal"], "output": str(args.output)}))
    return 0 if report["byte_equal"] else 1
