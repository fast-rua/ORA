"""The documented quickstart (``ora run --env synthetic``) runs on the Pi agent core, by default or named.

Same deterministic loopback HTTP model as ``examples/quickstart_smoke.py``; the
only difference between the runs is whether ``ORA_AGENT_CORE=pi`` is in the
CLI's environment. The CLI resolves, preflights, leases, supervises, evaluates
and seals as usual; both runs must be valid, sealed and on the Pi core, and
send the model requests of the same shape (separate CLI runs carry their own
run ids, so the bytes differ; the in-process episodes check bytes against the
goldens). Preflight admits only a clean source tree, so on a checkout with
local changes the test is skipped (CI runs it on the clean candidate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes

pytestmark = pytest.mark.usefixtures("pi_core", "clean_source_tree")


def quickstart(workspace: Path, core: str | None) -> tuple[dict[str, Any], list[bytes]]:
    from ora.agent.providers.tool_names import tool_alias

    bodies: list[bytes] = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            bodies.append(body)
            request = json.loads(body)
            name = tool_alias("robot." + ("read_manual" if len(bodies) == 1 else "submit_action"))
            assert name in {tool["function"]["name"] for tool in request["tools"]}
            arguments = {"section": "Actions"} if len(bodies) == 1 else {
                "arm": "left", "pose": {"position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]},
            }
            reply = json.dumps({"id": f"smoke-{len(bodies)}", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": f"call-{len(bodies)}",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("ORA_") and key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
        environment.update(ORA_API_KEY="local-smoke-placeholder", ORA_MODEL="smoke-model",
                           ORA_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                           ORA_WORKSPACE_ROOT=str(workspace))
        if core is not None:
            environment["ORA_AGENT_CORE"] = core
        result = subprocess.run([sys.executable, "-m", "ora.app", "run", "--env", "synthetic"],
                                env=environment, capture_output=True, text=True, timeout=300)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    roots = list(workspace.glob("outputs/quickstart/runs/*"))
    assert result.returncode == 0 and len(roots) == 1, (result.stdout[-2000:], result.stderr[-4000:], roots)
    root = roots[0]
    return {
        "core": episodes.ran_core(root),
        "result": json.loads((root / "result.json").read_text()),
        "evaluation": json.loads((root / "evaluation.json").read_text()),
        "sealed": (root / "seal.json").is_file(),
    }, bodies


def test_quickstart_runs_on_the_pi_core_by_default_and_named(tmp_path: Path) -> None:
    runs = {label: quickstart(tmp_path / label, core) for label, core in (("default", None), ("named", "pi"))}
    for run, bodies in runs.values():
        assert run["core"] == "pi"
        assert run["result"]["state"] == "completed" and run["result"]["official_success"] is None, run["result"]
        assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
        assert run["sealed"] and len(bodies) == 2
    def shape(body: bytes) -> dict[str, Any]:
        request = json.loads(body)
        return {"model": request["model"], "roles": [message["role"] for message in request["messages"]],
                "tools": [tool["function"]["name"] for tool in request["tools"]]}

    assert [shape(body) for body in runs["named"][1]] == [shape(body) for body in runs["default"][1]]
