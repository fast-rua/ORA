"""Run the documented quickstart against a deterministic local HTTP model.

No real API, credential, simulator or GPU is used. The CLI still resolves,
preflights, leases, supervises, evaluates and seals a native_api episode on the
Pi agent core; plan and session evidence must name that core.

    uv run --locked python -m examples.quickstart_smoke             in a temporary workspace, removed after (CI)
    uv run --locked python -m examples.quickstart_smoke --keep DIR  keep the run in the workspace DIR and print
                                                                    its run ID and the command that replays it
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMAND = "uv run --locked python -m examples.quickstart_smoke"


def smoke(keep: Path | None = None) -> Path:
    """Run the quickstart once and check its evidence; return the run directory.

    The run lands in a temporary workspace that is removed afterwards, or in
    the workspace ``keep`` (``ORA_WORKSPACE_ROOT``), where it stays.
    """
    from ora.agent.providers.tool_names import tool_alias

    calls = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            suffix = "read_manual" if len(calls) == 1 else "submit_action"
            name = tool_alias("robot." + suffix)
            assert name in {tool["function"]["name"] for tool in request["tools"]}
            arguments = {"section": "Actions"} if len(calls) == 1 else {
                "arm": "left", "pose": {"position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]},
            }
            body = json.dumps({"id": f"smoke-{len(calls)}", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": f"call-{len(calls)}",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with contextlib.ExitStack() as stack:
            if keep is None:
                workspace = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ora-quickstart-smoke-")))
            else:
                workspace = keep.expanduser().resolve()
                workspace.mkdir(parents=True, exist_ok=True)
            runs = workspace / "outputs/quickstart/runs"
            earlier = set(runs.iterdir()) if runs.is_dir() else set()
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("ORA_") and key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
            environment.update(ORA_API_KEY="local-smoke-placeholder", ORA_MODEL="smoke-model",
                ORA_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                ORA_WORKSPACE_ROOT=str(workspace))
            command = [sys.executable, "-m", "ora.app", "run", "--env", "synthetic"]
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=150)
            if result.returncode:
                logs = [path.read_text() for path in runs.glob("*/supervisor.log") if path.parent not in earlier]
                raise RuntimeError(result.stdout + result.stderr + "\n".join(logs))
            roots = [path for path in runs.iterdir() if path not in earlier]
            assert len(roots) == 1, roots
            root = roots[0]
            outcome = json.loads((root / "result.json").read_text())
            assert outcome["state"] == "completed" and outcome["official_success"] is None, outcome
            assert (root / "seal.json").is_file()
            assert json.loads((root / "evaluation.json").read_text())["status"] == "valid"
            assert len(calls) == 2
            assert json.loads((root / "plan.json").read_text())["policy"]["agent_core"] == "pi"
            assert json.loads((root / "native/agent-session.json").read_text())["agent_core"] == "pi"
            print("Quickstart passed on the Pi agent core: two HTTP model calls, one accepted action, "
                  "valid and sealed Synthetic evidence.")
            return root
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def watch_command(root: Path) -> str:
    """``ora watch`` (as this user runs ORA) with the run ID when this shell's ORA resolves it to ``root``, else the directory."""
    from ora.orchestrator import paths
    from ora.version import ora_command

    try:
        found = paths.resolve_run_root(root.name) == root.resolve()
    except (RuntimeError, OSError):
        found = False
    return f"{ora_command()} watch {root.name if found else root}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--keep", type=Path, metavar="DIR",
                        help="keep the run in this workspace directory and print how to replay it "
                             "(default: a temporary workspace, removed afterwards)")
    args = parser.parse_args(argv)
    try:
        from ora.agent.providers.tool_names import tool_alias  # noqa: F401  the fake model's import
    except ImportError as exc:
        print(f"quickstart_smoke: {sys.executable} cannot import ORA ({exc}); run it from the checkout as "
              f"`{COMMAND}`", file=sys.stderr)
        return 2
    root = smoke(args.keep)
    if args.keep is not None:
        print(f"run ID: {root.name}")
        print(f"kept in: {root}")
        print(f"replay it: {watch_command(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
