"""Test helpers for provider adapters, shared by several test modules.

They were defined in ``tests/ora/agent/providers/test_provider_adapters.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
import ssl
import subprocess
from pathlib import Path
from typing import Any

import httpx

from ora.orchestrator.spec.run_spec import file_sha256


ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "ora/policy/core/thin-react-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENT_RULES = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def native_plan(*, terminal_actions: int = 1) -> dict:
    return {
        "schema": "ora.run-plan.v1",
        "plan_sha256": "native-test-plan",
        "plan_id": "native-m0-test",
        "experiment": "native-harness",
        "config": "native-minimal",
        "milestone": "M0",
        "output_workspace": "outputs/native-harness",
        "task": "synthetic_tool_boundary",
        "task_instruction": "Move safely once.",
        "seed": 0,
        "evidence_tier": "diagnostic",
        "policy": {
            "kind": "native_api",
            "agent_core": "pi",
            "provider": "openai_compatible",
            "endpoint": "https://provider.invalid/v1",
            "model": "fake-vision-model",
            "reasoning_effort": "provider_default",
            "base_url_env": "NATIVE_MODEL_BASE_URL",
            "api_key_env": "NATIVE_MODEL_API_KEY",
            "context_window_tokens": 32768,
            "max_output_tokens": 2048,
            "stream": True,
            "supports_images": True,
            "context_profile": "native_minimal_v1",
            "summary_policy": "disabled",
            "memory_policy": "disabled",
            "tool_allowlist": [item for item in (
                "observation.dump_state", "vision.inspect", "geometry.query",
                "robot.read_manual", "robot.submit_action",
            )],
        },
        "runtime": {
            "kind": "synthetic",
            "workspace_bounds_m": BOUNDS,
            "terminal_actions": terminal_actions,
        },
        "budgets": {
            "max_robot_decisions": terminal_actions,
            "max_model_calls": 6 * terminal_actions,
            "max_sim_steps": terminal_actions,
            "wall_time_s": 30,
        },
        "policy_context": {
            "codex_project": [{
                "path": "ora/policy/planners/native/project-v1.md",
                "sha256": file_sha256(AGENT_RULES),
            }],
            "episode_prefix": [{
                "path": "ora/policy/core/thin-react-v1.md",
                "sha256": file_sha256(PROMPT),
            }],
            "tool_manual": [{
                "path": "ora/policy/embodiments/aloha-agilex/manual-v1.md",
                "sha256": file_sha256(MANUAL),
            }],
            "composition_sha256": "native-test-composition",
        },
        "source": {"commit": "test", "dirty": False},
        "qualification": {"required": False},
    }


def assistant_tool(call_id: str, name: str, arguments: dict, *, reasoning: str | None = None):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def usage(input_tokens: int, output_tokens: int = 10) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
    }


def native_run_spec(tmp_path: Path) -> dict[str, Any]:
    """A minimal OpenAI-compatible native RunSpec over a committed scratch repository."""
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    for name in ("prompt.md", "manual.md", "agents.md"):
        (tmp_path / name).write_text(
            "## overview\n\nmanual\n" if name == "manual.md" else name,
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True)
    spec = {
        "schema": "ora.run-spec.v1",
        "experiment": "native", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/native", "task": "synthetic",
        "task_instruction": "Move once.", "seed": 0, "evidence_tier": "diagnostic",
        "policy": {
            "kind": "native_api", "provider": "openai_compatible", "model": "model-a",
            "endpoint": "https://provider.invalid/v1/",
            "proxy_env": ["HTTPS_PROXY", "https_proxy"],
            "context_window_tokens": 32768,
        },
        "runtime": {"kind": "synthetic", "workspace_bounds_m": BOUNDS},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 2, "max_sim_steps": 1, "wall_time_s": 10},
        "policy_context": {
            "codex_project": ["agents.md"], "episode_prefix": ["prompt.md"],
            "tool_manual": ["manual.md"],
        },
        "qualification": {"required": False},
    }
    return spec


class _AnthropicAttemptClient:
    """Deterministic HTTP fixtures; never connects to a provider."""

    def __init__(self, modes, *, tool=False):
        self.modes = modes
        self.calls = []
        self.tool = tool

    def stream(self, *_args, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        mode = self.modes[min(index, len(self.modes)-1)]
        body = json.loads(kwargs["content"])
        tool = self.tool

        class Response:
            status_code = int(mode[4:]) if mode.startswith("http") else 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b"unauthorized"

            def iter_lines(self):
                def line(value):
                    return "data: " + json.dumps(value)
                yield line({"type": "message_start", "message": {"id": "fixture", "usage": {"input_tokens": 3}}})
                if tool:
                    name = next(t["name"] for t in body["tools"] if t["name"].startswith("robot_submit_action"))
                    block = {"type": "tool_use", "id": "accepted" if mode == "success" else "discarded", "name": name,
                             "input": {"action": "move_pose", "arm": "left", "position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]}}
                else:
                    block = {"type": "text", "text": "valid reply"}
                yield line({"type": "content_block_start", "index": 0, "content_block": block})
                if mode == "transport":
                    raise httpx.ReadError("fixture interrupted response")
                if mode == "timeout":
                    raise httpx.ReadTimeout("fixture read deadline")
                if mode == "tls_eof":
                    raise ssl.SSLEOFError("fixture TLS EOF")
                if mode == "tls_certificate":
                    raise ssl.SSLCertVerificationError("fixture invalid certificate")
                if mode == "tls_other":
                    raise ssl.SSLError("fixture non-EOF TLS error")
                if mode == "eof":
                    return
                if mode == "malformed_json":
                    yield "data: {broken"
                    return
                if mode == "parse_error":
                    yield line({"type": "content_block_delta", "index": 12, "delta": {"type": "text_delta", "text": "bad block"}})
                if mode == "credential_echo":
                    yield line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "fake-original-key"}})
                    raise httpx.ReadError("echo fixture fake-backup-key")
                yield line({"type": "content_block_stop", "index": 0})
                yield line({"type": "message_delta", "delta": {"stop_reason": "tool_use" if tool else "end_turn"}, "usage": {"output_tokens": 2}})
                yield line({"type": "message_stop"})
                if mode == "terminal_reset":
                    raise httpx.ReadError("ignored after terminal")

        return Response()
