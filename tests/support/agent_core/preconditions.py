"""Preconditions of the tests that start the Pi agent core: Node.js and the installed ``ora-pi`` packages.

Without them such a test is skipped with the reason. CI sets
``ORA_TEST_REQUIRE_AGENT_CORE=1``, which turns that skip into a failure
(``require_or_skip``), so a broken CI installation cannot pass by skipping.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REQUIRE_ENV = "ORA_TEST_REQUIRE_AGENT_CORE"
MINIMUM_NODE = (22, 19)


def require_or_skip(reason: str) -> None:
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{REQUIRE_ENV}=1: {reason}")
    pytest.skip(reason)


def node_unavailable() -> str | None:
    """Why ``node`` on PATH cannot run the TypeScript sources directly, or ``None``."""

    node = shutil.which("node")
    if node is None:
        return "node is not on PATH (the Pi agent core needs node >= 22.19)"
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    match = re.match(r"v(\d+)\.(\d+)", version)
    if match is None or (int(match[1]), int(match[2])) < MINIMUM_NODE:
        return f"node {version or '(unknown version)'} is older than 22.19"
    return None


def pi_core_unavailable() -> str | None:
    """Why the agent-core tests cannot start the Pi core from this checkout, or ``None``."""

    reason = node_unavailable()
    if reason is not None:
        return reason
    if not (ROOT / "ora-pi/node_modules/@earendil-works/pi-coding-agent/package.json").is_file():
        return "ora-pi/node_modules is not installed (npm ci in ora-pi)"
    return None


def require_pi_runtime() -> None:
    """Skip (or fail under ``ORA_TEST_REQUIRE_AGENT_CORE=1``) unless doctor and preflight find the Pi core ready.

    This is the product's own check (``agent_core.pi_runtime``), so the reason is
    the remedy ``ora doctor`` prints: install Node.js >= 22.19 or run ``npm ci`` in ``ora-pi``.
    """

    from ora.orchestrator.episode import agent_core

    status = agent_core.pi_runtime()
    if not status["passed"]:
        require_or_skip(status["remedy"])
