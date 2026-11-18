"""The agent-core tests: their ``agent_core`` marker and their preconditions.

Every test collected from this directory carries the ``agent_core`` marker
(registered in ``pyproject.toml``), which selects the agent-core lane: ``-m
agent_core`` runs it, ``-m "not agent_core"`` the rest of the suite.

The preconditions are node (``node_runtime``), node and the installed
``ora-pi`` packages (``pi_core``) and, for the CLI quickstart, a clean source tree
(``clean_source_tree``).

Without them the Pi-core tests are skipped with the reason. CI sets
``ORA_TEST_REQUIRE_AGENT_CORE=1``, which turns such a precondition skip into a
failure (``require_or_skip``, shared with ``tests/ora/app`` through
``tests/support/agent_core/preconditions.py``); the opt-in sealed-run checks still skip.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.support.agent_core.preconditions import ROOT, node_unavailable, pi_core_unavailable, require_or_skip

HERE = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark this directory's tests ``agent_core``, before ``-m`` selects by marker."""

    inside: dict[Path, bool] = {}
    for item in items:
        directory = item.path.parent
        if directory not in inside:
            inside[directory] = directory.resolve().is_relative_to(HERE)
        if inside[directory]:
            item.add_marker(pytest.mark.agent_core)


@pytest.fixture(scope="session")
def node_runtime() -> None:
    """Node.js >= 22.19 on PATH, which runs the TypeScript sources directly; no ora-pi packages needed."""

    reason = node_unavailable()
    if reason is not None:
        require_or_skip(reason)


@pytest.fixture(scope="session")
def pi_core() -> None:
    reason = pi_core_unavailable()
    if reason is not None:
        require_or_skip(reason)


@pytest.fixture(scope="session")
def clean_source_tree() -> None:
    """Preflight's ``source_identity`` check admits only a clean checkout of the plan's commit."""

    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if status.returncode != 0:
        require_or_skip(f"preflight's source_identity check needs a git checkout: {status.stderr.strip()}")
    if status.stdout.strip():
        require_or_skip("preflight's source_identity check blocks a source tree with local changes; "
                        "commit them (CI runs this on the clean candidate)")


@pytest.fixture(scope="session")
def sealed_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where sealed runs are extracted: ``ORA_TEST_SEALED_CACHE`` or a session temp directory."""

    configured = os.environ.get("ORA_TEST_SEALED_CACHE")
    return Path(configured) if configured else tmp_path_factory.mktemp("sealed-cache")

