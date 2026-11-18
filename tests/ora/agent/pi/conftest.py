"""Fixtures of the policy-session code-mode tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A tool runtime over the public recorded RoboDojo fixture (``tests/data/toolkit-general-pickup``) with two
    published observations, as ``tests/ora/agent/code_mode/conftest.py`` gives the library's tests."""

    from tests.support.agent.toolkit_atomic import timeline
    from tests.support.agent.toolkit_tools import make_host

    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    timeline(host)
    yield host
    client.close()
