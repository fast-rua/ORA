"""Fixtures of the code-mode tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A tool runtime over the public recorded RoboDojo fixture (``tests/data/toolkit-general-pickup``) with two
    published observations: the current one is ``observation-0001`` (step 1), its three cameras with RGB-D and
    calibration (the wrist cameras reuse the head's files, ``toolkit_atomic.complete_cameras``)."""

    from tests.support.agent.toolkit_atomic import timeline
    from tests.support.agent.toolkit_tools import make_host

    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    timeline(host)
    yield host
    client.close()
