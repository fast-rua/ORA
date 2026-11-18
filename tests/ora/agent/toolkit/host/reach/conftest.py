"""Fixtures shared by the reach tests: ``dojo``, a fake RoboDojo session for the reach check."""

from __future__ import annotations

import pytest


@pytest.fixture
def dojo(tmp_path, fake_sdk_candidate_boundary):
    """A sparse RoboDojo session on the fake plant, its kinematics outside the real candidate selector."""
    from tests.support.robodojo.backend import sparse_session

    instance = sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()
