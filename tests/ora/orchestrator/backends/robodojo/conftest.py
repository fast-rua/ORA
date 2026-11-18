"""Bind product code to the RoboDojo runtime the tests read when ``ORA_TEST_ROBODOJO_RUNTIME`` names one."""

from __future__ import annotations

import pytest

from tests.support.robodojo.runtime import bind_named_robodojo_runtime


@pytest.fixture(autouse=True)
def named_robodojo_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_named_robodojo_runtime(monkeypatch)
