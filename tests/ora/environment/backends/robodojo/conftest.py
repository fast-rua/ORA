"""Bind product code to the RoboDojo runtime the tests read when ``ORA_TEST_ROBODOJO_RUNTIME`` names one."""

from __future__ import annotations

import pytest

from tests.support.robodojo.runtime import bind_named_robodojo_runtime


@pytest.fixture(autouse=True)
def named_robodojo_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_named_robodojo_runtime(monkeypatch)


@pytest.fixture(autouse=True)
def private_start_gate(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Tests take a private start gate, never the node's.

    The node gate (``start_gate.root()``, per user under ``/dev/shm``) is shared
    with the live RoboDojo runs of the same user on the host. A test that
    reached it waited behind their Kit initializations until pytest-timeout
    stopped it, and while it held a slot it delayed their starts.
    """
    from ora.environment.backends.robodojo import start_gate

    folder = tmp_path_factory.mktemp("robodojo-start-gate")
    monkeypatch.setattr(start_gate, "root", lambda: folder)


@pytest.fixture
def sparse(tmp_path):
    """A sparse-motion RoboDojo session on the fake plant (``tests/support/robodojo/backend.py``)."""
    from tests.support.robodojo.backend import sparse_session

    instance = sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(autouse=True)
def explicit_recovery_for_direct_kernel_tests(monkeypatch):
    """Primitive tests call the kernel directly; act as their request-side owner.

    RPC integration tests supply their own execution context and bypass this
    adapter. Production recovery never chooses a stored anchor itself.
    """
    from ora.contracts.execution_state import FIELDS, pack, recovery_goals
    from ora.environment.backends.robodojo.session import RobodojoSession

    execute = RobodojoSession.execute

    def with_explicit_goal(session, action):
        if hasattr(session, "recovery_targets"):
            return execute(session, action)
        values = pack({name: getattr(session, name, {}) for name in FIELDS})
        session.recovery_targets = recovery_goals(action, values)
        try:
            return execute(session, action)
        finally:
            del session.recovery_targets

    monkeypatch.setattr(RobodojoSession, "execute", with_explicit_goal)
