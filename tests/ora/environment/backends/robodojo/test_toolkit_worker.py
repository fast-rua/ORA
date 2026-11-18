"""The diagnostic route never grants regular RPC before verified restoration."""

from types import SimpleNamespace
import pytest
from ora.environment.backends.robodojo import toolkit_worker as worker
from ora.environment.worker import WorkerProtocolError


@pytest.mark.parametrize(
    "method",
    ["observe", "align_plan", "probe_motion", "status", "execute", "toolkit_authorize"],
)
def test_initial_diagnostic_has_no_tool_authority(method):
    diagnostic = worker.Diagnostic(SimpleNamespace(), {})
    with pytest.raises(WorkerProtocolError):
        diagnostic.dispatch(method, {})


@pytest.mark.parametrize("execute", [False, True])
def test_restore_prepare_authorize_are_ordered_and_action_flag_is_enforced(
    monkeypatch, execute
):
    calls = []
    session = SimpleNamespace(start=lambda: calls.append("start"))
    diagnostic = worker.Diagnostic(
        session, dict(toolkit_boundary={}, toolkit_execute_action=execute)
    )
    monkeypatch.setattr(worker, "restore", lambda *args: calls.append("restore"))
    monkeypatch.setattr(
        worker, "_dispatch", lambda s, c, m, p: (calls.append(m), False)
    )
    diagnostic.dispatch("start", {})
    diagnostic.dispatch("toolkit_restore", {})
    with pytest.raises(WorkerProtocolError):
        diagnostic.dispatch("status", {})
    diagnostic.dispatch("toolkit_prepare", {})
    with pytest.raises(WorkerProtocolError):
        diagnostic.dispatch("status", {})
    diagnostic.dispatch("toolkit_authorize", {})
    diagnostic.dispatch("status", {})
    if execute:
        diagnostic.dispatch("execute", {})
    else:
        with pytest.raises(WorkerProtocolError):
            diagnostic.dispatch("execute", {})
    assert calls[:4] == ["start", "restore", "observe", "status"]
    for method in (
        "start",
        "toolkit_restore",
        "toolkit_prepare",
        "restart_round",
        "measure_jaw_state",
        "persist_official_verdict",
    ):
        with pytest.raises(WorkerProtocolError):
            diagnostic.dispatch(method, {})


def test_failed_restore_cannot_retry_or_prepare(monkeypatch):
    diagnostic = worker.Diagnostic(SimpleNamespace(start=lambda: {}), {})
    monkeypatch.setattr(
        worker, "restore", lambda *args: (_ for _ in ()).throw(ValueError("failed"))
    )
    diagnostic.dispatch("start", {})
    with pytest.raises(ValueError, match="failed"):
        diagnostic.dispatch("toolkit_restore", {})
    for method in ("toolkit_restore", "toolkit_prepare", "toolkit_authorize", "status"):
        with pytest.raises(WorkerProtocolError):
            diagnostic.dispatch(method, {})


def test_failed_render_still_requires_host_authorization(monkeypatch):
    diagnostic = worker.Diagnostic(SimpleNamespace(start=lambda: {}), {})
    monkeypatch.setattr(worker, "restore", lambda *args: {})
    monkeypatch.setattr(worker, "inspect", lambda *args: {"physics_clock": 10})
    monkeypatch.setattr(
        worker,
        "_dispatch",
        lambda *args: (_ for _ in ()).throw(RuntimeError("render failed")),
    )
    diagnostic.dispatch("start", {})
    diagnostic.dispatch("toolkit_restore", {})
    with pytest.raises(RuntimeError, match="render failed"):
        diagnostic.dispatch("toolkit_prepare", {})
    assert diagnostic.dispatch("toolkit_verify_state", {})[0] == {"physics_clock": 10}
    with pytest.raises(WorkerProtocolError):
        diagnostic.dispatch("status", {})
    diagnostic.dispatch("toolkit_authorize", {})
