"""An operator stop (``ora stop``: SIGTERM to the Supervisor) at chosen points of a Pi-core episode.

``hanging_fake`` is a FakeProvider whose scripted request ``hang_at`` never
answers: the Pi core (``pi_provider_spec`` ``hang_at``, see
``ora-pi/src/core/deps.ts``) waits until Pi aborts it. Its Python ``complete``,
which waited until the stop closed the provider, served the retired Native
core. ``stop_when_requested`` then plays ``ora stop``: SIGTERM to this process
(the Supervisor) once the request is in the trace.

The stops that land outside a provider request are sent from inside the
episode, where the Supervisor serves the core: ``stop_during_tool_call`` while
the Supervisor runs the core's ``nth`` read-only or world-mutating tool call
(after the reply that carried it completed), ``stop_during_execution`` while
the environment executes the ``nth`` action.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


class ProviderAborted(RuntimeError):
    """The Python fake's in-flight request ended by the stop (the Pi fake's error name)."""


def hanging_fake(responses: list[dict[str, Any]], hang_at: int) -> Any:
    from ora.agent.providers.fake import FakeProvider

    @dataclass
    class HangingFake(FakeProvider):
        hang_at: int = -1

        @property
        def pi_provider_spec(self) -> dict[str, Any]:
            return {"kind": "fake", "responses": list(self.responses), "hang_at": self.hang_at}

        def complete(self, *, request: Mapping[str, Any], model_call_id: str) -> dict[str, Any]:
            if self.index == self.hang_at:
                deadline = time.monotonic() + 60
                while not self.closed:  # the stop closes the provider under this request
                    if time.monotonic() > deadline:
                        raise AssertionError("the operator stop never closed the provider")
                    time.sleep(0.01)
                raise ProviderAborted("aborted")
            return super().complete(request=request, model_call_id=model_call_id)

    return HangingFake(responses=responses, hang_at=hang_at)


@contextlib.contextmanager
def stop_when_requested(run_root: Path, requests: int) -> Iterator[threading.Event]:
    """SIGTERM this process once ``run_root``'s trace holds ``requests`` provider requests.

    The signal is sent only while the episode runs (the Supervisor's handler is
    installed); the yielded event reports that it was sent.
    """

    sent, done = threading.Event(), threading.Event()

    def watch() -> None:
        events = run_root / "agent-events.jsonl"
        deadline = time.monotonic() + 120
        while not done.is_set() and time.monotonic() < deadline:
            if events.exists() and events.read_text(encoding="utf-8").count('"type": "provider_request"') >= requests:
                time.sleep(0.2)  # in flight, not just traced
                if not done.is_set():
                    sent.set()
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield sent
    finally:
        done.set()
        watcher.join(timeout=5)


def interrupt_delivered(monkeypatch: Any) -> threading.Event:
    """An event set once the Supervisor's stop handler has sent the core its ``interrupt``."""

    from ora.agent.pi.policy_session import PiPlanner

    delivered = threading.Event()
    original = PiPlanner.interrupt

    def interrupt(self: Any, reason: str = "operator_interrupt") -> None:
        try:
            original(self, reason)
        finally:
            delivered.set()

    monkeypatch.setattr(PiPlanner, "interrupt", interrupt)
    return delivered


def raise_stop(delivered: threading.Event) -> None:
    """SIGTERM this process now, then wait until the core has been sent its ``interrupt``.

    Whatever this process sends the core next (a tool result, a receipt)
    then reaches it after the interrupt.
    """

    signal.raise_signal(signal.SIGTERM)
    if not delivered.wait(timeout=5):
        raise AssertionError("the operator stop never reached the policy session")


def stop_during_tool_call(monkeypatch: Any, *, mutation: bool, nth: int) -> dict[str, Any]:
    """Stop while the Supervisor runs the core's ``nth`` read-only (or world-mutating) tool call.

    Returns a record whose ``stopped_at`` names the tool call the stop landed in.
    """

    from ora.agent.pi.tool_rpc import PiToolRpc

    original = PiToolRpc._rpc_invoke
    delivered = interrupt_delivered(monkeypatch)
    seen: dict[str, Any] = {"calls": 0, "stopped_at": None}

    def invoke(self: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        if bool(params.get("effect_is_mutation")) == mutation:
            seen["calls"] += 1
            if seen["calls"] == nth:
                seen["stopped_at"] = params["call_context"]["tool_call_id"]
                raise_stop(delivered)
        return original(self, params)

    monkeypatch.setattr(PiToolRpc, "_rpc_invoke", invoke)
    return seen


def stop_during_execution(monkeypatch: Any, *, nth: int) -> dict[str, Any]:
    """Stop while the Synthetic environment executes the ``nth`` action; ``stopped_at`` is its action id."""

    from ora.environment.synthetic import SyntheticSession

    original = SyntheticSession.execute
    delivered = interrupt_delivered(monkeypatch)
    seen: dict[str, Any] = {"calls": 0, "stopped_at": None}

    def execute(self: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        seen["calls"] += 1
        if seen["calls"] == nth:
            seen["stopped_at"] = payload.get("action_id")
            raise_stop(delivered)
        return original(self, payload)

    monkeypatch.setattr(SyntheticSession, "execute", execute)
    return seen
