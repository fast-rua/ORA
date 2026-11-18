"""A free-play session on the Pi core, driven the way the operator drives it.

The plan is ``ora play``'s own (``ora.app.play.play_spec``: play.scenes' Synthetic
world, the play framing, operator control). The operator writes instructions
and controls to the run's inbox, as the terminal UI does, and waits on the
Supervisor's events and interaction phase.

The Supervisor waits for the operator's next instruction as long as the session
lasts, so a run whose operator has given up would wait for good: an operator
that fails quits the session, and so does a session still running at its
deadline, with every thread's stack in the failure (``run_play``).
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[3]
#: Seconds a test's session may run before the operator quits it: far longer than any of them takes.
SESSION_DEADLINE_S = 300.0
#: Seconds a quit session has to end before the Supervisor is stopped as ``ora stop`` stops it (SIGTERM).
QUIT_GRACE_S = 30.0


def events(root: Path) -> list[dict[str, Any]]:
    """The Supervisor's events so far: each complete line of ``events.jsonl``.

    The Supervisor appends while the operator reads, and a read can see part of
    an append, so what follows the last newline is an event still being written,
    not one to parse.
    """

    path = root / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1] if line.strip()]


def wait(condition: Callable[[], bool], what: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def count(root: Path, kind: str) -> int:
    return sum(event.get("type") == kind for event in events(root))


def phase(root: Path) -> str | None:
    try:
        return json.loads((root / "native/episode-interaction.json").read_text())["phase"]
    except (OSError, ValueError, KeyError):
        return None


def send(root: Path, command_id: str, kind: str, message: str) -> None:
    from ora.orchestrator.episode.operator_control import submit_command

    submit_command(root, {"id": command_id, "type": kind, "message": message})


def play_plan(policy: Mapping[str, Any], *, scenes: int = 3, draw_seed: int = 5,
              budgets: Mapping[str, Any] | None = None,
              instruction_budgets: Mapping[str, Any] | None = None, code_mode: bool = False,
              runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The resolved plan of an ``ora play --env synthetic`` session for ``policy``.

    ``budgets`` and ``instruction_budgets`` override fields of the spec's session and per-instruction
    budgets (operator settings); ``code_mode`` is ``ora play --code-mode`` (``policy`` then names its
    ``code_mode``: Synthetic's baseline has none to default to); ``runtime`` overrides runtime fields.
    """

    from ora.app.play import play_spec
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    spec, _ = play_spec("synthetic", policy=policy, host=None, count=scenes, rng=random.Random(draw_seed),
                        draw={"draw_seed": draw_seed}, code_mode=code_mode)
    if runtime:
        spec["runtime"] = {**spec["runtime"], **runtime}
    if budgets:
        spec["budgets"] = {**spec["budgets"], **budgets}
    if instruction_budgets:
        spec["play"]["instruction_budgets"] = {**spec["play"]["instruction_budgets"], **instruction_budgets}
    return resolve_run_spec(spec, root=ROOT)


def _stacks() -> str:
    """Every thread's stack, for a session that did not end."""

    names = {thread.ident: thread.name for thread in threading.enumerate()}
    return "\n".join(f"thread {names.get(ident, ident)}:\n" + "".join(traceback.format_stack(frame))
                     for ident, frame in sys._current_frames().items())


def _quit(root: Path, command_id: str, why: str) -> None:
    """Quit the session from the operator's side, unless it is already over."""

    try:
        send(root, command_id, "quit", f"/quit ({why})")
    except Exception:  # noqa: BLE001 - the session closed meanwhile: nothing is waiting for an instruction
        pass


def run_play(root: Path, plan: Mapping[str, Any], operator: Callable[[Path], None], *,
             deadline_s: float = SESSION_DEADLINE_S) -> tuple[dict[str, Any], list[BaseException]]:
    """Run ``plan`` in ``root`` while ``operator(root)`` plays it; the result and the operator's failures.

    An operator that fails quits the session, which would otherwise wait for its
    next instruction. A session still running ``deadline_s`` after it started is
    quit, and stopped as ``ora stop`` stops it if it has not ended
    ``QUIT_GRACE_S`` later; the failure names it and carries every thread's stack.
    """

    from ora.orchestrator.episode.coordinator import run_episode

    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    failures: list[BaseException] = []
    ended = threading.Event()

    def play() -> None:
        try:
            operator(root)
        except BaseException as exc:  # reported by the test thread
            failures.append(exc)
            _quit(root, "test-operator-failed", f"the operator failed: {type(exc).__name__}")

    def watch() -> None:
        if ended.wait(deadline_s):
            return
        failures.append(AssertionError(
            f"the play session was still running {deadline_s:.0f} s after it started; the operator quit it\n"
            + _stacks()))
        _quit(root, "test-deadline", "the session deadline passed")
        if not ended.wait(QUIT_GRACE_S):
            os.kill(os.getpid(), signal.SIGTERM)  # the Supervisor's `ora stop` handler ends the session

    thread = threading.Thread(target=play, daemon=True)
    watchdog = threading.Thread(target=watch, name="play-session-deadline", daemon=True)
    thread.start()
    watchdog.start()
    try:
        result = run_episode(run_root=root, plan=plan, run_id=root.name)
    finally:
        ended.set()
    thread.join(timeout=10)
    return result, failures
