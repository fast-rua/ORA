"""Journaled native ticks with per-physics-step targets (journal and trace v2).

The executors of ``follow_joint_trajectory`` and the rate-limited
``set_gripper`` drive the plant one native tick at a time through
``bridge.native_tick_sequence``, which takes one requested control dict per
physics step. This module runs such a tick for them and keeps the evidence a
restore needs to send the plant exactly the same data again:

* every physics step is journaled (``ora.robodojo-controller-journal.v2``)
  before its successor: the ``requested`` dict the executor generated (before
  the SDK gripper limiter and the support-arm merge), the resolved
  ``command`` the SDK applied, the applied position **and velocity** targets
  read back from the articulation, the measured state and the executor's
  annotation (knot, step, guard state, a stop);
* the applied targets must equal the resolved command (positions on every
  controlled key, velocities on the arm keys) and every held key must keep
  its pre-action target, else the episode faults
  (``robodojo_applied_command_mismatch``);
* every tick appends a trace sample (``ora.robodojo-controller-trace.v2``)
  carrying the 10 requested dicts as ``control_steps``.

Restore (``control_replay``) feeds a v2 sample's recorded ``control_steps``
straight back through ``native_tick_sequence`` (``replay_recorded_tick``): no
interpolation, guard, planner or tool code runs again, so a later change of
any of them cannot change how an older run, or a fork of a fork, restores.
v1 samples keep their own ``native_tick`` path.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .bridge import RobodojoError
from .common import journal_lines, json_default

CONTROL_TRACE_V2_SCHEMA = "ora.robodojo-controller-trace.v2"
JOURNAL_V2_SCHEMA = "ora.robodojo-controller-journal.v2"
APPLIED_TARGET_ATOL = 1e-6


class StepSource(Protocol):
    """An executor's per-physics-step targets for one tick."""

    def request(self, index: int) -> dict[str, Any]:
        """The control dict of physics step ``index`` (0..9), called just before it."""

    def observe(self, index: int, measured: Mapping[str, Any]) -> None:
        """The measured end effectors right after physics step ``index``."""

    def annotation(self) -> dict[str, Any]:
        """Executor evidence for the journal line just written (knot, guard state, stop)."""

    def tick_record(self) -> dict[str, Any]:
        """Executor evidence for the trace sample of the tick just run."""


def applied_velocity_targets(session: Any, readings: Mapping[str, Any] | None = None) -> dict[str, list[float]]:
    """The arm joint velocity targets ``RobotManager.control_robot`` wrote, per arm key.

    ``readings`` are this physics step's ``session._arm_readings()`` when the caller already holds them.
    """
    manager = session.environment.robot_manager
    readings = session._arm_readings() if readings is None else readings
    result = {}
    for arm, robot in session.robots.items():
        values = readings[arm]["velocity_targets"][robot.arm_joint_indices]
        if not np.isfinite(values).all():
            raise RobodojoError("SDK applied an invalid joint velocity target")
        result[manager.process_name(robot.arm_name)] = values.tolist()
    return result


def _velocity(entry: Mapping[str, Any], width: int) -> np.ndarray:
    value = np.asarray(entry.get("velocity", 0), dtype=float)
    return np.broadcast_to(value, (width,)) if value.ndim == 0 else value


def check_applied(session: Any, command: Mapping[str, Any], applied: Mapping[str, Any],
                  velocities: Mapping[str, Any], held_targets: Mapping[str, Any]) -> None:
    """Applied targets equal the resolved command, and held keys their pre-action targets."""
    for key, target in command.items():
        if key not in applied:
            continue
        # A key that is not held is checked against the command once: the second comparison would repeat the first.
        expected = held_targets.get(key, target)
        positions_match = (np.allclose(applied[key], target["position"], rtol=0, atol=APPLIED_TARGET_ATOL)
                           and (expected is target
                                or np.allclose(applied[key], expected["position"], rtol=0, atol=APPLIED_TARGET_ATOL)))
        velocities_match = key not in velocities or (
            np.allclose(velocities[key], _velocity(target, len(velocities[key])), rtol=0, atol=APPLIED_TARGET_ATOL)
            and (expected is target
                 or np.allclose(velocities[key], _velocity(expected, len(velocities[key])), rtol=0,
                                atol=APPLIED_TARGET_ATOL)))
        if not (positions_match and velocities_match):
            session._fault("robodojo_applied_command_mismatch")
            raise RobodojoError(f"SDK applied a changed held target or an unrecorded command: {key}")


def run_tick(session: Any, *, action_id: str, segment: str, source: StepSource, held_targets: Mapping[str, Any],
             journal: Path, tracking: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Run and journal one native tick of ``source``'s targets; returns the official terminal, if any."""
    environment = session.environment
    requested_steps: list[dict[str, Any]] = []

    def requested(index: int) -> dict[str, Any]:
        value = deepcopy(source.request(index))
        requested_steps.append(value)
        return value

    def record(command: Mapping[str, Any]) -> None:
        index = len(requested_steps) - 1
        readings = session._arm_readings()  # one copy of every arm's buffers for this physics step
        applied = session._applied_position_targets(readings)
        velocities = applied_velocity_targets(session, readings)
        measured = session._proprioception(readings)["end_effectors"]
        source.observe(index, measured)
        line = {"schema": JOURNAL_V2_SCHEMA, "action_id": action_id, "segment": segment,
                "physics_clock": session._clock(), "native_action": int(environment.take_action_cnt[0]),
                "step_index": index, "requested": requested_steps[index], "command": command,
                "applied_position_targets": applied, "applied_velocity_targets": velocities,
                "measured": measured, **source.annotation()}
        journal_lines(session).write(journal, json.dumps(line, default=json_default, allow_nan=False,
                                                         sort_keys=True) + "\n")
        check_applied(session, command, applied, velocities, held_targets)

    clock_before = session._clock()
    environment.controller_trace_callback = record
    try:
        terminal = environment.native_tick_sequence(requested)
    except BaseException:
        session._fault("robodojo_native_execution_failed")
        raise
    finally:
        session.sim_steps += session._clock() - clock_before
        environment.controller_trace_callback = None
    if session._clock() - clock_before != 10 or len(requested_steps) != 10:
        session._fault("native_cadence_mismatch")
        raise RobodojoError("native action did not advance exactly 10 physics steps")
    session._ground_truth_tick(action_id, segment)
    current = session._proprioception()["end_effectors"]
    tracking.append({"native_action": int(environment.take_action_cnt[0]), "physics_steps": session.sim_steps,
                     "segment": segment, "control": deepcopy(requested_steps[-1]),
                     "control_steps": deepcopy(requested_steps), **source.tick_record(), "measured": current})
    if terminal is not None:
        session.official_terminal = deepcopy(terminal)
    return terminal


def replay_recorded_tick(session: Any, control_steps: list[Mapping[str, Any]]) -> None:
    """Send one recorded v2 tick's requested targets again (restore); nothing is recomputed."""
    session._require_started()
    if session.is_terminal:
        raise RobodojoError("terminated or aborted episode cannot accept actions")
    if len(control_steps) != 10:
        raise RobodojoError("a recorded native tick must hold exactly 10 physics-step targets")
    clock_before = session._clock()
    terminal = session.environment.native_tick_sequence([deepcopy(dict(step)) for step in control_steps])
    session.sim_steps += session._clock() - clock_before
    if session._clock() - clock_before != 10:
        session._fault("native_cadence_mismatch")
        raise RobodojoError("native action did not advance exactly 10 physics steps")
    session._ground_truth_tick(None, "conformance")
    if terminal is not None:
        session.official_terminal = deepcopy(terminal)


__all__ = [
    "APPLIED_TARGET_ATOL", "CONTROL_TRACE_V2_SCHEMA", "JOURNAL_V2_SCHEMA", "StepSource", "applied_velocity_targets",
    "check_applied", "replay_recorded_tick", "run_tick",
]
