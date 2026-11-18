"""On a real cell a stop at the cell ends a robot script, not only its current motion.

A free-play run on a real cell (plan runtime ``rig_mode`` ``read`` or ``on``) names in its play init the receipt
failure codes that mean the operator or a guard stopped a motion (``stop_scripts_on``, ``factory.py``). Once a
script action's receipt carries one, the robot socket (``robot_socket.py``) refuses that script's later
world-changing calls in the turn. Simulated and benchmark scripts name none and keep their behaviour.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.error_classes import ToolError


def stop_codes(play: Mapping[str, Any] | None) -> frozenset[str]:
    """The receipt failure codes that end a script on this run (none off a real cell)."""

    return frozenset((play or {}).get("stop_scripts_on") or ())


def stopped_by(outcome: Mapping[str, Any], codes: frozenset[str]) -> str | None:
    """The stop code a script action's receipt carries, if it is one of ``codes``."""

    receipt = outcome.get("receipt")
    code = receipt.get("failure_code") if isinstance(receipt, Mapping) else None
    return str(code) if code in codes else None


def refusal(code: str) -> ToolError:
    """The refusal of a script's world-changing call after a stop in the same turn."""

    return ToolError(
        f"the operator stopped a motion of this turn ({code}): a script's later world-changing calls are refused "
        "until the turn ends. End the turn to read the fresh observation and the operator's message.",
        code="script_stopped_by_operator", retryable=False, constraints={"failure_code": code})
