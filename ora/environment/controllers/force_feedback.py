"""Sensor-frame force projection and bounded force-stop policy.

No simulator imports or contact truth are used here. A wrist reaction includes
gravity and inertia; reversing it estimates environment-on-EEF loading rather
than identifying a collision or a grasped object.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FORCE_DIRECTION_CONVENTION = "environment_on_eef_reaction_estimate"
#: A contact move pushes on purpose, so force alone must not end it: turning a stove
#: knob took 130 N while the hand kept advancing. It stops once the force has stayed
#: above the threshold, over the move's lowest reading, for these ticks, and the target
#: error fell by less than this over them. The floor takes out a held object's weight
#: and the reaction a move starts in.
CONTACT_STALL_TICKS = 3
CONTACT_STALL_PROGRESS_M = 0.0005
#: A contact move is leaving its target once the target error has risen for these ticks
#: and stands this far above the lowest it reached. No contact is needed for that.
CONTACT_DIVERGENCE_RISING_TICKS = 2
CONTACT_DIVERGENCE_LIMIT_M = 0.005


def sensor_reaction_feedback(
    force_sensor_n: Sequence[float] | None,
    world_from_sensor_rotation: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    """Rotate and reverse MuJoCo's child-to-parent site-frame reaction.

The rotation must be the force sensor's actual site rotation at the same
measurement. A missing or malformed calibration never produces a direction.
"""
    result: dict[str, Any] = {
        "external_force_world_n": None,
        "external_force_direction_world": None,
        "force_frame": "world",
        "force_direction_convention": FORCE_DIRECTION_CONVENTION,
        "force_direction_status": "unavailable",
    }
    try:
        force = [float(value) for value in force_sensor_n]  # type: ignore[union-attr]
        rotation = [[float(value) for value in row]
                    for row in world_from_sensor_rotation]  # type: ignore[union-attr]
        if len(force) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            return result
        if not all(math.isfinite(value) for value in force + sum(rotation, [])):
            return result
        for i in range(3):
            for j in range(3):
                if abs(sum(rotation[k][i] * rotation[k][j] for k in range(3))
                       - float(i == j)) > 1e-6:
                    return result
        determinant = sum(rotation[0][i] * (
            rotation[1][(i + 1) % 3] * rotation[2][(i + 2) % 3]
            - rotation[1][(i + 2) % 3] * rotation[2][(i + 1) % 3]
        ) for i in range(3))
        if abs(determinant - 1.0) > 1e-6:
            return result
    except (TypeError, ValueError, OverflowError):
        return result
    world = [-sum(row[i] * force[i] for i in range(3)) for row in rotation]
    magnitude = math.sqrt(sum(value * value for value in world))
    result.update(
        external_force_world_n=world,
        external_force_direction_world=(
            [value / magnitude for value in world] if magnitude > 1e-12 else None
        ),
        force_direction_status="available" if magnitude > 1e-12 else "zero_force",
    )
    return result


def public_force_direction(state: Mapping[str, Any]) -> dict[str, Any]:
    """Select calibrated scalar/vector fields without leaking simulator names."""
    return {key: state.get(key) for key in (
        "external_force_world_n", "external_force_direction_world", "force_frame",
        "force_direction_convention", "force_direction_status",
    )}


def precision_force_stop_reason(
    *, interaction: str | None, magnitude_n: float | None,
    threshold_n: float, enabled: bool,
) -> str | None:
    """One stop decision for the legacy precision adapter and bounded Servo."""
    if not enabled or interaction != "precision" or magnitude_n is None:
        return None
    if (math.isfinite(magnitude_n) and math.isfinite(threshold_n)
            and magnitude_n > threshold_n > 0.0):
        return "contact_stop"
    return None


def contact_force_stop_reason(
    *, interaction: str | None, forces_n: Sequence[float], errors_m: Sequence[float],
    threshold_n: float, enabled: bool,
) -> str | None:
    """A contact move pressing without advancing, from one reading per tick."""
    ticks = CONTACT_STALL_TICKS
    if not enabled or interaction != "contact_approach" or len(forces_n) <= ticks:
        return None
    floor = min(forces_n[:-ticks])
    pressing = all(force - floor > threshold_n for force in forces_n[-ticks:])
    if pressing and errors_m[-ticks] - errors_m[-1] < CONTACT_STALL_PROGRESS_M:
        return "contact_stop"
    return None


def contact_divergence_stop_reason(
    *, interaction: str | None, errors_m: Sequence[float], enabled: bool,
) -> str | None:
    """A contact move whose hand is leaving its target, from one error per tick."""
    ticks = CONTACT_DIVERGENCE_RISING_TICKS
    if not enabled or interaction != "contact_approach" or len(errors_m) <= ticks:
        return None
    recent = errors_m[-ticks - 1:]
    rising = all(later > earlier for earlier, later in zip(recent, recent[1:]))
    if rising and errors_m[-1] - min(errors_m) > CONTACT_DIVERGENCE_LIMIT_M:
        return "tracking_diverged"
    return None
