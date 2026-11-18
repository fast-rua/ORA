"""Tool policy: named approach directions and uniformly sampled jaw rolls."""
import math
from typing import Any, Mapping, Sequence
from ...lib.pose import approach_orientation
from ...lib.values import JsonValue
from ..errors import ToolRefusal

APPROACH_SIDES = ("top", "left", "right", "front", "back")
MAXIMUM_POSES = 12
REQUEST_SCHEMA = "ora.align-plan-request.v1"


def _refusal(message, *, field, received, allowed):
    return ToolRefusal(message, field=field, received=received, constraints={"allowed": allowed})

ARGUMENT_FIELDS = frozenset({"target_ref", "approach_side", "arm"})
OPTIONAL_FIELDS = frozenset({"standoff_m", "max_candidates"})

DEFAULT_STANDOFF_M = 0.10
MINIMUM_STANDOFF_M = 0.02
MAXIMUM_STANDOFF_M = 0.30
DEFAULT_CANDIDATES = 3

# World-frame unit vector pointing *from* the approach position *toward* the
# target, per side. Named directions toward the object, never an arm.
_APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "right": (0.0, 1.0, 0.0),
}


def synthesise_approach_poses(
    centre_xyz_m: Sequence[float], *, approach_side: str, standoff_m: float,
    candidate_count: int, local_axes: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Poses standing off the target along one side, differing only in jaw roll.

    The policy should not have to compose a quaternion by hand to ask whether an
    approach is reachable; that is exactly the arithmetic the tool surface exists
    to remove.
    """

    direction = _APPROACH_DIRECTIONS[approach_side]
    centre = [float(value) for value in centre_xyz_m]
    position = [centre[axis] - direction[axis] * standoff_m for axis in range(3)]
    poses: list[list[float]] = []
    for index in range(candidate_count):
        roll = math.pi * index / max(candidate_count, 1)
        poses.append([
            *position, *approach_orientation(tuple(direction), roll, local_axes=JsonValue.of(local_axes)),
        ])
    return poses

def validate_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a malformed request by naming the field, never by a bare message."""

    unknown = set(arguments) - ARGUMENT_FIELDS - OPTIONAL_FIELDS
    missing = ARGUMENT_FIELDS - set(arguments)
    if unknown or missing:
        raise _refusal(
            "align_plan requires target_ref, approach_side and arm",
            field=sorted(unknown or missing)[0],
            received=sorted(arguments),
            allowed=sorted(ARGUMENT_FIELDS | OPTIONAL_FIELDS),
        )
    if arguments["approach_side"] not in APPROACH_SIDES:
        raise _refusal(
            "approach side is not a declared direction", field="approach_side",
            received=arguments["approach_side"], allowed=list(APPROACH_SIDES),
        )
    standoff = arguments.get("standoff_m", DEFAULT_STANDOFF_M)
    if not isinstance(standoff, (int, float)) or isinstance(standoff, bool) or not (
        MINIMUM_STANDOFF_M <= float(standoff) <= MAXIMUM_STANDOFF_M
    ):
        raise _refusal(
            "standoff is outside the declared range", field="standoff_m",
            received=standoff,
            allowed=[MINIMUM_STANDOFF_M, MAXIMUM_STANDOFF_M],
        )
    count = arguments.get("max_candidates", DEFAULT_CANDIDATES)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 6:
        raise _refusal(
            "max_candidates is outside the declared range", field="max_candidates",
            received=count, allowed=[1, 6],
        )
    return {
        "target_ref": arguments["target_ref"],
        "approach_side": arguments["approach_side"],
        "arm": arguments["arm"],
        "standoff_m": float(standoff),
        "max_candidates": int(count),
    }

def build_request(
    validated: Mapping[str, Any], *, centre_xyz_m: Sequence[float],
    local_axes: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The closed backend request; its field set is checked on arrival."""

    poses = synthesise_approach_poses(
        centre_xyz_m,
        approach_side=validated["approach_side"],
        standoff_m=validated["standoff_m"],
        candidate_count=min(validated["max_candidates"], MAXIMUM_POSES),
        local_axes=local_axes,
    )
    return {
        "schema": REQUEST_SCHEMA,
        "target_ref": validated["target_ref"],
        "approach_side": validated["approach_side"],
        "arm": validated["arm"],
        "target_poses_xyzw": poses,
    }

