"""Motion refusals in the public argument names, with the bounds they enforce.

Three defects of the pose tools' refusals:

* ``move_pose`` attached ``constraints.examples`` in the nested spelling
  ``{"position": {...}, "orientation": {...}}``. Every surface that publishes
  the flat ``position_mode``/``orientation_mode`` targets refuses that shape, so
  each refusal taught a request its own schema rejects (17 recorded refusals).
  ``move_pose_examples`` writes them in the bound schema's spelling, through
  ``motion_arguments.arm_target_fields``, and keeps only examples that schema
  admits.
* The pose compiler (``contracts/motion.py``, a RoboDojo worker-kernel file)
  names ``move_pose`` for ``move_to``/``rotate_*`` and internal fields
  (``angle_deg``, ``target_orientation_xyzw``), and states no bounds. The
  public bounds are checked here first, under the public field names, and a
  compiler message is re-raised under the primitive that was called.
* The workspace refusal named ``xyz_m``, which is not an argument of the
  sparse pose tools; ``target_field`` names the argument the target came from.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from ora.agent.composition.surface.schema_validation import (
    ToolSurfaceError, validate_schema_instance,
)

from . import arguments as motion_arguments
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.tools.move_pose._refusals import (
    ANGLE_BOUND_DEG, PrimitiveAngleError, QUATERNION_NORM_BOUNDS, _finite_number, primitive_field, target_field,
)


# Every rejected move_pose argument payload gets
# these worked examples in the structured error so a format mistake costs one
# call, not a trial-and-error loop. Kept in one place and asserted against
# compile_enhanced_move_pose by a regression test so they cannot drift from
# the accepted contract.
ENHANCED_MOVE_POSE_EXAMPLES: tuple[dict[str, Any], ...] = (
    {"targets": {"right": {
        "interaction": "free_space",
        "position": {"mode": "absolute", "position_m": [-0.10, 0.02, 0.95]},
        "orientation": {"mode": "hold_current"},
    }}},
    {"targets": {"left": {
        "interaction": "precision",
        "position": {
            "mode": "delta",
            "translation_m": [0.0, 0.0, -0.03],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }}},
    {"targets": {"right": {
        "interaction": "contact_approach",
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta",
            "axis": "x",
            "angle_deg": 25.0,
            "frame": "end_effector",
        },
    }}},
    {"targets": {"left": {
        "position_m": [-0.12, 0.05, 0.92],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }}},
    {"plan_ref": "plan-0123456789abcdef0123456789abcdef"},
)


def check_rotation_angle(value: Any, field: str) -> None:
    """A signed angle in degrees inside the compiler's bound."""

    if not _finite_number(value) or abs(float(value)) > ANGLE_BOUND_DEG:
        raise ToolError(
            f"{field} must be a finite angle in degrees within "
            f"[-{ANGLE_BOUND_DEG:g}, {ANGLE_BOUND_DEG:g}]",
            code="invalid_motion_arguments", field=field, received=value,
            constraints={"minimum": -ANGLE_BOUND_DEG, "maximum": ANGLE_BOUND_DEG,
                         "unit": "degrees"},
        )


def check_quaternion(value: Any, field: str) -> None:
    """Four finite numbers whose norm the compiler accepts."""

    low, high = QUATERNION_NORM_BOUNDS
    norm = (math.sqrt(sum(float(item) ** 2 for item in value))
            if isinstance(value, list) and len(value) == 4 and all(map(_finite_number, value))
            else None)
    if norm is None or not low <= norm <= high:
        raise ToolError(
            f"{field} must be a unit quaternion [x,y,z,w] with norm in [{low}, {high}]",
            code="invalid_motion_arguments", field=field, received=value,
            constraints={"length": 4, "norm_minimum": low, "norm_maximum": high,
                         "received_norm": norm},
        )


def check_flat_request(request: Mapping[str, Any], prefix: str) -> None:
    """The public bounds of one flat arm target, read from its compiled choice."""

    orientation = request.get("orientation", {})
    if orientation.get("mode") == "axis_angle_delta":
        check_rotation_angle(orientation.get("angle_deg"), f"{prefix}.rotation_angle_deg")
    elif orientation.get("mode") == "absolute_quaternion":
        check_quaternion(orientation.get("quaternion_xyzw"), f"{prefix}.orientation_xyzw")


def primitive_refusal(error: Exception, *, primitive: str, field: str | None) -> ToolError:
    """A compiler or request error re-raised under the primitive that was called."""

    message = str(error).replace("move_pose", primitive)
    constraints = dict(getattr(error, "constraints", None) or {})
    return ToolError(message, code="invalid_motion_arguments", field=field,
                     constraints=constraints)


def _flat_target(schema: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    if "position" in target:
        return motion_arguments.arm_target_fields(
            schema, position=target["position"], orientation=target["orientation"],
            interaction=target.get("interaction"),
        )
    return motion_arguments.arm_target_fields(
        schema, position={"mode": "absolute", "position_m": target["position_m"]},
        orientation={"mode": "absolute_quaternion", "quaternion_xyzw": target["orientation_xyzw"]},
    )


def move_pose_examples(surface: Any, examples: Any) -> list[dict[str, Any]]:
    """``examples`` in the spelling ``surface``'s own ``move_pose`` schema admits."""

    schema = motion_arguments.surface_move_pose_schema(surface)
    flat = motion_arguments.declares_flat_targets(schema)
    single_arm = surface.supports("single_arm_flat_targets_v1")
    written = []
    for example in examples:
        if "plan_ref" in example:
            if surface.supports("direct_pose_submission_v1"):
                continue
            candidate = copy.deepcopy(dict(example))
        else:
            ((arm, target),) = example["targets"].items()
            target = _flat_target(schema, target) if flat else copy.deepcopy(dict(target))
            candidate = target if single_arm else {"targets": {arm: target}}
        admitted = _admitted(candidate, schema, single_arm)
        if admitted is not None:
            written.append(admitted)
    return written


def _admitted(candidate: dict[str, Any], schema: Mapping[str, Any],
              single_arm: bool) -> dict[str, Any] | None:
    """``candidate``, or it without ``interaction`` on a schema that has none, or ``None``."""

    if not schema:
        return candidate
    try:
        validate_schema_instance(candidate, schema, root_path="$input")
        return candidate
    except ToolSurfaceError:
        pass
    target = candidate if single_arm else next(iter(candidate.get("targets", {}).values()), {})
    if "interaction" not in target:
        return None
    target.pop("interaction")
    return _admitted(candidate, schema, single_arm)


__all__ = [
    "ANGLE_BOUND_DEG", "PrimitiveAngleError", "QUATERNION_NORM_BOUNDS",
    "check_flat_request", "check_quaternion", "check_rotation_angle",
    "move_pose_examples", "primitive_field", "primitive_refusal", "target_field",
]
