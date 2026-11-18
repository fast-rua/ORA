"""The one place either published spelling of a move_pose target is read.

``place.*`` was flattened in ``5ddc3f0a`` because a single-branch ``oneOf``
swallows each branch's specific failure: a malformed position yielded ``$input
must match exactly one input branch; oneOf[0]: $input.targets.left must match
exactly one input branch`` with empty ``constraints``, naming neither the field
nor the bound. The same shape stayed on ``move_pose``, which the sampled corpus
calls 1,033 times against ``vision.ground``'s 395 -- the most expensive place in
the surface to keep it.

``move_pose``'s union is worse than ``place.*``'s was, because it is three deep:
``targets.<arm>`` is a two-branch ``oneOf`` whose second branch carries a
three-branch ``position`` and a three-branch ``orientation``. ``gateway.py``
flattens a union only at the schema root, so every one of those reaches the
provider intact.

The flat spelling names the choice as an enum and the choice's arguments as
siblings::

    {"position_mode": "delta", "translation_m": [0, 0, -0.05],
     "position_frame": "end_effector",
     "orientation_mode": "hold_current", "interaction": "contact_approach"}

What the schema can no longer state -- "delta also needs translation_m" -- is a
structured ``ToolError`` naming the field, never an ``anyOf``. A union anywhere
in this schema reproduces a recorded failure while looking like a schema improvement.

Two nested spellings still read, and both must: the ergonomic
``{"position": {...}, "orientation": {...}}`` form that frozen surfaces publish,
and the bare ``{"position_m", "orientation_xyzw"}`` canonical TCP pose that
every pose-producing tool hands over. ``compile_enhanced_move_pose`` already
normalises the second into the first, so this module converts the flat form into
the first and leaves both others untouched.

``arm_target_fields`` is the write half. A producer that publishes a target for
the policy to copy -- ``grasp.plan``'s ``approach_target`` -- asks the bound
schema which spelling this surface advertises rather than writing a literal,
because a copyable target published in a shape its own surface refuses is worse
than no copyable target at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.agent.composition.surface.motion_modes import (
    DECLARED_ORIENTATION_FIELDS, POSITION_FIELDS, WORLD_AXES_MODE,
)

from ora.agent.error_classes import ToolError
from ora.agent.composition.surface import tool_names
from ora.agent.toolkit.tools.move_pose._arguments import (
    CAPABILITY, FRAMES, INTERACTIONS, POSITION_MODES, arm_target_fields, declares_flat_targets,
)


FLAT_KEYS = frozenset(
    {"position_mode", "orientation_mode", "interaction"}
    | {field for fields in POSITION_FIELDS.values() for field in fields}
    | {field for fields in DECLARED_ORIENTATION_FIELDS.values() for field in fields}
)


def _reject(message: str, field: str, **constraints: Any) -> ToolError:
    return ToolError(
        message, code="invalid_arguments", retryable=True, field=field,
        constraints=constraints or None,
    )


def _one_choice(
    target: Mapping[str, Any], *, kind: str, modes: Sequence[str],
    fields: Mapping[str, Sequence[str]], prefix: str,
) -> dict[str, Any]:
    """Read one mode plus exactly the arguments that mode requires."""

    key = f"{kind}_mode"
    mode = target.get(key)
    if mode not in modes:
        raise _reject(
            f"{key} must name one of {', '.join(modes)}",
            f"{prefix}.{key}", received=mode, allowed=list(modes),
        )
    required = fields[mode]
    missing = [name for name in required if target.get(name) is None]
    if missing:
        raise _reject(
            f"{key} {mode!r} also needs {', '.join(missing)}",
            f"{prefix}.{missing[0]}", received=sorted(target), requires=list(required),
        )
    surplus = [
        name
        for other, names in fields.items() if other != mode
        for name in names
        if name not in required and target.get(name) is not None
    ]
    if surplus:
        raise _reject(
            f"{key} {mode!r} does not take {', '.join(sorted(surplus))}",
            f"{prefix}.{sorted(surplus)[0]}", received=sorted(surplus),
            allowed=list(required),
        )
    choice: dict[str, Any] = {"mode": mode}
    if mode == "absolute":
        choice["position_m"] = target["position_m"]
    elif mode == "delta":
        choice["translation_m"] = target["translation_m"]
        choice["frame"] = target["position_frame"]
    elif mode == "absolute_quaternion":
        choice["quaternion_xyzw"] = target["orientation_xyzw"]
    elif mode == "axis_angle_delta":
        choice["axis"] = target["rotation_axis"]
        choice["angle_deg"] = target["rotation_angle_deg"]
        choice["frame"] = target["orientation_frame"]
    elif mode == WORLD_AXES_MODE:
        choice["approach_world"] = target["approach_world"]
        choice["jaw_line_world"] = target["jaw_line_world"]
    return choice


def arm_target(target: Any, *, arm: str = "target") -> dict[str, Any]:
    """One arm's request, in the nested shape the pose compiler already takes.

    A nested request passes through verbatim; so does the bare canonical TCP
    pose, which ``compile_enhanced_move_pose`` normalises itself. Only the flat
    spelling is converted, and only here.
    """

    if not isinstance(target, Mapping):
        return target
    # The flat spelling announces itself with its mode enums. Anything else --
    # the ergonomic position/orientation pair, the bare canonical TCP pose, and
    # a malformed request on a surface still publishing the nested union --
    # passes through untouched, so the existing admission keeps raising its own
    # error with the examples that teach the accepted contract. Reading the
    # arguments rather than the surface is what the retired `placement_arguments` did, and
    # it keeps this module usable by anything that holds no surface at all.
    if not ({"position_mode", "orientation_mode"} & set(target)):
        return dict(target)
    unknown = sorted(set(target) - FLAT_KEYS)
    if unknown:
        raise _reject(
            "unknown move_pose field", f"{arm}.{unknown[0]}",
            received=unknown, allowed=sorted(FLAT_KEYS),
        )
    # Report both halves together, so a correction does not discover one
    # missing companion field per model round. Never infer a frame or target.
    required_by_mode = {
        kind: list(fields[target[f"{kind}_mode"]])
        for kind, fields in (("position", POSITION_FIELDS), ("orientation", DECLARED_ORIENTATION_FIELDS))
        if isinstance(target.get(f"{kind}_mode"), str) and target[f"{kind}_mode"] in fields
    }
    missing = [name for names in required_by_mode.values() for name in names
               if target.get(name) is None]
    if missing:
        raise _reject(
            "move_pose is missing mode-dependent fields: " + ", ".join(missing)
            + ". Supply every listed field in the same target; coordinate frames are never inferred.",
            f"{arm}.{missing[0]}", missing_fields=missing,
            requires=[name for names in required_by_mode.values() for name in names],
            required_by_mode=required_by_mode,
            missing_paths=[f"{arm}.{name}" for name in missing],
            preserved_target=dict(target),
            frame_choices=list(FRAMES), effect="no_action_submitted",
        )
    request: dict[str, Any] = {
        "position": _one_choice(
            target, kind="position", modes=POSITION_MODES,
            fields=POSITION_FIELDS, prefix=arm,
        ),
        "orientation": _one_choice(
            target, kind="orientation", modes=tuple(DECLARED_ORIENTATION_FIELDS),
            fields=DECLARED_ORIENTATION_FIELDS, prefix=arm,
        ),
    }
    from .refusals import check_flat_request  # the compiler's bounds, public names
    check_flat_request(request, arm)
    interaction = target.get("interaction")
    if interaction is not None:
        if interaction not in INTERACTIONS:
            raise _reject(
                "interaction names a physical intent, never a controller",
                f"{arm}.interaction", received=interaction, allowed=list(INTERACTIONS),
            )
        request["interaction"] = interaction
    return request


def arm_targets(targets: Mapping[str, Any]) -> dict[str, Any]:
    """Read every arm's request, naming the arm in any refusal.

    Reads the arguments, not the surface: a request that does not announce
    itself as flat passes through, so a surface still publishing the nested
    union keeps teaching the nested contract with the examples its own refusal
    carries.
    """

    result, missing_errors = {}, {}
    for arm, request in targets.items():
        try:
            result[arm] = arm_target(request, arm=f"targets.{arm}")
        except ToolError as exc:
            if not (exc.constraints or {}).get("missing_fields"):
                raise
            missing_errors[arm] = exc
    if len(missing_errors) == 1:
        raise next(iter(missing_errors.values()))
    if missing_errors:
        missing_paths = [path for error in missing_errors.values()
                         for path in error.constraints["missing_paths"]]
        raise _reject(
            "move_pose is missing fields in multiple arm targets: " + ", ".join(missing_paths)
            + ". Correct all selected arms in one request; no coordinates or frames are inferred.",
            missing_paths[0], missing_paths=missing_paths,
            missing_fields_by_target={arm: error.constraints["missing_fields"]
                                      for arm, error in missing_errors.items()},
            required_by_target={arm: error.constraints["required_by_mode"]
                                for arm, error in missing_errors.items()},
            preserved_targets={arm: dict(request) for arm, request in targets.items()},
            frame_choices=list(FRAMES), effect="no_action_submitted",
        )
    return result


def surface_move_pose_schema(surface: Any) -> dict[str, Any]:
    """The bound ``move_pose`` input schema of one surface, or an empty one.

    A surface that declares no ``move_pose`` gets ``{}`` back, which
    ``declares_flat_targets`` reads as "nested", so a producer keeps publishing
    the shape such a surface was sealed with.
    """

    for item in getattr(surface, "definitions", ()) or ():
        if tool_names.canonical(str(item.get("name"))) == "move_pose":
            return item.get("inputSchema", {}) or {}
    return {}


def surface_flat_targets(surface: Any) -> bool:
    """Whether producers publish move targets to this surface in the flat spelling.

    The bound ``move_pose`` schema decides when the surface has one. A surface
    that disabled ``move_pose`` in favour of its own motion tool (an RSI
    toolset's ``move_pose_paced``, I-0015) still declares its argument spelling
    through ``flat_motion_arguments_v1``; without that fallback the toolkit's
    ``grasp_plan`` published nested targets that its own flat resultSchema
    refuses.
    """

    schema = surface_move_pose_schema(surface)
    if schema:
        return declares_flat_targets(schema)
    supports = getattr(surface, "supports", None)
    return bool(supports(CAPABILITY)) if callable(supports) else False

def project_copyable_approach_targets(
    result: Mapping[str, Any], *, semantic_interaction: bool = False,
    allow_grip_site_alias: bool = False,
    move_pose_schema: Mapping[str, Any] | None = None,
) -> None:
    """Replace legacy compiler deltas with exact enhanced move_pose targets.

    Written in whichever spelling the surface's own ``move_pose`` advertises.
    A target published for the policy to copy unchanged, in a shape that
    surface's ``move_pose`` refuses, is worse than publishing none: it reads as
    an instruction and fails on submission.
    """

    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ToolError(
            "grasp result has no candidate list",
            code="grasp_approach_contract",
            retryable=False,
        )
    for candidate in candidates:
        legacy_delta = candidate.get("approach_delta")
        # "grip_site" is the Panda embodiment's private name for its
        # end-effector frame. Only a surface that declares the Panda frame
        # axes may spell it; every other embodiment must use the public
        # "end_effector" name, so a Panda-specific frame can never be
        # silently accepted on a different robot.
        allowed_frames = (
            {"end_effector", "grip_site"} if allow_grip_site_alias
            else {"end_effector"}
        )
        if (
            not isinstance(legacy_delta, Mapping)
            or legacy_delta.get("frame") not in allowed_frames
            or legacy_delta.get("rotation_rotvec_rad") != [0.0, 0.0, 0.0]
        ):
            raise ToolError(
                "grasp approach is not a copyable end-effector translation",
                code="grasp_approach_contract",
                retryable=False,
            )
        candidate["approach_delta"] = arm_target_fields(
            move_pose_schema or {},
            position={
                "mode": "delta",
                "translation_m": list(legacy_delta["translation_m"]),
                "frame": "end_effector",
            },
            orientation={"mode": "hold_current"},
            interaction="contact_approach" if semantic_interaction else None,
        )
