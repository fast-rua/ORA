"""What a script reads of a call, in its surface's terms (``library: extended``; ``agent/pi/code_mode.py`` asks).

A call a script cannot make is refused in this surface's terms before its tool runs (``refusal``): a ``targets``
wrapper around the arguments of a tool whose schema takes one arm's fields directly names the form it takes
(``unwrapped``; the schema alone would call it missing mode fields), and a name of no tool lists what a script can call
and, for a name of ``ora_lib`` or ``robot``, the import that gives it (``unknown``). A not-ok answer with no ``error``
whose items failed with codes of their own (``mark_point``'s ``marks``, ``vision_segment``'s ``queries``) reads as one
that counts them (``item_errors``), so the script's ``ToolRefused`` and the result's ``calls`` say why. A dry run's
arm records add the predicted pose's world axes (``predicted_axes``), as the published end-effector axes give the
measured pose's; tool-call mode's projection of a dry run (``session_core/domain.py``) adds them with the same function.
Its own line is ``preview_line``.
"""

from __future__ import annotations

import functools
import math
from collections import Counter
from typing import Any, Callable, Collection, Iterable, Mapping

from ora.agent.composition.embodiment.descriptor import EmbodimentDescriptorError
from ora.agent.composition.embodiment.geometry import public_local_axes
from ora.agent.error_classes import ToolError

from . import docs, library_card
from .contract import DIRECT_TOOLS, NOT_FROM_SCRIPTS, WITHHELD

#: The decimals of a dry run's predicted axes, as its poses carry them.
PREDICTED_DIGITS = 5
#: How a single-arm surface's call is written where a script wrapped it in ``targets``: what it takes directly and the
#: call; another tool's refusal shows its signature.
UNWRAPPED_CALLS = {"move_pose": ("one arm's target fields", "tools.move_pose(**target(p_world))"),
                   "move_to": ("its fields", "tools.move_to(position_m=p_world)")}


def refusal(canonical: str, name: str, arguments: Any, functions: Iterable[str], *, reads: Collection[str],
            definitions: Callable[[], Iterable[Mapping[str, Any]]] = tuple,
            lifecycle: Collection[str] = ()) -> ToolError | None:
    """What a script call (``name``, the tool ``canonical``) is refused before its tool runs, None for a call the tool
    takes up: a name of none of the ``functions`` a script may name (``unknown``, the model's own tools and the
    ``lifecycle`` ones left out of ``allowed``), or a ``targets`` wrapper that the tool's schema, one of
    ``definitions()``, has no room for (``unwrapped``)."""

    functions = set(functions)
    if canonical not in functions:
        own = DIRECT_TOOLS | NOT_FROM_SCRIPTS | WITHHELD | set(lifecycle or ())
        return unknown(name, [tool for tool in functions if tool not in own], reads)
    if not isinstance(arguments, Mapping) or "targets" not in arguments:
        return None
    return unwrapped(canonical, arguments, next((item for item in definitions() if item["name"] == canonical), None),
                     reads)


def unwrapped(name: str, arguments: Any, definition: Mapping[str, Any] | None,
              reads: Collection[str]) -> ToolError | None:
    """The refusal of a ``targets`` wrapper around the arguments of a tool whose schema (``definition``) has none (a
    single-arm surface's ``move_pose``, ``move_to``, ...), naming the form it takes; None for any other call."""

    if not isinstance(arguments, Mapping) or "targets" not in arguments or definition is None:
        return None
    schema = definition.get("inputSchema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping) or "targets" in properties:
        return None
    tool = name.rsplit(".", 1)[-1]
    fields, call = UNWRAPPED_CALLS.get(tool) or ("its arguments", docs.signature(definition, reads=reads))
    targets = arguments["targets"]
    return ToolError(f"this surface's {tool} takes {fields} directly, with no targets wrapper: {call}",
                     code="invalid_arguments", retryable=True, field="targets",
                     received=sorted(map(str, targets)) if isinstance(targets, Mapping) else type(targets).__name__)


def unknown(name: str, functions: Iterable[str], reads: Collection[str]) -> ToolError:
    """The refusal of a script call of no tool: ``allowed`` the script ``functions`` by the name a script calls them,
    the nearest of them, and for a name of ``ora_lib`` or ``robot`` the import that gives it."""

    allowed = sorted({docs.wire_name(tool) for tool in functions})
    near = docs.nearest(str(name), allowed)
    module = library_card.library_module(str(name), reads)
    hint = (f"; {name} is {module}'s: from {module} import {name}" if module is not None
            else f"; the nearest: {', '.join(near)}" if near else "; robot.docs() lists the functions")
    return ToolError(f"unknown tool: {name}{hint}", code="unknown_tool", retryable=False, field="name",
                     received=name, constraints={"allowed": allowed, "nearest": near})


def item_errors(answer: Mapping[str, Any]) -> dict[str, Any] | None:
    """The error a not-ok answer with none reads as, where its result's items failed with codes of their own: the most
    frequent code, and a message that counts every code with the first message given for it. None otherwise."""

    result = answer.get("result")
    if answer.get("ok") or isinstance(answer.get("error"), Mapping) or not isinstance(result, Mapping):
        return None
    for key, items in result.items():
        listed = [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []
        failed = [item for item in listed if isinstance(item.get("error_code"), str)]
        if not failed:
            continue
        counts = Counter(item["error_code"] for item in failed)
        said = {code: next((item["error_message"] for item in failed if item["error_code"] == code
                            and isinstance(item.get("error_message"), str)), None) for code in counts}
        parts = [f"{code} x{count}" + (f" ({said[code]})" if said[code] else "") for code, count in counts.items()]
        return {"code": counts.most_common(1)[0][0], "retryable": True,
                "message": f"{len(failed)} of {len(listed)} {key} failed: {', '.join(parts)}",
                "constraints": {"failed": dict(counts), "of": len(listed)}}
    return None


def predicted_axes(record: Mapping[str, Any], view_config: Any) -> Mapping[str, Any]:
    """A dry run's record with each arm's ``predicted_approach_world`` and ``predicted_jaw_line_world`` after its
    ``predicted_pose``: the pose's rotation times the controlled point's local approach and closing axes of the
    surface's ``view_config.controlled_frame_convention``, rounded as the pose is. The record as it was without a
    convention, or for an arm without a pose."""

    convention = view_config.get("controlled_frame_convention") if isinstance(view_config, Mapping) else None
    arms = record.get("arms")
    if not isinstance(convention, str) or not isinstance(arms, Mapping):
        return record
    try:
        local = _local_axes(convention)
    except EmbodimentDescriptorError:
        return record
    shown: dict[str, Any] = {}
    for arm, fields in arms.items():
        pose = fields.get("predicted_pose") if isinstance(fields, Mapping) else None
        rotation = _rotation(pose[3:]) if isinstance(pose, list) and len(pose) == 7 else None
        if rotation is None:
            shown[arm] = fields
            continue
        shown[arm] = {}
        for key, value in fields.items():
            shown[arm][key] = value
            if key == "predicted_pose":
                for name, axis in (("predicted_approach_world", "approach"), ("predicted_jaw_line_world", "closing")):
                    shown[arm][name] = [round(sum(row[i] * local[axis][i] for i in range(3)), PREDICTED_DIGITS)
                                        for row in rotation]
    return {**record, "arms": shown}


def preview_line(answer: Mapping[str, Any]) -> Any:
    """A dry run's own line: its numbers, or why the real call would refuse."""

    result = answer.get("result")
    if not isinstance(result, Mapping):
        return None
    refusal = result.get("refusal")
    if isinstance(refusal, Mapping) and isinstance(refusal.get("message"), str):
        return refusal["message"]
    return result.get("summary")


@functools.lru_cache(maxsize=8)
def _local_axes(convention: str) -> dict[str, list[float]]:
    """The controlled point's local approach and closing axes of a controlled-frame convention."""

    return public_local_axes(convention)


def _rotation(quaternion: Any) -> list[list[float]] | None:
    """The rotation matrix of a quaternion ``[qx, qy, qz, qw]``; None for one that is not finite or has no length."""

    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in quaternion):
        return None
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-9:
        return None
    x, y, z, w = (value / norm for value in quaternion)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


__all__ = ["PREDICTED_DIGITS", "UNWRAPPED_CALLS", "item_errors", "predicted_axes", "preview_line", "refusal", "unknown",
           "unwrapped"]
