"""Generic retention hints: Python decides, the TypeScript core executes.

Domain rules -- which observation or tool-result fields carry no information
(``observation_delivery_trim_v2``), which conventions a payload states once
(``within_payload_deduplication_v1``) and which ``current_actual`` facts are
static within a Session round (``round_static_facts_slot_v1``) -- stay in
Python. Python sends the untrimmed payload plus an ordered list of hints; the
TypeScript core (``ora-pi/src/core/hints.ts``) applies them without knowing
any robot vocabulary:

``{"op": "omit", "path": [...], "rule": ...}``
    Leave the member at ``path`` out of the delivered copy.
``{"op": "hoist_shared", "at": [...], "into": ..., "collections": [{"collection": ...,
  "placeholder": ..., "fields": [[...], ...]}, ...]}``
    Under ``at``, when a collection has at least two entries and every entry
    carries the same string at a field path, state it once under
    ``into["<collection>.<placeholder>"]`` and drop it from the entries.
``{"op": "round_static_facts", "at": [...], "scope": round, ...}``
    Capture the subtree at ``at`` once per scope into a protected slot message
    and deliver later occurrences as a reduction against it (byte-identical
    members omitted, see :mod:`ora.agent.context.delivery.round_static_facts`).

``omit`` hints are derived mechanically from the Python rule itself: the rule
runs, and every member it removed becomes one hint (:func:`omit_hints`). A rule
that rewrites a value instead of removing it cannot be expressed this way and
fails loudly.

:func:`apply_hints` is the Python reference applier; the parity test checks
that the TypeScript applier produces the same bytes.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ora.agent.context.delivery import round_static_facts
from ora.agent.context.delivery.dedup import (
    CAMERA_GEOMETRY_CONVENTION_FIELDS, CONVENTIONS_KEY, _hoist as python_hoist,
)


class HintError(ValueError):
    """A domain rule produced a change that is not a generic hint."""


_MISSING = object()


def omit_hints(before: Any, after: Any, *, prefix: Sequence[Any] = (), rule: str) -> list[dict[str, Any]]:
    """One ``omit`` hint per member ``after`` removed from ``before``.

    ``after`` must equal ``before`` with members removed and nothing else
    changed; anything else raises :class:`HintError`.
    """

    hints: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: list[Any]) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            extra = set(right) - set(left)
            if extra:
                raise HintError(f"{rule} added {sorted(extra)} at {path}")
            for key, child in left.items():
                if key not in right:
                    hints.append({"op": "omit", "path": [*path, key], "rule": rule})
                else:
                    walk(child, right[key], [*path, key])
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise HintError(f"{rule} changed a list length at {path}")
            for index, (item, other) in enumerate(zip(left, right)):
                walk(item, other, [*path, index])
            return
        if type(left) is not type(right) or left != right:
            raise HintError(f"{rule} rewrote a value at {path}")

    walk(before, after, list(prefix))
    return hints


def hoist_hint(at: Sequence[Any], table: Mapping[str, Any] = CAMERA_GEOMETRY_CONVENTION_FIELDS) -> dict[str, Any]:
    return {
        "op": "hoist_shared", "at": list(at), "into": CONVENTIONS_KEY,
        "collections": [
            {"collection": collection, "placeholder": placeholder,
             "fields": [list(path) for path in fields]}
            for collection, (placeholder, fields) in table.items()
        ],
    }


def round_static_facts_hint(*, at: Sequence[Any], scope: int, listing: str) -> dict[str, Any]:
    return {
        "op": "round_static_facts", "at": list(at), "scope": scope, "listing": listing,
        "identity_keys": sorted(round_static_facts.IDENTITY_KEYS),
        "listing_key": round_static_facts.STATIC_FACTS_KEY,
        "scope_field": "round_slot",
        "slot_key": round_static_facts.SLOT_KEY,
        "slot_scope_field": "round",
        "slot_sequence_field": "captured_at_actual_sequence",
        "sequence_member": "actual_sequence",
        "rule": round_static_facts.RULES[listing],
        "slot_context_source": f"round_static_facts_{scope}",
        "slot_flag": round_static_facts.MESSAGE_FLAG,
    }


# -- Reference applier -------------------------------------------------------------


def _get(value: Any, path: Sequence[Any]) -> Any:
    for step in path:
        if isinstance(value, Mapping) and step in value:
            value = value[step]
        elif isinstance(value, list) and isinstance(step, int) and 0 <= step < len(value):
            value = value[step]
        else:
            return _MISSING
    return value


def _omit(value: Any, path: Sequence[Any]) -> None:
    parent = _get(value, path[:-1])
    if isinstance(parent, dict) and path[-1] in parent:
        del parent[path[-1]]


def _hoist(container: dict[str, Any], hint: Mapping[str, Any]) -> None:
    table = {item["collection"]: (item["placeholder"], tuple(tuple(p) for p in item["fields"]))
             for item in hint["collections"]}
    hoisted = python_hoist(container, table)
    container.clear()
    container.update(hoisted)


class StaticFactsState:
    """The scope a ``round_static_facts`` hint last captured, and its slot."""

    def __init__(self) -> None:
        self.scope: Any = None
        self.fields: dict[str, Any] | None = None


def apply_hints(value: Any, hints: Sequence[Mapping[str, Any]], state: StaticFactsState,
                ) -> tuple[Any, dict[str, Any] | None]:
    """The delivered copy and the slot message a new scope produced (or None)."""

    delivered = copy.deepcopy(value)
    slot = None
    for hint in hints:
        op = hint["op"]
        if op == "omit":
            _omit(delivered, hint["path"])
        elif op == "hoist_shared":
            target = _get(delivered, hint["at"])
            if isinstance(target, dict):
                _hoist(target, hint)
        elif op == "round_static_facts":
            parent = _get(delivered, hint["at"][:-1])
            current = parent[hint["at"][-1]]
            if state.scope != hint["scope"] or state.fields is None:
                state.scope = hint["scope"]
                state.fields = round_static_facts.capture(current)
                slot = round_static_facts.slot_message(
                    round_index=hint["scope"], actual_sequence=current[hint["sequence_member"]],
                    fields=state.fields, listing=hint["listing"],
                )
            parent[hint["at"][-1]] = round_static_facts.reduce(
                current, state.fields, round_index=hint["scope"], listing=hint["listing"],
            )
        else:
            raise HintError(f"unknown hint op {op!r}")
    return delivered, slot
