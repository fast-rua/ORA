"""Immutable public wire values; no worker or client objects cross this edge."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


class InterfaceError(ValueError):
    """A protocol failure, distinct from a recorded backend failure."""


class NotRecorded(InterfaceError):
    """未记录: this boundary has no matching evidence, with optional exact requests."""

    def __init__(self, message, *, missing_requests=()):
        super().__init__(message)
        self.missing_requests = list(missing_requests)


class Divergence(InterfaceError):
    """分歧: the proposed side effect differs from the recorded invocation."""


class AmbiguousRecord(InterfaceError):
    """More than one invocation occupies one identity or sequence slot."""


class CorruptRecord(InterfaceError):
    """Evidence does not verify against its declared identity."""


class AccessDenied(InterfaceError):
    """The request exceeds the public data or action capability."""


class ActionNotExecuted(AccessDenied):
    """A replay without a simulator reached an action: it carries the proposal and never executes it.

    Raised by the snapshot replay's ``act`` capability at the tool's first
    ``execute_action`` / ``submit_action``; nothing is executed or queued, so
    the replay has no receipt or measured outcome for it.
    """

    def __init__(self, method: str, action: Any, budget: int | None):
        super().__init__(f"phase-one replay cannot execute actions: {method} proposed, not executed")
        self.method, self.action, self.budget = method, action, budget


class SimulatorRequired(InterfaceError):
    """需要仿真器: the requested capability requires a live planner."""


class ActionNotStarted(InterfaceError):
    """Trusted execution owner proved this request failed before physical dispatch."""

    def __init__(self, message: str, *, stage: str = "before_intent"):
        super().__init__(message)
        if stage not in {"before_intent", "begin"}:
            raise ValueError("invalid no-dispatch evidence stage")
        self.stage = stage


class ServiceUnavailable(InterfaceError):
    """No external service capability was explicitly supplied for this replay."""


class ReferenceFailure(InterfaceError):
    """Public advisory-reference exception type, with no worker object or message."""

    def __init__(self, exception_type: str):
        if not isinstance(exception_type, str) or not exception_type.isidentifier():
            raise CorruptRecord("invalid reference failure type")
        self.exception_type = exception_type
        super().__init__(exception_type)


class PublicToolFailure(InterfaceError):
    """An explicit public tool-error projection; never a serialized exception object."""

    def __init__(self, error: dict):
        required = {'code', 'message', 'retryable', 'field', 'received', 'constraints', 'infrastructure'}
        if not required <= error.keys() or set(error) - required - {'retry_requires_changed_arguments'}:
            raise CorruptRecord('invalid public tool failure projection')
        self.error_bytes = canonical(error)
        super().__init__(error['message'])
        self.code, self.retryable = error['code'], error['retryable']

    def public(self) -> dict:
        return decode(self.error_bytes)


def canonical(value: Any) -> bytes:
    """Strict JSON, preserving list order and int/float/bool distinctions."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise InterfaceError("strict finite JSON required")
    check(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode(data: bytes) -> Any:
    """Read strict finite JSON and refuse duplicate keys instead of discarding them."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CorruptRecord("duplicate JSON object key")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs)
        canonical(value)
        return value
    except (ValueError, UnicodeError) as error:
        raise CorruptRecord("malformed public JSON evidence") from error


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Response:
    """Exact recorded semantic bytes; comparison never changes those bytes."""

    payload: bytes
    comparison_bytes: bytes | None = None

    def json(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def comparison(self) -> dict[str, Any] | None:
        return None if self.comparison_bytes is None else json.loads(self.comparison_bytes)


@dataclass(frozen=True)
class PublicData:
    """A field-projected payload produced by a trusted adapter, not a worker dump."""

    data: bytes
    sources: bytes = b"[]"
    cache_hit: bool = False

    @classmethod
    def from_json(cls, data: dict[str, Any], *, sources: list | None = None,
                  cache_hit: bool = False) -> PublicData:
        return cls(canonical(data), canonical(sources or []), cache_hit)


@dataclass(frozen=True)
class CallContext:
    """An immutable parent tool identity and its published boundary."""

    encoded: bytes

    @classmethod
    def create(cls, *, run_id: str, round_index: int, tool_call_id: str,
               boundary: dict, state: dict, check_task: bool = False) -> CallContext:
        if type(check_task) is not bool:
            raise InterfaceError("check_task capability must be boolean")
        allowed_state = {"observation", "published_seq", "world_mutation_sequence", "action_sequence", "native_action", "physics_clock"}
        if set(state) - allowed_state:
            raise InterfaceError("context state contains undeclared public fields")
        if not run_id or not tool_call_id or type(round_index) is not int or round_index < 0:
            raise InterfaceError("run, round and parent tool identity are required")
        if boundary.get("kind") not in ("before_call", "after_native_action", "snapshot_step"):
            raise InterfaceError("explicit snapshot boundary required")
        if boundary.get("kind") == "snapshot_step" and (type(boundary.get("step")) is not int or boundary["step"] < 0):
            raise InterfaceError("snapshot_step requires a nonnegative published step")
        return cls(canonical(dict(run_id=run_id, round_index=round_index,
                                 tool_call_id=tool_call_id, boundary=boundary,
                                 state=state, check_task=check_task)))

    def json(self) -> dict:
        return json.loads(self.encoded)

    def identity(self, ordinal: int) -> dict:
        value = self.json()
        identity = {key: value[key] for key in ("run_id", "round_index", "tool_call_id", "boundary")}
        identity["interface_ordinal"] = ordinal
        for phase in ("before", "after"):
            identity[phase] = dict(tool_call_id=value["tool_call_id"], interface_ordinal=ordinal, phase=phase)
        return identity
