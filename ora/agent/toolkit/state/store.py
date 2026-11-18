"""An explicit owner for state, with copy-on-read finite JSON values.

Accessors are same-process capability boundaries, not process isolation.
Neither a data interface nor a worker owns this store. The Supervisor gives
each caller the appropriate accessor; tools receive only the tool writer.
"""

import json
import math
from typing import Any, Mapping

from .keys import StateAccessDenied, StateError, key_definition, require_read


def encode(value: Any) -> bytes:
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
        raise StateError("state values must be finite JSON with string object keys")
    check(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


class StateStore:
    """One live state. Environment/execution updates are atomic across their keys."""

    def __init__(self, *, read_only: bool = False) -> None:
        self._values: dict[str, bytes] = {}
        self.read_only = read_only
        self.complete_boundary = False
        self.tool_journal = None
        self.tool_watermark = None
        self.exec_ledger = None

    def access(self, *, reader: str = "robot", writer: str | None = None,
               privileged_keys: tuple[str, ...] = (), privileged_tools: tuple[str, ...] = ()) -> "StateAccess":
        if reader not in {"robot", "rsi", "supervisor"} or writer not in {None, "environment", "execution", "tool"}:
            raise StateAccessDenied("invalid state accessor")
        if writer in {"environment", "execution"} and reader != "supervisor":
            raise StateAccessDenied("only the Supervisor assembles environment/execution writers")
        if privileged_keys and (set(privileged_keys) != {"privileged.task_check"} or "check_task" not in privileged_tools):
            raise StateAccessDenied("privileged task check requires both tool and key grants")
        return StateAccess(self, reader, writer, frozenset(privileged_keys))

    def action_started(self) -> None:
        if self.read_only:
            raise StateAccessDenied("loaded state is read-only")
        self.complete_boundary = False

    def publication_completed(self) -> None:
        if self.read_only:
            raise StateAccessDenied("loaded state is read-only")
        self.complete_boundary = True


class StateAccess:
    """Read current values; tool writes accept only tool.*. No filesystem access."""

    def __init__(self, store: StateStore, reader: str, writer: str | None, privileged_keys: frozenset[str]):
        self._store, self._reader, self._writer = store, reader, writer
        self._privileged_keys = privileged_keys

    def read(self, key: str) -> Any:
        require_read(key, reader=self._reader, privileged_keys=self._privileged_keys)
        if key not in self._store._values:
            raise StateError(f"state not recorded: {key}")
        return json.loads(self._store._values[key])

    def values(self) -> dict[str, Any]:
        result = {}
        for key in sorted(self._store._values):
            if key.startswith("privileged.") and self._reader == "robot" and key not in self._privileged_keys:
                continue
            result[key] = self.read(key)
        return result

    def write(self, key: str, value: Any) -> None:
        self.update({key: value})

    def update(self, values: Mapping[str, Any]) -> None:
        if self._store.read_only or self._writer is None:
            raise StateAccessDenied("state accessor is read-only")
        encoded = {}
        for key, value in values.items():
            definition = key_definition(key)
            if definition.writer != self._writer:
                raise StateAccessDenied(f"{self._writer} cannot write {key}")
            encoded[key] = encode(value)
        previous = {key: self._store._values.get(key) for key in encoded}
        self._store._values.update(encoded)
        if self._writer == "tool" and self._store.tool_journal is not None:
            # Evidence failures must not undo or replace a successful tool write.
            self._store.tool_journal.changed(previous, encoded)
