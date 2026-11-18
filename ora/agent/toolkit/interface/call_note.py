"""What a tool knows about a toolkit call that the call itself does not carry: evidence only.

``noted`` holds fields while the calls in its block run, and the journal writes them as ``note`` on those calls' events
(``native/toolkit-io/calls.jsonl``). A note stays outside the semantic request: replay matches the call as it always
did, and the environment never sees it. A call outside a block has no ``note`` and its events are unchanged.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator

_note: ContextVar[dict] = ContextVar("toolkit_call_note", default={})


def current() -> dict:
    """The note of the call in progress, empty outside a block."""

    return dict(_note.get())


@contextlib.contextmanager
def noted(**fields: object) -> Iterator[None]:
    """The calls made inside the block carry ``fields``, leaving out those that are ``None``."""

    token = _note.set({key: value for key, value in fields.items() if value is not None})
    try:
        yield
    finally:
        _note.reset(token)


__all__ = ["current", "noted"]
