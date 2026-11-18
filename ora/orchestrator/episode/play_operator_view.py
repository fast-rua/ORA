"""Free play's operator view: the environment's picture for the human, stored beside the run.

The environment session's optional ``operator_view()`` (base.py) is asked after the
policy has its observation. The PNG is written to ``native/operator-view/<step:06d>.png``
and an ``operator_view_frame`` event names it; it is the operator's alone: no observation,
context, tool result or agent event carries it. A failure is recorded and the session
goes on; after a few in a row it stops asking.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from ora.environment.base import validate_operator_view

from .records import _append

FAILURE_LIMIT = 3


def store_operator_view(loop: Any, step: int) -> None:
    """Store the operator view of observation ``step`` for a ``PlayLoop``, if the environment has one."""

    view_of = getattr(loop.runtime, "operator_view", None)
    if not callable(view_of) or loop.operator_view_failures >= FAILURE_LIMIT:
        return
    try:
        view = validate_operator_view(view_of())
        if view is None:
            return
        target = loop.native / "operator-view" / f"{step:06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(view["png"])
        temporary.replace(target)
    except Exception as exc:
        loop.operator_view_failures += 1
        loop.record_auxiliary_failure("play:operator_view", exc)
        return
    loop.operator_view_failures = 0
    record = {
        "observation_step": step, "path": target.relative_to(loop.run_root).as_posix(),
        "camera": view["camera"], "width": view["width"], "height": view["height"],
        "sha256": hashlib.sha256(view["png"]).hexdigest(),
    }
    loop.operator_view = record
    _append(loop.events, {"type": "operator_view_frame", **record, "time": time.time()})
    loop.publish_live("operator_view_frame", record)
