"""The recovery route a RoboDojo runtime may name.

The RunSpec derives ``runtime.recovery_route = "retract_then_unwind"`` from a
surface declaring ``robodojo_retract_then_unwind_recovery_v1``, and plans sealed
with it carry the key, so ``session.__init__`` still validates it here. The
recovery itself is planned by the toolkit (``lib.motion_plan.recovery``:
return_home and recover_pose) and runs as one ``follow_joint_trajectory``; the
SDK executor's ``recover_motion`` that used to read this route is gone.
"""

from __future__ import annotations

from typing import Any, Mapping

RETRACT_THEN_UNWIND = "retract_then_unwind"


def configured_route(config: Mapping[str, Any]) -> str | None:
    """Absent keeps the released straight joint-line recovery."""
    route = config.get("recovery_route")
    if route not in (None, RETRACT_THEN_UNWIND):
        from .bridge import RobodojoError
        raise RobodojoError("recovery_route names no supported route")
    return route


__all__ = ["RETRACT_THEN_UNWIND", "configured_route"]
