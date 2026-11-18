"""One more admission of a pose move the pacing refused for its wrist: the same empty parallel jaw, half a turn round.

A Panda's last joint stops at +/-2.8973 rad. From some starts every in-limit goal of the requested orientation lies
at the far end of that joint, so the planned path turns the hand the long way round and the pacing refuses it
(``paced_path_wrist_unwind``, ``motion_pace.check_unwind``); no other goal of the same orientation helps, because
the limit, not the planner, chose the long way. A symmetric parallel jaw turned a half-turn about its approach axis
grips the same way, and its goal lies half a turn away on that joint, usually well inside the range. So a backend
whose pose move the pacing refused for the wrist admits the move once more, to that twin (``jaw_twin``), when its
jaws are commanded open. A jaw that may hold something keeps the request: the load would turn with the hand.

The twin runs only when everything that admits a move admits it (the planner and the pacing); its receipt then says
the executed orientation is the twin (``motion.wrist.jaw_flip_retry``, ``applied: true``), and the pose error and
arrival are judged against it. Otherwise the original refusal stands and says the twin was tried and what refused it
(``motion.refused.jaw_flip_retry``, ``applied: false``). A joint goal, a recovery and a probe's immutable plan are
never retried. A dry run asks the same retry as the real call, so a preview never promises what the real call
refuses, nor refuses what it would run.

Pure computation: LIBERO (``planned_motion.py``), RoboCasa365 (``motion.py``) and RoboLab (``shared_motion.py``)
call it with their own admission.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Sequence

from .motion_pace import PacedPath
from .pace_refusal import WRIST_UNWIND_CODE, PaceRefusal

#: The refusal a move is retried for: the pacing's refusal of an unrequested wrist turn.
RETRY_CODE = WRIST_UNWIND_CODE
#: The approach axis of the Panda's grip site and of the Robotiq's TCP, in their own frame.
APPROACH_LOCAL_Z = (0.0, 0.0, 1.0)
#: How far the twin turns the target about its approach axis.
TWIN_TURN_DEG = 180.0


def jaw_twin(pose: Sequence[float], approach_local: Sequence[float] = APPROACH_LOCAL_Z) -> list[float]:
    """Pure computation: ``pose`` (x, y, z, qx, qy, qz, qw) turned a half-turn about its own approach axis (w >= 0).

    The orientation is ``q`` times the half-turn ``(axis, 0)``; about local +z that is LIBERO's probe variant
    ``[y, -x, w, -z]`` and RoboCasa's ``kinematics.jaw_equivalent``, up to the quaternion's sign.
    Example: jaw_twin([0, 0, 0, 0, 0, 0, 1]) == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]."""
    x, y, z, w = (float(value) for value in pose[3:7])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    axis = [float(value) for value in approach_local]
    length = math.sqrt(sum(value * value for value in axis))
    if len(pose) != 7 or not math.isfinite(norm) or norm < 1e-9 or len(axis) != 3 or length < 1e-9:
        raise ValueError("a jaw twin needs one pose (x, y, z, qx, qy, qz, qw) and a non-zero approach axis")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    ax, ay, az = (value / length for value in axis)
    turned = [w * ax + y * az - z * ay, w * ay - x * az + z * ax, w * az + x * ay - y * ax, -(x * ax + y * ay + z * az)]
    sign = -1.0 if turned[3] < 0.0 else 1.0
    return [*(float(value) for value in pose[:3]), *(sign * value + 0.0 for value in turned)]


def retry(pose: Sequence[float], refusal: PaceRefusal, admit: Callable[[list[float]], Any], *,
          jaws_open: Callable[[], bool], approach_local: Sequence[float] = APPROACH_LOCAL_Z
          ) -> tuple[list[float], Any, dict[str, Any]] | None:
    """The move to ``pose`` that ``refusal`` stopped, admitted once more to its jaw twin: ``(twin, admitted, record)``.

    ``None`` leaves ``refusal`` standing: it was not a wrist refusal, the jaws are not open (``jaws_open`` is asked only
    for a wrist refusal), or the twin was refused too, which ``refusal.constraints["jaw_flip_retry"]`` then records.
    ``admit(twin)`` returns what the backend would run, or the code of the twin's refusal; a ``PaceRefusal`` it raises
    is the twin's refusal as well. ``record`` is what the receipt carries (``recorded``)."""
    if refusal.code != RETRY_CODE or not jaws_open():
        return None
    twin = jaw_twin(pose, approach_local)
    try:
        admitted = admit(twin)
    except PaceRefusal as again:
        admitted = again.code
    tried = {"applied": False, "reason": RETRY_CODE, "turn_deg": TWIN_TURN_DEG,
             "refused_unrequested_turn_deg": refusal.constraints.get("unrequested_turn_deg")}
    if isinstance(admitted, str):
        refusal.constraints["jaw_flip_retry"] = {**tried, "twin_refused": admitted}
        return None
    return twin, admitted, {**tried, "applied": True, "requested_orientation_xyzw": [float(v) for v in pose[3:7]],
                            "executed_orientation_xyzw": twin[3:]}


def recorded(paced: PacedPath, record: dict[str, Any]) -> PacedPath:
    """``paced`` whose wrist report carries ``record``: the receipt's ``motion.wrist.jaw_flip_retry``."""
    return replace(paced, wrist={**paced.wrist, "jaw_flip_retry": dict(record)})


__all__ = ["APPROACH_LOCAL_Z", "RETRY_CODE", "TWIN_TURN_DEG", "jaw_twin", "recorded", "retry"]
