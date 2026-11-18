"""A read-only plan of one action: the question a dry run asks and the shape of its answer (``ora.action-plan.v1``).

Where the environment plans a move when it executes it, a dry run asks the same planner the same question over the
read-only ``align_plan`` bridge: ``request`` wraps the action, a session's ``plan_action`` answers with what it would run,
and nothing steps the world. The answer is planner output and forward kinematics only: no contact, no payload, no
physics rollout, so it means the same on a real robot. Where the policy side plans, it builds the same segments.

An answer is one or more segments in the order they would run. A ``planned`` segment is a path a planner or a kinematic
schedule produced; an unplanned one is a closed-loop servo's commanded line, which has no path until it runs. Poses are
the public controlled point in the world frame the observation publishes, ``[x, y, z, qx, qy, qz, qw]``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

REQUEST_SCHEMA = "ora.action-plan-request.v1"
PLAN_SCHEMA = "ora.action-plan.v1"
#: The tools that move the robot, which alone take ``dry_run``. ``wait_ticks`` moves nothing, and reads have none.
MOVING_TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch", "recover_pose", "return_home", "set_gripper",
                "grasp_execute", "move_base")
#: The name of the argument, in a tool call and a script.
DRY_RUN_ARGUMENT = "dry_run"
#: A segment carries at most this many poses (a preview needs the shape, not every native action).
MAX_POSES = 96
#: The kinds of segment: an arm's path, the base, a lift and the jaws.
KINDS = frozenset({"arm", "base", "torso", "gripper"})


def request(action: Mapping[str, Any]) -> dict[str, Any]:
    """The read-only request for ``action`` (a validated runtime action)."""
    return {"schema": REQUEST_SCHEMA, "action": dict(action)}


def is_request(value: Any) -> bool:
    """Whether ``value`` is a plan request (the ``align_plan`` bridge also carries other schemas)."""
    return isinstance(value, Mapping) and value.get("schema") == REQUEST_SCHEMA


def decimated(rows: Sequence[Sequence[float]], limit: int = MAX_POSES) -> list[list[float]]:
    """At most ``limit`` rows of a path, evenly along it, its first and last kept."""
    values = [[float(value) for value in row] for row in rows]
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    return [values[round(index * step)] for index in range(limit)]


def segment(kind: str, arm: str | None, label: str, *, poses: Sequence[Sequence[float]], planned: bool,
            native_actions: int | None = None, duration_s: float | None = None,
            not_predicted: bool = False, **detail: Any) -> dict[str, Any]:
    """One stretch of the action. ``native_actions`` is None for a servo, whose length only its run says.

    ``not_predicted`` marks what follows the first contact: it is commanded, and nothing here predicts it."""
    if kind not in KINDS:
        raise ValueError(f"unknown segment kind {kind!r}")
    return {"kind": kind, "arm": arm, "label": label, "planned": bool(planned), "poses": decimated(poses),
            "native_actions": native_actions, "duration_s": duration_s, "not_predicted": bool(not_predicted), **detail}


def body_points(fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], joints: Sequence[float],
                step_rad: float = 0.2) -> list[list[float]]:
    """Points along a serial arm at ``joints``, from forward kinematics alone, the controlled point last.

    Turning one joint either way moves the controlled point on a circle about that joint's axis, so the circle's centre
    lies on the axis, whatever the robot. ``fk`` answers hypothetical joints without moving anything. A preview leaves
    these out of what it measures against: its depth images show the arm too. Example: body_points(fk, q)."""
    import numpy as np

    start = [float(value) for value in joints]
    rows = [start]
    for index in range(len(start)):
        for sign in (1.0, -1.0):
            rows.append([value + sign * step_rad if place == index else value for place, value in enumerate(start)])
    points = np.asarray([pose[:3] for pose in fk(rows)], dtype=np.float64)
    here = points[0]
    found = []
    for index in range(len(start)):
        first, second = points[1 + 2 * index] - here, points[2 + 2 * index] - here
        normal = np.cross(first, second)
        area = float(normal @ normal)
        if area > 1e-12:
            found.append(here + ((first @ first) * np.cross(second, normal)
                                 + (second @ second) * np.cross(normal, first)) / (2.0 * area))
    return [[float(value) for value in point] for point in (*found, here)]


def planned(route: str, segments: Sequence[Mapping[str, Any]], **detail: Any) -> dict[str, Any]:
    """The answer of an action that would run."""
    return {"schema": PLAN_SCHEMA, "status": "planned", "route": route, "segments": [dict(item) for item in segments],
            **detail}


def refused(code: str, stage: str, message: str | None = None, **constraints: Any) -> dict[str, Any]:
    """The answer of an action the real call would refuse before anything moved."""
    return {"schema": PLAN_SCHEMA, "status": "refused", "failure_code": str(code), "failure_stage": str(stage),
            "message": message, "constraints": constraints}


def unavailable(reason: str) -> dict[str, Any]:
    """The answer where no plan can be made without stepping the world; never a guess at one."""
    return {"schema": PLAN_SCHEMA, "status": "unavailable", "reason": str(reason)}


__all__ = ["DRY_RUN_ARGUMENT", "KINDS", "MAX_POSES", "MOVING_TOOLS", "PLAN_SCHEMA", "REQUEST_SCHEMA", "body_points",
           "decimated", "is_request", "planned", "refused", "request", "segment", "unavailable"]
