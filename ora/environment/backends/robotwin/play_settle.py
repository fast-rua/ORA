"""The post-action settle of ORA's RoboTwin free-play scenes (scene sets only).

RoboTwin advances physics only inside an action, so an object the arm knocked
over hangs in mid-fall until the next action starts: in a sealed play run
(table_setting_sequence, 2026-10-20) a knife and a plate thrown off the table
stayed frozen in the air for 11 s of the model's thinking. After each action
of a play scene of one of ORA's scene sets, the session now steps the scene
on, every joint and gripper drive target held as ``wait`` holds them
(nothing is written to a drive), ``check_interval_physics_steps`` (10) at a
time, until over one such window no dynamic object of the scene moved more
than ``max_displacement_m`` (1 mm) or turned more than ``max_rotation_deg``
(1 degree), at most ``max_physics_steps`` (500, 2 s of simulated time):
``robotwin_play_scenes.PLAY_POST_ACTION_SETTLE``. Stillness is judged by
poses over a window, not by velocities, which jitter at rest. An object more
than ``off_table_drop_m`` (0.44 m) below the table top has fallen to the
floor: its fall is in the settle that saw it, but it no longer holds one open
(one rolling on the floor would otherwise keep every later settle at its cap),
and the record names it (``off_table``).

The settle runs from the execution-capture ``execute`` wrapper after the
action's receipt is written and before its ``action_end`` capture boundary,
so its frames are the action's (they carry its action id) and the next
observation sees the settled scene. It changes no receipt and no
observation, writes no drive target, does not count toward the episode's
semantic action limit (``take_action_cnt``) and does not run the official
success latch (a play scene has none). Each settle is recorded as one line
of ``post-action-settle.jsonl`` beside the execution capture
(``POST_ACTION_SETTLE_SCHEMA``): the action's id (a skill stage's settle
carries its parent action's id and its own as ``dispatched_action_id``), the
capture clock's physics step it started at (``physics_step_start``, steps
since the round's reset, the frames' ``sim_step``) and how many it took. A
benchmark task, and play on an official RoboTwin task, never settle.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Mapping

from ora.environment.benchmarks.robotwin_play_scenes import PLAY_POST_ACTION_SETTLE

POST_ACTION_SETTLE_SCHEMA = "ora.robotwin-play-post-action-settle.v1"
POST_ACTION_SETTLE_RECORD = "post-action-settle.jsonl"


def rotation_deg(first: Any, second: Any) -> float:
    """The angle (degrees) between two unit quaternions."""
    dot = min(1.0, abs(sum(float(a) * float(b) for a, b in zip(first, second))))
    return math.degrees(2.0 * math.acos(dot))


def window_motion(before: Mapping[str, Any], after: Mapping[str, Any], *,
                  ignore: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    """The largest displacement (m) and turn (degrees) of any object between two ``object_poses`` readings,
    and the object that moved most (displacement in millimetres against turn in degrees)."""
    moved, turned, mover, most = 0.0, 0.0, None, 0.0
    for name, (position, orientation) in after.items():
        if name not in before or name in ignore:
            continue
        previous_position, previous_orientation = before[name]
        distance = math.dist([float(value) for value in position], [float(value) for value in previous_position])
        angle = rotation_deg(previous_orientation, orientation)
        moved, turned = max(moved, distance), max(turned, angle)
        if max(distance * 1000.0, angle) > most:
            mover, most = name, max(distance * 1000.0, angle)
    return {"max_displacement_m": round(moved, 6), "max_rotation_deg": round(turned, 4), "object": mover}


def settle_after_action(session: Any, payload: Mapping[str, Any], *,
                        bounds: Mapping[str, Any] = PLAY_POST_ACTION_SETTLE) -> dict[str, Any] | None:
    """Step a scene-set play scene on until its objects rest (or the cap); append and return its record.

    Returns None (nothing stepped, nothing recorded) for anything but a live scene-set play scene that is
    not terminal: a benchmark task, play on an official task, or a scene whose environment has no
    ``object_poses``.
    """
    if getattr(session, "play_scene_set", None) is None or session.is_terminal:
        return None
    environment = session._require_environment()
    poses = getattr(environment, "object_poses", None)
    if not callable(poses):
        return None
    interval, limit = int(bounds["check_interval_physics_steps"]), int(bounds["max_physics_steps"])
    floor = float(environment.table_top_z) - float(bounds["off_table_drop_m"])
    scene = environment.scene
    started = time.monotonic()
    step_start = getattr(session, "_execution_physics_step", None)
    context = getattr(session, "_execution_action", None) or {}
    before = poses()
    off_table: set[str] = set()
    steps, windows, ended = 0, 0, "step_limit"
    first: dict[str, Any] | None = None
    last: dict[str, Any] = {"max_displacement_m": 0.0, "max_rotation_deg": 0.0, "object": None}
    while steps < limit:
        chunk = min(interval, limit - steps)
        for _ in range(chunk):
            scene.step()  # drive targets untouched: every arm and gripper holds its last command
        steps += chunk
        windows += 1
        after = poses()
        off_table = {name for name, (position, _) in after.items() if float(position[2]) < floor}
        last = window_motion(before, after, ignore=off_table)
        first = first or last
        if (last["max_displacement_m"] <= float(bounds["max_displacement_m"])
                and last["max_rotation_deg"] <= float(bounds["max_rotation_deg"])):
            ended = "at_rest"
            break
        before = after
    dispatched = str(payload.get("action_id"))
    record = {
        "schema": POST_ACTION_SETTLE_SCHEMA,
        "action_id": dispatched.split(":", 1)[0],
        **({"dispatched_action_id": dispatched} if ":" in dispatched else {}),
        "action": payload.get("action"),
        "action_sequence": context.get("action_sequence"),
        "round_index": int(getattr(session, "round_index", 0)),
        "physics_step_start": step_start,
        "physics_steps": steps,
        "physics_dt_s": float(scene.get_timestep()),
        "simulated_s": round(steps * float(scene.get_timestep()), 6),
        "windows": windows,
        "ended": ended,
        "moving_at_start": bool(first is not None and (
            first["max_displacement_m"] > float(bounds["max_displacement_m"])
            or first["max_rotation_deg"] > float(bounds["max_rotation_deg"]))),
        "first_window": first,
        "last_window": last,
        "off_table": sorted(off_table),
        "bounds": dict(bounds),
        "wall_s": round(time.monotonic() - started, 4),
        "visibility": "control_only",
    }
    with (session.output_root / POST_ACTION_SETTLE_RECORD).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


__all__ = ["POST_ACTION_SETTLE_RECORD", "POST_ACTION_SETTLE_SCHEMA", "rotation_deg", "settle_after_action",
           "window_motion"]
