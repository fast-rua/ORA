"""A run's scene trace for the 3D replay: every simulated body's mesh and pose, on the simulator-step clock.

``ora replay RUN --scene`` writes the trace (format: ``ora/environment/scene_trace.py``) to ``scene-trace/<run_id>/``
beside the run's ``runs/`` directory, or under ``--scene-traces DIR``. This module only reads it. The page gets the
rows of one round and the bodies it draws, then fetches ``poses.bin`` and each body's ``meshes/<id>.glb`` through
``api/scene-trace``. It draws objects and static bodies, and the trace's robot links when the bundle has no URDF for
the robot (``robots.py``); the backdrop (the wall and the ground) would stand between the camera and the table. A trace
that names each arm's TCP (``tcp``) gives the page the hand's own path between commands. A row's step is on the
capture's clock; ``per_unit`` turns it into the page's (RoboDojo: 10 physics steps per native action; RoboCasa365
and RoboLab, which have no capture: 25 MuJoCo steps and 8 PhysX steps per native action).
"""

from __future__ import annotations

import os
import re

from viewer.dashboard.config import Config
from viewer.dashboard.replay import _json, _jsonl
from viewer.dashboard.store import memo, stat_sig

SCHEMA = "ora.scene-trace.v1"
DIRECTORY = "scene-trace"
DRAWN = ("object", "static")
ROBOT = "robot"
POSES = "poses.bin"
POSE_BYTES = 7 * 4
#: The files ``api/scene-trace`` serves, relative to the trace's directory.
SERVED = re.compile(r"^(?:poses\.bin|meshes/[A-Za-z0-9][A-Za-z0-9._-]*\.glb)$")


def directory(run_dir: str, cfg: Config) -> str:
    run_dir = run_dir.rstrip(os.sep)
    root = cfg.scene_traces or os.path.join(os.path.dirname(os.path.dirname(run_dir)), DIRECTORY)
    return os.path.join(root, os.path.basename(run_dir))


def missing_note(run_dir: str, options: str = "") -> str:
    return "no scene trace: export one with `ora replay %s --scene%s`" % (os.path.basename(run_dir.rstrip(os.sep)), options)


def _rows(path: str) -> list:
    return [[r.get("index"), r.get("round"), r.get("physics_step")] for r in _jsonl(path)]


def _tcp(manifest: dict) -> dict | None:
    """Each arm's TCP as the column of its body and the offset in that body's frame; None when the trace has none."""
    columns = {b.get("name"): i for i, b in enumerate(manifest["bodies"])}
    links = {arm: {"index": columns[link["body"]], "offset": link["offset_xyz_xyzw"]}
             for arm, link in (manifest.get("tcp") or {}).items() if link.get("body") in columns}
    return links or None


def bundle(run_dir: str, cfg: Config, round_index: int, robot: bool = False, per_unit: float = 1) -> dict | None:
    """What the page needs for one round, or None when the run has no complete trace (or none for this round).
    ``robot``: draw the trace's robot links, because the bundle has no URDF for the robot."""
    root = directory(run_dir, cfg)
    path = os.path.join(root, "manifest.json")
    sig = stat_sig(path)
    manifest = memo(("scene-trace", path), sig, lambda: _json(path)) if sig else None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or len(manifest.get("shape") or []) != 3:
        return None
    count, width = manifest["shape"][0], manifest["shape"][1]
    poses = stat_sig(os.path.join(root, POSES))
    steps = os.path.join(root, "steps.jsonl")
    rows = memo(("scene-trace-rows", steps), stat_sig(steps), lambda: _rows(steps))
    if not poses or poses[1] != count * width * POSE_BYTES or len(rows) != count:
        return None
    here = [[index, step / per_unit if per_unit != 1 else step] for index, rnd, step in rows if rnd == round_index]
    if not here:
        return None
    drawn = DRAWN + (ROBOT,) if robot else DRAWN
    return {"version": sig[0], "source": manifest.get("source"), "clock": manifest.get("clock"), "body_count": width,
            "rows": here, "tcp": _tcp(manifest), "rounds": manifest.get("rounds") or [round_index],
            "spans": manifest.get("spans") or [],
            "bodies": [{"index": i, "id": b["id"], "kind": b["kind"], "mesh": b["mesh"]}
                       for i, b in enumerate(manifest["bodies"]) if b["kind"] in drawn and b.get("mesh")]}


def trace_file(run_dir: str, rel: str, cfg: Config) -> str | None:
    """A served file of the run's trace, only from inside its directory (after resolving links)."""
    if not SERVED.match(rel):
        return None
    root = os.path.realpath(directory(run_dir, cfg))
    path = os.path.realpath(os.path.join(root, rel))
    if not path.startswith(root.rstrip(os.sep) + os.sep) or not os.path.isfile(path):
        return None
    return path
