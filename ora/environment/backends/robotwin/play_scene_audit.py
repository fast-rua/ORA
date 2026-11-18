"""Trusted play-scene audit worker: build, settle and render ORA's RoboTwin free-play scenes.

Run by ``ora/orchestrator/backends/robotwin/play_audit.py`` under the RoboTwin
interpreter, with the RoboTwin checkout as working directory, one scene per
process. For every seed it builds the scene exactly as the live session does
(``common._load_task_args``, ``setup_demo``; RoboTwin's own stability check
runs inside), then checks what a play scene needs and nothing a task would:

- built: ``setup_demo`` returned and every object the scene placed exists;
- settled: over ``settle_steps`` more physics steps no object moves more than
  ``max_displacement_m`` or turns more than ``max_rotation_deg``, and at the end
  no dynamic object is faster than the speed limits;
- on the table: every object's lowest point is within
  ``table_contact_tolerance_m`` of the table top and its centre over the table;
- inside the arms' workspace: every object's centre lies in
  ``robotwin_play_scenes.PLAY_WORKSPACE_XY_M``;
- not interpenetrating: no contact between two scene objects, an object and
  the table or an object and the robot is deeper than ``max_penetration_m``,
  and no object touches the robot;
- renders all cameras: every RoboTwin camera (head, front and both wrists, the
  observer and both world cameras) and ORA's operator camera
  (``play_scenes.base.OPERATOR_CAMERA``) returns a frame that is not blank.

It records poses, contacts and camera images as qualification evidence: they
never reach a policy. No expert runs, no robot action is published and there
is no verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

SEED_SCHEMA = "ora.robotwin-play-scene-audit-seed.v1"
CRITERIA = {
    "settle_steps": 250,
    "max_displacement_m": 0.002,
    "max_rotation_deg": 2.0,
    "max_linear_speed_m_s": 0.01,
    "max_angular_speed_rad_s": 0.1,
    "table_contact_tolerance_m": 0.004,
    "table_half_extent_xy_m": [0.6, 0.35],
    "max_penetration_m": 0.002,
    "min_image_std": 2.0,
}
#: RoboTwin's world point-cloud cameras, as the prototypes named their views.
WORLD_CAMERAS = {"world_left": "world_camera1", "world_right": "world_camera2"}
#: RoboTwin files a play scene's construction reads at the pinned revision.
ROBOTWIN_SOURCES = ("envs/_base_task.py", "envs/camera/camera.py", "envs/robot/robot.py",
                    "task_config/_camera_config.yml", "task_config/_embodiment_config.yml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_deg(q1: Any, q2: Any) -> float:
    dot = min(1.0, abs(sum(float(a) * float(b) for a, b in zip(q1, q2))))
    return math.degrees(2.0 * math.acos(dot))


def _vector(value: Any) -> list[float]:
    return [round(float(item), 6) for item in value]


def _save_image(rgb: Any, path: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    array = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return {"path": path.name, "sha256": _sha256(path), "shape": list(array.shape),
            "mean": round(float(array.mean()), 3), "std": round(float(array.std()), 3)}


def render_cameras(environment: Any, directory: Path) -> dict[str, dict[str, Any]]:
    """Every RoboTwin camera's RGB frame and the operator camera's, saved as ``<camera>.png``."""
    environment._update_render()
    cameras = environment.cameras
    cameras.update_picture()
    images = {str(name): value["rgb"] for name, value in cameras.get_rgb().items()}
    images["observer_camera"] = cameras.get_observer_rgb()
    for name, attribute in WORLD_CAMERAS.items():
        camera = getattr(cameras, attribute)
        camera.take_picture()
        images[name] = (camera.get_picture("Color") * 255).clip(0, 255)
    operator = getattr(environment, "operator_rgb", None)
    if callable(operator):  # ORA's operator camera, outside RoboTwin's camera manager
        images[str(environment.operator_camera_name)] = operator()
    return {name: _save_image(rgb, directory / f"{name}.png") for name, rgb in sorted(images.items())}


def _classify(entity: Any, objects: Mapping[str, Any]) -> str:
    name = entity.get_name() if entity is not None else ""
    if name in objects:
        return name
    if name in {"table", "wall", "ground", ""}:
        return name or "ground"
    return "robot"


def contact_report(environment: Any, objects: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each touching pair that involves a scene object, with its deepest separation."""
    pairs: dict[tuple[str, str], float] = {}
    for contact in environment.scene.get_contacts():
        bodies = [_classify(getattr(body, "entity", None), objects) for body in contact.bodies]
        if not any(body in objects for body in bodies) or not contact.points:
            continue
        key = tuple(sorted(bodies))
        separation = min(float(point.separation) for point in contact.points)
        pairs[key] = min(separation, pairs.get(key, separation))  # type: ignore[index]
    return [{"bodies": list(key), "min_separation_m": round(value, 6)} for key, value in sorted(pairs.items())]


def check_scene(environment: Any, criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """Settle, measure and judge one built scene (see the module docstring)."""
    import numpy as np

    from ora.environment.benchmarks.robotwin_play_scenes import PLAY_WORKSPACE_XY_M
    from .play_scenes.base import rigid_component

    objects = {item.name: item for item in environment.play_objects}
    before = {name: item.entity.get_pose() for name, item in objects.items()}
    for _ in range(int(criteria["settle_steps"])):
        environment.scene.step()
    top = float(environment.table_top_z)
    half_x, half_y = criteria["table_half_extent_xy_m"]
    rows, failures = [], []
    for name, item in objects.items():
        component = rigid_component(item.entity)
        low, high = (np.asarray(value, dtype=float) for value in component.compute_global_aabb_tight())
        centre = (low + high) / 2
        pose = item.entity.get_pose()
        moved = float(np.linalg.norm(np.asarray(pose.p) - np.asarray(before[name].p)))
        turned = _rotation_deg(pose.q, before[name].q)
        linear = float(np.linalg.norm(component.get_linear_velocity())) if item.dynamic else 0.0
        angular = float(np.linalg.norm(component.get_angular_velocity())) if item.dynamic else 0.0
        checks = {
            "at_rest": (moved <= criteria["max_displacement_m"] and turned <= criteria["max_rotation_deg"]
                        and linear <= criteria["max_linear_speed_m_s"]
                        and angular <= criteria["max_angular_speed_rad_s"]),
            "on_table": (abs(float(low[2]) - top) <= criteria["table_contact_tolerance_m"]
                         and abs(float(centre[0])) <= half_x and abs(float(centre[1])) <= half_y),
            "in_workspace": (PLAY_WORKSPACE_XY_M["x"][0] <= float(centre[0]) <= PLAY_WORKSPACE_XY_M["x"][1]
                             and PLAY_WORKSPACE_XY_M["y"][0] <= float(centre[1]) <= PLAY_WORKSPACE_XY_M["y"][1]),
        }
        failures += [f"{name}:{check}" for check, passed in checks.items() if not passed]
        rows.append({"name": name, "asset": item.asset, "dynamic": item.dynamic,
                     "mass_kg": round(float(component.mass), 4) if item.dynamic else None,
                     "position_m": _vector(pose.p), "quaternion_wxyz": _vector(pose.q),
                     "aabb_m": [_vector(low), _vector(high)], "settle_displacement_m": round(moved, 6),
                     "settle_rotation_deg": round(turned, 4), "linear_speed_m_s": round(linear, 6),
                     "angular_speed_rad_s": round(angular, 6), "checks": checks})
    contacts = contact_report(environment, objects)
    for contact in contacts:
        if contact["min_separation_m"] < -criteria["max_penetration_m"]:
            failures.append("interpenetration:" + "+".join(contact["bodies"]))
        if "robot" in contact["bodies"]:
            failures.append("robot_contact:" + "+".join(contact["bodies"]))
    return {"objects": rows, "contacts": contacts, "failures": failures}


def asset_files(robotwin_root: Path, environment: Any) -> list[dict[str, str]]:
    files = sorted({path for item in environment.play_objects for path in item.files})
    return [{"path": path, "sha256": _sha256(robotwin_root / path)} for path in files]


def audit_seed(*, robotwin_root: Path, scene_set: str, scene: str, task_config: str, seed: int,
               output: Path) -> dict[str, Any]:
    """Build one scene at ``seed``, check it, render it and close it."""
    from .common import _load_task_args
    from ora.environment.benchmarks.robotwin_play_scenes import scene_instance

    record: dict[str, Any] = {"schema": SEED_SCHEMA, "scene_set": scene_set, "scene": scene, "seed": seed,
                              "task_config": task_config, "started_unix_s": time.time(), "built": False,
                              "passed": False, "failures": [], "error": None}
    environment = None
    try:
        environment = scene_instance(scene_set, scene)
        args = _load_task_args(robotwin_root=robotwin_root, task=scene, task_config=task_config,
                               runtime_output=output / "runtime")
        environment.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        if environment.check_success() is not False:
            raise RuntimeError("a play scene's predicate must read false")
        record["built"] = True
        record["object_count"] = len(environment.play_objects)
        record.update(check_scene(environment))
        record["cameras"] = render_cameras(environment, output / "cameras")
        blank = [name for name, image in record["cameras"].items() if image["std"] < CRITERIA["min_image_std"]]
        record["failures"] += [f"blank_camera:{name}" for name in blank]
        record["assets"] = asset_files(robotwin_root, environment)
        record["passed"] = not record["failures"]
    except Exception as exc:  # an unbuildable or unstable seed is rejected, never retried here
        unstable = type(exc).__name__ == "UnStableError"
        record["failures"].append("unstable_at_reset" if unstable else "build_error")
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc()[-2000:]
    finally:
        if environment is not None:
            try:
                environment.close_env(clear_cache=False)
            except Exception as exc:
                record["close_error"] = f"{type(exc).__name__}: {exc}"
    record["finished_unix_s"] = time.time()
    path = output / "record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ORA's RoboTwin free-play scenes (worker).")
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-set", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    root = args.robotwin_root.resolve()
    for path in (str(root), str(root / "envs/curobo/src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(root)  # RoboTwin reads its assets and configs relative to its root
    from ora.environment.benchmarks.robotwin_play_scenes import source_binding

    output = args.output_root.resolve()
    records = [audit_seed(robotwin_root=root, scene_set=args.scene_set, scene=args.scene,
                          task_config=args.task_config, seed=seed, output=output / f"seed-{seed}")
               for seed in args.seeds]
    summary = {
        "schema": "ora.robotwin-play-scene-audit-worker.v1", "scene_set": args.scene_set, "scene": args.scene,
        "task_config": args.task_config, "criteria": CRITERIA, "scene_source": source_binding(args.scene_set),
        "robotwin_sources": [{"path": path, "sha256": _sha256(root / path)} for path in ROBOTWIN_SOURCES],
        "seeds": {str(record["seed"]): f"seed-{record['seed']}/record.json" for record in records},
        "passed_seeds": [record["seed"] for record in records if record["passed"]],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scene": args.scene, "passed_seeds": summary["passed_seeds"],
                      "failures": {str(record["seed"]): record["failures"] for record in records}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
