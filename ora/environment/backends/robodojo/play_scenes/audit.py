"""Play-scene audit worker: build, settle and judge ORA's RoboDojo play scenes (RoboDojo interpreter).

Run by ``ora/orchestrator/robodojo_play_audit.py`` with the RoboDojo
interpreter, one process per scene and pass, one process on a card at a time.
Each pass starts the live play session (``RobodojoPlaySession``, the code a
``ora play`` session runs) on its first seed and reaches every later seed
through ``restart_round(seed=...)``, the relaunch a ``/reset`` takes. For each
seed it checks what free play needs and nothing a task would:

- settle: every layout object exists, rests on the table (``on_table``) and
  moved at most ``max_settle_drift_m`` / tilted at most
  ``max_settle_tilt_deg`` from its generated spawn pose while the worker's
  reset settled the scene, or, placed in or on another object, came to rest
  within that object's footprint (grapes rolling into their bowl);
- stability window: over ``window_steps`` more physics steps no object moves
  more than ``max_window_displacement_m`` (peak ``max_window_peak_m``) or turns
  more than ``max_window_rotation_deg``, and no articulation joint drifts more
  than ``max_joint_drift``;
- contacts (CPU physics only: PhysX delivers contact reports there): no
  object penetrates another object or the table deeper than
  ``max_penetration_m`` and no object touches the robot. A CUDA scene gets a
  second, ``contacts`` pass on CPU physics (``audit_physics_device``);
- reach envelope: every object's centre lies in
  ``PLAY_REACH_ENVELOPE_XY_M`` within ``MAX_NEARER_BASE_DISTANCE_M`` of the
  nearer arm base; an object resting in or on another one is checked through
  its table-level support (a pen leaning out of its holder is where the
  holder is), and the IK check below still aims at the object itself;
- IK reachability: every movable object can be approached the way the
  policy's tools approach it. For each declared approach side
  (``ora/contracts/alignment.py`` ``APPROACH_SIDES``: top, left, right,
  front, back) the audit synthesises align_plan's own poses: the controlled
  frame (``arx_x5_link6``) against the side's direction from the object's
  near face (its settled box top for ``top``, else the footprint's nearest
  extent at the box centre's height), at the standoff that stops the open
  fingertips (``FINGERTIP_M`` along link6's approach) ``ik_fingertip_clearance_m``
  short of it, the jaw rolled about the approach axis every
  ``ik_roll_step_deg`` over the full turn (align_plan's candidates and their
  jaw-symmetric flips). align_plan's default 0.10 m from the centre would put
  those fingertips 5.8 cm past the centre: for a top approach to an object
  less than about 10 cm tall, below the table, where the shared planner
  refuses every pose (``goal_below_support``). The object passes when any
  side gives either arm a pose the session's read-only ``align_plan`` holds
  (the shared planner's IK in the observed depth world, as a plan to the
  pose would); every side's result is recorded. An object no side reaches
  passes when a tilted approach plans (``tilted_approaches``): the
  approaches of ``tilt_sides`` turned about those sides' fingertips by
  ``tilt_approach_deg``, a top approach leaning toward each of
  ``tilt_top_azimuths_deg``, a side approach tilting down, at each of
  ``tilt_rolls_deg``, smallest tilt first, each a plan of the shared planner
  from the measured joints (``shared_motion.plan_path``, which moves
  nothing), the arm with the nearer base first, until one plans;
  ``ik.tilted`` records what planned (arm, side, tilt, azimuth, roll and
  link6 pose), the plans tried, their refusals and the time. The seed's
  observation is taken first, so the IK and the plans answer in this seed's
  world;
- gripper clearance: every table-level object's settled footprint keeps
  ``min_gripper_clearance_m`` from the measured horizontal footprint of each
  arm's gripper (link6 and finger collision meshes at their live poses);
- cameras: the session's own observation returns all three RGB-D cameras,
  none blank.

The contact pass instruments the scene for measurement only: PhysX contact
reports on every scene body and a zero sleep threshold, so the window sees
live bodies (stricter than the runtime default). Poses, contacts and images
are qualification evidence and never reach a policy. No robot action is
published, no model is called and there is no verdict. The worker runtime is
the one a play plan of the scene resolves to, so the session starts the shared
motion planner it names, as a play session does, and a planner that does not
start fails the pass. The process ends with ``os._exit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from functools import partial
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.alignment import APPROACH_SIDES, MAXIMUM_POSES, REQUEST_SCHEMA as ALIGN_REQUEST_SCHEMA
from ora.environment.benchmarks import robodojo_play_layouts as geometry
from ora.environment.benchmarks.robodojo_play_scenes import (
    AUDIT_CRITERIA, AUDIT_PASSES, AUDIT_SEED_SCHEMA, AUDIT_WORKER_REPORT_NAME, AUDIT_WORKER_REPORT_SCHEMA,
    MAX_NEARER_BASE_DISTANCE_M, PLAY_REACH_ENVELOPE_XY_M, TABLE_TOP_Z_M, TABLE_XY_M, X5_BASES_XY_M,
)
from ora.environment.controllers.depth_world import PUBLIC_DEPTH_VOXEL_X5_V1

SEED_SCHEMA = AUDIT_SEED_SCHEMA
WORKER_REPORT_SCHEMA = AUDIT_WORKER_REPORT_SCHEMA
WORKER_REPORT_NAME = AUDIT_WORKER_REPORT_NAME
PASSES = AUDIT_PASSES
VISIBILITY = "qualification_only_never_policy_visible"
#: The audit's thresholds (host-safe, ``robodojo_play_scenes.AUDIT_CRITERIA``, recorded in every record).
CRITERIA = AUDIT_CRITERIA
#: align_plan's approach directions (``ora/agent/tools/alignment.py``): world unit vectors from the
#: approach position toward the target, per declared side. A host test keeps the two equal.
APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "right": (0.0, 1.0, 0.0),
}
#: The RoboDojo surface's controlled frame, ``arx_x5_link6``: link6 approaches along its local +x and
#: closes along local +y (``ora/policy/embodiments/arx-x5/descriptor-public-geometry-v5.json``).
CONTROLLED_FRAME = "arx_x5_link6"
LINK6_LOCAL_AXES = {"approach": (1.0, 0.0, 0.0), "closing": (0.0, 1.0, 0.0)}
#: How far along link6's approach axis the open fingertips reach: the point the shared planner refuses below the
#: support (``depth_world.PUBLIC_DEPTH_VOXEL_X5_V1``), 0.1576 m, past align_plan's 0.10 m default standoff.
FINGERTIP_M = PUBLIC_DEPTH_VOXEL_X5_V1.fingertip_m
CLOSE_TIMEOUT_S = 60.0


# --- pure judgement (unit-tested without a simulator) -------------------------------------------

def rotation_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    n1 = math.sqrt(sum(v * v for v in q1)) or 1.0
    n2 = math.sqrt(sum(v * v for v in q2)) or 1.0
    dot = abs(sum(a * b for a, b in zip(q1, q2)) / (n1 * n2))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def up_tilt_deg(q_wxyz: Sequence[float]) -> float:
    w, x, y, z = geometry.normalized(q_wxyz)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def roll_angles_deg(criteria: Mapping[str, Any] = CRITERIA) -> list[float]:
    """Jaw rolls about the approach axis, every ``ik_roll_step_deg`` over the full turn."""
    step = float(criteria["ik_roll_step_deg"])
    count = int(round(360.0 / step))
    if count < 1 or count > MAXIMUM_POSES or abs(count * step - 360.0) > 1e-9:
        raise ValueError(f"ik_roll_step_deg must divide the turn into 1..{MAXIMUM_POSES} rolls")
    return [step * index for index in range(count)]


def approach_orientation(direction: Sequence[float], roll_rad: float,
                         local_axes: Mapping[str, Sequence[float]] = LINK6_LOCAL_AXES) -> list[float]:
    """align_plan's xyzw orientation: the declared approach axis along ``direction``, the jaw rolled by ``roll_rad``.

    The same construction as ``_quaternion_from_approach`` in ``ora/agent/tools/alignment.py`` (a host
    test compares them); restated here because the worker does not import policy-side code.
    """
    import numpy as np

    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(reference, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = up * cos_roll - right * sin_roll
    approach_local = np.asarray(local_axes["approach"], dtype=np.float64)
    closing_local = np.asarray(local_axes["closing"], dtype=np.float64)
    approach_local /= np.linalg.norm(approach_local)
    closing_local /= np.linalg.norm(closing_local)
    world = np.stack([forward, rolled_right, rolled_up], axis=1)
    local = np.stack([approach_local, closing_local, np.cross(approach_local, closing_local)], axis=1)
    matrix = world @ local.T
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [float((matrix[2, 1] - matrix[1, 2]) / scale), float((matrix[0, 2] - matrix[2, 0]) / scale),
                float((matrix[1, 0] - matrix[0, 1]) / scale), float(0.25 * scale)]
    index = int(np.argmax(np.diag(matrix)))
    i, j, k = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][index]
    scale = math.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
    parts = [0.0, 0.0, 0.0]
    parts[i] = 0.25 * scale
    parts[j] = float((matrix[j, i] + matrix[i, j]) / scale)
    parts[k] = float((matrix[k, i] + matrix[i, k]) / scale)
    return [*parts, float((matrix[k, j] - matrix[j, k]) / scale)]


def approach_standoff_m(criteria: Mapping[str, Any] = CRITERIA) -> float:
    """link6's distance from the near face: the fingertips' reach plus ``ik_fingertip_clearance_m``."""
    return FINGERTIP_M + float(criteria["ik_fingertip_clearance_m"])


def near_face(record: Mapping[str, Any], side: str) -> list[float]:
    """Where a side's approach meets the object's settled box: its top for ``top``, else the footprint's nearest
    extent against the side's direction, at the box centre's height."""
    face = [float(value) for value in record["centre_m"]]
    if side == "top":
        face[2] = float(record["top_z_m"])
        return face
    axis = 0 if APPROACH_DIRECTIONS[side][0] else 1
    extents = [float(point[axis]) for point in record["footprint_xy"]]
    face[axis] = min(extents) if APPROACH_DIRECTIONS[side][axis] > 0 else max(extents)
    return face


def approach_poses(face_xyz_m: Sequence[float], side: str, criteria: Mapping[str, Any] = CRITERIA) -> list[list[float]]:
    """align_plan's poses for one side: link6 ``approach_standoff_m`` from the near face against the side's direction."""
    direction = APPROACH_DIRECTIONS[side]
    standoff = approach_standoff_m(criteria)
    position = [float(face_xyz_m[axis]) - direction[axis] * standoff for axis in range(3)]
    return [[*position, *approach_orientation(direction, math.radians(roll))] for roll in roll_angles_deg(criteria)]


def tilted_approaches(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> list[dict[str, Any]]:
    """The tilted approaches planned for an object no declared side reaches, in the order they are tried.

    Smallest tilt first, then ``tilt_sides`` in order. Each keeps its side's fingertips where the side's own pose
    puts them, ``ik_fingertip_clearance_m`` short of the near face, and turns the approach about them: a top
    approach leans ``tilt_deg`` toward the world azimuth ``azimuth_deg`` (degrees from +x), a side approach tilts
    down by ``tilt_deg``. ``pose_xyzw`` is link6's."""
    clearance = float(criteria["ik_fingertip_clearance_m"])
    candidates = []
    for tilt in criteria["tilt_approach_deg"]:
        for side in criteria["tilt_sides"]:
            direction = APPROACH_DIRECTIONS[side]
            tips = [value - direction[axis] * clearance for axis, value in enumerate(near_face(record, side))]
            angle = math.radians(tilt)
            if side == "top":
                azimuths = [float(value) for value in criteria["tilt_top_azimuths_deg"]]
            else:
                azimuths = [round(math.degrees(math.atan2(direction[1], direction[0])) % 360.0, 6)]
            for azimuth in azimuths:
                if side == "top":
                    turn = math.radians(azimuth)
                    approach = [math.sin(angle) * math.cos(turn), math.sin(angle) * math.sin(turn), -math.cos(angle)]
                else:
                    approach = [direction[0] * math.cos(angle), direction[1] * math.cos(angle), -math.sin(angle)]
                for roll in criteria["tilt_rolls_deg"]:
                    position = [tips[axis] - approach[axis] * FINGERTIP_M for axis in range(3)]
                    candidates.append({"side": side, "tilt_deg": float(tilt), "azimuth_deg": azimuth,
                                       "roll_deg": float(roll), "approach": [round(value, 9) for value in approach],
                                       "pose_xyzw": [*position, *approach_orientation(approach, math.radians(roll))]})
    return candidates


def box_centre(vertices: Sequence[Sequence[float]], position: Sequence[float], q_wxyz: Sequence[float]) -> list[float]:
    """The world centre of an asset's aligned box at a pose (the object's measurable centroid)."""
    rotation = geometry.rotation_matrix(q_wxyz)
    local = [sum(vertex[axis] for vertex in vertices) / len(vertices) for axis in range(3)]
    return [float(position[row]) + sum(rotation[row][k] * local[k] for k in range(3)) for row in range(3)]


def judge_object(item: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> list[str]:
    """The failed checks of one object's measurements (see the module docstring)."""
    failures = []
    settled, final = item["settled"]["position_m"], item["final"]["position_m"]
    (table_x0, table_x1), (table_y0, table_y1) = TABLE_XY_M["x"], TABLE_XY_M["y"]
    if (final[2] < TABLE_TOP_Z_M - criteria["left_table_below_top_m"]
            or not table_x0 <= final[0] <= table_x1 or not table_y0 <= final[1] <= table_y1):
        failures.append("on_table")
    settle = item["settle"]
    if not settle.get("on_support") and (settle["drift_m"] > criteria["max_settle_drift_m"]
                                         or settle["tilt_change_deg"] > criteria["max_settle_tilt_deg"]):
        failures.append("settle")
    window = item["window"]
    if (window["displacement_m"] > criteria["max_window_displacement_m"]
            or window["peak_displacement_m"] > criteria["max_window_peak_m"]
            or window["rotation_deg"] > criteria["max_window_rotation_deg"]
            or any(abs(value) > criteria["max_joint_drift"] for value in (window.get("joint_drift") or {}).values())):
        failures.append("stability_window")
    contacts = item.get("contacts")
    if contacts is not None:
        others = [sep for body, sep in contacts.items() if not body.startswith("<") or body == "<table>"]
        if any(sep < -criteria["max_penetration_m"] for sep in others):
            failures.append("penetration")
        if "<robot>" in contacts:
            failures.append("robot_contact")
    # An object resting in or on another one is checked through its table-level support.
    x, y = (item.get("envelope_position_m") or settled)[:2]
    envelope = PLAY_REACH_ENVELOPE_XY_M
    nearer = min(math.dist((x, y), base) for base in X5_BASES_XY_M.values())
    if not (envelope["x"][0] <= x <= envelope["x"][1] and envelope["y"][0] <= y <= envelope["y"][1]
            and nearer <= MAX_NEARER_BASE_DISTANCE_M):
        failures.append("reach_envelope")
    reach = item.get("ik")
    if reach is not None and not reach.get("reachable_arms") and not (reach.get("tilted") or {}).get("planned"):
        failures.append("ik_reachability")
    clearance = item.get("gripper_clearance_m")
    if clearance is not None and clearance < criteria["min_gripper_clearance_m"]:
        failures.append("gripper_clearance")
    return failures


def table_support(label: str, placements: Mapping[str, Mapping[str, Any]]) -> str:
    """The table-level object ``label`` rests on, through every ``relative_plane`` above it (itself when on the table)."""
    seen = {label}
    while True:
        parent = (placements.get(label) or {}).get("relative_plane")
        if parent in (None, "Table") or parent not in placements or parent in seen:
            return label
        seen.add(parent)
        label = parent


def judge_seed(objects: Mapping[str, Mapping[str, Any]], *, missing: Sequence[str], cameras: Mapping[str, Any] | None,
               criteria: Mapping[str, Any] = CRITERIA) -> list[str]:
    failures = [f"{label}:missing" for label in missing]
    for label, item in sorted(objects.items()):
        failures += [f"{label}:{check}" for check in judge_object(item, criteria)]
    for name, image in sorted((cameras or {}).items()):
        if image.get("std", 0.0) < criteria["min_image_std"]:
            failures.append(f"blank_camera:{name}")
    return failures


# --- measurement (Isaac) -------------------------------------------------------------------------

def _numbers(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [float(item) for item in (value.reshape(-1) if hasattr(value, "reshape") else value)]


def object_inventory(session: Any) -> dict[str, dict[str, Any]]:
    """label -> the live layout instance: name, kind, record and prim path (articulations included)."""
    environment = session.environment
    manager = environment.scene_manager
    layout = manager.layout_manager
    records = {}
    for section in ("Rigid", "Articulation"):
        for record in layout.object_records_by_type[section].layout_records_by_env[0]:
            records[record["inst_name"]] = (record["label"], section, record)
    inventory = {}
    for collection, kind in ((manager._rigid_and_dynamic_objects[0], "rigid"),
                             (manager._articulation_objects[0], "articulation")):
        for inst_name, obj in collection.items():
            if inst_name not in records:
                continue
            label, section, record = records[inst_name]
            inventory[label] = {"inst_name": inst_name, "kind": kind, "obj": obj, "record": record,
                                "prim_path": str(obj.prim_path)}
    return inventory


def poses(session: Any, inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    layout = session.environment.scene_manager.layout_manager
    result = {}
    for label, item in inventory.items():
        position, quaternion = layout.get_instance_pose(env_idx=0, inst_name=item["inst_name"])
        result[label] = (_numbers(position)[:3], _numbers(quaternion)[:4])
    return result


def joint_positions(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, list[float]]:
    result = {}
    for label, item in inventory.items():
        if item["kind"] == "articulation":
            result[label] = _numbers(item["obj"].get_joint_positions())
    return result


def instrument_contacts(environment: Any) -> dict[str, Any]:
    """Contact reports and a zero sleep threshold on every scene body, now and after each respawn."""
    tagged: set[str] = set()

    def tag() -> None:
        from pxr import PhysxSchema, Usd, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        manager = environment.scene_manager
        for collection in (manager._rigid_and_dynamic_objects[0], manager._articulation_objects[0]):
            for obj in collection.values():
                root = stage.GetPrimAtPath(str(obj.prim_path))
                if not root or not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    path = str(prim.GetPath())
                    if path in tagged:
                        continue
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
                        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr().Set(0.0)
                        tagged.add(path)
                    elif prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSleepThresholdAttr().Set(0.0)
                        tagged.add(path)

    spawn = environment.scene_manager.spawn_scene_objects

    def spawn_and_tag(env_id: int, exclude_types: Any = None) -> Any:
        result = spawn(env_id=env_id, exclude_types=exclude_types)
        tag()
        return result

    environment.scene_manager.spawn_scene_objects = spawn_and_tag
    post_setup = environment._post_setup_scene

    def post_setup_and_tag(sim: Any) -> None:
        tagged.clear()  # a relaunch builds a new stage: every body is new, whatever its path
        post_setup(sim)
        tag()

    environment._post_setup_scene = post_setup_and_tag
    return {"tagged": tagged}


def _owner(path: str, prefixes: Sequence[tuple[str, str]]) -> str:
    for prefix, label in prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return label
    if "/Table" in path:
        return "<table>"
    if "/robot" in path.lower():
        return "<robot>"
    return "<other>"


def read_contacts(store: dict[tuple[str, str], float], prefixes: Sequence[tuple[str, str]]) -> int:
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    headers, data = get_physx_simulation_interface().get_contact_report()
    samples = 0
    for header in headers:
        a = _owner(str(PhysicsSchemaTools.intToSdfPath(header.actor0)), prefixes)
        b = _owner(str(PhysicsSchemaTools.intToSdfPath(header.actor1)), prefixes)
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
            separation = float(data[index].separation)
            if key not in store or separation < store[key]:
                store[key] = separation
            samples += 1
    return samples


def gripper_hulls(session: Any) -> dict[str, list[list[float]]]:
    """Each arm's measured horizontal gripper footprint: link6 and finger collision meshes at their live poses."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    manager = session.environment.robot_manager
    origin = np.asarray(_numbers(session.environment.sim.scene.env_origins[0])[:3])
    cache = UsdGeom.XformCache()
    result = {}
    for arm, robot in session.robots.items():
        articulation = manager.robot_key[manager.robot_list.index(robot)]
        root = str(articulation.root_physx_view.prim_paths[0])
        names = list(articulation.body_names)
        positions = np.asarray(_numbers(articulation.data.body_pos_w[0])).reshape(-1, 3)
        quaternions = np.asarray(_numbers(articulation.data.body_quat_w[0])).reshape(-1, 4)
        bodies = [robot.ee_link_name, *(name.replace("joint", "link") for name in robot.gripper_joints_name)]
        points = []
        for body in bodies:
            prim = stage.GetPrimAtPath(f"{root}/{body}")
            if not prim or not prim.IsValid() or body not in names:
                raise ValueError(f"ARX gripper body {body} is absent")
            rotation = np.asarray(geometry.rotation_matrix(quaternions[names.index(body)]))
            translation = positions[names.index(body)] - origin
            for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                if not mesh.IsA(UsdGeom.Mesh) or "/collisions/" not in str(mesh.GetPath()):
                    continue
                ancestor, collision = mesh, False
                while ancestor and ancestor != prim:
                    collision |= ancestor.HasAPI(UsdPhysics.CollisionAPI)
                    ancestor = ancestor.GetParent()
                if not collision:
                    continue
                relative = cache.ComputeRelativeTransform(mesh, prim)[0]
                for point in UsdGeom.Mesh(mesh).GetPointsAttr().Get() or ():
                    local = relative.Transform(point)
                    world = rotation @ np.asarray([local[0], local[1], local[2]], dtype=float) + translation
                    points.append((float(world[0]), float(world[1])))
        if not points:
            raise ValueError(f"no gripper collision mesh on the {arm} arm")
        result[arm] = [[round(x, 6), round(y, 6)] for x, y in geometry.hull(points)]
    return result


def side_reach(answer: Mapping[str, Any], rolls_deg: Sequence[float]) -> dict[str, Any]:
    """One side's align_plan answer: reachable rolls per arm and the rejection reasons, counted."""
    reachable: dict[str, list[float]] = {}
    rejections: dict[str, int] = {}
    for candidate in answer["candidates"]:
        if candidate["reachable"]:
            reachable.setdefault(str(candidate["arm"]), []).append(float(rolls_deg[candidate["source_index"]]))
        else:
            reason = str(candidate.get("rejection_reason") or "unreachable")
            rejections[reason] = rejections.get(reason, 0) + 1
    return {"reachable_arms": sorted(reachable), "reachable_rolls_deg": {arm: sorted(rolls) for arm, rolls in
                                                                          sorted(reachable.items())},
            "rejections": dict(sorted(rejections.items()))}


def tilted_reach(plan_path: Callable[[str, Sequence[float]], Mapping[str, Any]], record: Mapping[str, Any],
                 criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """Plan ``tilted_approaches`` in turn, the nearer arm first, until one plans; what planned, or ``None``."""
    x, y = record["centre_m"][:2]
    arms = sorted(X5_BASES_XY_M, key=lambda arm: math.dist((x, y), X5_BASES_XY_M[arm]))
    started, tried, reasons = time.monotonic(), 0, {}
    planned = None
    for candidate in tilted_approaches(record, criteria):
        for arm in arms:
            tried += 1
            answer = plan_path(arm, candidate["pose_xyzw"])
            if answer.get("status") == "planned":
                planned = {"arm": arm, **{key: candidate[key] for key in ("side", "tilt_deg", "azimuth_deg", "roll_deg")},
                           "approach": candidate["approach"],
                           "link6_pose_xyzw": [round(value, 6) for value in candidate["pose_xyzw"]]}
                break
            reason = str(answer.get("reason") or answer.get("status"))
            reasons[reason] = reasons.get(reason, 0) + 1
        if planned is not None:
            break
    return {"rule": "first_tilted_plan", "tilts_deg": [float(value) for value in criteria["tilt_approach_deg"]],
            "plans_tried": tried, "refusals": dict(sorted(reasons.items())), "planned": planned,
            "seconds": round(time.monotonic() - started, 3)}


def approach_reach(plan: Callable[[Mapping[str, Any]], Mapping[str, Any]], label: str, record: Mapping[str, Any],
                   criteria: Mapping[str, Any] = CRITERIA,
                   plan_path: Callable[[str, Sequence[float]], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Ask align_plan, side by side, whether either arm reaches the object, and when none does, plan the tilted
    approaches through ``plan_path`` (see the module docstring)."""
    rolls = roll_angles_deg(criteria)
    sides = {}
    for side in APPROACH_SIDES:
        face = near_face(record, side)
        poses = approach_poses(face, side, criteria)
        answer = plan({"schema": ALIGN_REQUEST_SCHEMA, "target_ref": f"play-audit-{label}", "approach_side": side,
                       "arm": "both", "target_poses_xyzw": poses})
        sides[side] = {"face_m": [round(value, 6) for value in face],
                       "link6_position_m": [round(value, 6) for value in poses[0][:3]], **side_reach(answer, rolls)}
    reachable = sorted({arm for item in sides.values() for arm in item["reachable_arms"]})
    return {"rule": "any_declared_side_else_tilted_plan", "controlled_frame": CONTROLLED_FRAME,
            "standoff_m": round(approach_standoff_m(criteria), 6), "fingertip_m": FINGERTIP_M,
            "fingertip_clearance_m": float(criteria["ik_fingertip_clearance_m"]), "rolls_deg": rolls,
            "centre_m": [round(float(value), 6) for value in record["centre_m"]], "sides": sides,
            "reachable_sides": [side for side in APPROACH_SIDES if sides[side]["reachable_arms"]],
            "reachable_arms": reachable,
            "tilted": tilted_reach(plan_path, record, criteria) if plan_path is not None and not reachable else None}


def planned_from_measured_joints(session: Any, arm: str, pose_xyzw: Sequence[float]) -> Mapping[str, Any]:
    """The shared planner's plan of ``arm`` to link6 ``pose_xyzw`` from its measured joints, in the observed world;
    nothing moves."""
    from .. import shared_motion

    return shared_motion.plan_path(session, {"arm": arm, "goal_pose_xyz_xyzw": list(pose_xyzw), "goal_joints": None})


def camera_images(session: Any, destination: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    observation, images = session.observe(destination)
    result = {}
    for path in images:
        array = np.asarray(Image.open(path).convert("RGB"))
        camera = Path(path).parent.name
        result[camera] = {"path": Path(path).relative_to(destination).as_posix(),
                          "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          "mean": round(float(array.mean()), 3), "std": round(float(array.std()), 3)}
    return {"observation_id": observation["observation_id"], "cameras": result}


def audit_seed(session: Any, *, seed: int, pass_name: str, output: Path, contacts: bool,
               criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """Measure and judge the live scene at ``seed`` (already loaded by start or restart_round)."""
    environment = session.environment
    generated = session.source["play_layouts"][str(seed)]
    placements = {item["label"]: item for item in generated["placements"]}
    inventory = object_inventory(session)
    missing = sorted(set(placements) - set(inventory))
    prefixes = sorted(((item["prim_path"], label) for label, item in inventory.items()), key=lambda pair: -len(pair[0]))
    settled = poses(session, inventory)
    joints_settled = joint_positions(inventory)
    store: dict[tuple[str, str], float] = {}
    samples = 0
    if contacts:
        # IsaacLab turns omni.physx contact processing off whenever it builds a simulation (after a
        # relaunch too); reports reach the window only with it back on.
        import carb.settings

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        read_contacts({}, prefixes)  # drop reports from before the window
    peak = {label: 0.0 for label in inventory}
    for step in range(int(criteria["window_steps"])):
        environment.sim_step(render=False)
        if contacts and step % 10 == 9:
            samples += read_contacts(store, prefixes)
        if step % 25 == 24:
            for label, (position, _) in poses(session, inventory).items():
                peak[label] = max(peak[label], math.dist(position, settled[label][0]))
    final = poses(session, inventory)
    joints_final = joint_positions(inventory)
    objects: dict[str, Any] = {}
    hulls = gripper_hulls(session) if pass_name == "session" else None
    assets = Path(session.config["asset_root"])
    for label, item in inventory.items():
        placement = placements.get(label, {})
        spawn_position, spawn_q = placement.get("spawn_pos"), placement.get("spawn_ori_wxyz")
        (p1, q1), (p2, q2) = settled[label], final[label]
        record = {
            "section": placement.get("section"), "category": placement.get("category"),
            "category_idx": placement.get("category_idx"), "relative_plane": placement.get("relative_plane"),
            "spawn": {"position_m": spawn_position, "quaternion_wxyz": spawn_q},
            "settled": {"position_m": [round(v, 6) for v in p1], "quaternion_wxyz": [round(v, 6) for v in q1]},
            "final": {"position_m": [round(v, 6) for v in p2], "quaternion_wxyz": [round(v, 6) for v in q2]},
            "settle": {"drift_m": round(math.dist(p1, spawn_position), 6) if spawn_position else None,
                       "tilt_change_deg": (round(abs(up_tilt_deg(q1) - up_tilt_deg(spawn_q)), 4)
                                           if spawn_q else None)},
            "window": {"displacement_m": round(math.dist(p2, p1), 6), "peak_displacement_m": round(peak[label], 6),
                       "rotation_deg": round(rotation_deg(q1, q2), 4)},
        }
        if record["settle"]["drift_m"] is None:
            record["settle"] = {"drift_m": math.inf, "tilt_change_deg": math.inf}
        support = table_support(label, placements)
        if support != label and support in settled:
            record["envelope_support"] = support
            record["envelope_position_m"] = [round(v, 6) for v in settled[support][0]]
        if label in joints_settled:
            record["window"]["joint_drift"] = {str(index): round(b - a, 6) for index, (a, b) in
                                              enumerate(zip(joints_settled[label], joints_final[label]))}
        if contacts:
            record["contacts"] = {(b if a == label else a): round(sep, 6) for (a, b), sep in store.items()
                                  if label in (a, b)}
        if placement:
            metadata = json.loads((assets / placement["metadata"]).read_text())
            vertices = metadata["geometry"]["aligned_bbox"]["vertices"]
            outline = geometry.footprint(vertices, p1, q1)
            top = p1[2] + geometry.placed_box(vertices, q1)["zmax"]
            record["footprint_xy"] = [[round(x, 6), round(y, 6)] for x, y in outline]
            record["top_z_m"] = round(top, 6)
            record["centre_m"] = [round(value, 6) for value in box_centre(vertices, p1, q1)]
        objects[label] = record
    for label, record in objects.items():
        # A child that settled within its support's footprint is settled, however far it slid there
        # (grapes rolling into the bottom of their bowl).
        parent = objects.get(str((placements.get(label) or {}).get("relative_plane")))
        if parent is not None and parent.get("footprint_xy") and record.get("footprint_xy"):
            x, y = record["settled"]["position_m"][:2]
            point = [(x - 1e-4, y - 1e-4), (x + 1e-4, y - 1e-4), (x + 1e-4, y + 1e-4), (x - 1e-4, y + 1e-4)]
            record["settle"]["on_support"] = geometry.separation(point, parent["footprint_xy"]) < 0
    # The observation before the IK: an IK in the observed world (the shared planner's) answers only in the world
    # the latest observation published at the current physics step (shared_motion.world_input).
    cameras = camera_images(session, output / "observations") if pass_name == "session" else None
    for label, record in objects.items():
        placement = placements.get(label, {})
        if pass_name == "session" and record.get("footprint_xy"):
            outline = [tuple(point) for point in record["footprint_xy"]]
            if placement.get("relative_plane") == "Table" and hulls:
                record["gripper_clearance_m"] = round(min(geometry.separation(outline, hull) for hull in hulls.values()), 6)
            if placement.get("section") != "Articulation":
                record["ik"] = approach_reach(session.align_plan, label, record, criteria, plan_path=partial(
                    planned_from_measured_joints, session))
    failures = judge_seed(objects, missing=missing, cameras=(cameras or {}).get("cameras"), criteria=criteria)
    if contacts and not samples:
        failures.append("contact_report_empty")  # every object rests on something: no report is no evidence
    layout = session._loaded_layout()
    return {
        "schema": SEED_SCHEMA, "seed": seed, "pass": pass_name,
        "physics_device": session.source.get("physics_device"),
        "layout_sha256": layout["layout_sha256"], "planned_layout_sha256": layout["planned_layout_sha256"],
        "object_count": len(inventory), "missing": missing, "objects": objects,
        "gripper_hulls_xy": hulls, "contact_points_sampled": samples if contacts else None,
        "cameras": cameras, "failures": failures, "passed": not failures and layout["layout_sha256"] == layout[
            "planned_layout_sha256"],
        "reset_physics_steps": int(session.reset_physics_steps), "visibility": VISIBILITY,
        "robot_actions_published": 0, "verdict": None, "finished_unix_s": round(time.time(), 3),
        "home_end_effectors": {arm: value["pose_xyz_xyzw"] for arm, value in
                               session._proprioception()["end_effectors"].items()},
    }


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item),
                                 allow_nan=True))


def run_worker(config: Mapping[str, Any], output: Path, *, seeds: Sequence[int], pass_name: str,
               session_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Start the play session on the first seed, relaunch on each later one, judge each; never raise."""
    from .bootstrap import build_play_environment
    from .session import RobodojoPlaySession

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contacts = config.get("physics_device") == "cpu"
    report: dict[str, Any] = {"schema": WORKER_REPORT_SCHEMA, "task": config.get("task"),
                              "scene_set": config.get("play_scene_set"), "pass": pass_name,
                              "physics_device": config.get("physics_device"), "contacts": contacts,
                              "criteria": CRITERIA, "seeds": {}, "error": None, "visibility": VISIBILITY}

    def factory(worker_config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        environment, app, manifest = build_play_environment(worker_config)
        if contacts:
            instrument_contacts(environment)
        return environment, app, manifest

    def save() -> None:
        (output / WORKER_REPORT_NAME).write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")

    session = execution = None
    try:
        session = (session_factory or RobodojoPlaySession)(
            config={**config, "play_seed": int(seeds[0])}, output_root=Path(config["output_root"]),
            environment_factory=factory)
        session.exit_process_on_close = False  # this process writes its report, then ends with os._exit
        from ora.contracts.execution_state import SCHEMA
        if callable(getattr(session, "execution_request", None)):
            execution = session.execution_request({"schema": SCHEMA, "values": {}})
            execution.__enter__()
        started = time.monotonic()
        session.start()
        report["start_s"] = round(time.monotonic() - started, 1)
        report["source_manifest_sha256"] = session.source_hash
        report["graphics_compatibility"] = session.source.get("graphics_compatibility")
        report["device_binding"] = {key: value for key, value in (session.source.get("device_binding") or {}).items()
                                    if key != "startup_log_prefix"}
        report["scene_source"] = session.source.get("scene_source")
        for index, seed in enumerate(seeds):
            started = time.monotonic()
            if index:
                session.restart_round(index, seed=int(seed))
            try:
                record = audit_seed(session, seed=int(seed), pass_name=pass_name, output=output / f"seed-{seed}",
                                    contacts=contacts)
            except Exception as exc:  # a measurement failure rejects the seed, the pass goes on
                record = {"schema": SEED_SCHEMA, "seed": int(seed), "pass": pass_name, "passed": False,
                          "failures": ["audit_error"], "error": f"{type(exc).__name__}: {exc}",
                          "traceback_tail": traceback.format_exc()[-2000:]}
            record["seconds"] = round(time.monotonic() - started, 1)
            report["seeds"][str(seed)] = record
            save()
        report["completed"] = True
    except BaseException as exc:  # noqa: BLE001 - recorded for the host; unaudited seeds stay rejected
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:],
                      completed=False)
    finally:
        if execution is not None:
            execution.__exit__(None, None, None)
        save()
        if session is not None:
            thread = threading.Thread(target=session.close, daemon=True)
            thread.start()
            thread.join(CLOSE_TIMEOUT_S)
            report["closed_within_timeout"] = not thread.is_alive()
        save()
    return report


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one ORA RoboDojo play scene (worker).")
    parser.add_argument("--config", type=Path, required=True, help="worker config (robodojo_play_worker_config)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    report = run_worker(config, arguments.output, seeds=arguments.seeds, pass_name=arguments.pass_name)
    return 0 if report.get("error") is None else 1


__all__ = ["APPROACH_DIRECTIONS", "CRITERIA", "FINGERTIP_M", "PASSES", "SEED_SCHEMA", "WORKER_REPORT_NAME",
           "WORKER_REPORT_SCHEMA", "approach_orientation", "approach_poses", "approach_reach", "approach_standoff_m",
           "audit_seed", "box_centre", "judge_object", "judge_seed", "main", "near_face", "roll_angles_deg",
           "rotation_deg", "run_worker", "side_reach", "table_support", "up_tilt_deg"]


if __name__ == "__main__":  # pragma: no cover - runs in the RoboDojo interpreter
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Never tear Kit down in-process: a crashing teardown leaves the card's Vulkan state unusable.
    os._exit(code)
