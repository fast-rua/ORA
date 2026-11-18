"""The seed-0 task-contract witness, run inside the RoboLab worker (private; never policy-visible).

``RobolabSession.start`` hands an ``ora.robolab-witness-job.v1`` here when the worker config names
``contract_witness_job``, the way a replay hands its program (host side:
``orchestrator/backends/robolab/contract_witness.py``). The job carries one task's witness plan
from its review (``ora.robolab-witness-plan.v1``, written by ``ora dev robolab-tasks generate``).

Every case starts from a fresh reset of the seed and holds the arm at its reset pose. Its ops edit
object poses directly (place into, onto or beside an object, stack, lift, turn upright or upside
down, move beside a reference in the robot's frame, set down at a spot and turn the plan states,
park clutter off the table) and, for the pick-up family, close the gripper on an object held at
the TCP (the plan may state which axes, how deep and how far along the jaw line). The verdict is
RoboLab's own success termination, read every native action for the case's ticks; RoboLab latches
it at the first true step, and a case matches when any tick's verdict equals the expectation. The reset
case also projects every named object into the two public cameras, as the evidence a reviewer needs
for the contract's observability check. Object identities, poses and verdicts stay in this record.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.environment.controllers import robolab_droid as D

from .common import RobolabError

JOB_SCHEMA = "ora.robolab-witness-job.v1"
WITNESS_SCHEMA = "ora.robolab-task-contract-witness.v2"
#: Ticks the arm holds after a reset before any edit: RoboLab resets a termination within an episode's
#: first two steps as a physics artifact, so no case may reach success before its third step.
LEAD_TICKS = 3
#: Ticks of settling after each placement op before the next one.
SETTLE_TICKS = 12
CLEARANCE_M = 0.05
DROP_M = 0.01


def _rotation(quat_wxyz: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat(matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return [float(w), float(x), float(y), float(z)]


class Scene:
    """Direct pose edits in the env frame (world minus the env origin), with the robot held."""

    def __init__(self, session: Any, keep: Any = ()) -> None:
        from robolab.core.world.world_state import get_world

        self.session = session
        #: Objects the plan names (targets, containers, references): never parked, never piled onto.
        self.keep = set(keep)
        #: Boxes of objects placed since the last physics step (the sim's bbox may not show them yet).
        self.placed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        #: Unnamed clutter moved off the table to make room (recorded with the case).
        self.parked: list[str] = []
        #: Ticks at the start of a case whose verdict does not count (objects still landing).
        self.ignore_until = 0
        self.env = session.bridge.env
        self.world = get_world(self.env)
        self.origin = np.asarray(self.env.scene.env_origins[0].detach().cpu().numpy(), dtype=float)
        self.finger = D.GRIPPER_OPEN_ACTION
        self.results: list[bool] = []

    # ------------------------------------------------------------------ reading
    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        data = self.env.scene[name].data
        position = np.asarray(data.root_pos_w[0].detach().cpu().numpy(), dtype=float) - self.origin
        return position, np.asarray(data.root_quat_w[0].detach().cpu().numpy(), dtype=float)

    def corners(self, name: str) -> np.ndarray:
        corners, _centroid = self.world.get_bbox(name, env_id=0)
        return np.array([[float(c[0]), float(c[1]), float(c[2])] for c in corners])

    def box(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name in self.placed:
            low, high = self.placed[name]
            return low, high, (low + high) / 2
        points = self.corners(name)
        return points.min(axis=0), points.max(axis=0), points.mean(axis=0)

    def local_corners(self, name: str) -> np.ndarray:
        position, quat = self.pose(name)
        return (self.corners(name) - position) @ _rotation(quat)

    def table_top(self) -> float:
        return float(self.box("table")[1][2])

    def robot_frame(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.session.bridge.robot.data
        position = np.asarray(data.root_pos_w[0].detach().cpu().numpy(), dtype=float) - self.origin
        return position, _rotation(data.root_quat_w[0].detach().cpu().numpy())

    # ------------------------------------------------------------------ writing
    def put(self, name: str, *, center: np.ndarray | None = None, bottom: float | None = None,
            rotation: np.ndarray | None = None) -> None:
        """Place ``name`` with its box centroid at ``center`` (xy), its lowest point at ``bottom``."""
        import torch

        _position, quat = self.pose(name)
        local = self.local_corners(name)
        new_rotation = _rotation(quat) if rotation is None else rotation
        moved = local @ new_rotation.T
        target = np.array(center if center is not None else self.box(name)[2], dtype=float)
        offset = moved.mean(axis=0)
        root = target - offset
        if bottom is not None:
            root[2] = bottom - moved[:, 2].min()
        asset = self.env.scene[name]
        pose = asset.data.root_pose_w.clone()
        pose[0, :3] = torch.tensor(root + self.origin, dtype=pose.dtype, device=pose.device)
        pose[0, 3:7] = torch.tensor(_quat(new_rotation), dtype=pose.dtype, device=pose.device)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=pose.dtype, device=pose.device))
        placed = moved + root
        self.placed[name] = (placed.min(axis=0), placed.max(axis=0))

    def hold(self, ticks: int) -> None:
        bridge = self.session.bridge
        target = bridge.tcp_pose()
        for _ in range(ticks):
            result = bridge.step(target, self.finger)
            self.placed.clear()
            self.results.append(bool(result["official_result"]))
            if result["official_result"] and len(self.results) > self.ignore_until:
                break

    # -------------------------------------------------------------------- ops
    def _others(self, exclude: set[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Boxes of every object resting at or above the table, but those in ``exclude``."""
        floor = self.table_top() - 0.05  # objects parked out of the scene do not occupy the table
        out = []
        for other in self.env.scene.rigid_objects:
            if other in exclude or other == "table":
                continue
            low, high, _centroid = self.box(other)
            if high[2] > floor:
                out.append((other, low, high))
        return out

    @staticmethod
    def _blockers(x: float, y: float, half: np.ndarray, others: list, gap: float = 0.005) -> list:
        return [item for item in others if x + half[0] + gap > item[1][0] and x - half[0] - gap < item[2][0]
                and y + half[1] + gap > item[1][1] and y - half[1] - gap < item[2][1]]

    @staticmethod
    def _grid(low: float, high: float, half: float) -> np.ndarray:
        if high - low <= 2 * half:
            return np.array([(low + high) / 2])
        return np.arange(low + half, high - half + 1e-9, 0.01)

    def _rotations(self, name: str, vertical: bool) -> list[np.ndarray]:
        """The object's own rotation, turned a quarter about z, and (``vertical``) its longest axis up."""
        _position, quat = self.pose(name)
        own = _rotation(quat)
        quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        options = [own, quarter @ own]
        if vertical:
            local = self.local_corners(name)
            order = np.argsort(local.max(axis=0) - local.min(axis=0))[::-1]  # longest first
            up = np.zeros((3, 3))
            up[:, order[0]] = [0.0, 0.0, 1.0]
            up[:, order[1]] = [1.0, 0.0, 0.0]
            up[:, order[2]] = [0.0, 1.0, 0.0]
            if np.linalg.det(up) < 0:
                up[:, order[2]] *= -1
            options.append(up)
        return options

    def _half(self, name: str, rotation: np.ndarray) -> np.ndarray:
        moved = self.local_corners(name) @ rotation.T
        return (moved.max(axis=0) - moved.min(axis=0))[:2] / 2

    def place_over(self, name: str, reference: str, index: int = 0, count: int = 1, mode: str = "into") -> None:
        """Set ``name`` down inside ``reference`` (``into``: low inside the walls) or on its top (``onto``).

        The object keeps its rotation, or turns a quarter about z, or (``into`` only) stands on its long axis,
        whichever first fits the opening; it takes the free spot nearest the middle, clear of everything
        already there, and piles on top only when no spot is free.
        """
        low, high, centroid = self.box(reference)
        size = high - low
        margin = np.maximum(size[:2] * (0.12 if mode == "into" else 0.15), 0.01)
        region_low, region_high = low[:2] + margin, high[:2] - margin
        opening = region_high - region_low
        options = self._rotations(name, vertical=mode == "into")
        rotation = next((r for r in options if np.all(2 * self._half(name, r) <= opening)),
                        options[-1] if mode == "into" else min(options, key=lambda r: float(
                            np.max(2 * self._half(name, r) - opening))))
        half = self._half(name, rotation)
        floor = float(low[2]) + min(0.25 * float(size[2]), 0.04) if mode == "into" else float(high[2]) + 0.005
        others = [item for item in self._others({name, reference})
                  if item[2][0] > region_low[0] and item[1][0] < region_high[0]
                  and item[2][1] > region_low[1] and item[1][1] < region_high[1]]
        best = None
        for x in self._grid(region_low[0], region_high[0], half[0]):
            for y in self._grid(region_low[1], region_high[1], half[1]):
                blockers = self._blockers(x, y, half, others)
                pile = max([floor] + [float(item[2][2]) + 0.005 for item in blockers])
                score = (pile > floor, pile, float(np.hypot(x - centroid[0], y - centroid[1])))
                if best is None or score < best[0]:
                    best = (score, x, y, pile)
        _score, x, y, bottom = best
        self.put(name, center=np.array([x, y, 0.0]), bottom=bottom, rotation=rotation)

    def beside(self, name: str, reference: str) -> None:
        """Set ``name`` down on the free stretch of table nearest to ``reference``.

        On a table with no free stretch, the nearest stretch blocked only by clutter the plan does not name
        is cleared: that clutter is parked off the table, and the case records it.
        """
        _low, _high, centroid = self.box(reference)
        table_low, table_high, _table = self.box("table")
        low, high, _own = self.box(name)
        half = (high - low)[:2] / 2 + 0.005
        others = self._others({name})
        best = None
        for x in np.arange(table_low[0] + half[0] + 0.01, table_high[0] - half[0] - 0.01, 0.01):
            for y in np.arange(table_low[1] + half[1] + 0.01, table_high[1] - half[1] - 0.01, 0.01):
                blockers = self._blockers(x, y, half, others, gap=0.0)
                if any(item[0] in self.keep or item[0] == reference for item in blockers):
                    continue
                score = (len(blockers), float(np.hypot(x - centroid[0], y - centroid[1])))
                if best is None or score < best[0]:
                    best = (score, x, y, blockers)
        if best is None:
            raise RobolabError(f"no free table spot for {name}")
        _score, x, y, blockers = best
        for index, (other, _l, _h) in enumerate(blockers):
            self.park(other, index)
        self.put(name, center=np.array([x, y, 0.0]), bottom=self.table_top() + DROP_M)

    def park(self, name: str, index: int) -> None:
        """Move unnamed clutter well clear of the table (it falls to the floor there)."""
        _low, high, _centroid = self.box("table")
        self.put(name, center=np.array([float(high[0]) + 2.0 + 0.5 * index, float(high[1]) + 2.0, 0.0]),
                 bottom=self.table_top())
        self.parked.append(name)

    def relative(self, name: str, reference: str, direction: str) -> None:
        low, high, centroid = self.box(reference)
        _position, rotation = self.robot_frame()
        axis = {"left": rotation[:, 1], "right": -rotation[:, 1], "front": -rotation[:, 0],
                "behind": rotation[:, 0]}[direction]
        local = self.local_corners(name)
        reach = float(np.abs((local.max(axis=0) - local.min(axis=0))[:2]).max()) / 2 + 0.06
        extent = float(np.abs((high - low)[:2] @ axis[:2])) / 2
        self.put(name, center=centroid + axis * (extent + reach), bottom=self.table_top() + DROP_M)

    def stack(self, names: list[str]) -> None:
        identity = np.eye(3)
        base = names[0]
        self.put(base, bottom=self.table_top() + DROP_M, rotation=identity)
        self.hold(SETTLE_TICKS)
        for lower, upper in zip(names, names[1:]):
            _low, high, centroid = self.box(lower)
            self.put(upper, center=centroid, bottom=float(high[2]) + 0.005, rotation=identity)
            self.hold(SETTLE_TICKS)

    def orient(self, name: str, pose: str) -> None:
        rotation = np.eye(3) if pose == "upright" else np.diag([1.0, -1.0, -1.0])
        self.put(name, bottom=self.table_top() + 0.02, rotation=rotation)

    def lift(self, name: str, dz: float) -> None:
        low, _high, centroid = self.box(name)
        self.put(name, center=centroid, bottom=float(low[2]) + dz)

    def table_center(self, name: str) -> None:
        _low, _high, centroid = self.box("table")
        self.put(name, center=centroid, bottom=self.table_top() + DROP_M)

    def turned(self, name: str, spec: str) -> np.ndarray:
        """A rotation for ``set_down``: ``own``; ``yaw:D`` (own, turned D degrees about world up); ``flat:D``
        (longest axis level along world yaw D, shortest axis up); ``long_up``; ``tilt:T:A`` (long axis up,
        then tipped T degrees toward world azimuth A)."""
        from scipy.spatial.transform import Rotation

        _position, quat = self.pose(name)
        own = _rotation(quat)
        kind, _, rest = spec.partition(":")
        if kind == "own":
            return own
        if kind == "yaw":
            return Rotation.from_euler("z", float(rest), degrees=True).as_matrix() @ own
        if kind == "long_up":
            return self._rotations(name, vertical=True)[-1]
        if kind == "tilt":
            tilt, azimuth = (np.radians(float(value)) for value in rest.split(":"))
            axis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
            return Rotation.from_rotvec(axis * tilt).as_matrix() @ self._rotations(name, vertical=True)[-1]
        if kind == "flat":
            yaw = np.radians(float(rest))
            local = self.local_corners(name)
            order = np.argsort(local.max(axis=0) - local.min(axis=0))[::-1]  # longest first
            rotation = np.zeros((3, 3))
            rotation[:, order[0]] = [np.cos(yaw), np.sin(yaw), 0.0]
            rotation[:, order[1]] = [-np.sin(yaw), np.cos(yaw), 0.0]
            rotation[:, order[2]] = [0.0, 0.0, 1.0]
            if np.linalg.det(rotation) < 0:
                rotation[:, order[2]] *= -1
            return rotation
        raise RobolabError(f"unknown witness rotation {spec!r}")

    def set_down(self, name: str, *, reference: str | None = None, center: Any = None, dx: float = 0.0,
                 dy: float = 0.0, bottom: Any = "table", rotation: str = "own") -> None:
        """Set ``name`` down at a spot the plan states: its box centroid at ``center`` (env-frame xy) or at
        ``reference``'s centroid moved by ``dx``/``dy``; its lowest point at ``bottom`` (a height, ``table``,
        ``top:<object>`` or ``floor:<object>[+dz]``); turned by ``rotation`` (``turned``)."""
        if center is not None:
            xy = np.array([float(center[0]), float(center[1]), 0.0])
        else:
            _low, _high, centroid = self.box(reference)
            xy = np.array([float(centroid[0]) + float(dx), float(centroid[1]) + float(dy), 0.0])
        if bottom == "table":
            height = self.table_top() + DROP_M
        elif isinstance(bottom, str) and bottom.startswith("top:"):
            height = float(self.box(bottom[4:])[1][2]) + 0.005
        elif isinstance(bottom, str) and bottom.startswith("floor:"):
            other, _, lift = bottom[6:].partition("+")
            height = float(self.box(other)[0][2]) + (float(lift) if lift else 0.0)
        else:
            height = float(bottom)
        self.put(name, center=xy, bottom=height, rotation=self.turned(name, rotation))

    def count_in(self, objects: list[str], container: str, k: int, mode: str) -> None:
        """End with the ``k`` objects nearest ``container`` in (or on) it; the others stay where they are.

        Nearest first, so an object that starts inside (even one leaning over the rim) is among the ``k`` and one
        that starts farthest away is the one left out.
        """
        low, high, centroid = self.box(container)

        def inside(name: str) -> bool:
            own = self.box(name)[2]
            return bool(low[0] <= own[0] <= high[0] and low[1] <= own[1] <= high[1])

        def distance(name: str) -> float:
            own = self.box(name)[2]
            return float(np.hypot(own[0] - centroid[0], own[1] - centroid[1]))

        chosen = sorted(objects, key=distance)[:k]
        extra = [name for name in chosen if not inside(name)]
        for index, name in enumerate(extra):
            self.place_over(name, container, index, len(extra), mode)
            self.hold(SETTLE_TICKS)

    def grasp(self, name: str, order: Any = None, depth: float | None = None, flip: bool = False,
              shift: float = 0.0) -> None:
        """Hold ``name`` between the pads: its narrowest axis on the jaw line, then close the gripper.

        ``order`` names the object's local axes for the jaw line, the approach and across the pads (default:
        narrowest, middle, widest); ``depth`` is how far along the approach its box centroid sits past the TCP
        (default: only its near 2 cm between the pads); ``flip`` turns it half a turn about the jaw line, so its
        other end faces the gripper; ``shift`` moves it along the jaw line, so a wall or a handle off its centre
        sits between the pads.
        """
        from scipy.spatial.transform import Rotation

        bridge = self.session.bridge
        tcp = bridge.tcp_pose()
        root_position, root_rotation = self.robot_frame()
        tcp_rotation = root_rotation @ Rotation.from_quat(tcp[3:7]).as_matrix()
        tcp_position = root_position + root_rotation @ np.asarray(tcp[:3], dtype=float)
        local = self.local_corners(name)
        size = local.max(axis=0) - local.min(axis=0)
        # narrowest -> jaw line (TCP x), widest -> across the pads (TCP y)
        order = np.argsort(size) if order is None else np.asarray(order, dtype=int)
        align = np.zeros((3, 3))
        align[:, order[0]] = tcp_rotation[:, 0]
        align[:, order[2]] = tcp_rotation[:, 1]
        align[:, order[1]] = np.cross(tcp_rotation[:, 0], tcp_rotation[:, 1])
        if np.linalg.det(align) < 0:
            align[:, order[1]] *= -1
        if flip:
            align[:, order[1]] *= -1
            align[:, order[2]] *= -1
        # Out along the approach axis so only the object's near part sits between the pads.
        offset = max(0.0, size[order[1]] / 2 - 0.02) if depth is None else float(depth)
        center = tcp_position + tcp_rotation[:, 2] * offset + tcp_rotation[:, 0] * float(shift)
        self.finger = D.GRIPPER_CLOSE_ACTION
        start = previous = bridge.jaw_separation_m()
        for step in range(D.GRIPPER_STEPS):
            self.put(name, center=center, rotation=align)
            result = bridge.step(tcp, self.finger)
            self.results.append(bool(result["official_result"]))
            separation = bridge.jaw_separation_m()
            # The jaw takes a few steps to start; it has closed on the object once it moved and then stalled.
            if step >= 5 and start - separation > 1e-3 and abs(previous - separation) < 1e-4:
                break
            previous = separation

    # ------------------------------------------------------------- visibility
    def visibility(self, names: list[str]) -> dict[str, Any]:
        cameras = self.session.bridge.cameras()
        root_position, root_rotation = self.robot_frame()
        report: dict[str, Any] = {}
        for name in names:
            points = np.vstack([self.corners(name), self.box(name)[2]])
            in_root = (points - root_position) @ root_rotation
            size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / 2
            seen = {}
            for public, data in cameras.items():
                world_to_camera = np.linalg.inv(np.asarray(data["cam2world_cv"], dtype=float))
                camera = in_root @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                intrinsic = np.asarray(data["intrinsic_cv"], dtype=float)
                depth = np.asarray(data["depth"], dtype=float)
                height, width = depth.shape[:2]
                framed = unoccluded = 0
                for x, y, z in camera:
                    if z <= 0:
                        continue
                    u, v = intrinsic[0, 0] * x / z + intrinsic[0, 2], intrinsic[1, 1] * y / z + intrinsic[1, 2]
                    if not (0 <= u < width and 0 <= v < height):
                        continue
                    framed += 1
                    measured = depth[int(v), int(u)]
                    if np.isfinite(measured) and measured >= z - size - 0.02:
                        unoccluded += 1
                seen[public] = {"points_in_frame": framed, "points_unoccluded": unoccluded}
            report[name] = {"cameras": seen, "visible": any(item["points_unoccluded"] for item in seen.values())}
        return report


def _run_case(scene: Scene, case: Mapping[str, Any]) -> None:
    for op in case["ops"]:
        kind = op["op"]
        if kind == "place":
            if "beside" in op:
                scene.beside(op["object"], op["beside"])
            else:
                mode = "into" if op.get("into") else "onto"
                scene.place_over(op["object"], op.get("into") or op.get("onto"), op.get("slot", 0),
                                 op.get("slots", 1), mode)
        elif kind == "stack":
            scene.stack(list(op["objects"]))
        elif kind == "orient":
            scene.orient(op["object"], op["pose"])
        elif kind == "lift":
            scene.lift(op["object"], float(op["dz"]))
        elif kind == "relative":
            scene.relative(op["object"], op["reference"], op["direction"])
        elif kind == "table_center":
            scene.table_center(op["object"])
        elif kind == "count_in":
            scene.count_in(list(op["objects"]), op["container"], int(op["k"]), op.get("mode", "into"))
        elif kind == "grasp":
            scene.grasp(op["object"], op.get("order"), op.get("depth"), bool(op.get("flip", False)),
                        float(op.get("shift", 0.0)))
        elif kind == "put":
            scene.set_down(op["object"], reference=op.get("reference"), center=op.get("center"),
                           dx=float(op.get("dx", 0.0)), dy=float(op.get("dy", 0.0)), bottom=op.get("bottom", "table"),
                           rotation=op.get("rotation", "own"))
        elif kind == "park":
            scene.park(op["object"], int(op.get("index", 0)))
        else:
            raise RobolabError(f"unknown witness op {kind!r}")
        # An exact-count negative places all its objects before physics runs, so the count jumps past K at once.
        if kind in ("place", "put", "orient", "relative", "table_center") and op.get("settle", True):
            scene.hold(SETTLE_TICKS if kind in ("place", "put") else 2)


def _named(plan: Mapping[str, Any]) -> set[str]:
    """Every object the plan names: its visibility objects and each op's objects and references."""
    names = set(plan.get("visibility_objects") or [])
    for case in plan.get("cases", []):
        for op in case.get("ops", []):
            for key in ("object", "into", "onto", "beside", "reference", "container"):
                if isinstance(op.get(key), str):
                    names.add(op[key])
            names.update(op.get("objects") or [])
    return names


def run_witness(job: Mapping[str, Any], *, session: Any) -> dict[str, Any]:
    """Run every case of the job's plan on this session and write the witness record."""
    if job.get("schema") != JOB_SCHEMA or job.get("task") != session.task:
        raise RobolabError("the witness job does not name this session's task")
    plan = job["plan"]
    named = list(plan.get("visibility_objects") or [])
    cases, visibility = [], None
    for index, case in enumerate(plan["cases"]):
        entry: dict[str, Any] = {"case": case["case"], "expected": bool(case["expected"])}
        try:
            if index:
                session.restart_round(index, seed=int(plan.get("seed", 0)))
            scene = Scene(session, keep=_named(plan))
            scene.hold(LEAD_TICKS)
            if case["case"] == "reset_state" and named:
                visibility = scene.visibility(named)
            scene.results = []
            ignore = int(case.get("ignore_ticks", 0))
            scene.ignore_until = ignore
            _run_case(scene, case)
            scene.hold(int(case.get("ticks", 40)))
            counted = scene.results[ignore:]
            value = any(counted)
            entry.update(official_success=value, matches=value == entry["expected"], ticks_held=len(scene.results),
                         first_success_tick=ignore + counted.index(True) if value else None,
                         ignored_ticks=ignore, parked=list(scene.parked),
                         object_positions_m={name: [round(float(v), 5) for v in scene.box(name)[2]]
                                             for name in named})
        except Exception as exc:  # noqa: BLE001 - a case that cannot run is a failed case, with its reason
            entry.update(official_success=None, matches=False, error=f"{type(exc).__name__}: {exc}")
        session.bridge.take_ticks()
        cases.append(entry)
    record = {"schema": WITNESS_SCHEMA, "task": session.task, "seed": int(plan.get("seed", 0)),
              "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest(),
              "cases": cases, "all_cases_match": all(case["matches"] for case in cases),
              "visibility_at_reset": visibility,
              "scope": "private offline witness of the official termination; object poses set directly with the "
                       "robot held (the pick-up family closes the gripper on an object placed at the TCP), each case "
                       "from a fresh reset, not a policy rollout",
              "created_unix_s": time.time()}
    path = Path(job["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


__all__ = ["JOB_SCHEMA", "WITNESS_SCHEMA", "run_witness"]
