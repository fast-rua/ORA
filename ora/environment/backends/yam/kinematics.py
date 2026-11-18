"""Kinematics of the bimanual YAM station on i2rt's station MJCF.

One MuJoCo model carries both arms, both wrist cameras and the 0.61 m base offset, so
forward kinematics, inverse kinematics (mink, damped differential IK with joint limits)
and the inter-arm and table clearance checks all read the same geometry the CAD gives.
Every public pose is in ``yam_station_world`` with the public TCP axes
(``controllers.yam_station``); the station model's own frame is internal.

A cell's calibration may correct the CAD (``ora.cell.calibration.apply``): where an arm's
base really stands (``set_base_corrections``, which moves that arm's bodies in the model,
so IK, clearance and its wrist camera all follow) and each joint's encoder zero
(``joint_offsets``). Joint values in and out of this class are always the encoders'
readings; the model sees ``reading + offset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ora.environment.controllers import yam_station as Y

from .common import YamError


def station_xml_path() -> Path:
    """The station MJCF shipped inside the installed i2rt package."""

    import i2rt

    root = Path(i2rt.__file__).resolve().parent / "robot_models" / "station" / Y.STATION_MODEL
    path = root / f"{Y.STATION_MODEL}.xml"
    if not path.is_file():
        raise YamError(f"i2rt station model is missing: {path}")
    return path


_TCP_FROM_SITE = np.asarray(Y.TCP_AXES_IN_SITE, dtype=float).T   # columns: public axes in the site frame
_STATION_TO_WORLD = np.asarray(Y.STATION_TO_WORLD_XYZ_M, dtype=float)


def _frame(position: Sequence[float], xmat: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(xmat, dtype=float).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix


def pose_matrix(position: Sequence[float], xyzw: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([float(v) for v in xyzw]).as_matrix()
    matrix[:3, 3] = [float(v) for v in position]
    return matrix


def pose_xyz_xyzw(matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if xyzw[3] < 0:
        xyzw = -xyzw
    return [*map(float, matrix[:3, 3]), *map(float, xyzw)]


@dataclass(frozen=True)
class IkResult:
    ok: bool
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class StationKinematics:
    """FK, IK and clearance on the station model. Not thread-safe; one per worker."""

    def __init__(self, model=None) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(station_xml_path()))
        self.data = mujoco.MjData(self.model)
        name = lambda kind, n: mujoco.mj_name2id(self.model, kind, n)
        J, B, S = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE
        self.arm_qadr = {arm: np.array([self.model.jnt_qposadr[name(J, f"{arm}_joint{i}")] for i in range(1, 7)])
                         for arm in Y.ARMS}
        self.finger_qadr = {arm: np.array([self.model.jnt_qposadr[name(J, j)] for j in Y.FINGER_JOINTS[arm]])
                            for arm in Y.ARMS}
        self.site = {arm: name(S, Y.TCP_SITES[arm]) for arm in Y.ARMS}
        self.camera_body = {arm: name(B, Y.WRIST_CAMERA_BODIES[arm]) for arm in Y.ARMS}
        if min(self.site.values()) < 0 or min(self.camera_body.values()) < 0:
            raise YamError("station model lacks a TCP site or a wrist camera body")
        self.arm_joints = {arm: np.array([name(J, f"{arm}_joint{i}") for i in range(1, 7)]) for arm in Y.ARMS}
        self.model_limits = {arm: np.array([self.model.jnt_range[name(J, f"{arm}_joint{i}")] for i in range(1, 7)])
                             for arm in Y.ARMS}
        # Moving geoms of each arm: every body below link1 (the base and link1 never leave the mount).
        self.arm_geoms: dict[str, list[int]] = {}
        for arm in Y.ARMS:
            root = name(B, f"{arm}_link2")
            bodies = {b for b in range(self.model.nbody) if self._descends(b, root)}
            self.arm_geoms[arm] = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies]
        # Each arm's wrist (link5 and below) against its own base, shoulder and upper arm.
        self.self_pairs: dict[str, list[tuple[int, int]]] = {}
        for arm in Y.ARMS:
            low = {name(B, f"{arm}_{b}") for b in ("base", "link1", "link2")}
            tip = {b for b in range(self.model.nbody) if self._descends(b, name(B, f"{arm}_link5"))}
            low_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in low]
            tip_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in tip]
            self.self_pairs[arm] = [(a, b) for a in tip_geoms for b in low_geoms]
        self.table_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "ora_table")
        self.qpos = self.data.qpos.copy()
        self.world_offset = _STATION_TO_WORLD.copy()      # model frame -> yam_station_world
        # Measured corrections of the wrist cameras' CAD mounts (cell calibration), camera frame.
        self.camera_corrections: dict[str, np.ndarray] = {}
        # Measured encoder zero errors (cell calibration): model angle = reading + offset.
        self.joint_offsets: dict[str, np.ndarray] = {}
        self.base_body = {arm: name(B, f"{arm}_base") for arm in Y.ARMS}
        self._cad_base = {arm: (self.model.body_pos[b].copy(), self.model.body_quat[b].copy())
                          for arm, b in self.base_body.items()}
        self.base_corrections: dict[str, np.ndarray] = {}

    def _offset(self, arm: str) -> np.ndarray:
        return self.joint_offsets.get(arm, np.zeros(6))

    @property
    def arm_limits(self) -> dict[str, np.ndarray]:
        """Each arm's joint limits as encoder readings."""

        return {arm: limits - self._offset(arm)[:, None] for arm, limits in self.model_limits.items()}

    def set_base_corrections(self, corrections: Mapping[str, np.ndarray]) -> None:
        """Move each named arm's base by a 4x4 given in ``yam_station_world`` (applied before
        the CAD pose: corrected = correction @ CAD), every other arm staying at its CAD pose."""

        import mujoco

        self.base_corrections = {arm: np.asarray(m, float) for arm, m in corrections.items()}
        for arm, (pos, quat) in self._cad_base.items():
            self.model.body_pos[self.base_body[arm]], self.model.body_quat[self.base_body[arm]] = pos, quat
        self.data.qpos[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        cad = {arm: _frame(self.data.xpos[b], self.data.xmat[b]) for arm, b in self.base_body.items()}
        to_model = np.eye(4)
        to_model[:3, 3] = -_STATION_TO_WORLD
        wanted = {arm: to_model @ self.base_corrections.get(arm, np.eye(4)) @ np.linalg.inv(to_model) @ cad[arm]
                  for arm in Y.ARMS}
        for arm, b in self.base_body.items():
            parent = int(self.model.body_parentid[b])
            above = next((other for other, ob in self.base_body.items() if ob == parent), None)
            if above is not None:
                parent_pose = wanted[above]
            else:
                parent_pose = _frame(self.data.xpos[parent], self.data.xmat[parent])
            local = np.linalg.inv(parent_pose) @ wanted[arm]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, local[:3, :3].ravel())
            self.model.body_pos[b], self.model.body_quat[b] = local[:3, 3], quat
        self.set_state({arm: self.arm_q(arm) for arm in Y.ARMS})

    def _descends(self, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return False

    # ----------------------------------------------------------- state
    def set_state(self, arm_q: Mapping[str, Sequence[float]], finger_m: Mapping[str, float] | None = None) -> None:
        for arm, q in arm_q.items():
            self.qpos[self.arm_qadr[arm]] = np.asarray(q, dtype=float) + self._offset(arm)
        for arm, value in (finger_m or {}).items():
            self.qpos[self.finger_qadr[arm]] = float(value)
        self.data.qpos[:] = self.qpos
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)

    def arm_q(self, arm: str) -> np.ndarray:
        return self.qpos[self.arm_qadr[arm]] - self._offset(arm)

    # ----------------------------------------------------------- frames
    def tcp(self, arm: str) -> np.ndarray:
        """Public TCP pose (4x4) of ``arm`` in yam_station_world for the current state."""

        site = self.site[arm]
        matrix = np.eye(4)
        matrix[:3, :3] = self.data.site_xmat[site].reshape(3, 3) @ _TCP_FROM_SITE
        matrix[:3, 3] = self.data.site_xpos[site] + _STATION_TO_WORLD
        return matrix

    CHAIN_BODIES = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

    def chain(self, arm: str) -> list[list[float]]:
        """World positions of the arm's bodies from base to gripper, then the TCP: its skeleton."""

        ids = [self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{name}")
               for name in self.CHAIN_BODIES]
        points = [(self.data.xpos[i] + _STATION_TO_WORLD).tolist() for i in ids]
        return points + [self.tcp(arm)[:3, 3].tolist()]

    def fk(self, arm: str, q: Sequence[float]) -> np.ndarray:
        self.set_state({arm: q})
        return self.tcp(arm)

    def contact_geometry(self) -> dict[str, dict]:
        """Each gripper's finger and pad boxes in its TCP frame, measured on the loaded station model.

        The fingers stand fully open (the descriptor's reference opening); the collision box is
        each fingertip mesh within 60 mm behind the TCP, the pad its inner 10.1 mm toward the
        closing line within 50 mm behind it, rounded to 0.1 mm as the public descriptor states
        them. Its own MjData, so the live state is untouched. Robot geometry only, never contact.
        """

        mujoco = self.mujoco
        data = mujoco.MjData(self.model)
        for arm in Y.ARMS:
            data.qpos[self.finger_qadr[arm]] = Y.FINGER_OPEN_M
        mujoco.mj_kinematics(self.model, data)
        body = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

        def vertices(name: str) -> np.ndarray:
            found = []
            for geom in range(self.model.ngeom):
                if (self.model.geom_bodyid[geom] != body(name)
                        or self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH):
                    continue
                mesh = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mesh]
                local = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]]
                found.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
            if not found:
                raise YamError(f"station model has no finger mesh on {name}")
            return np.vstack(found)

        def box(points: np.ndarray) -> dict[str, list[float]]:
            return {"minimum": [round(float(v), 4) for v in points.min(0)],
                    "maximum": [round(float(v), 4) for v in points.max(0)]}

        result = {}
        for arm in Y.ARMS:
            site = self.site[arm]
            tcp = np.eye(4)
            tcp[:3, :3] = data.site_xmat[site].reshape(3, 3) @ _TCP_FROM_SITE
            tcp[:3, 3] = data.site_xpos[site]
            to_tcp = np.linalg.inv(tcp)
            fingers = []
            for finger, name, sign in (("finger_1", f"{arm}_tip_left", 1.0), ("finger_2", f"{arm}_tip_right", -1.0)):
                points = vertices(name)
                local = (to_tcp @ np.c_[points, np.ones(len(points))].T).T[:, :3]
                collision = local[local[:, 2] > -0.06]
                inner = collision[:, 0].min() if sign > 0 else collision[:, 0].max()
                pad = collision[(collision[:, 2] > -0.05) & (np.abs(collision[:, 0] - inner) <= 0.0101)]
                fingers.append({"id": finger, "opening_direction_local": [sign, 0.0, 0.0],
                                "collision_bounds_m": box(collision), "pad_bounds_m": box(pad)})
            result[f"{arm}_gripper"] = {
                "claim": "robot_geometry_not_contact_or_grasp_guarantee", "controlled_point": f"{arm}_tcp",
                "reference_finger_qpos_m": [Y.FINGER_OPEN_M, Y.FINGER_OPEN_M], "unit": "metre",
                "fingers": fingers}
        return result

    def wrist_camera(self, arm: str) -> np.ndarray:
        """cam2world (OpenCV optical frame) of ``arm``'s wrist D405 for the current state."""

        body = self.camera_body[arm]
        matrix = np.eye(4)
        matrix[:3, :3] = self.data.xmat[body].reshape(3, 3)
        matrix[:3, 3] = self.data.xpos[body] + _STATION_TO_WORLD
        correction = self.camera_corrections.get(arm)
        return matrix if correction is None else matrix @ correction

    # ----------------------------------------------------------- inverse kinematics
    def ik(self, arm: str, target_world: np.ndarray, seed: Sequence[float], *,
           max_iterations: int = 150, start: Sequence[float] | None = None) -> IkResult:
        """Damped differential IK of the public TCP onto ``target_world``, within joint limits.

        ``start`` (encoder readings, the motion's measured start): a joint already outside its
        limit there may stay at that angle or move inward, never further out (``widened_limits``).
        """

        if start is None:
            return self._ik(arm, target_world, seed, max_iterations)
        joints = self.arm_joints[arm]
        saved = self.model.jnt_range[joints].copy()
        self.model.jnt_range[joints] = self.widened_limits(arm, start) + self._offset(arm)[:, None]
        try:
            return self._ik(arm, target_world, seed, max_iterations)
        finally:
            self.model.jnt_range[joints] = saved

    def widened_limits(self, arm: str, start: Sequence[float]) -> np.ndarray:
        """``arm``'s limits as encoder readings, each widened to include the start angle.

        At rest a joint can read past its limit: the arm sits on its hard stop, and the cell's
        measured encoder zero (``joint_offsets``) puts that reading a little outside the model's
        range (2026-11-14: joints 2 and 3 by 0.5-1 deg). Without this, every motion from rest
        was refused at its first waypoint or tick.
        """

        limits = self.arm_limits[arm].copy()
        q = np.asarray(start, dtype=float)
        limits[:, 0] = np.minimum(limits[:, 0], q)
        limits[:, 1] = np.maximum(limits[:, 1], q)
        return limits

    def _ik(self, arm: str, target_world: np.ndarray, seed: Sequence[float], max_iterations: int) -> IkResult:
        import mink

        site_target = np.eye(4)
        site_target[:3, :3] = target_world[:3, :3] @ _TCP_FROM_SITE.T
        site_target[:3, 3] = target_world[:3, 3] - _STATION_TO_WORLD
        configuration = mink.Configuration(self.model)
        q0 = self.qpos.copy()
        q0[self.arm_qadr[arm]] = np.asarray(seed, dtype=float) + self._offset(arm)
        configuration.update(q0)
        task = mink.FrameTask(frame_name=Y.TCP_SITES[arm], frame_type="site",
                              position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        task.set_target(mink.SE3.from_matrix(site_target))
        limits = [mink.ConfigurationLimit(self.model)]
        error = task.compute_error(configuration)
        used = 0
        for used in range(1, max_iterations + 1):
            velocity = mink.solve_ik(configuration, [task], 0.02, "daqp", damping=1e-4, limits=limits)
            configuration.integrate_inplace(velocity, 0.02)
            error = task.compute_error(configuration)
            if (np.linalg.norm(error[:3]) <= Y.IK_POSITION_TOLERANCE_M
                    and np.linalg.norm(error[3:]) <= Y.IK_ORIENTATION_TOLERANCE_RAD):
                break
        q = np.asarray(configuration.q)[self.arm_qadr[arm]] - self._offset(arm)
        position_error, orientation_error = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        ok = position_error <= Y.IK_POSITION_TOLERANCE_M and orientation_error <= Y.IK_ORIENTATION_TOLERANCE_RAD
        return IkResult(ok, q, position_error, orientation_error, used)

    # ----------------------------------------------------------- clearance
    def _distance(self, a: int, b: int, cap: float) -> float:
        return float(self.mujoco.mj_geomDistance(self.model, self.data, a, b, cap, None))

    def inter_arm_distance(self, cap: float = 0.5) -> float:
        best = cap
        for a in self.arm_geoms["left"]:
            for b in self.arm_geoms["right"]:
                best = min(best, self._distance(a, b, best))
        return best

    def inter_arm_closest_parts(self, cap: float = 0.5) -> dict[str, str]:
        """The two arms' nearest parts at the current state, by body name without the arm prefix."""

        best, pair = cap, None
        for a in self.arm_geoms["left"]:
            for b in self.arm_geoms["right"]:
                distance = self._distance(a, b, best)
                if distance < best or pair is None:
                    best, pair = min(best, distance), (a, b)
        body = lambda geom, arm: self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom]).removeprefix(f"{arm}_")
        return {"left": body(pair[0], "left"), "right": body(pair[1], "right")}

    def self_distances(self, arm: str, cap: float = 0.1) -> np.ndarray:
        """Distance of each pair of ``arm``'s wrist geoms and its own base, shoulder and upper-arm geoms."""

        return np.array([self._distance(a, b, cap) for a, b in self.self_pairs[arm]])

    def table_distance(self, arm: str, cap: float = 0.5) -> float | None:
        """Smallest distance from ``arm``'s moving geoms to the table; None when no table is modelled."""

        if self.table_geom < 0:
            return None
        return min(self._distance(g, self.table_geom, cap) for g in self.arm_geoms[arm])

    def within_limits(self, arm: str, q: Sequence[float], start: Sequence[float] | None = None) -> bool:
        """``q`` within ``arm``'s limits; with ``start``, each widened to include that angle (``widened_limits``)."""

        limits = self.arm_limits[arm] if start is None else self.widened_limits(arm, start)
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= limits[:, 0] - 1e-6) and np.all(q <= limits[:, 1] + 1e-6))
