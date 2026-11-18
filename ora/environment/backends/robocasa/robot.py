"""Trusted PandaOmron state reads and one-control-step commands.

Every motion module goes through :class:`OmronIO`. Public poses are in the
fixed RoboCasa world frame. The arm follows joint targets, held from one step
to the next until a motion replaces them, and the base velocity is
converted to the current base frame here, so callers never handle
controller frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ora.environment.controllers import robocasa_omron as C

from .common import RoboCasaError, quaternion_xyzw_from_matrix, wrap_angle


@dataclass
class StepResult:
    reward: float
    official_success: bool


@dataclass
class OmronIO:
    environment: Any
    control_steps: int = 0
    gripper_directive: float = C.GRIPPER_OPEN_COMMAND
    arm_goal: list[float] = field(init=False, default_factory=list)
    #: Every action vector sent since ``take_commands``, in order: what a replay feeds back, one per control step.
    commands: list[Any] = field(init=False, default_factory=list)
    _robot_geoms: frozenset[int] = field(init=False, default=frozenset())
    _object_geoms: frozenset[int] = field(init=False, default=frozenset())
    _gripper_geoms: frozenset[int] = field(init=False, default=frozenset())
    _base_geoms: frozenset[int] = field(init=False, default=frozenset())

    def __post_init__(self) -> None:
        self.bind()
        self.hold_measured_arm()

    # ----------------------------------------------------------------- model
    @property
    def model(self) -> Any:
        sim = self.environment.sim
        return getattr(sim.model, "_model", sim.model)

    @property
    def data(self) -> Any:
        sim = self.environment.sim
        return getattr(sim.data, "_data", sim.data)

    def bind(self) -> None:
        """Index robot geoms after every (re)load of the MuJoCo model."""

        import mujoco

        model = self.model
        # A movable object is any body below a free joint; everything else in
        # the scene (counters, cabinets, their doors and drawers) is a fixture.
        free = set()
        for body in range(model.nbody):
            first, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
            if any(int(model.jnt_type[first + k]) == mujoco.mjtJoint.mjJNT_FREE for k in range(count)):
                free.add(body)
        robot, gripper, base, objects = set(), set(), set(), set()
        for geom in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body.startswith(C.ROBOT_BODY_PREFIXES):
                robot.add(geom)
            if body.startswith("gripper0_") or body == "robot0_right_hand":
                gripper.add(geom)
            if body.startswith("mobilebase0_"):
                base.add(geom)
            ancestor = body_id
            while ancestor > 0:
                if ancestor in free:
                    objects.add(geom)
                    break
                ancestor = int(model.body_parentid[ancestor])
        if not gripper or not base:
            raise RoboCasaError("PandaOmron gripper or base geoms are missing")
        self._robot_geoms = frozenset(robot)
        self._gripper_geoms = frozenset(gripper)
        self._base_geoms = frozenset(base)
        self._object_geoms = frozenset(objects - robot)

    def _site_id(self, name: str) -> int:
        import mujoco

        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if index < 0:
            raise RoboCasaError(f"site is missing: {name}")
        return int(index)

    def _body_id(self, name: str) -> int:
        import mujoco

        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise RoboCasaError(f"body is missing: {name}")
        return int(index)

    def _joint_qpos(self, name: str) -> float:
        import mujoco

        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise RoboCasaError(f"joint is missing: {name}")
        return float(self.data.qpos[self.model.jnt_qposadr[index]])

    # ----------------------------------------------------------------- state
    def public_gripper_geometry(self) -> dict[str, Any]:
        """Loaded Panda finger and pad extents in the grip-site frame (robot geoms only)."""

        from ora.environment.backends.libero_pro.robot_asset_geometry import panda_geometry_from_model

        return panda_geometry_from_model(self.model, self.data, prefix=C.GRIP_SITE[: -len("grip_site")])

    def grip_pose(self) -> list[float]:
        """Grip-site pose (xyz + xyzw) in the world frame."""

        site = self._site_id(C.GRIP_SITE)
        position = [float(value) for value in self.data.site_xpos[site]]
        return position + quaternion_xyzw_from_matrix(self.data.site_xmat[site])

    def _body_pose(self, name: str) -> list[float]:
        body = self._body_id(name)
        return [float(value) for value in self.data.xpos[body]] + quaternion_xyzw_from_matrix(self.data.xmat[body])

    def hand_pose(self) -> list[float]:
        """The hand body's pose (xyz + xyzw) in the world frame: the planner's tool frame."""

        return self._body_pose(C.HAND_BODY)

    def arm_base_pose(self) -> list[float]:
        """The arm's base link pose (xyz + xyzw) in the world frame: the planner's base."""

        return self._body_pose(C.ARM_BASE_BODY)

    def arm_base_position(self) -> list[float]:
        """World position of the arm's mount on the torso (reach is measured from it)."""

        return [float(value) for value in self.data.site_xpos[self._site_id(C.ARM_BASE_SITE)]]

    def base_pose(self) -> list[float]:
        """Base footprint ``(x, y, yaw)`` in the world frame."""

        import numpy as np

        body = self._body_id(C.BASE_BODY)
        position = self.data.xpos[body]
        matrix = np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3)
        return [float(position[0]), float(position[1]), float(math.atan2(matrix[1, 0], matrix[0, 0]))]

    def base_anchor(self) -> list[float]:
        """World xy of the yaw hinge anchor (the base's rotation centre)."""

        import mujoco

        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, C.BASE_YAW_JOINT)
        if joint < 0:
            raise RoboCasaError("base yaw joint is missing")
        anchor = self.data.xanchor[joint]
        return [float(anchor[0]), float(anchor[1])]

    def anchor_offset(self) -> list[float]:
        """Anchor minus body origin, expressed in the base frame (constant)."""

        x, y, yaw = self.base_pose()
        ax, ay = self.base_anchor()
        dx, dy = ax - x, ay - y
        return [math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy]

    def slide_axes(self) -> list[list[float]]:
        """World directions of the forward and side slide joints."""

        import mujoco

        axes = []
        for name in C.BASE_JOINTS[:2]:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            axes.append([float(value) for value in self.data.xaxis[joint][:2]])
        return axes

    def finger_qpos(self) -> list[float]:
        return [self._joint_qpos(name) for name in C.FINGER_JOINTS]

    def gripper_opening(self) -> float:
        """Normalized measured opening: 0 closed, 1 fully open."""

        values = [abs(value) for value in self.finger_qpos()]
        return float(min(1.0, max(0.0, (sum(values) / len(values)) / C.PANDA_FINGER_OPEN_M)))

    def arm_qpos(self) -> list[float]:
        return [self._joint_qpos(name) for name in C.ARM_JOINTS]

    def hold_measured_arm(self) -> None:
        """Make the measured arm joints the held target (after a reset or a blocked stop)."""

        self.arm_goal = self.arm_qpos()

    def torso_height(self) -> float:
        return self._joint_qpos(C.TORSO_JOINT)

    def contact_forces(self) -> dict[str, float]:
        """Summed normal force between the robot and the scene, by robot part."""

        import mujoco
        import numpy as np

        model, data = self.model, self.data
        buffer = np.zeros(6, dtype=np.float64)
        totals = {"gripper": 0.0, "base": 0.0, "arm": 0.0}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            robot_side = [geom in self._robot_geoms for geom in geoms]
            if robot_side[0] == robot_side[1]:
                continue  # robot-robot self contact or scene-scene contact
            robot_geom = geoms[0] if robot_side[0] else geoms[1]
            mujoco.mj_contactForce(model, data, index, buffer)
            force = abs(float(buffer[0]))
            if robot_geom in self._gripper_geoms:
                totals["gripper"] += force
            elif robot_geom in self._base_geoms:
                totals["base"] += force
            else:
                totals["arm"] += force
        return totals

    def state_sha256(self) -> str:
        """The whole scene's time, positions and velocities, objects and fixtures as well as the robot."""

        import hashlib

        import numpy as np

        data = self.data
        return hashlib.sha256(np.concatenate(([data.time], data.qpos, data.qvel)).tobytes()).hexdigest()

    def state(self) -> dict[str, Any]:
        return {
            "pose_xyz_xyzw": self.grip_pose(),
            "gripper": self.gripper_opening(),
            "gripper_finger_qpos": self.finger_qpos(),
            "arm_joint_qpos": self.arm_qpos(),
            "base_pose_xy_yaw": self.base_pose(),
            "torso_height_m": self.torso_height(),
        }

    # --------------------------------------------------------------- command
    def step(
        self,
        *,
        arm_joints: Sequence[float] | None = None,
        base_velocity_world: Sequence[float] | None = None,
        gripper: float | None = None,
        torso_delta_m: float = 0.0,
    ) -> StepResult:
        """Apply one 20 Hz control step.

        ``arm_joints`` replaces the held arm joint target; without it the arm
        keeps its last target. The controller takes a delta from the measured
        joints, so the command is the remaining difference. ``base_velocity_world`` is ``(vx, vy,
        yaw_rate)`` in the world frame; when given the base mode bit is set.
        ``gripper`` updates the held jaw directive (-1 open, +1 close).
        """

        import numpy as np

        if gripper is not None:
            self.gripper_directive = float(gripper)
        if arm_joints is not None:
            self.arm_goal = [float(value) for value in arm_joints]
        action = np.zeros(C.ACTION_DIM, dtype=np.float64)
        start, end = C.ARM_SLICE
        delta = np.asarray(self.arm_goal, dtype=np.float64) - np.asarray(self.arm_qpos(), dtype=np.float64)
        action[start:end] = np.clip(delta / C.ARM_DELTA_LIMIT_RAD, -1.0, 1.0)
        action[C.GRIPPER_INDEX] = self.gripper_directive
        action[C.TORSO_INDEX] = float(np.clip(torso_delta_m / C.TORSO_MAX_DELTA_PER_STEP_M, -1.0, 1.0))
        if base_velocity_world is not None:
            vx, vy, yaw_rate = (float(value) for value in base_velocity_world)
            yaw = self.base_pose()[2]
            forward = math.cos(yaw) * vx + math.sin(yaw) * vy
            side = -math.sin(yaw) * vx + math.cos(yaw) * vy
            start, end = C.BASE_SLICE
            action[start:end] = np.clip([forward, side, yaw_rate], -1.0, 1.0)
            action[C.MODE_INDEX] = C.MODE_BASE
        else:
            action[C.MODE_INDEX] = C.MODE_ARM
        self.commands.append(action.copy())
        return self.apply(action)

    def apply(self, action: Any) -> StepResult:
        """Send one action vector and read the official predicate: one control step, run or replayed."""

        _obs, reward, _done, _info = self.environment.step(action)
        self.control_steps += 1
        return StepResult(reward=float(reward), official_success=bool(self.environment._check_success()))

    def take_commands(self) -> list[Any]:
        """The action vectors sent since the last call."""

        commands, self.commands = self.commands, []
        return commands


def yaw_error(target: float, current: float) -> float:
    return wrap_angle(float(target) - float(current))
