"""Test helpers for robodojo backend, shared by several test modules.

They were defined in ``tests/ora/environment/backends/robodojo/test_robodojo_backend.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from ora.environment.backends.robodojo.session import RobodojoSession


class Robot:
    def __init__(self, side):
        self.type = "target"
        self.robot_name = side
        self.entity_origin_pose = [0, 0, 0, 1, 0, 0, 0]
        self.arm_name, self.gripper_name = f"{side}_arm", f"{side}_ee"
        self.gripper_scale = [-0.01, 0.044]
        self.gripper_move = {"sign": 1, "mimic": ["joint8", 1, 0]}
        self.ee_type = "gripper"
        self.ee_link_name = "link6"
        self.arm_joints_name = [f"joint{index}" for index in range(1, 7)]
        self.gripper_joints_name = ["joint7", "joint8"]
        self.arm_joint_indices = list(range(6))
        self.gripper_joint_indices = [6, 7]


class ArticulationData(SimpleNamespace):
    """An articulation's buffers as ``common.arm_readings`` copies them.

    ``joint_pos`` and ``body_link_pose_w`` come from the fake SDK readers
    (``get_joint``, ``get_end_effector_real_val``, ``get_real_endpose``) at
    every read, so a test that moves the fake plant or replaces a reader feeds
    the session as before.
    """

    @property
    def joint_pos(self):
        return self.state()[0]

    @property
    def body_link_pose_w(self):
        return self.state()[1]


class Manager:
    def __init__(self):
        self.robot_list = [Robot("left"), Robot("right")]
        self.ik_solver = {robot.robot_name: SimpleNamespace(manager=self, robot=robot)
                          for robot in self.robot_list}
        self.robot_key = [SimpleNamespace(
            is_fixed_base=True, joint_names=[f"joint{index}" for index in range(1, 9)],
            body_names=["base_link", "link6", "camera"],
        ) for _ in self.robot_list]
        self.fail_ik = False
        # The synthetic home joint vector, which the fake kinematics below read
        # back as the link6 pose; ``set_home`` moves it so a test can place the
        # fake plant inside a specific action envelope.
        self.home = [0, 0, 0.8, 0, 0, 0]
        self.joints = {robot.arm_name: list(self.home) for robot in self.robot_list}
        self.fingers = {robot.arm_name: [0.044, 0.044] for robot in self.robot_list}
        self.control_manager = SimpleNamespace(prev_control=[{
            key: {"position": list(value)}
            for robot in self.robot_list
            for key, value in ((self.process_name(robot.arm_name), self.joints[robot.arm_name]),
                               (self.process_name(robot.gripper_name), self.fingers[robot.arm_name]))
        }])
        self.unreachable = None
        self.scene = SimpleNamespace(env_origins=np.zeros((1, 3)))
        for robot, articulation in zip(self.robot_list, self.robot_key):
            articulation.data = ArticulationData(joint_pos_target=np.array([[*self.home, .044, .044]]),
                                                 joint_vel_target=np.zeros((1, 8)),
                                                 # The measured joint velocities (a test sets a moving arm's).
                                                 joint_vel=np.zeros((1, 8)),
                                                 joint_pos_limits=np.array([[[-3.14, 3.14]] * 8]),
                                                 state=lambda robot=robot: self.articulation_state(robot))
        for robot, articulation in zip(self.robot_list, self.robot_key):
            articulation.write_joint_state_to_sim = (
                lambda position, velocity, joint_ids=None, env_ids=None, _arm=robot.arm_name:
                self.write_joint_state(_arm, position, joint_ids))

    def write_joint_state(self, arm, position, joint_ids=None):
        """Teleport ``arm`` joints like Isaac Lab's ``write_joint_state_to_sim``."""
        ids = list(range(6)) if joint_ids is None else list(joint_ids)
        for column, index in enumerate(ids):
            if index < 6:
                self.joints[arm][index] = float(position[0][column])

    def set_home(self, joints):
        """Rebind the synthetic home pose and settle the plant on it."""
        self.home = [float(value) for value in joints]
        self.reset_home()

    def reset_home(self):
        for robot in self.robot_list:
            self.joints[robot.arm_name] = list(self.home)
            self.fingers[robot.arm_name] = [0.044, 0.044]
            self.control_manager.prev_control[0][self.process_name(robot.arm_name)] = {"position": list(self.home)}
            self.control_manager.prev_control[0][self.process_name(robot.gripper_name)] = {"position": [0.044, 0.044]}
        for articulation in self.robot_key:
            articulation.data.joint_pos_target[0] = [*self.home, .044, .044]
            articulation.data.joint_vel_target[0] = np.zeros(8)

    def articulation_state(self, robot):
        """``robot``'s joint positions and body poses (environment-relative, wxyz) as Isaac Lab lays them out."""
        joints = np.zeros((1, 8))
        joints[0, robot.arm_joint_indices] = self.get_joint(robot)[0]
        joints[0, robot.gripper_joint_indices] = self.get_end_effector_real_val(robot)[0]
        names = self.robot_key[self.robot_list.index(robot)].body_names
        bodies = np.zeros((1, len(names), 7))
        bodies[0, names.index(robot.ee_link_name)] = self.get_real_endpose(robot)[0]
        return joints, bodies

    @staticmethod
    def process_name(name):
        return name + "_joint_state"

    def get_joint(self, robot, env_idx_list=None):
        return {0: self.joints[robot.arm_name]}

    def get_real_endpose(self, robot, env_idx_list=None):
        # Fake kinematics: joints 0-2 are the link6 position, joints 3-5 the
        # xyz quaternion components on the w >= 0 hemisphere.
        joints = self.joints[robot.arm_name]
        vector = [float(value) for value in joints[3:6]]
        w = math.sqrt(max(0.0, 1.0 - sum(value * value for value in vector)))
        return {0: list(joints[:3]) + [w] + vector}

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        return {0: self.fingers[robot.arm_name]}

    def solve_ik(self, target_pose, env_idx, robot):
        if self.fail_ik or (self.unreachable is not None and self.unreachable(target_pose)):
            return {"status": "Fail"}
        sign = -1.0 if target_pose[3] < 0 else 1.0
        return {"status": "Success", "joint_value": list(target_pose[:3]) + [sign * v for v in target_pose[4:7]]}


def sdk_gripper_limit(robot, position, measured):
    """MetaControl.get_action's gripper limiter: an explicit target at most 0.2 of the range from the finger."""
    span = robot.gripper_scale[1] - robot.gripper_scale[0]
    if abs(position - measured) / span < 0.2:
        value = position
    else:
        value = measured + span * 0.2 if position > measured else measured - span * 0.2
    return [value, value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]


class Plant:
    def __init__(self):
        self.robot_manager = Manager()
        camera_names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        mounts = [None, "robot0/camera", "robot1/camera"]
        self.camera_manager = SimpleNamespace(
            camera_names=[camera_names],
            cameras=[[SimpleNamespace(prim_path=f"/World/envs/env_0/{mount + '/' if mount else ''}Camera_{index}/optical")
                      for index, mount in enumerate(mounts)]],
            camera_config={name: SimpleNamespace(camera={"mount_link": mount})
                           for name, mount in zip(camera_names, mounts)},
        )
        self.sim = SimpleNamespace(_sim_step_counter=500)
        self.sim.sim = SimpleNamespace(current_time_step_index=500)
        self.take_action_cnt = [0]
        self.step_lim = 1050
        self.reward_registered = False
        # The upstream episode-lifecycle latches the round-isolation audit
        # measures: ``success`` is the failure latch (True until the evaluator
        # marks a failure) and ``end_flag`` the episode's own terminal flag.
        self.success = [True]
        self.end_flag = [False]
        self._official_terminal = None
        # The layout instances upstream registers as scene objects, and the
        # monotone id counter ``SceneManager.reload_env_scene`` regenerates
        # them from. ``rename_on_reset`` reproduces exactly that upstream
        # behaviour, so a test can watch the identity check fail closed.
        self.layout_generation = 0
        self.rename_on_reset = False
        self.scene_registration = None
        self.obs_manager = SimpleNamespace(instruction=["Pick up the displayed object by 10 cm."])
        self.terminal_at = None
        self.commands = []
        self.closed = False
        self.leak_clock = False

    def reset(self, seed):
        from ora.environment.benchmarks.robodojo import (
            build_scene_registration,
        )

        assert seed == [0]
        self.take_action_cnt = [0]
        self.reward_registered = True
        self.success = [True]
        self.end_flag = [False]
        self._official_terminal = None
        self.resets = getattr(self, "resets", 0) + 1
        # Upstream only reloads the saved layout on a *later* reset of a live
        # plant, so the first reset keeps generation 0 either way.
        if self.rename_on_reset and getattr(self, "resets", 0) > 1:
            self.layout_generation += 1
        suffix = self.layout_generation
        self.scene_registration = build_scene_registration(
            {f"mug_{suffix}": "rigid", f"rack_{suffix}": "geometry"},
            evaluator_registration={
                "check_list": True, "final_check_list": False,
                "trigger_check_list": False,
            },
            native_action_limit=self.step_lim,
        )
        self.robot_manager.reset_home()

    def native_tick(self, control):
        self.commands.append(deepcopy(control))
        self.take_action_cnt[0] += 1
        manager = self.robot_manager
        fraction = getattr(self, "step_fraction", 1.0)
        for robot in manager.robot_list:
            arm_key, ee_key = manager.process_name(robot.arm_name), manager.process_name(robot.gripper_name)
            if arm_key in control:
                manager.joints[robot.arm_name] = [
                    current + fraction * (target - current)
                    for current, target in zip(manager.joints[robot.arm_name], control[arm_key]["position"])
                ]
            if ee_key in control:
                limits = manager.robot_key[manager.robot_list.index(robot)].data.joint_pos_limits[0]
                manager.fingers[robot.arm_name] = [
                    max(float(limits[index][0]), value)
                    for index, value in zip(robot.gripper_joint_indices, control[ee_key]["position"])
                ]
        manager.control_manager.prev_control[0].update(deepcopy(control))
        for robot, articulation in zip(manager.robot_list, manager.robot_key):
            targets = manager.control_manager.prev_control[0]
            articulation.data.joint_pos_target[0] = (
                targets[manager.process_name(robot.arm_name)]["position"]
                + targets[manager.process_name(robot.gripper_name)]["position"]
            )
        for _ in range(10):
            self.sim._sim_step_counter += 1
            self.sim.sim.current_time_step_index += 1
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.control_manager.prev_control[0]))
        if self.take_action_cnt[0] == self.terminal_at:
            return {"official_success": True, "native_actions": self.take_action_cnt[0]}
        if self.take_action_cnt[0] == self.step_lim:
            return {"official_success": False, "native_actions": self.step_lim}
        return None

    def native_tick_sequence(self, requested):
        """``bridge.native_tick_sequence`` on the fake plant: one target per physics step.

        Each step resolves the requested keys into ``prev_control`` (velocity 0
        when absent; an explicit gripper target passes the SDK's own limiter,
        ``sdk_gripper_limit``), writes the articulation's position and velocity targets,
        then moves the plant: an arm joint goes ``track_gain`` of the way to its
        target (1.0 tracks exactly) unless ``blocked[arm]`` freezes it; a finger
        follows its target but stops at ``jaw_object[arm]`` (metres) when closing.
        """
        self.take_action_cnt[0] += 1
        manager = self.robot_manager
        gain = getattr(self, "track_gain", 1.0)
        blocked = getattr(self, "blocked", {})
        jaw_object = getattr(self, "jaw_object", {})
        self.sequence_steps = getattr(self, "sequence_steps", [])
        for index in range(10):
            request = deepcopy(dict(requested(index) if callable(requested) else requested[index]))
            self.sequence_steps.append(request)
            retained = manager.control_manager.prev_control[0]
            for key, value in request.items():
                position = list(value["position"])
                robot = next((robot for robot in manager.robot_list
                              if manager.process_name(robot.gripper_name) == key), None)
                if robot is not None:
                    position = sdk_gripper_limit(robot, position[0], manager.fingers[robot.arm_name][0])
                retained[key] = {"position": position, "velocity": deepcopy(value.get("velocity", 0))}
            for robot, articulation in zip(manager.robot_list, manager.robot_key):
                arm_key, ee_key = manager.process_name(robot.arm_name), manager.process_name(robot.gripper_name)
                arm_target = retained[arm_key]["position"]
                velocity = np.broadcast_to(np.asarray(retained[arm_key].get("velocity", 0), dtype=float), (6,))
                articulation.data.joint_pos_target[0] = list(arm_target) + list(retained[ee_key]["position"])
                articulation.data.joint_vel_target[0] = [*velocity, 0.0, 0.0]
                if robot.arm_name not in blocked:
                    manager.joints[robot.arm_name] = [
                        current + gain * (target - current)
                        for current, target in zip(manager.joints[robot.arm_name], arm_target)]
                limits = articulation.data.joint_pos_limits[0]
                floor = jaw_object.get(robot.arm_name)
                manager.fingers[robot.arm_name] = [
                    max(float(limits[i][0]), value if floor is None else max(floor, value))
                    for i, value in zip(robot.gripper_joint_indices, retained[ee_key]["position"])]
            self.sim._sim_step_counter += 1
            self.sim.sim.current_time_step_index += 1
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.control_manager.prev_control[0]))
        if self.take_action_cnt[0] == self.terminal_at:
            return {"official_success": True, "native_actions": self.take_action_cnt[0]}
        if self.take_action_cnt[0] == self.step_lim:
            return {"official_success": False, "native_actions": self.step_lim}
        return None

    def get_obs(self):
        self.sim._sim_step_counter += int(self.leak_clock)
        self.sim.sim.current_time_step_index += int(self.leak_clock)
        return {
            "object_gt": "must not reach the public manifest",
            "vision": {name: {"color": np.full((8, 10, 3), 100, dtype=np.uint8),
                              "depth": np.full((8, 10), 1.0, dtype=np.float32)}
                       for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        }

    def close(self):
        self.closed = True


def _fake_source_manifest(tmp_path):
    """A source manifest whose layout file really exists on disk.

    The round-isolation audit re-hashes the layout the session was started
    against, so the fixture has to bind a real file rather than a name.
    """

    layout = tmp_path / "layout" / "hang_mugs_0.json"
    layout.parent.mkdir(parents=True, exist_ok=True)
    if not layout.is_file():
        layout.write_text('{"objects": []}', encoding="utf-8")
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "source_revisions": {"robodojo": "1" * 40},
        "layout_file": {
            "path": str(layout),
            "sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        },
    }


def install_fake_sdk_candidate_boundary(monkeypatch):
    """Keep fake plant kinematics outside the real bounded candidate selector.

    The root ``conftest.py`` registers this as the ``fake_sdk_candidate_boundary``
    fixture; a module that drives the fake plant names it in its ``pytestmark``.
    """
    from ora.environment.backends.robodojo import ik, reachability
    from ora.environment.backends.robodojo.common import native_pose

    def candidates(planner, current, target, base, explicit_seeds=False):
        # The fake solve draws no seeds, so the kinematics query's explicit seeding changes nothing here.
        result = planner.manager.solve_ik(native_pose(target), 0, planner.robot)
        success = result["status"] == "Success"
        return {"joint_names": planner.robot.arm_joints_name,
                "solutions": [result.get("joint_value", current)], "success": [success],
                "feasible": [success], "position_tolerance_m": .001, "orientation_tolerance_rad": .02}

    def fk(session, arm, q):
        vector = list(q[3:6])
        return list(q[:3]) + vector + [math.sqrt(max(0.0, 1.0 - sum(x * x for x in vector)))]

    monkeypatch.setattr(ik, "sdk_candidates", candidates)
    monkeypatch.setattr(reachability, "fk_link6", fk)
    # The fake plant has no SDK collision spheres; the straight-path screen is
    # covered in test_robodojo_ik_continuity with planar and pinned X5 models.
    monkeypatch.setattr(ik, "self_collision_model",
                        lambda *a: SimpleNamespace(straight_path=lambda start, end: {"status": "clear"}))
    # Nor a second placed sphere model: the inter-arm screen is covered in
    # test_robodojo_inter_arm_clearance.
    monkeypatch.setattr(ik, "inter_arm_screen", lambda session: CLEAR_INTER_ARM)
    # Nor the SDK planner's table cuboid: the table screen is covered in test_robodojo_table_screen.
    from ora.environment.backends.robodojo import kinematics_query
    monkeypatch.setattr(kinematics_query, "table_screen", lambda session: CLEAR_TABLE)


#: A stand-in inter-arm screen that finds every lockstep path clear.
CLEAR_INTER_ARM = SimpleNamespace(straight_paths=lambda starts, ends: {
    "status": "clear", "min_clearance_m": 1.0, "required_clearance_m": 0.005, "start_clearance_m": 1.0,
    "limiting_links": {"left": "link6", "right": "link6"}, "limiting_fraction": 1.0,
    "moving_arms": sorted(arm for arm in starts if list(starts[arm]) != list(ends[arm]))})


#: A stand-in table screen that finds every joint path above the table.
CLEAR_TABLE = SimpleNamespace(straight_path=lambda arm, start, end: {"status": "clear", "min_clearance_m": 1.0})


def sparse_session(tmp_path, **config):
    plant = Plant()
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                "action_codec": "motion_action_v3", **config},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    instance.start()
    # Direct primitive callers have no RPC host. Supply its explicit recovery
    # goal selection here, outside the production worker.
    execute = instance.execute
    def with_goal(action):
        from ora.contracts.execution_state import FIELDS, pack, recovery_goals
        if hasattr(instance, "recovery_targets"):
            return execute(action)
        instance.recovery_targets = recovery_goals(action, pack({name: getattr(instance, name) for name in FIELDS}))
        try:
            return execute(action)
        finally:
            del instance.recovery_targets
    instance.execute = with_goal
    return instance


def pose(position, orientation=(0, 0, 0, 1)):
    return {"position_m": list(position), "orientation_xyzw": list(orientation)}


def v3(session, tmp_path, *, kind="move_pose", targets=None, interaction="free_space", plan_ref=None):
    observation, _ = session.observe(tmp_path / "observations")
    return v3_for(observation, session, kind=kind, targets=targets, interaction=interaction, plan_ref=plan_ref)


def v3_for(observation, session, *, kind="move_pose", targets=None, interaction="free_space", plan_ref=None):
    if plan_ref is not None:
        arguments = {"plan_ref": plan_ref}
    elif kind in {"move_pose", "move_free_space"}:
        arguments = {"targets": targets, "interactions": {arm: interaction for arm in targets}}
    else:
        arguments = {"targets": targets}
    return {"schema": "ora-motion-action.v3", "observation_id": observation["observation_id"],
            "action_id": f"action-{session.action_sequence}", "action": kind, "arguments": arguments,
            "bound_observation_sha256": "a" * 64}


def trajectory_to(session, tmp_path, targets, *, step=0.04):
    """One ``follow_joint_trajectory`` taking each arm to its position (the fake plant's joints 0-2 are its link6).

    The knots carry the link6 pose each should reach, so the receipt judges its endpoint as a toolkit move's does.
    """
    effectors = session._proprioception()["end_effectors"]
    starts = {arm: [float(value) for value in effectors[arm]["arm_joint_qpos"]] for arm in targets}
    goals = {arm: [*map(float, position), *starts[arm][3:]] for arm, position in targets.items()}
    count = max(1, max(math.ceil(max(abs(g - s) for g, s in zip(goals[arm], starts[arm])) / step) for arm in targets))
    plan = {}
    for arm in targets:
        knots = [[s + (g - s) * index / count for s, g in zip(starts[arm], goals[arm])] for index in range(1, count + 1)]
        plan[arm] = {"knots_rad": knots, "expected_link6_xyz_xyzw": [
            [*knot[:3], *knot[3:6], math.sqrt(max(0.0, 1.0 - sum(x * x for x in knot[3:6])))] for knot in knots]}
    action = v3(session, tmp_path, kind="follow_joint_trajectory", targets={arm: {} for arm in targets})
    action["arguments"] = {"arms": plan}
    return session.execute(action)


def probe_request(session, tmp_path, position, *, policy="exact", arm="left", orientation=(0, 0, 0, 1)):
    observation, _ = session.observe(tmp_path / "observations")
    return observation, {
        "schema": "ora.motion-probe-request.v1", "observation_id": observation["observation_id"],
        "observation_sha256": "a" * 64, "variant_policy": policy,
        "queries": [{"query_id": "q0", "arm": arm, "pose": pose(position, orientation)}],
    }
