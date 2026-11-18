"""Test helpers for robotwin correctness repair, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robotwin/test_robotwin_correctness_repair.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ora.environment.backends.robotwin.session import RoboTwinLiveSession


ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = ROOT / 'ora/policy/embodiments/aloha-agilex/control-profile-v10.json'
PROFILE = json.loads(PROFILE_PATH.read_text())
WORKSPACE = {'x': [-1., 1.], 'y': [-1., 1.], 'z': [0., 2.]}


class Plant:
    def __init__(self, *, success_step=None, gain=1.):
        self.q = {arm: np.zeros(6) for arm in ('left', 'right')}
        self.targets = {arm: np.zeros(6) for arm in ('left', 'right')}
        self.steps = 0
        self.success_step = success_step
        self.gain = gain
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 100
        self.calls = []
        self.observation_reads = 0
        self.predicate_reads = 0
        self.scene = SimpleNamespace(step=self.step)
        self.robot = SimpleNamespace(set_arm_joints=self.write, set_gripper=self.write_gripper)
        for arm in ('left', 'right'):
            setattr(self.robot, f'{arm}_entity', SimpleNamespace(get_qpos=lambda a=arm: self.q[a].copy()))
            setattr(self.robot, f'{arm}_homestate', [0.] * 6)

    def write(self, position, velocity, arm):
        assert not self.eval_success, 'joint command after official terminal'
        self.calls.append(('joint', arm, self.steps))
        self.targets[arm] = np.asarray(position)

    def write_gripper(self, value, arm, per_step=0.):
        assert not self.eval_success, 'gripper command after official terminal'
        self.calls.append(('gripper', arm, self.steps))

    def set_gripper(self, set_tag, **kwargs):
        path = {'num_step': 300, 'per_step': .01, 'result': np.linspace(0., 1., 300)}
        return (copy.deepcopy(path), copy.deepcopy(path)) if set_tag == 'together' else path

    def step(self):
        assert not self.eval_success, 'physics after official terminal'
        self.steps += 1
        for arm in self.q:
            self.q[arm] += self.gain * (self.targets[arm] - self.q[arm])

    def check_success(self):
        self.predicate_reads += 1
        return self.steps == self.success_step

    def pose(self, arm, joints=None):
        q = self.q[arm] if joints is None else np.asarray(joints)
        xyz = q[:3] + np.asarray([-.2 if arm == 'left' else .2, 0., .8])
        return [*xyz, *Rotation.from_rotvec(q[3:]).as_quat()]

    def get_arm_pose(self, arm):
        p = self.pose(arm)
        return [*p[:3], p[6], *p[3:6]]

    def _update_render(self):
        pass

    def close_env(self, **kwargs):
        pass

    def get_obs(self):
        self.observation_reads += 1
        return {
            'observation': {'head_camera': {
                'rgb': np.full((3, 4, 3), self.steps, dtype=np.uint8),
                'depth': np.full((3, 4), 1000. + self.steps, dtype=np.float32),
                'intrinsic_cv': np.eye(3), 'extrinsic_cv': np.eye(4), 'cam2world_gl': np.eye(4),
            }},
            'endpose': {
                f'{arm}_{key}': (self.get_arm_pose(arm) if key == 'endpose' else .5)
                for arm in self.q for key in ('endpose', 'gripper')
            },
            'joint_action': {f'{arm}_arm': self.q[arm].copy() for arm in self.q},
        }


def live_session(tmp_path, plant):
    live = RoboTwinLiveSession(
        robotwin_root=tmp_path, output_root=tmp_path / 'native', task='hanging_mug',
        actual_seed=0, episode=0, action_codec='motion_action_v3',
        control_profile={'profile': copy.deepcopy(PROFILE), 'sha256': hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()},
        workspace_bounds_m=WORKSPACE, task_factory=lambda task: plant,
    )
    live.environment = plant
    live._measured_arm_joints = lambda env, arm: env.q[arm].tolist()
    live._public_pose_wxyz = lambda env, arm: env.get_arm_pose(arm)
    live._pinocchio_public_pose_xyzw = lambda env, arm, joints: env.pose(arm, joints)
    live._pinocchio_link6_pose_xyzw = lambda env, arm, joints: env.pose(arm, joints)
    live._pinocchio_chain_points = lambda env, arm, joints: [[-.3 if arm == 'left' else .3, 0., .8]]
    live._arm_joint_limits = lambda env, arm: [[-2., 2.]] * 6
    return live


def action(kind='set_gripper', arguments=None):
    return {
        'schema': 'ora-motion-action.v3', 'action_id': 'action-0',
        'observation_id': 'obs-0', 'bound_observation_sha256': '0' * 64,
        'action': kind, 'arguments': arguments or {'targets': {'left': 'open'}},
    }


def branch_candidates(arm='left'):
    # Both IK branches have the same public TCP endpoint. The short one sweeps
    # an elbow through the peer; the longer one goes around it.
    sign = -1. if arm == 'left' else 1.
    positions = [
        [[0, 0, 0, 0, 0, 0], [.01, 0, sign * .2, 0, 0, 0], [.02, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0], [.006, 0, -sign * .2, 0, 0, 0], [.014, 0, -sign * .2, 0, 0, 0], [.02, 0, 0, 0, 0, 0]],
    ]
    return {'schema': 'robotwin.curobo-plan-result.v2', 'status': 'Success',
            'joint_names': PROFILE['arms'][arm]['planner_joint_names'],
            'candidates': [
                {'schema': 'robotwin.curobo-plan-result.v1', 'status': 'Success',
                 'planner_status': 'SUCCESS', 'ik_candidate_index': index,
                 'position': np.asarray(q, dtype=float), 'velocity': np.zeros_like(q, dtype=float),
                 'interpolation_dt_s': .01}
                for index, q in enumerate(positions)
            ]}


def vendored_planner_with_fake_solver(error, *, all_candidates_error=False):
    tree = ast.parse((ROOT/'ora/environment/backends/robotwin/vendored/planner.py').read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'CuroboPlanner')
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cuda(self): return self
        def reshape(self, *shape): return Tensor(self.data.reshape(*shape))
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
        def detach(self): return self
        def to(self, *args): return self
        def numpy(self): return self.data
    class IK:
        def __init__(self, cfg):
            self.kinematics = SimpleNamespace(get_joint_limits=lambda: SimpleNamespace(position=[[-1] * 6, [1] * 6]))
        def solve_single(self, *args, **kwargs):
            return SimpleNamespace(success=[True, True], solution=Tensor([[.1, 0, 0, 0, 0, 0], [-.1, .2, 0, 0, 0, 0]]), position_error=[0, 0], rotation_error=[0, 0])
    class MotionGen:
        tensor_args = object()
        def __init__(self): self.calls = 0
        def plan_single_js(self, *args):
            self.calls += 1
            if self.calls > 1 or all_candidates_error:
                raise error('candidate solver exception')
            return {'status': 'Success', 'position': np.zeros((2, 6)), 'velocity': np.zeros((2, 6)), 'interpolation_dt_s': .01, 'timings_s': {'total': .001}}
    namespace = {
        'np': np, 'time': time, 'IKSolver': IK,
        'IKSolverConfig': SimpleNamespace(load_from_robot_config=lambda *args, **kwargs: None),
        'torch': SimpleNamespace(tensor=lambda data, **kwargs: Tensor(data), float32='float32'),
        'JointState': SimpleNamespace(from_position=lambda *args, **kwargs: object()),
        'MotionGenPlanConfig': lambda **kwargs: kwargs, '_planner_value': lambda value: value,
        '_motion_result': lambda result, **kwargs: result,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<production-vendored-planner>', 'exec'), namespace)
    planner = object.__new__(namespace['CuroboPlanner'])
    planner.motion_gen_config = SimpleNamespace(robot_cfg={})
    planner.world_config = {}
    planner.active_joints_name = PROFILE['arms']['left']['planner_joint_names']
    planner.motion_gen = MotionGen()
    return planner
