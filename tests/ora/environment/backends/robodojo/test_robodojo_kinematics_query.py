"""The read-only kinematics query and the IK seed generator as episode state.

CuRobo needs a GPU, so the solvers here are stubs with CuRobo's two seeding
shapes: the SDK's default LM mode (``use_lm_seed=True``: a ``SeedIKSolver``
whose ``act_sample_gen`` completes the current joints with bounded draws, and
no seed manager on the path) and the plain mode (a seed manager whose
``prepare_action_seeds`` completes a partial ``seed_config``).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ora.contracts.kinematics import REQUEST_SCHEMA, validate_request
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.backends.robodojo import ik, ik_sampler, kinematics_query

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")
#: The real candidate solve, before the fake plant's boundary fixture replaces it.
SDK_CANDIDATES = ik.sdk_candidates

HOME = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]


class StubGenerator:
    """An index generator: its state is a counter, every draw advances it."""

    def __init__(self):
        self.state = np.zeros(8, dtype=np.uint8)

    def get_state(self):
        return self.state.copy()

    def set_state(self, value):
        self.state = np.asarray(value, dtype=np.uint8).copy()

    def draw(self, count):
        start = int.from_bytes(self.state.tobytes(), "little")
        self.state = np.frombuffer((start + count).to_bytes(8, "little"), dtype=np.uint8).copy()
        return [(start + index) * 0.001 for index in range(count)]


class StubSeedManager:
    def __init__(self):
        self.action_sample_generator = SimpleNamespace(_int_gen=StubGenerator(), sequencer=SimpleNamespace(seed=123))

    def prepare_action_seeds(self, batch_size, num_seeds, seed_config=None):
        provided = [] if seed_config is None else [list(row) for row in np.asarray(seed_config).reshape(-1, 6)]
        drawn = [[value] * 6 for value in self.action_sample_generator._int_gen.draw(num_seeds - len(provided))]
        return np.asarray(provided + drawn).reshape(-1, 1, 6)


class StubSolver:
    """CuRobo's IKSolver seeding: without seed_config, the current joints then internal draws."""

    def __init__(self):
        self.config = SimpleNamespace(num_seeds=4, use_lm_seed=False)
        self.seed_manager = StubSeedManager()
        self.used = []

    def solve_pose(self, current_state, seed_config=None):
        if seed_config is None:
            seed_config = self.seed_manager.prepare_action_seeds(1, 4, current_state.position.reshape(1, 1, -1))
        self.used.append(np.asarray(seed_config).reshape(-1, 6).tolist())
        return self.used[-1]


class StubSampleBuffer:
    """CuRobo's SampleBuffer: an index generator (``_int_gen``) and bounded draws from the stored samples."""

    def __init__(self, seed):
        self._int_gen, self.sequencer = StubGenerator(), SimpleNamespace(seed=seed)

    def get_samples(self, num_samples, bounded=False):
        return np.asarray([[value] * 6 for value in self._int_gen.draw(num_samples)])

    def reset(self):
        self._int_gen.set_state(np.zeros(8, dtype=np.uint8))


class StubSeedIKSolver:
    """CuRobo's SeedIKSolver: ``_generate_seed_configs`` completes a partial seed_config from act_sample_gen."""

    def __init__(self, num_seeds=8):
        self.config = SimpleNamespace(num_seeds=num_seeds)
        self.act_sample_gen = StubSampleBuffer(451)

    def generate(self, seed_config):
        provided = np.asarray(seed_config, dtype=float).reshape(1, -1, 6)
        if provided.shape[1] > self.config.num_seeds:
            raise ValueError("too many seeds")
        if provided.shape[1] == self.config.num_seeds:
            return provided
        extra = self.act_sample_gen.get_samples(self.config.num_seeds - provided.shape[1], bounded=True)
        return np.concatenate([provided, extra.reshape(1, -1, 6)], axis=1)


class StubLMSolver:
    """CuRobo's IKSolver as the SDK runs it (use_lm_seed=True): no seed manager on the seeding path; the
    seed solver draws and hands the main solver a full seed_config (its best num_seeds; here its first)."""

    def __init__(self):
        self.config = SimpleNamespace(num_seeds=4, use_lm_seed=True, max_batch_size=1)
        self.seed_ik_solver = StubSeedIKSolver()
        self.used = []

    def solve_pose(self, current_state, seed_config=None):
        if seed_config is None:
            seed_config = current_state.position.reshape(1, 1, -1)
        seeds = self.seed_ik_solver.generate(seed_config)
        self.used.append(seeds.reshape(-1, 6).tolist())
        return self.used[-1][:self.config.num_seeds]


SOLVERS = {"lm_seed_solver": StubLMSolver, "seed_manager": StubSolver}


def _generator(solver):
    if isinstance(solver, StubLMSolver):
        return solver.seed_ik_solver.act_sample_gen._int_gen
    return solver.seed_manager.action_sample_generator._int_gen


@pytest.mark.parametrize("mode", sorted(SOLVERS))
def test_explicit_seeds_equal_the_default_draw_for_the_same_generator_state(mode):
    current = SimpleNamespace(position=np.asarray(HOME))
    default, explicit = SOLVERS[mode](), SOLVERS[mode]()
    planner = SimpleNamespace(ik_solver=explicit)
    total = 8 if mode == "lm_seed_solver" else 4
    for _ in range(3):  # the sequence advances the same way call after call
        default.solve_pose(current)
        seeds, seeding = ik._explicit_seed_config(planner, current, 4)
        explicit.solve_pose(current, seed_config=seeds)
        assert explicit.used[-1] == default.used[-1]
        assert ik.generator_state(_generator(explicit)) == ik.generator_state(_generator(default))
        assert seeding["rule"] == ik.SEEDING_EXPLICIT and seeding["seeds_passed"] == total
        assert seeding["generator"] == ("seed_ik_solver" if mode == "lm_seed_solver" else "ik_solver")
    # Restoring the recorded state before a draw reproduces that draw without the history.
    recorded = seeding["generator_state_before_hex"]
    ik.restore_generator_state(planner, recorded)
    again, _ = ik._explicit_seed_config(planner, current, 4)
    assert np.asarray(again).tolist() == np.asarray(seeds).tolist()


def test_a_solver_without_a_seed_generator_is_a_typed_refusal():
    current = SimpleNamespace(position=np.asarray(HOME))
    bare = StubLMSolver()
    del bare.seed_ik_solver.act_sample_gen
    for solver in (bare, SimpleNamespace(config=SimpleNamespace(num_seeds=4, use_lm_seed=False))):
        with pytest.raises(EnvironmentRefusal, match="seeding_unavailable"):
            ik._explicit_seed_config(SimpleNamespace(ik_solver=solver), current, 4)
        with pytest.raises(EnvironmentRefusal, match="seeding_unavailable"):
            ik.restore_generator_state(SimpleNamespace(ik_solver=solver), "00")


def test_sdk_candidates_record_the_generator_state_before_and_after_the_solve(monkeypatch):
    import sys
    from types import ModuleType

    transformer = ModuleType("utils.transformer")
    transformer.calculate_target_pose = lambda base, origin, pose: pose
    monkeypatch.setitem(sys.modules, "utils", ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.transformer", transformer)
    solver = StubLMSolver()
    solver.config.position_tolerance, solver.config.orientation_tolerance = 0.001, 0.02

    def solve_pose(goal_tool_poses, current_state, seed_config=None, return_seeds=1):
        StubLMSolver.solve_pose(solver, current_state, seed_config)
        names = [f"joint{index}" for index in range(1, 7)]
        return SimpleNamespace(js_solution=SimpleNamespace(joint_names=names,
                                                           position=np.zeros((1, return_seeds, 6))),
                               success=np.ones((1, return_seeds), bool), feasible=np.ones((1, return_seeds), bool))

    solver.solve_pose = solve_pose
    planner = SimpleNamespace(
        ik_solver=solver, robot_origin_pose=[0.0] * 3 + [1.0, 0.0, 0.0, 0.0], frame_bias=[0.0] * 3,
        active_joints_name=[f"joint{index}" for index in range(1, 7)],
        _trans_from_world_to_base=lambda origin, target: (np.asarray(target[:3]), np.asarray(target[3:])),
        _build_joint_state=lambda joints: SimpleNamespace(position=np.asarray(joints, dtype=float)),
        _build_goal_pose=lambda position, quaternion: None)
    before = ik.generator_state(_generator(solver))
    seeding = SDK_CANDIDATES(planner, HOME, [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], [0.0] * 7,
                             explicit_seeds=True)["seeding"]
    after = ik.generator_state(_generator(solver))
    assert (seeding["generator_state_before_hex"], seeding["generator_state_after_hex"]) == (before, after)
    assert before != after and seeding["generator"] == "seed_ik_solver"


def _session_with_planner(sparse, mode="seed_manager"):
    planner = SimpleNamespace(ik_solver=SOLVERS[mode]())
    for robot in sparse.robots.values():
        robot.robot_name = "x5"
    sparse.environment.robot_manager.ik_solver = {"x5": planner}
    return planner


@pytest.mark.parametrize("mode", sorted(SOLVERS))
def test_the_sampler_state_round_trips_through_the_execution_state(sparse, mode):
    planner = _session_with_planner(sparse, mode)
    generator = _generator(planner.ik_solver)
    generator.draw(5)
    state = ik_sampler.sampler_states(sparse)
    expected = (451, "seed_ik_solver") if mode == "lm_seed_solver" else (123, "ik_solver")
    assert (state["planners"]["x5"]["random_seed"], state["planners"]["x5"]["generator"]) == expected
    generator.draw(7)
    assert ik_sampler.restore_sampler_states(sparse, state) == ["x5"]
    assert ik_sampler.sampler_states(sparse) == state
    assert ik_sampler.resume(sparse, None) == ik_sampler.UNRECORDED
    assert ik_sampler.provenance(sparse)["ik_sampler_state"] == "unrecorded"


def _candidates(solutions):
    def candidates(planner, current, pose, base, explicit_seeds=False):
        return {"joint_names": [f"joint{i}" for i in range(1, 7)], "solutions": solutions(pose),
                "success": [True] * len(solutions(pose)), "feasible": [True] * len(solutions(pose)),
                "position_tolerance_m": 0.001, "orientation_tolerance_rad": 0.02,
                "seeding": {"rule": ik.SEEDING_EXPLICIT if explicit_seeds else ik.SEEDING_INTERNAL}}
    return candidates


def _request(poses, **overrides):
    return {"schema": REQUEST_SCHEMA, "operation": "ik_chain", "arm": "left", "seed_joints_rad": None,
            "poses_xyz_xyzw": poses, "ranking": "nearest", "max_joint_step_rad": 0.1, "screen": "none", **overrides}


def test_an_ik_chain_takes_the_nearest_solution_and_reports_a_branch_jump(sparse, monkeypatch):
    poses = [[0.0, 0.0, 0.8 + 0.01 * k, 0.0, 0.0, 0.0, 1.0] for k in range(1, 4)]

    def solutions(pose):
        near = [pose[0], pose[1], pose[2], 0.0, 0.0, 0.0]
        return [[value + 2.0 for value in near], near] if pose[2] < 0.825 else [[value + 2.0 for value in near]]

    monkeypatch.setattr(ik, "sdk_candidates", _candidates(solutions))
    answer = kinematics_query.answer(sparse, _request(poses))
    rows = answer["rows"]
    assert [row["status"] for row in rows] == ["ok", "ok", "discontinuous"]
    assert rows[1]["joints_rad"][2] == pytest.approx(0.82) and rows[0]["seeding"]["rule"] == ik.SEEDING_EXPLICIT
    assert answer["first_failure_index"] == 2 and answer["seeding"] == kinematics_query.SEEDING_RULE


def test_requests_are_validated_and_other_backends_refuse(sparse):
    with pytest.raises(EnvironmentRefusal):
        validate_request(_request([[0.0] * 7], ranking="anything"))
    from ora.environment.controllers.alignment import plan_alignment
    with pytest.raises(EnvironmentRefusal):
        plan_alignment(_request([[0.0, 0.0, 0.8, 0, 0, 0, 1]]), adapter=SimpleNamespace(),
                       control_profile={}, available_arms=("left",))


def test_the_kinematics_query_never_advances_physics(sparse, monkeypatch):
    monkeypatch.setattr(ik, "sdk_candidates", _candidates(lambda pose: [[*pose[:3], 0.0, 0.0, 0.0]]))
    clock = sparse._clock()
    kinematics_query.answer(sparse, _request([[0.0, 0.0, 0.81, 0, 0, 0, 1]]))
    assert sparse._clock() == clock


@pytest.mark.parametrize("mode", sorted(SOLVERS))
def test_every_request_reports_the_held_joint_target_and_the_sampler_state_and_restores_it(sparse, mode):
    from ora.contracts.execution_state import FIELDS, SCHEMA, pack

    planner = _session_with_planner(sparse, mode)
    generator = _generator(planner.ik_solver)
    fields = {name: getattr(sparse, name) for name in FIELDS}
    for name in FIELDS:
        delattr(sparse, name)
    request = sparse.execution_request({"schema": SCHEMA, "values": pack(fields)})
    with request:
        generator.draw(3)
    values = request.updates["values"]
    assert values["exec.command.left"]["joint_target_rad"] == pytest.approx(HOME)
    recorded = values["exec.ik_sampler_state"]
    generator.draw(9)
    with sparse.execution_request({"schema": SCHEMA, "values": values}):
        assert ik_sampler.sampler_states(sparse) == recorded

