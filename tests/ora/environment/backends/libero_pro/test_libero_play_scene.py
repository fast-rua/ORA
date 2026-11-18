"""A LIBERO play scene: the task's scene loads and its official predicate is withheld."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ora.environment.backends.libero_pro.direct import (
    DRIVER, PLAY_DRIVER, DirectLiberoEnvironment, withhold_official_predicate,
)
from ora.environment.backends.libero_pro.session import LiberoProError, LiberoProSession


class Problem:
    """Stock LIBERO's BDDL problem: ``step`` returns ``done = _check_success()``."""

    def __init__(self):
        self.predicate_reads = 0

    def _check_success(self):
        self.predicate_reads += 1
        return True

    def step(self, action):
        done = self._check_success()
        return {"image": np.zeros((2, 2, 3))}, float(done), done, {}


class ControlEnv:
    """``OffScreenRenderEnv``: delegates to its one problem instance."""

    def __init__(self):
        self.env = Problem()
        self.unwrapped = self

    def seed(self, value): pass
    def reset(self): pass
    def set_init_state(self, value): pass
    def step(self, action): return self.env.step(action)
    def check_success(self): return self.env._check_success()


def _driver(scene):
    return DirectLiberoEnvironment(factory=lambda: scene, init_states=["a"], first_reset_id=0,
                                   reset_id=0, seed=0, task_language="task", max_sim_steps=20)


def test_the_play_driver_never_reports_success_or_reward_and_keeps_the_step_limit():
    benchmark = _driver(ControlEnv())
    benchmark.reset()
    assert benchmark.step(np.zeros((1, 7)))[2][0]  # the stock boundary is unchanged
    scene = withhold_official_predicate(ControlEnv())
    play = _driver(scene)
    play.reset()  # 15 settling steps
    for _ in range(19):
        _, reward, done, truncated, _ = play.step(np.zeros((1, 7)))
        assert reward[0] == 0.0 and not done[0] and not truncated[0]
    assert play.step(np.zeros((1, 7)))[3][0]  # the step limit still ends a scene
    assert scene.check_success() is False and scene.env.predicate_reads == 0
    with pytest.raises(RuntimeError, match="no BDDL problem predicate"):
        withhold_official_predicate(object())


def _play_session(tmp_path, *, driver=PLAY_DRIVER):
    session = object.__new__(LiberoProSession)
    session.environment_driver = driver
    session.output_root = tmp_path
    session.execution_capture = {"enabled": False}
    session.suite, session.task_id, session.seed = "libero_goal", 7, 3
    session.sim_steps = session.round_index = session.action_sequence = 0
    session.max_sim_steps = 100
    session.terminated = session.truncated = False
    session.first_termination = session.abort_reason = session.control_fault_reason = None
    session.robustness_case = session.robustness_episode = None
    return session


def test_a_play_session_persists_a_null_verdict_and_fails_closed_on_a_reported_predicate(tmp_path):
    session = _play_session(tmp_path)
    assert session.play_scene and DRIVER != PLAY_DRIVER
    action = {"action_id": "a-1"}
    session._latch_environment_outcome(terminated=np.asarray([False]), truncated=np.asarray([False]),
                                       steps=4, action=action, controller="OSC_POSE")
    session.abort("operator_quit")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is None and verdict["sim_steps"] == 4
    assert "withheld" in verdict["source"]
    assert json.loads((tmp_path / "official-verdict.json").read_text())["official_success"] is None
    with pytest.raises(LiberoProError, match="withholds"):
        session._latch_environment_outcome(terminated=np.asarray([True]), truncated=np.asarray([False]),
                                           steps=1, action=action, controller="OSC_POSE")
    benchmark = _play_session(tmp_path / "benchmark", driver=DRIVER)
    benchmark._latch_environment_outcome(terminated=np.asarray([True]), truncated=np.asarray([False]),
                                         steps=1, action=action, controller="OSC_POSE")
    assert benchmark.persist_official_verdict()["official_success"] is True
