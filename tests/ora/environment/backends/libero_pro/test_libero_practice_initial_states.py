"""LIBERO practice episodes sample their initial state; every other episode is in the evaluation set."""
from __future__ import annotations

import numpy as np
import pytest

from ora.environment.backends.libero_pro.direct import DRIVERS, DirectLiberoEnvironment
from ora.environment.benchmarks import evaluation_sets as sets
from ora.environment.benchmarks.libero_pro import (
    OFFICIAL_CORE_INIT_STATES, PRACTICE_DRIVER, PRACTICE_SEED_FLOOR,
)
from ora.environment.benchmarks.robocasa import SPLIT as ROBOCASA_SPLIT
from ora.environment.benchmarks.robotwin import evaluation_protocol
from ora.orchestrator.backends.libero_pro.configuration import validate_libero_configuration
from ora.orchestrator.spec.values import RunSpecError

OFFICIAL = [np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])]


class ControlEnv:
    """``OffScreenRenderEnv``: seeding then reset places the scene from the global RNG."""

    def __init__(self):
        self.seeded, self.loaded = [], []
        self.state = None

    def seed(self, value):
        self.seeded.append(value)
        self.state = np.array([float(value % 7), 0.5, 0.25])

    def reset(self): pass
    def set_init_state(self, value): self.loaded.append(value)
    def get_sim_state(self): return self.state
    def step(self, action): return {"image": np.zeros((2, 2, 3))}, 0.0, False, {}


def _driver(scene, *, practice, seed):
    return DirectLiberoEnvironment(factory=lambda: scene, init_states=OFFICIAL, first_reset_id=0, reset_id=1,
                                   seed=seed, task_language="task", max_sim_steps=20, practice=practice)


def test_a_practice_reset_samples_from_its_seed_and_loads_no_official_state():
    scene = ControlEnv()
    practice = _driver(scene, practice=True, seed=PRACTICE_SEED_FLOOR + 3)
    practice.reset()
    record = practice.initial_state_record
    assert scene.seeded == [PRACTICE_SEED_FLOOR + 3] and scene.loaded == []
    assert record["source"] == "practice_sampled" and record["official"] is False
    assert record["practice_seed"] == PRACTICE_SEED_FLOOR + 3 and record["official_states_compared"] == 2
    assert len(record["sampled_state_sha256"]) == 64
    official = _driver(ControlEnv(), practice=False, seed=1)
    official.reset()
    assert official.initial_state_record == {"source": "official", "official": True, "init_state_index": 1}
    assert PRACTICE_DRIVER in DRIVERS


def test_a_sample_equal_to_an_official_state_is_refused():
    scene = ControlEnv()
    scene.seed = lambda value: setattr(scene, "state", OFFICIAL[1].copy())
    with pytest.raises(ValueError, match="reproduces official state 1"):
        _driver(scene, practice=True, seed=PRACTICE_SEED_FLOOR).reset()


def test_runspec_resolution_refuses_a_low_practice_seed_and_robustness_cases():
    runtime = {"transport": "local_rpc", "environment_driver": PRACTICE_DRIVER}
    # A practice seed passes the practice checks and reaches the ordinary runtime fields.
    with pytest.raises(RunSpecError, match="python_env"):
        validate_libero_configuration(dict(runtime), seed=PRACTICE_SEED_FLOOR)
    with pytest.raises(RunSpecError, match="seed of at least"):
        validate_libero_configuration(dict(runtime), seed=49)
    with pytest.raises(RunSpecError, match="robustness"):
        validate_libero_configuration({**runtime, "robustness_case": "object"}, seed=PRACTICE_SEED_FLOOR)


@pytest.mark.parametrize("plan, hit", [
    ({"seed": 11, "runtime": {"kind": "libero_pro", "environment_driver": "direct_libero_v1"}}, True),
    ({"seed": 11, "runtime": {"kind": "libero_pro"}}, True),
    ({"seed": 2_000_000, "runtime": {"kind": "libero_pro", "environment_driver": PRACTICE_DRIVER}}, False),
    ({"seed": 100001, "runtime": {"kind": "robotwin"}}, True),
    ({"seed": 5, "runtime": {"kind": "robotwin"}}, False),
    ({"seed": 0, "runtime": {"kind": "robocasa", "split": "target"}}, True),
    ({"seed": 0, "runtime": {"kind": "robocasa"}}, True),
    ({"seed": 0, "runtime": {"kind": "robocasa", "split": "pretrain"}}, False),
    ({"seed": 0, "runtime": {"kind": "robolab"}}, True),
])
def test_evaluation_set_hits(plan, hit):
    assert (sets.evaluation_set_hit(plan) is not None) is hit


def test_the_manifest_agrees_with_the_code_it_describes():
    environments = sets.evaluation_sets()["environments"]
    libero = environments["libero_pro"]
    assert libero["match"]["practice_values"] == [PRACTICE_DRIVER]
    assert libero["practice"]["runtime.environment_driver"] == PRACTICE_DRIVER
    assert libero["practice"]["seed_minimum"] == PRACTICE_SEED_FLOOR
    assert libero["evaluation_set"]["official_initial_states_per_task"] == OFFICIAL_CORE_INIT_STATES
    assert libero["evaluation_set"]["official_initial_state_indices"] == [0, OFFICIAL_CORE_INIT_STATES - 1]
    robotwin = environments["robotwin"]
    assert robotwin["match"]["evaluation_minimum"] == evaluation_protocol()["start_requested_seed"]
    assert robotwin["evaluation_set"]["seed_minimum"] == robotwin["practice"]["seed_maximum_exclusive"]
    assert environments["robocasa"]["match"]["evaluation_values"] == [ROBOCASA_SPLIT]
