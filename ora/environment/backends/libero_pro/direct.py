"""Single LIBERO simulator owned by the isolated Harness worker."""
from __future__ import annotations

from typing import Any, Callable

from ora.environment.benchmarks.libero_pro import PRACTICE_DRIVER

DRIVER = "direct_libero_v1"
#: The same driver on a play scene: the task's scene and initial state load,
#: and its official predicate (BDDL ``_check_success``) is withheld, so no step
#: reports success, reward or a success termination. The step limit stays.
PLAY_DRIVER = "direct_libero_play_v1"
#: The same driver on a practice episode (benchmarks.libero_pro): the scene is
#: placed by the BDDL region samplers from the episode seed, no official
#: initial state is loaded, and the official predicate stays.
DRIVERS = (DRIVER, PLAY_DRIVER, PRACTICE_DRIVER)


def _withheld_predicate() -> bool:
    return False


def withhold_official_predicate(environment: Any) -> Any:
    """Replace one constructed LIBERO environment's success predicate with false.

    Stock LIBERO's ``step`` sets ``done`` and its sparse reward from the BDDL
    problem's ``_check_success``; every Harness path (OSC steps, joint
    trajectories, replay, reset settling) reaches the simulator through that
    ``step``. Shadowing the predicate on the one problem instance therefore
    removes the success boundary everywhere without evaluating it.
    """

    problem = getattr(environment, "env", None)
    if problem is None or not callable(getattr(problem, "_check_success", None)):
        raise RuntimeError("LIBERO play scene has no BDDL problem predicate to withhold")
    problem._check_success = _withheld_predicate
    return environment


class DirectLiberoEnvironment:
    """Adapt stock LIBERO reset/step to the existing trusted Session port.

    There is no vector scheduler or auto-reset. The outer Runtime already owns
    process isolation, rounds and action admission. The tiny attribute port is
    private to the trusted Session/controller wrapper, never a policy tool.
    """

    def __init__(self, *, factory: Callable[[], Any], init_states: Any,
                 first_reset_id: int, reset_id: int, seed: int,
                 task_language: str, max_sim_steps: int, practice: bool = False) -> None:
        self._factory = factory
        # Practice: the BDDL region samplers place the scene from ``seed`` and
        # no official initial state is loaded (benchmarks.libero_pro
        # PRACTICE_INITIAL_STATE). ``initial_state_record`` says which kind
        # of state the last reset produced.
        self._practice = practice
        self.initial_state_record: dict[str, Any] | None = None
        self._delegate = None
        self._init_states = init_states
        self._first_reset_id = first_reset_id
        self.specific_reset_id = reset_id
        self.seed = seed
        self.task_descriptions = [task_language]
        self.max_sim_steps = max_sim_steps
        self._elapsed_steps = 0
        self.current_raw_obs = None
        self.env = self

    def reset(self, *, reset_state_ids=None):
        import numpy as np
        if reset_state_ids is not None:
            ids = np.asarray(reset_state_ids)
            if ids.shape != (1,):
                raise ValueError("direct LIBERO reset requires exactly one state id")
            self.specific_reset_id = int(ids[0])
        if self._delegate is None:
            self._delegate = self._factory()
        if self._practice:
            self._delegate.seed(self.seed)
            self._delegate.reset()
            self.initial_state_record = self._practice_record(self._delegate.get_sim_state())
        else:
            trial = self.specific_reset_id - self._first_reset_id
            if not 0 <= trial < len(self._init_states):
                raise ValueError("direct LIBERO reset state belongs to another task")
            self._delegate.seed(self.seed)
            self._delegate.reset()
            self._delegate.set_init_state(self._init_states[trial])
            self.initial_state_record = {"source": "official", "official": True, "init_state_index": trial}
        # Retain the source treatment's 15 settling steps with gripper open.
        # These are reset preparation, not policy actions or official outcomes.
        command = np.zeros(7, dtype=np.float32)
        command[-1] = -1
        for _ in range(15):
            raw, _reward, _done, _info = self._delegate.step(command)
        self.current_raw_obs = [dict(raw)]
        self._elapsed_steps = 0
        return self.current_raw_obs, {}

    def _practice_record(self, sampled: Any) -> dict[str, Any]:
        """Refuse a sampled state equal to an official one; describe the rest."""

        import hashlib

        import numpy as np

        state = np.asarray(sampled, dtype=np.float64)
        compared = 0
        for index, official in enumerate(self._init_states):
            official = np.asarray(official, dtype=np.float64)
            if official.shape != state.shape:
                continue
            compared += 1
            if np.allclose(official, state, atol=1e-6, rtol=0.0):
                raise ValueError(f"practice initial state reproduces official state {index}")
        if compared == 0:
            raise ValueError("practice initial state could not be compared with the official states")
        return {"source": "practice_sampled", "official": False, "practice_seed": int(self.seed),
                "sampled_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "official_states_compared": compared}

    def step(self, actions):
        import numpy as np
        command = np.asarray(actions)
        if command.shape != (1, 7) or not np.isfinite(command).all():
            raise ValueError("direct LIBERO action must be a finite 1x7 command")
        if self._delegate is None:
            raise RuntimeError("direct LIBERO environment must be reset before step")
        raw, reward, terminated, info = self._delegate.step(command[0])
        self.current_raw_obs = [dict(raw)]
        self._elapsed_steps += 1
        # Stock LIBERO's step returns its official success boundary. Keep the
        # independently owned time limit separate from that Boolean verdict.
        return (self.current_raw_obs, np.asarray([reward]), np.asarray([bool(terminated)]),
                np.asarray([self._elapsed_steps >= self.max_sim_steps]), dict(info))

    def get_env_attr(self, name: str):
        if self._delegate is None:
            raise RuntimeError("direct LIBERO worker has no environment")
        return [getattr(self._delegate.unwrapped, name)]

    def set_env_attr(self, name: str, value: Any):
        if self._delegate is None:
            raise RuntimeError("direct LIBERO worker has no environment")
        setattr(self._delegate.unwrapped, name, value)

    def close(self):
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            delegate.close()
