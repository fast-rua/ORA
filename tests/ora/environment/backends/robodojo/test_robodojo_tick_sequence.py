"""``tick_sequence.run_native_tick_sequence`` itself, not the fake plant's stand-in.

The environment below reproduces the SDK pieces the tick drives: the control
queue (``push``/``pop``/``get_empty``), ``MetaControl.get_action`` with the
SDK's gripper limiter (an explicit jaw target at most 0.2 of its range from the
measured finger) and ``None`` for dimensions absent from a proposal, the
support-arm queue, the task step, the reward step and the official terminal.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ora.environment.backends.robodojo.bridge import RobodojoError
from ora.environment.backends.robodojo.tick_sequence import run_native_tick_sequence

ARM, EE, SUPPORT = "left_arm_joint_state", "left_ee_joint_state", "support_arm_joint_state"
SPAN = 0.054


class MetaControl:
    def __init__(self, control, environment):
        self.control, self.environment = control, environment

    def get_action(self, robot_manager, env_idx):
        result = {}
        for key in (ARM, EE, SUPPORT):
            if key not in self.control:
                result[key] = None
            elif key == EE:
                self.environment.finger_reads += 1
                position, measured = self.control[key]["position"][0], self.environment.finger
                value = position if abs(position - measured) / SPAN < 0.2 else (
                    measured + 0.2 * SPAN if position > measured else measured - 0.2 * SPAN)
                result[key] = {"position": [value, value], "velocity": 0}
            else:
                result[key] = {"position": list(self.control[key]["position"]),
                               "velocity": self.control[key].get("velocity", 0)}
        return result


class ControlManager:
    def __init__(self, environment):
        self.environment, self.queue = environment, []
        self.prev_control = [{ARM: {"position": [0.0] * 6, "velocity": 0}, EE: {"position": [0.03, 0.03]},
                              SUPPORT: {"position": [9.0] * 6}}]

    def push(self, env_ids, controls):
        self.queue.extend(MetaControl(control, self.environment) for control in controls[0])

    def pop(self, env_ids):
        """``ControlManager.pop``: absent dimensions filled from ``prev_control``, the resolved action left there."""
        filled = self.queue.pop(0).get_action(None, 0)
        for key, value in filled.items():
            if value is None:
                filled[key] = self.prev_control[0][key]
        control = MetaControl(filled, self.environment)
        for key, value in control.get_action(None, 0).items():
            if value is not None:
                self.prev_control[0][key] = value
        return [control]

    def get_empty(self, env_ids):
        return not self.queue


class Environment:
    def __init__(self):
        self.finger, self.applied, self.events, self.finger_reads = 0.03, [], [], 0
        self.control_manager = ControlManager(self)
        self.robot_manager = SimpleNamespace(control_manager=self.control_manager,
                                             robot_list=[SimpleNamespace(type="target"),
                                                         SimpleNamespace(type="support")],
                                             get_robot_obs_name=lambda: [ARM, EE, SUPPORT])
        self._official_terminal, self.end_flag, self.reward_registered = None, [False], True
        self.take_action_cnt, self.step_lim = [0], 100
        self.native_action_ceiling = 100
        self.obs_manager = SimpleNamespace(collect_interval=10)
        self.unstable_envs = []
        self.reward_manager = SimpleNamespace(step=lambda env_idx_list: self.events.append("reward"))
        self.support_arm_action = [[{SUPPORT: {"position": [float(i)] * 6}} for i in range(3)]]
        self.interact = False
        self.terminal = None

    def step(self, meta_control_list):
        action = meta_control_list[0].get_action(self.robot_manager, env_idx=0)
        self.applied.append(deepcopy(action))

    def sim_step(self, render):
        self.events.append("sim_step")

    def advance_official_terminal(self):
        return self.terminal


def _step(index):
    return {ARM: {"position": [0.01 * (index + 1)] * 6, "velocity": [0.25] * 6}}


def test_each_physics_step_resolves_the_caller_target_with_the_official_lifecycle():
    environment = Environment()
    traces = []
    environment.controller_trace_callback = traces.append
    assert run_native_tick_sequence(environment, _step, environment.step) is None
    assert environment.take_action_cnt == [1] and len(environment.applied) == 10
    # The caller's arm target and velocity reach the task step at every physics step.
    assert [step[ARM]["position"][0] for step in environment.applied] == pytest.approx(
        [0.01 * (k + 1) for k in range(10)])
    assert environment.applied[0][ARM]["velocity"] == [0.25] * 6
    # A dimension the proposal does not name keeps the held target, not the SDK's None.
    assert all(step[EE] == {"position": [0.03, 0.03]} for step in environment.applied)
    # The support-arm queue is popped at the tick start, as interpolate pops it, and merged in.
    assert [step[SUPPORT]["position"][0] for step in environment.applied[:3]] == [0.0, 1.0, 2.0]
    assert environment.support_arm_action == [[]]
    assert environment.events.count("sim_step") == 10 and environment.events[-1] == "reward"
    assert len(traces) == 10 and environment.control_manager.prev_control[0][ARM]["position"][0] == pytest.approx(0.1)


def test_an_explicit_jaw_target_passes_the_sdk_limiter():
    environment = Environment()
    run_native_tick_sequence(environment, [{EE: {"position": [0.0, 0.0]}}] * 10, environment.step)
    # 0.03 m closing to 0: the SDK holds the target 0.2 of the range (10.8 mm) from the finger.
    assert environment.applied[0][EE]["position"][0] == pytest.approx(0.03 - 0.2 * SPAN)
    assert environment.control_manager.prev_control[0][EE]["position"][0] == pytest.approx(0.03 - 0.2 * SPAN)


def test_the_official_terminal_is_returned_and_ends_further_actions():
    environment = Environment()
    environment.terminal = {"official_success": True, "native_actions": 1}
    assert run_native_tick_sequence(environment, _step, environment.step) == environment.terminal
    environment._official_terminal = environment.terminal
    with pytest.raises(RobodojoError, match="officially terminated"):
        run_native_tick_sequence(environment, _step, environment.step)


def test_malformed_ticks_and_an_undrained_queue_fail_closed():
    environment = Environment()
    with pytest.raises(RobodojoError, match="exactly 10"):
        run_native_tick_sequence(environment, [_step(0)] * 9, environment.step)
    environment = Environment()
    environment.control_manager.get_empty = lambda env_ids: False
    with pytest.raises(RobodojoError, match="did not drain"):
        run_native_tick_sequence(environment, _step, environment.step)
    environment = Environment()
    environment.take_action_cnt = [100]
    with pytest.raises(RobodojoError, match="native action limit"):
        run_native_tick_sequence(environment, _step, environment.step)


@pytest.mark.parametrize("requested", [_step, [{EE: {"position": [0.0, 0.0]}}] * 10,
                                       [{ARM: {"position": [0.2] * 6}, EE: {"position": [0.05, 0.05]}}] * 10])
def test_reusing_the_action_pop_resolved_changes_nothing_and_reads_the_fingers_less(monkeypatch, requested):
    from ora.environment.backends.robodojo import tick_sequence

    def run(reuse):
        environment, traces = Environment(), []
        environment.controller_trace_callback = lambda command: traces.append(deepcopy(command))
        if not reuse:
            original = tick_sequence._fixed_meta_control
            monkeypatch.setattr(tick_sequence, "_fixed_meta_control", lambda *args: original(*args[:4]))
        run_native_tick_sequence(environment, requested, environment.step)
        monkeypatch.undo()
        return environment, traces

    reused, reused_traces = run(True)
    recomputed, recomputed_traces = run(False)
    assert reused.applied == recomputed.applied and reused_traces == recomputed_traces
    assert reused.control_manager.prev_control == recomputed.control_manager.prev_control
    explicit = EE in (requested(0) if callable(requested) else requested[0])
    # ``pop`` limits the jaw twice when it is commanded, then ``_fixed_meta_control`` a third time unless reused.
    assert reused.finger_reads == 10 * (2 if explicit else 1)
    assert recomputed.finger_reads == 10 * (3 if explicit else 2)
