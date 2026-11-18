"""Rate- and lead-limited ``set_gripper`` (the object target form).

The released ``set_gripper`` writes the directive as one target per native
action; ``bridge.interpolate`` restarts it from the measured finger every
tick, so the jaws always close at full drive (about 0.24 m/s, bounded only by
the SDK limiter's 10.8 mm lead) and a jaw stopped on an object is squeezed in
40 ms pulses between about 3 and 10.8 mm of lead. This executor instead moves
the primary finger's drive target once per physics step:

* from the *previously commanded* target (at the first step, the drive
  target the plant still applies, so a held squeeze is kept; one beyond the
  lead comes back at the rate) toward the directive at ``rate_per_s``
  (normalized opening per second; ``null`` steps straight to the directive);
* never more than ``max_lead_m`` from the measured first finger (``null``: the
  SDK's 10.8 mm), which bounds the free-air finger speed and the squeeze on an
  object (both scale with the lead through the drive stiffness);
* the second finger follows through the SDK mimic, and the SDK's own limiter
  still runs (a smaller lead passes it unchanged).

The command ends when every named jaw has arrived within ``end.tolerance`` or
stalled (``end.stall_window_native_actions`` native actions each moving less
than ``end.stall_still_step``), or the budget runs out; at least one native
action always runs so the directive reaches the drive. A jaw stopped on an
object keeps the last applied target (measured finger less the lead): a
constant squeeze, journaled, with no pulse. ``hold_lead_m`` (optional) is the
lead from the native action after the first one in which the first finger
moved less than ``end.stall_still_step``: a close may approach at the SDK's
lead and keep a smaller squeeze once its finger has stopped; the stall and
arrival rules are unchanged. ``arm_settle`` (optional) first holds every
command until the arms of the named jaws move slower than
``joint_speed_rad_s`` (at most ``max_native_actions``): a close then starts
on an arm at rest instead of on one still converging from its last move. Its
first reading is after one held native action, or, with ``first_reading``
``joint_velocity``, the arms' measured joint velocities before any: an arm
already at rest then costs none.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.joint_trajectory import NATIVE_ACTION_S, PHYSICS_STEP_S, SDK_GRIPPER_LEAD_M
from ora.environment.controllers.gripper_stall import GripperStallRule, gripper_outcome_state, gripper_stalled

from .common import array, physical_gripper
from .native_sequence import run_tick
from .trajectory import execution_budget, gripper_key, held_gripper_targets

GRIPPER_RAMP_EXECUTOR = "robodojo_gripper_rate_and_lead_limited"
GRIPPER_RAMP_SCHEMA = "ora.robodojo-gripper-ramp.v1"


def _directive(opening: Any) -> float:
    return 1.0 if opening == "open" else 0.0 if opening == "close" else float(opening)


class GripperRampSource:
    """Per-physics-step drive targets of the named jaws (``arm_settle`` ticks name none)."""

    def __init__(self, session: Any, targets: Mapping[str, Mapping[str, Any]], before: Mapping[str, Any]) -> None:
        self.robots = {arm: session.robots[arm] for arm in targets}
        self.keys = {arm: gripper_key(session, arm) for arm in targets}
        self.goal = {arm: float(physical_gripper(self.robots[arm], _directive(target["opening"]))[0])
                     for arm, target in targets.items()}
        span = {arm: float(robot.gripper_scale[1] - robot.gripper_scale[0]) for arm, robot in self.robots.items()}
        self.step = {arm: (None if target["rate_per_s"] is None
                           else float(target["rate_per_s"]) * span[arm] * PHYSICS_STEP_S)
                     for arm, target in targets.items()}
        self.lead = {arm: float(target["max_lead_m"] or SDK_GRIPPER_LEAD_M) for arm, target in targets.items()}
        # The lead once the first finger has stopped, and the finger travel (metres) one still native action stays below.
        self.hold_lead = {arm: (None if target.get("hold_lead_m") is None else float(target["hold_lead_m"]))
                          for arm, target in targets.items()}
        self.still_m = {arm: float(target["end"]["stall_still_step"]) * span[arm] for arm, target in targets.items()}
        self.holding = {arm: False for arm in targets}
        self.fingers = {arm: float(before[arm]["gripper_finger_qpos"][0]) for arm in targets}
        self.tick_start = dict(self.fingers)
        # The ramp starts from the drive target the plant still applies (a held
        # squeeze stays a squeeze), never from the measured finger.
        self.commanded = held_gripper_targets(session, targets, self.fingers)
        self.joints = {arm: np.asarray(before[arm]["arm_joint_qpos"], dtype=float) for arm in targets}
        self.previous_joints = dict(self.joints)
        self.mode = "gripper"
        self.lead_limited = {arm: False for arm in targets}

    def lead_of(self, arm: str) -> float:
        """The lead this physics step allows: ``hold_lead_m`` once the jaw's first finger has stopped."""
        return self.hold_lead[arm] if self.holding[arm] else self.lead[arm]

    def request(self, index: int) -> dict[str, Any]:
        if self.mode == "arm_settle":
            return {}
        if index == 0:
            self.tick_start = dict(self.fingers)
        result = {}
        for arm, robot in self.robots.items():
            previous, goal, step = self.commanded[arm], self.goal[arm], self.step[arm]
            value = goal if step is None else previous + float(np.clip(goal - previous, -step, step))
            lead = self.lead_of(arm)
            limited = float(np.clip(value, self.fingers[arm] - lead, self.fingers[arm] + lead))
            self.lead_limited[arm] = abs(limited - value) > 1e-12
            if step is not None:
                # A held target beyond the lead (a legacy close's squeeze) comes back at the rate, not in one step.
                limited = previous + float(np.clip(limited - previous, -step, step))
            value = float(np.clip(limited, *robot.gripper_scale))
            self.commanded[arm] = value
            mimic = robot.gripper_move["mimic"]
            result[self.keys[arm]] = {"position": [value, value * mimic[1] + mimic[2]]}
        return result

    def observe(self, index: int, measured: Mapping[str, Any]) -> None:
        self.previous_joints = dict(self.joints)
        for arm in self.robots:
            self.fingers[arm] = float(measured[arm]["gripper_finger_qpos"][0])
            self.joints[arm] = np.asarray(measured[arm]["arm_joint_qpos"], dtype=float)
            if (index == 9 and self.mode == "gripper" and self.hold_lead[arm] is not None and not self.holding[arm]
                    and abs(self.fingers[arm] - self.tick_start[arm]) < self.still_m[arm]):
                self.holding[arm] = True

    def arm_speed(self) -> float:
        return max(float(np.max(np.abs(self.joints[arm] - self.previous_joints[arm]))) / PHYSICS_STEP_S
                   for arm in self.robots)

    def _holding(self) -> dict[str, Any]:
        """The arms that have a hold lead and whether each now runs at it (nothing without one: v1's bytes)."""
        return {arm: self.holding[arm] for arm in self.robots if self.hold_lead[arm] is not None}

    def annotation(self) -> dict[str, Any]:
        holding = self._holding()
        return {"gripper_ramp": {"mode": self.mode,
                                 "commanded_primary_m": {arm: round(value, 7) for arm, value in self.commanded.items()},
                                 "lead_limited": dict(self.lead_limited), **({"holding": holding} if holding else {})}}

    def tick_record(self) -> dict[str, Any]:
        holding = self._holding()
        return {"gripper_ramp": {"mode": self.mode,
                                 "commanded_primary_m": {arm: round(value, 7) for arm, value in self.commanded.items()},
                                 "arm_joint_speed_rad_s": round(self.arm_speed(), 6),
                                 **({"holding": holding} if holding else {})}}


def measured_arm_speed(session: Any, arms: Sequence[str]) -> float:
    """The largest measured arm joint speed (rad/s) of ``arms``: the articulations' own joint velocities."""
    manager = session.environment.robot_manager
    speeds = []
    for arm in arms:
        robot = session.robots[arm]
        data = manager.robot_key[manager.robot_list.index(robot)].data
        values = array(data.joint_vel[0])[list(robot.arm_joint_indices)]
        speeds.append(float(np.max(np.abs(values))))
    return max(speeds)


def _rule(end: Mapping[str, Any]) -> GripperStallRule:
    window = int(end["stall_window_native_actions"])
    return GripperStallRule(still_step=float(end["stall_still_step"]), window=window, minimum_samples=window + 1,
                            arrival_tolerance=float(end["tolerance"]), tick_period_s=NATIVE_ACTION_S,
                            tick_unit="native_action")


def execute_rate_limited_gripper(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    from .sparse_motion import ARMS, _SparseAction, gripper_closed_floor, gripper_failure_code

    targets = action["arguments"]["targets"]
    arms = tuple(arm for arm in ARMS if arm in targets)
    execution = _SparseAction(session, action, arms)
    execution.budget = execution_budget(session)
    execution.begin([gripper_key(session, arm) for arm in arms])
    directives = {arm: _directive(targets[arm]["opening"]) for arm in arms}
    for arm, value in directives.items():
        session.gripper_directives[arm] = value
    effective = {arm: max(value, gripper_closed_floor(session, arm)) for arm, value in directives.items()}
    rules = {arm: _rule(targets[arm]["end"]) for arm in arms}
    source = GripperRampSource(session, {arm: targets[arm] for arm in arms}, execution.before)
    action_id = str(action["action_id"])

    def tick(mode: str) -> None:
        source.mode = mode
        run_tick(session, action_id=action_id, segment=mode, source=source, held_targets=execution.held_targets,
                 journal=execution.journal, tracking=execution.tracking)
        execution.used += 1

    settle_requests = [targets[arm]["arm_settle"] for arm in arms if targets[arm]["arm_settle"] is not None]
    arm_settle = None
    if settle_requests:
        speed_limit = min(item["joint_speed_rad_s"] for item in settle_requests)
        most = min(max(item["max_native_actions"] for item in settle_requests), max(execution.remaining() - 1, 0))
        arm_settle = {"joint_speed_limit_rad_s": speed_limit, "native_actions": 0, "settled": False}
        speed = None
        if all(item.get("first_reading") == "joint_velocity" for item in settle_requests):
            # The articulations' measured joint velocities: an arm already at rest needs no look.
            speed = measured_arm_speed(session, [arm for arm in arms if targets[arm]["arm_settle"] is not None])
            arm_settle.update(first_reading="joint_velocity", settled=speed <= speed_limit)
        # Without it the start state has no speed reading: look at least once when asked.
        while not arm_settle["settled"] and arm_settle["native_actions"] < most and not session.is_terminal:
            tick("arm_settle")
            arm_settle["native_actions"] += 1
            speed = source.arm_speed()
            arm_settle["settled"] = speed <= speed_limit
        arm_settle["arm_joint_speed_rad_s"] = round(source.arm_speed() if speed is None else speed, 6)
    traces = {arm: [float(execution.before[arm]["gripper"])] for arm in arms}
    ramp_ticks, outcome = 0, {}
    while execution.remaining() > 0 and not session.is_terminal:
        tick("gripper")
        ramp_ticks += 1
        current = session._proprioception()["end_effectors"]
        outcome = {}
        for arm in arms:
            traces[arm].append(float(current[arm]["gripper"]))
            reached = rules[arm].arrived(current[arm]["gripper"], effective[arm])
            outcome[arm] = (reached, gripper_stalled(traces[arm], target=effective[arm], rule=rules[arm]))
        if all(reached or stalled for reached, stalled in outcome.values()):
            break
    states = {arm: gripper_outcome_state(reached=outcome.get(arm, (False, False))[0],
                                         stalled=outcome.get(arm, (False, False))[1],
                                         ticks_used=execution.used, ticks_budget=execution.budget) for arm in arms}
    code = gripper_failure_code(states, terminal=session.is_terminal) if ramp_ticks else "native_action_budget_exhausted"
    summary = {"schema": GRIPPER_RAMP_SCHEMA, "executor": GRIPPER_RAMP_EXECUTOR, "native_actions": execution.used,
               "ramp_native_actions": ramp_ticks, "arm_settle": arm_settle,
               "arms": {arm: {"directive": directives[arm], "rate_per_s": targets[arm]["rate_per_s"],
                              "max_lead_m": source.lead[arm], "end": dict(targets[arm]["end"]),
                              "last_commanded_primary_m": round(source.commanded[arm], 7),
                              "lead_limited_at_end": source.lead_limited[arm],
                              **({} if source.hold_lead[arm] is None else
                                 {"hold_lead_m": source.hold_lead[arm], "held_at_end": source.holding[arm]})}
                        for arm in arms}}
    verified = session._held_targets_unchanged(execution.held_targets)
    return execution.finish(
        status="reached" if code is None else "not_reached", strategy=None,
        stage=None if code is None else "execution", code=code, executor=[GRIPPER_RAMP_EXECUTOR],
        gripper_targets=effective, gripper_states=states,
        extra={"gripper_ramp": summary, "command_invariant": {
            "schema": "ora.command-invariant.v3", "controlled_dimension": "gripper",
            "held_dimension": "eef_pose_target", "verified": verified,
            "applied_targets": "per_physics_step_sdk_joint_position_and_velocity_buffers"}})


__all__ = ["GRIPPER_RAMP_EXECUTOR", "GRIPPER_RAMP_SCHEMA", "GripperRampSource", "execute_rate_limited_gripper",
           "measured_arm_speed"]
