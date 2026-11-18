"""RoboTwin paces every global route at the public TCP and checks its wrist (``end_effector_speed_v1``).

The plant is the correctness fake: the first three joints are the TCP's metres and the last three its
rotation vector, so a joint path's TCP speed is read directly from the issued targets. The real timing is
``ora/environment/controllers/motion_pace.py``, which has its own tests; these hold the wiring: the knots
are what the transaction issues, what replay records and what the receipt reports; a refusal writes
nothing; both arms stay on one timeline; and the local servo keeps the speed without ever re-anchoring
to a smaller step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.environment.backends.robotwin import pacing
from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
from ora.environment.controllers.motion_pace import step_caps
from tests.support.robotwin.correctness_repair import PROFILE, action
from tests.support.robotwin.motion_v2 import transaction_plant, transaction_session

TICK = pacing.PHYSICS_STEP_S
#: The left arm's fake TCP at zero joints.
LEFT_ORIGIN = np.asarray([-0.2, 0.0, 0.8])
RIGHT_ORIGIN = np.asarray([0.2, 0.0, 0.8])


def _candidate(arm: str, rows: Any) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=float)
    return {"schema": "robotwin.curobo-plan-result.v1", "status": "Success", "planner_status": "SUCCESS",
            "joint_names": PROFILE["arms"][arm]["planner_joint_names"], "position": rows,
            "velocity": np.zeros_like(rows), "interpolation_dt_s": TICK}


def _plan(plant: Any, arm: str, rows: Any) -> None:
    setattr(plant.robot, f"{arm}_plan_path", lambda *_args, **_kwargs: _candidate(arm, rows))


def _line(end: list[float], rows: int = 60) -> np.ndarray:
    return np.linspace(np.zeros(6), np.asarray(end, dtype=float), rows)


def _move(targets: dict[str, list[float]], *, speed: dict[str, Any] | None = None,
          interaction: str = "free_space") -> dict[str, Any]:
    """A move_pose to the fake plant's pose at these joints."""

    arguments: dict[str, Any] = {"targets": {}, "interactions": {arm: interaction for arm in targets}}
    for arm, joints in targets.items():
        origin = LEFT_ORIGIN if arm == "left" else RIGHT_ORIGIN
        arguments["targets"][arm] = {
            "position_m": list(np.asarray(joints[:3]) + origin),
            "orientation_xyzw": list(Rotation.from_rotvec(joints[3:]).as_quat()),
        }
    if speed is not None:
        arguments["speed"] = speed
    return action("move_pose", arguments)


def _issued(live: Any) -> dict[str, Any]:
    transactions = sorted((Path(live.output_root) / "motion-transactions").glob("action-0000-*"))
    return dict(np.load(transactions[-1] / "arrays.npz"))


def _session(tmp_path: Path) -> tuple[Any, Any]:
    plant = transaction_plant()
    live, _calls = transaction_session(tmp_path, plant)
    return plant, live


def test_the_default_move_runs_the_paced_knots_one_per_physics_step(tmp_path):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}))["receipt"]
    assert receipt["execution_status"] == "reached"
    motion = receipt["motion"]
    assert motion["speed"]["speed"] == "fast" and motion["speed"]["linear_m_s"] is None
    assert motion["speed"]["joint_speed_deg_s"] == pytest.approx(math.degrees(pacing.JOINT_SPEED_CEILING_RAD_S), abs=1e-3)
    assert set(motion["wrist"]) == {"left"}
    assert motion["planned_ticks"] > 1 and motion["duration_s"] == pytest.approx(motion["planned_ticks"] * TICK)
    arrays = _issued(live)
    paced = arrays["paced_left_joint__position"]
    assert len(paced) == motion["planned_ticks"]
    # The transaction issues the paced knots, each until converged, ending exactly on the goal.
    issued = arrays["issued_left_arm__position"]
    assert issued.shape[0] >= len(paced)
    np.testing.assert_allclose(np.unique(issued.round(5), axis=0), np.unique(paced.round(5), axis=0), atol=1e-5)
    assert paced[-1] == pytest.approx([0.15, 0, 0, 0, 0, 0])
    steps = np.abs(np.diff(np.vstack([np.zeros(6), paced]), axis=0)).max(axis=1)
    assert steps.max() <= pacing.JOINT_SPEED_CEILING_RAD_S * TICK + 1e-9


@pytest.mark.parametrize("speed,cap", [
    ({"speed": "careful", "slow_approaching": False}, 0.1),
    ({"speed": "normal", "slow_approaching": False}, 0.3),
    ({"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False}, 0.05),
])
def test_a_requested_speed_holds_at_the_tcp_and_the_receipt_says_so(tmp_path, speed, cap):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}, speed=speed))["receipt"]
    assert receipt["execution_status"] == "reached"
    motion = receipt["motion"]
    assert motion["speed"]["speed"] == speed["speed"] and motion["speed"]["linear_m_s"] == cap
    assert motion["speed"]["planned_peak_linear_m_s"] <= cap * 1.001
    paced = _issued(live)["paced_left_joint__position"]
    tcp = np.linalg.norm(np.diff(np.vstack([np.zeros(6), paced])[:, :3], axis=0), axis=1) / TICK
    assert tcp.max() <= cap * 1.001


def test_slow_approaching_holds_the_end_of_the_path_to_careful(tmp_path):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    speed = {"speed": "normal", "slow_approaching": True}
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}, speed=speed))["receipt"]
    assert receipt["motion"]["speed"]["slow"]["linear_m_s"] == 0.1
    paced = _issued(live)["paced_left_joint__position"]
    tcp = np.diff(np.vstack([np.zeros(6), paced])[:, 0]) / TICK
    last = paced[:, 0] >= 0.15 - 0.02 + 1e-6
    assert tcp[last].max() <= 0.1 * 1.01 < tcp.max()


def test_a_wrist_unwind_is_refused_before_any_controller_write(tmp_path):
    plant, live = _session(tmp_path)
    # The wrist joint turns 1.9 rad out and back, for a target that needs no turn at all.
    rows = np.vstack([_line([0.03, 0, 0, 0, 0, 1.9], 60), _line([0.03, 0, 0, 0, 0, 1.9], 60)[::-1] * [0, 0, 0, 0, 0, 1]
                      + [0.03, 0, 0, 0, 0, 0]])
    _plan(plant, "left", rows)
    receipt = live.execute(_move({"left": [0.03, 0, 0, 0, 0, 0]}))["receipt"]
    assert receipt["execution_status"] == "plan_rejected"
    assert (receipt["failure_stage"], receipt["failure_code"]) == ("pacing", "paced_path_wrist_unwind")
    assert plant.steps == 0 and not plant.calls and receipt["state_changed"] is False
    refused = receipt["motion"]["refused"]
    assert refused["arm"] == "left" and refused["unrequested_turn_deg"] > 45.0
    assert refused["wrist"]["travel_deg"] > 200.0 and receipt["steps_used"] == 0


def test_a_route_over_the_budget_is_refused_whole(tmp_path, monkeypatch):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    monkeypatch.setattr(pacing, "tick_budget", lambda _profile: 5)
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}, speed={"speed": "careful", "slow_approaching": False}))["receipt"]
    assert (receipt["failure_stage"], receipt["failure_code"]) == ("pacing", "paced_motion_budget")
    assert plant.steps == 0 and receipt["motion"]["refused"]["tick_budget"] == 5


def test_the_tick_budget_is_the_tighter_of_the_waypoint_and_physics_limits():
    assert pacing.tick_budget(PROFILE) == 5000
    assert pacing.tick_budget({**PROFILE, "plan_validation": {"maximum_waypoints": 30000}}) == 19936


def test_two_arms_share_one_timeline_and_their_paced_knots_are_clearance_checked(tmp_path, monkeypatch):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    _plan(plant, "right", _line([0.03, 0, 0, 0, 0, 0]))
    checked = []
    original = RoboTwinMotionBackendAdapter.validate_collision

    def spy(self, plans, action_start):
        checked.append({arm: len(plan.position) for arm, plan in plans.items()})
        return original(self, plans, action_start)

    monkeypatch.setattr(RoboTwinMotionBackendAdapter, "validate_collision", spy)
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0], "right": [0.03, 0, 0, 0, 0, 0]},
                                 speed={"speed": "careful", "slow_approaching": False}))["receipt"]
    assert receipt["execution_status"] == "reached"
    motion = receipt["motion"]
    assert set(motion["wrist"]) == {"left", "right"}
    arrays = _issued(live)
    left, right = arrays["paced_left_joint__position"], arrays["paced_right_joint__position"]
    assert len(left) == len(right) == motion["planned_ticks"]
    assert right[-1] == pytest.approx([0.03, 0, 0, 0, 0, 0])
    # The shorter arm is slowed uniformly: it never exceeds its cap and it never stops halfway.
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros(6), right])[:, :3], axis=0), axis=1) / TICK
    assert steps.max() <= 0.1 * 1.001 and steps[1:-1].min() > 0.0
    # The second check is the paced one: pacing changed when each arm is where.
    assert checked[-1] == {"left": len(left), "right": len(right)}


def test_a_paced_two_arm_clash_is_refused_before_any_write(tmp_path, monkeypatch):
    from ora.environment.backends.robotwin.planning import RoboTwinPlanRejected

    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    _plan(plant, "right", _line([0.03, 0, 0, 0, 0, 0]))
    original = RoboTwinMotionBackendAdapter.validate_collision
    calls = []

    def clash_on_the_paced_knots(self, plans, action_start):
        calls.append(1)
        if len(calls) > 1:
            raise RoboTwinPlanRejected("inter_arm_collision", "planned arm link spheres collide with the other arm")
        return original(self, plans, action_start)

    monkeypatch.setattr(RoboTwinMotionBackendAdapter, "validate_collision", clash_on_the_paced_knots)
    receipt = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0], "right": [0.03, 0, 0, 0, 0, 0]}))["receipt"]
    assert (receipt["failure_stage"], receipt["failure_code"]) == ("plan_validation", "inter_arm_collision")
    assert plant.steps == 0 and not plant.calls


def test_replay_records_the_paced_targets(tmp_path):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}, speed={"speed": "careful", "slow_approaching": False}))
    replay = np.load(Path(live.output_root) / "controller-replay" / "action-0000.npz")
    arrays = _issued(live)
    np.testing.assert_array_equal(replay["left_arm__position"], arrays["issued_left_arm__position"])


def test_recover_pose_is_paced_at_the_default_speed(tmp_path):
    plant, live = _session(tmp_path)
    plant.q["left"][:] = [0.1, 0, 0, 0, 0, 0]
    live.home_joints["left"] = [0.0] * 6
    live.last_safe_joints["left"] = [0.0] * 6
    live._arm_profile = lambda arm: PROFILE["arms"][arm]
    setattr(plant.robot, "left_plan_joint_path",
            lambda *_args, **_kwargs: _candidate("left", np.linspace(plant.q["left"].copy(), np.zeros(6), 40)))
    receipt = live.execute(action("recover_motion", {"targets": {"left": "home"}}))["receipt"]
    assert receipt["motion"]["speed"]["speed"] == "fast" and receipt["motion"]["planned_ticks"] >= 1


def _local_session(tmp_path, monkeypatch):
    plant, live = _session(tmp_path)
    monkeypatch.setattr(RoboTwinMotionBackendAdapter, "public_tcp_jacobian", lambda self, arm, joints: np.eye(6))
    return plant, live


def _local_issued(live: Any) -> np.ndarray:
    transactions = sorted((Path(live.output_root) / "motion-transactions").glob("action-0000-local-*"))
    return dict(np.load(transactions[-1] / "arrays.npz"))["issued_left_arm__position"]


def test_the_default_local_route_is_unchanged(tmp_path, monkeypatch):
    """No caps, one joint target per servo iteration, as before."""

    plant, live = _local_session(tmp_path, monkeypatch)
    receipt = live.execute(_move({"left": [0.006, 0, 0, 0, 0, 0]}, interaction="contact_approach"))["receipt"]
    assert receipt["motion_strategy"]["strategy"] == "local"
    assert receipt["execution_status"] == "reached" and set(receipt["motion"]) == {"speed"}
    assert receipt["motion"]["speed"]["linear_m_s"] is None
    issued = np.vstack([np.zeros(6), _local_issued(live)])
    assert np.abs(np.diff(issued[:, 0])).max() == pytest.approx(PROFILE["local_cartesian"]["maximum_translation_step_m"])


def test_a_requested_speed_spreads_each_local_step_over_physics_steps(tmp_path, monkeypatch):
    plant, live = _local_session(tmp_path, monkeypatch)
    speed = {"speed": "careful", "slow_approaching": False}
    receipt = live.execute(_move({"left": [0.006, 0, 0, 0, 0, 0]}, speed=speed, interaction="contact_approach"))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["motion"]["speed"]["linear_m_s"] == 0.1
    # The plant tracks every target in one step, so the issued targets are the TCP's own motion.
    issued = np.vstack([np.zeros(6), _local_issued(live)])
    speeds = np.linalg.norm(np.diff(issued[:, :3], axis=0), axis=1) / TICK
    assert speeds.max() <= 0.1 * 1.001
    assert (speeds > 0.05).sum() >= 5


def test_local_substeps_divide_a_step_by_the_caps_that_bind():
    caps = step_caps({"speed": "careful", "slow_approaching": False}, tick_s=TICK)
    assert pacing.local_substeps(caps, [0.005, 0, 0], [0, 0, 0], 0.05) == 13
    assert pacing.local_substeps(caps, [0.0001, 0, 0], [0, 0, 0], 0.05) == 1
    assert pacing.local_substeps(caps, [0, 0, 0], [0.035, 0, 0], 0.05) == 26
    near = step_caps({"speed": "fast", "slow_approaching": True}, tick_s=TICK)
    assert pacing.local_substeps(near, [0.005, 0, 0], [0, 0, 0], 0.05) == 1
    assert pacing.local_substeps(near, [0.005, 0, 0], [0, 0, 0], 0.01) == 13


def _context(declared: bool) -> ReceiptProjectionContext:
    return ReceiptProjectionContext(last_action_id="action-0", last_action_public_name="move_pose",
                                    surface_supports=lambda name: name in (
                                        {"public_motion_effect_v1", "end_effector_speed_v1"} if declared
                                        else {"public_motion_effect_v1"}))


def test_the_policy_sees_the_pacing_where_the_speed_is_published(tmp_path):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    receipt = json.loads(json.dumps(live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}))["receipt"]))
    project = resolve_receipt_effect_projector(receipt["schema"])
    effect = project(receipt, _context(True))
    assert set(effect["detail"]["motion"]) == {"speed", "wrist", "planned_ticks", "duration_s"}
    assert "motion" not in project(receipt, _context(False))["detail"]


def test_a_pacing_refusal_reaches_the_policy_with_its_numbers(tmp_path):
    plant, live = _session(tmp_path)
    rows = np.vstack([_line([0.03, 0, 0, 0, 0, 1.9], 60), _line([0.03, 0, 0, 0, 0, 1.9], 60)[::-1] * [0, 0, 0, 0, 0, 1]
                      + [0.03, 0, 0, 0, 0, 0]])
    _plan(plant, "left", rows)
    receipt = json.loads(json.dumps(live.execute(_move({"left": [0.03, 0, 0, 0, 0, 0]}))["receipt"]))
    effect = resolve_receipt_effect_projector(receipt["schema"])(receipt, _context(True))
    assert (effect["status"], effect["failure_stage"], effect["failure_code"]) == (
        "plan_rejected", "pacing", "paced_path_wrist_unwind")
    assert "code" not in effect["detail"]["motion"]["refused"]
    assert effect["detail"]["motion"]["refused"]["unrequested_turn_deg"] > 45.0


#: What no policy-readable copy may hold: the timing evidence and the names of its libraries.
FORBIDDEN = ('"timing":', "motion_pace", "toppra", "ruckig", "jerk_source")


def test_the_observation_embeds_the_receipt_without_its_timing_evidence(tmp_path):
    plant, live = _session(tmp_path)
    _plan(plant, "left", _line([0.15, 0, 0, 0, 0, 0]))
    sealed = live.execute(_move({"left": [0.15, 0, 0, 0, 0, 0]}))["receipt"]
    # The sealed receipt says which implementation timed each arm's route ...
    assert set(sealed["motion"]["timing"]) == {"left"}
    assert sealed["motion"]["timing"]["left"]["implementation"] in {"motion_pace", "toppra_ruckig"}
    # ... and the observation, the policy's, embeds the move without it.
    observation, _images = live.observe(tmp_path / "policy")
    copy_ = observation["previous_execution_receipt"]["receipt"]
    assert "timing" not in copy_["motion"] and copy_["motion"]["planned_ticks"] == sealed["motion"]["planned_ticks"]
    files = "".join(path.read_text(errors="replace") for path in (tmp_path / "policy").rglob("*.json"))
    assert files and not [token for token in FORBIDDEN if token in files]
    assert '"timing":' in (Path(live.output_root) / "receipts" / "action-0000.json").read_text()


def test_the_timing_worker_certifies_robotwin_at_the_unverified_default_jerk():
    from ora.environment.controllers import motion_timing

    assert pacing.ROBOT == "aloha_agilex"
    assert motion_timing.jerk_limits(pacing.ROBOT, 6) == {
        "rad_s3": [motion_timing.DEFAULT_JERK_RAD_S3] * 6, "source": motion_timing.UNVERIFIED}


def test_the_first_round_binds_the_runs_timing_once(tmp_path, monkeypatch):
    """session.py is pinned, so its round-binding mixin is where the worker takes the run's timing."""

    from ora.environment.controllers import motion_timing

    plant, live = _session(tmp_path)
    bound = []
    monkeypatch.setattr(motion_timing, "bind_preflight", bound.append)
    monkeypatch.setattr(live, "_pinocchio_context", lambda environment, arm: {})
    live._bind_episode_state(plant)
    live._bind_episode_state(plant)
    assert bound == [live.output_root]
