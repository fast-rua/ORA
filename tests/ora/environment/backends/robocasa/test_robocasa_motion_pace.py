"""RoboCasa365 times every arm path at the grip site and checks its wrist (``end_effector_speed_v1``).

The worker has no simulator here: a fake plant tracks its joint target in one control tick, and its grip
site is ``0.4 *`` the first three joints in metres plus the last three as a rotation vector, so a path's
grip-site speed reads straight from the commanded targets. The timing itself is
``ora/environment/controllers/motion_pace.py``; these hold the wiring.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.environment.backends.robocasa import action_plan, kinematics, motion
from ora.environment.controllers import robocasa_omron as C
from ora.environment.controllers.jaw_flip_retry import jaw_twin
from ora.environment.controllers.motion_pace import PaceRefusal

TICK = C.CONTROL_PERIOD_S
LIMITS = (-2.9, 2.9)
LEVER_M = 0.4


def pose_of(joints: Any) -> list[float]:
    joints = np.asarray(joints, dtype=float)
    return [*(LEVER_M * joints[:3]), *Rotation.from_rotvec(joints[4:7]).as_quat()]


def fk(rows: list[list[float]]) -> list[list[float]]:
    return [pose_of(row) for row in rows]


class Plant:
    """One-tick joint tracking; records every commanded target."""

    model = data = None
    control_steps = 0
    #: The jaws start commanded open, as ``OmronIO``'s do.
    gripper_directive = C.GRIPPER_OPEN_COMMAND

    def __init__(self, start: Any = None) -> None:
        self.q = np.zeros(7) if start is None else np.asarray(start, dtype=float)
        self.commands: list[np.ndarray] = []

    def arm_qpos(self) -> list[float]:
        return self.q.tolist()

    def grip_pose(self) -> list[float]:
        return pose_of(self.q)

    def base_pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def contact_forces(self) -> dict[str, float]:
        return {"gripper": 0.0, "base": 0.0}

    def step(self, arm_joints: Any = None, **_command: Any) -> dict[str, Any]:
        self.control_steps += 1
        if arm_joints is not None:
            self.q = np.asarray(arm_joints, dtype=float)
            self.commands.append(self.q.copy())
        return {}

    def hold_measured_arm(self) -> None:
        pass


class Session:
    round_index = 0
    is_terminal = False
    physics_steps_per_tick = 25
    last_safe_arm_qpos = None
    home_arm_qpos = None

    def __init__(self, plant: Plant, path: Any, *, torso: float | None = None) -> None:
        self.io = plant
        self.path = [list(map(float, row)) for row in path]
        self.torso = torso
        self.plans: list[bool] = []
        self.after_lift: dict[str, Any] | None = None

    def arm_state(self) -> dict[str, Any]:
        return {"right": {"pose_xyz_xyzw": self.io.grip_pose()}}

    def must_stop(self) -> bool:
        return False

    def after_control_step(self, _result: Any) -> None:
        pass

    def remember_safe_joints(self) -> None:
        self.last_safe_arm_qpos = self.io.arm_qpos()

    def workspace_refusal(self, _pose: Any) -> None:
        return None

    def arm_path_plan(self, _pose: Any, _interaction: str, *, allow_torso: bool = True,
                      speed: Any = None) -> dict[str, Any]:
        self.plans.append(allow_torso)
        if self.torso is not None and allow_torso:
            return {"refusal": None, "code": None, "path": None, "poses": None, "torso_height_m": self.torso,
                    "kind": "joint"}
        if self.after_lift is not None:
            return self.after_lift
        return {"refusal": None, "code": None, "path": self.path, "poses": fk(self.path),
                "torso_height_m": None, "kind": "joint"}


@pytest.fixture(autouse=True)
def _kinematics(monkeypatch):
    monkeypatch.setattr(kinematics, "grip_poses", lambda _model, _data, rows: fk(rows))
    monkeypatch.setattr(kinematics, "joint_range", lambda _model, _name: LIMITS)


def _action(end: list[float], *, speed: Any = None, interaction: str = "free_space") -> dict[str, Any]:
    pose = pose_of(end)
    arguments: dict[str, Any] = {"targets": {"right": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}},
                                 "interactions": {"right": interaction}}
    if speed is not None:
        arguments["speed"] = speed
    return {"action_id": "action-1", "action": "move_pose", "arguments": arguments}


def _line(end: list[float], count: int = 25) -> np.ndarray:
    return np.linspace(np.zeros(7), np.asarray(end, dtype=float), count + 1)[1:]


END = [0.5, 0, 0, 0, 0, 0, 0]


def _run(path: Any, action: dict[str, Any]) -> tuple[dict[str, Any], Plant, Session]:
    plant = Plant()
    session = Session(plant, path)
    return motion.execute_motion_v3(session, action), plant, session


def _grip_speeds(plant: Plant) -> np.ndarray:
    positions = np.asarray([pose_of(joints)[:3] for joints in [np.zeros(7), *plant.commands]])
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) / TICK


def test_the_default_move_runs_one_paced_target_per_tick_within_the_interaction_bounds():
    receipt, plant, session = _run(_line(END), _action(END))
    assert receipt["execution_status"] == "reached"
    motion_block = receipt["motion"]
    assert motion_block["speed"]["speed"] == "fast" and motion_block["speed"]["linear_m_s"] is None
    assert set(motion_block["wrist"]) >= {"start_deg", "travel_deg", "min_limit_margin_deg", "path_turn_deg"}
    # The last knot's tick is the first of the settle ticks.
    assert len(plant.commands) == motion_block["planned_ticks"] + C.ARM_SETTLE_STEPS - 1
    assert plant.commands[-1] == pytest.approx(END)
    steps = np.abs(np.diff(np.vstack([np.zeros(7), plant.commands]), axis=0)).max(axis=1)
    assert steps.max() <= C.ARM_JOINT_STEP_RAD["free_space"] + 1e-9
    assert session.last_safe_arm_qpos == [0.0] * 7


@pytest.mark.parametrize("speed,cap", [
    ({"speed": "careful", "slow_approaching": False}, 0.1),
    ({"speed": "normal", "slow_approaching": False}, 0.3),
    ({"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False}, 0.05),
])
def test_a_requested_speed_holds_at_the_grip_site(speed, cap):
    receipt, plant, _session = _run(_line(END), _action(END, speed=speed))
    assert receipt["execution_status"] == "reached"
    assert receipt["motion"]["speed"]["linear_m_s"] == cap
    assert receipt["motion"]["speed"]["planned_peak_linear_m_s"] <= cap * 1.001
    assert _grip_speeds(plant).max() <= cap * 1.01


def test_slow_approaching_holds_the_end_of_the_path_to_careful():
    speed = {"speed": "normal", "slow_approaching": True}
    receipt, plant, _session = _run(_line(END), _action(END, speed=speed))
    assert receipt["motion"]["speed"]["slow"]["linear_m_s"] == 0.1
    x = np.asarray([pose_of(joints)[0] for joints in plant.commands])
    speeds = _grip_speeds(plant)
    assert speeds[np.nonzero(x >= LEVER_M * 0.5 - 0.02 + 1e-6)[0]].max() <= 0.1 * 1.01 < speeds.max()


@pytest.mark.parametrize("interaction", ["precision", "contact_approach"])
def test_the_interaction_bounds_hold_at_every_speed(interaction):
    """A joint path whose grip site outruns the interaction is timed again at a lower joint ceiling."""

    lever = np.asarray([1.0] * 3 + [0.0] * 4)
    far = (0.3 * lever).tolist()
    paced = motion.pace_arm_path(np.zeros(7), _line(far, 30), fk, speed={"speed": "normal", "slow_approaching": False},
                                 interaction=interaction, wrist_limits_rad=LIMITS)
    assert motion.grip_step_ratio(pose_of(np.zeros(7)), fk(paced.rows()), interaction) <= 1.0 + 1e-9
    unbounded = motion.pace_arm_path(np.zeros(7), _line(far, 30), fk, speed=None, interaction="free_space",
                                     wrist_limits_rad=LIMITS)
    assert unbounded.report()["planned_ticks"] < paced.report()["planned_ticks"]


def test_a_wrist_unwind_is_refused_before_anything_moves():
    out = np.linspace([0, 0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0, 2.0], 40)
    back = np.linspace([0.1, 0, 0, 0, 0, 0, 2.0], [0.1, 0, 0, 0, 0, 0, 0.0], 40)
    path = np.vstack([out[1:], back[1:]])
    receipt, plant, session = _run(path, _action([0.1, 0, 0, 0, 0, 0, 0]))
    assert (receipt["execution_status"], receipt["failure_stage"], receipt["failure_code"]) == (
        "plan_rejected", "pacing", "paced_path_wrist_unwind")
    assert not plant.commands and receipt["state_changed"] is False
    assert receipt["motion"]["refused"]["unrequested_turn_deg"] > 45.0
    # A refused move is not admitted, so recover_pose's previous stays where it was.
    assert session.last_safe_arm_qpos is None


def test_a_path_over_the_budget_is_refused_whole():
    speed = {"speed": "custom", "speed_m_s": 0.02, "slow_approaching": False}
    receipt, plant, _session = _run(_line(END), _action(END, speed=speed))
    assert (receipt["failure_stage"], receipt["failure_code"]) == ("pacing", "paced_motion_budget")
    assert receipt["motion"]["refused"]["tick_budget"] == C.ARM_MAX_STEPS - C.ARM_SETTLE_MAX_STEPS
    assert not plant.commands


def _lifted(monkeypatch: Any, session: Session) -> None:
    """The first admission names a height and no path; after the lift ``session.after_lift`` answers."""
    monkeypatch.setattr(motion.MotionRun, "move_torso", lambda _self, target: {
        "stop_reason": "reached", "from_m": 0.0, "to_m": target, "target_m": target})


def test_the_path_is_planned_and_paced_after_the_lift(monkeypatch):
    plant = Plant()
    session = Session(plant, _line(END), torso=0.2)
    _lifted(monkeypatch, session)
    receipt = motion.execute_motion_v3(session, _action(END))
    assert receipt["execution_status"] == "reached" and receipt["torso"]["arm_path_replanned"] is True
    assert session.plans == [True, False] and receipt["motion"]["planned_ticks"] >= 1


def test_a_plan_refused_after_the_lift_leaves_the_arm_where_it_is(monkeypatch):
    plant = Plant()
    session = Session(plant, _line(END), torso=0.2)
    session.after_lift = {"refusal": {"path_kind": "joint"}, "code": "path_in_collision_at_base_pose", "path": None,
                          "poses": None, "torso_height_m": None, "kind": "joint"}
    _lifted(monkeypatch, session)
    receipt = motion.execute_motion_v3(session, _action(END))
    assert receipt["execution_status"] == "not_reached" and receipt["failure_code"] == "path_in_collision_at_base_pose"
    assert receipt["torso"]["arm_path_replanned"] is False and not plant.commands
    assert session.last_safe_arm_qpos is not None


def test_a_pacing_refused_after_the_lift_leaves_the_arm_where_it_is(monkeypatch):
    plant = Plant()
    session = Session(plant, _line(END), torso=0.2)
    _lifted(monkeypatch, session)
    monkeypatch.setattr(motion, "_paced_plan", lambda *_args: (_ for _ in ()).throw(
        PaceRefusal("budget", code="paced_motion_budget")))
    receipt = motion.execute_motion_v3(session, _action(END))
    assert (receipt["execution_status"], receipt["failure_code"]) == ("not_reached", "paced_motion_budget")
    assert not plant.commands


def test_recover_pose_is_paced_at_the_default_speed():
    plant = Plant(start=[0.2, 0, 0, 0, 0, 0, 0])
    session = Session(plant, _line([0.0] * 7))
    session.home_arm_qpos = [0.0] * 7
    session.joint_target_plan = lambda joints: {
        "refusal": None, "code": None, "path": np.linspace(plant.q, joints, 20)[1:].tolist(),
        "poses": [], "torso_height_m": None, "kind": "joint"}
    action = {"action_id": "action-1", "action": "recover_motion", "arguments": {"targets": {"right": "home"}}}
    receipt = motion.execute_motion_v3(session, action)
    assert receipt["execution_status"] == "reached" and receipt["motion"]["speed"]["speed"] == "fast"
    assert session.last_safe_arm_qpos is None


def test_the_policy_sees_the_pacing_where_the_speed_is_published():
    receipt, _plant, _session = _run(_line(END), _action(END))
    context = ReceiptProjectionContext(last_action_id="action-1", last_action_public_name="move_pose",
                                       surface_supports=lambda name: name == "end_effector_speed_v1")
    effect = resolve_receipt_effect_projector(receipt["schema"])(json.loads(json.dumps(receipt)), context)
    assert set(effect["detail"]["motion"]) == {"speed", "wrist", "planned_ticks", "duration_s"}


#: What no policy-readable copy may hold: the timing evidence and the names of its libraries.
FORBIDDEN = ('"timing":', "motion_pace", "toppra", "ruckig", "jerk_source")


def test_the_observation_embeds_the_receipt_without_its_timing_evidence(tmp_path, monkeypatch):
    from ora.environment.backends.robocasa.session import RoboCasaSession

    receipt, plant, _session = _run(_line(END), _action(END))
    assert receipt["motion"]["timing"]["implementation"] in {"motion_pace", "toppra_ruckig"}
    session = RoboCasaSession(config={"task": "OpenDrawer", "seed": 0}, output_root=tmp_path / "native")
    plant.gripper_directive, plant.torso_height = -1.0, lambda: 0.0
    plant.gripper_opening, plant.finger_qpos = (lambda: 1.0), (lambda: [0.04, 0.04])
    session.io = plant
    session.horizon = 500
    session.last_receipt = {"path": "receipts/action-0000.json", "sha256": "0" * 64, "receipt": receipt}
    meta = {"intrinsic": np.eye(3), "cam2world": np.eye(4), "near": 0.01, "far": 10.0}
    monkeypatch.setattr(session, "_camera", lambda _source: (
        np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), meta))
    observation, _images = session.observe(tmp_path / "policy")
    copy_ = observation["previous_execution_receipt"]["receipt"]
    assert "timing" not in copy_["motion"] and copy_["motion"]["planned_ticks"] == receipt["motion"]["planned_ticks"]
    files = "".join(path.read_text(errors="replace") for path in (tmp_path / "policy").rglob("*.json"))
    assert files and not [token for token in FORBIDDEN if token in files]


def test_the_timing_worker_certifies_the_panda_at_its_published_jerk():
    from ora.environment.controllers import motion_timing

    assert motion.ROBOT == "franka_panda"
    assert motion_timing.jerk_limits(motion.ROBOT, 7)["source"] != motion_timing.UNVERIFIED


def test_start_binds_the_runs_timing_and_close_stops_the_worker(tmp_path, monkeypatch):
    from ora.environment.backends.robocasa import session as session_module
    from ora.environment.controllers import motion_timing

    calls = []
    monkeypatch.setattr(motion_timing, "bind_preflight", lambda root: calls.append(("bind", root)))
    monkeypatch.setattr(motion_timing, "close", lambda: calls.append(("close",)))
    session = session_module.RoboCasaSession(config={"task": "OpenDrawer", "seed": 0}, output_root=tmp_path / "native")
    monkeypatch.setattr(session, "_reset", lambda *, seed, round_index: {"seed": seed})
    session.environment_factory = object()
    assert session.start() == {"seed": 0}
    session.close()
    assert calls == [("bind", session.output_root), ("close",)]


#: The request turns the hand 149 degrees about its approach; the planner reaches it the long way (211 degrees),
#: while its jaw twin is a 31-degree turn the other way, as at the Panda's last-joint limit.
REQUEST_END = [0.1, 0, 0, 0, 0, 0, 2.6]
TWIN_END = [0.1, 0, 0, 0, 0, 0, 2.6 - np.pi]


class TwinSession(Session):
    """Admits the long way round to the requested pose and a straight path to its jaw twin."""

    motion_planner = None

    def __init__(self, plant: Plant) -> None:
        super().__init__(plant, [])
        self.twin = jaw_twin(pose_of(REQUEST_END))
        self.asked: list[str] = []

    def arm_path_plan(self, pose: Any, _interaction: str, *, allow_torso: bool = True,
                      speed: Any = None) -> dict[str, Any]:
        twin = list(pose) == self.twin
        self.asked.append("twin" if twin else "request")
        path = _line(TWIN_END) if twin else _line([0.1, 0, 0, 0, 0, 0, 2.6 - 2 * np.pi])
        return {"refusal": None, "code": None, "path": path.tolist(), "poses": fk(path), "torso_height_m": None,
                "kind": "joint"}


def test_a_wrist_refusal_with_open_jaws_runs_the_jaw_twin_and_says_so():
    plant = Plant()
    session = TwinSession(plant)
    receipt = motion.execute_motion_v3(session, _action(REQUEST_END))
    assert session.asked == ["request", "twin"] and plant.commands[-1] == pytest.approx(TWIN_END)
    assert receipt["execution_status"] == "reached"
    retry = receipt["motion"]["wrist"]["jaw_flip_retry"]
    assert retry["applied"] is True and retry["executed_orientation_xyzw"] == session.twin[3:]
    assert retry["refused_unrequested_turn_deg"] == pytest.approx(62.0, abs=0.5)
    assert receipt["motion"]["wrist"]["path_turn_deg"] == pytest.approx(31.0, abs=0.5)
    assert receipt["arms"]["right"]["pose_error"]["orientation_rad"] == pytest.approx(0.0, abs=1e-6)
    assert session.last_safe_arm_qpos == [0.0] * 7


@pytest.mark.parametrize("directive", [C.GRIPPER_OPEN_COMMAND, C.GRIPPER_CLOSE_COMMAND])
def test_a_dry_run_answers_what_the_real_call_does(directive):
    plant = Plant()
    plant.gripper_directive = directive
    session = TwinSession(plant)
    plan = action_plan.plan_action(session, _action(REQUEST_END))
    receipt = motion.execute_motion_v3(session, _action(REQUEST_END))
    if directive == C.GRIPPER_OPEN_COMMAND:
        (path,) = plan["segments"]
        assert plan["status"] == "planned" and session.asked == ["request", "twin"] * 2
        assert path["poses"][-1] == pytest.approx(session.twin, abs=1e-9)
        assert path["jaw_flip_retry"] == receipt["motion"]["wrist"]["jaw_flip_retry"]
        assert receipt["execution_status"] == "reached"
    else:
        # A jaw that may hold something keeps the request: the load would turn with the hand.
        assert (plan["status"], plan["failure_code"]) == ("refused", "paced_path_wrist_unwind")
        assert "jaw_flip_retry" not in plan["constraints"] and session.asked == ["request"] * 2
        assert (receipt["failure_stage"], receipt["failure_code"]) == ("pacing", "paced_path_wrist_unwind")
        assert not plant.commands
