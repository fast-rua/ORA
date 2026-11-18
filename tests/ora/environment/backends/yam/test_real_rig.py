"""``RealRig`` asks the operator through the cell's approval broker, never a terminal: enabling the arms,
each motion plan, and releasing arms that could not get home. Closing at rest releases without a
question; a blocked way home holds and asks, and a decline keeps holding and tries home again.

No bus, camera or i2rt: ``_start`` is replaced by the measured start pose, i2rt by arms that follow
every command (or, ``blocked``, never move), and the operator by a thread answering ``pending()``.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import types
from typing import Any, Callable

import numpy as np
import pytest

from ora.cell import calibration as cell_calibration
from ora.cell import alerts as cell_alerts
from ora.cell.lock import AT_REST, HOMING, NOT_CONTROLLED, holder
from ora.cell.operator import approval
from ora.environment.backends.yam import grippers, health, real_rig
from ora.environment.backends.yam.common import YamError
from ora.environment.backends.yam.real_rig import RealRig
from ora.environment.controllers import yam_station as Y
from tests.support.cell.fake import FakeKinematics, redirect_lock, write_config

REST = {arm: [0.0] * 6 for arm in Y.ARMS}
START = {arm: [0.3, 0.2, 0.0, 0.0, 0.0, 0.0] for arm in Y.ARMS}
CONFIG = """schema: ora.cell-config.v1
cell_id: yam-test-01
robot:
  environment: yam
  model: yam_station_linear_4310_d405
  buses: {left: can_left, right: can_right}
cameras:
  overhead:    {serial: "1", mount: fixed_world}
  front:       {serial: "2", mount: fixed_world}
  wrist_left:  {serial: "3", mount: end_effector}
  wrist_right: {serial: "4", mount: end_effector}
calibration: CALIBRATION
safety:
  operator_present_required: true
"""


class FakeArm:
    """An i2rt YAM robot: commands move the joints, unless ``blocked``."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.q = np.asarray(START["left"], float).copy()
        self.blocked = False
        self.closed = False
        self.motor_chain = types.SimpleNamespace(running=True)
        self._gripper_limits = [0.0, 1.0]
        self._gripper_index = 6
        self._kp, self._kd = np.full(7, 20.0), np.full(7, 1.0)
        self.gravity_comp_factor = np.ones(7)
        self._command_lock = threading.Lock()
        self._commands = types.SimpleNamespace(kp=self._kp, kd=self._kd, torques=np.zeros(7))

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._kp, self._kd = kp, kd

    @property
    def gripper_gain(self) -> float:
        """What the control loop applies to the gripper now."""

        return float(self._commands.kp[6])

    sag = 0.0                                   # rad below the command, as gravity leaves a held arm

    temperature_c = 30.0
    gripper_effort_nm = 0.0
    jaws = 1.0                                  # i2rt's reading: fully open, at the stop
    last_command: np.ndarray | None = None

    @property
    def gripper_torque(self) -> float:
        """The torque the control loop sends the gripper now (force control)."""

        return float(self._commands.torques[6])

    states: list[tuple[float, np.ndarray, np.ndarray]] = []       # (time, position, velocity) via command_joint_state

    def command_joint_state(self, state: dict[str, np.ndarray]) -> None:
        self.states = [*self.states, (time.monotonic(), np.asarray(state["pos"], float).copy(),
                                      np.asarray(state["vel"], float).copy())]
        self.command_joint_pos(state["pos"])

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.last_command = np.asarray(target, float).copy()
        self._commands.torques = np.zeros(7)        # as i2rt: a position command zeroes the torques
        if not self.blocked:
            self.q = np.asarray(target[:6], float).copy() - self.sag

    def get_observations(self) -> dict[str, Any]:
        temperatures = np.full(7, 30.0)
        temperatures[6] = self.temperature_c
        return {"joint_pos": self.q.copy(), "gripper_pos": np.array([self.jaws]), "joint_eff": np.zeros(len(Y.ARM_MOTORS)),
                "gripper_eff": np.array([self.gripper_effort_nm]), "temp_mos": temperatures, "temp_rotor": temperatures}

    def close(self) -> None:
        self.closed = True


class Cell:
    """The fake cell's state a test inspects: arms i2rt created, DISABLE sweeps, questions asked."""

    def __init__(self) -> None:
        self.arms: dict[str, FakeArm] = {}
        self.disabled = 0
        self.asked: list[dict[str, Any]] = []
        self.kinematics = FakeKinematics()

    def released(self) -> bool:
        return self.disabled > 0 or any(arm.closed for arm in self.arms.values())


@pytest.fixture
def cell(tmp_path, monkeypatch):
    redirect_lock(monkeypatch, tmp_path / "ora-cell.lock")
    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)
    monkeypatch.setenv("ORA_CELL_ALERTS", str(tmp_path / "ora-cell-alerts.json"))
    monkeypatch.setattr(real_rig, "MONITOR_PERIOD_S", 0.02)
    monkeypatch.setattr(health, "GRIPPER_EFFORT_WARN_S", 0.05)
    state = Cell()

    def get_yam_robot(*, channel, gripper_type, zero_gravity_mode, gripper_limits_override):
        arm = FakeArm(channel)
        state.arms[{"can_left": "left", "can_right": "right"}[channel]] = arm
        return arm

    modules = {"i2rt": types.ModuleType("i2rt"), "i2rt.robots": types.ModuleType("i2rt.robots"),
               "i2rt.robots.get_robot": types.ModuleType("i2rt.robots.get_robot"),
               "i2rt.robots.utils": types.ModuleType("i2rt.robots.utils")}
    modules["i2rt.robots.get_robot"].get_yam_robot = get_yam_robot
    modules["i2rt.robots.utils"].GripperType = lambda value: value
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def start(self, config, serials):
        self.kinematics = state.kinematics             # the station model the way home is checked with
        self._q = {arm: np.asarray(START[arm], float) for arm in Y.ARMS}
        self._finger = {arm: Y.FINGER_OPEN_M for arm in Y.ARMS}
        if self.mode == "on":
            self._enable(at_rest=False)
        self.start_q = {arm: q.copy() for arm, q in self._q.items()}
        self._commanded = {arm: q.copy() for arm, q in self._q.items()}
        self._over = {arm: {"tracking": 0, "effort": 0} for arm in Y.ARMS}
        self._next_tick = time.monotonic()

    monkeypatch.setattr(RealRig, "_start", start)
    monkeypatch.setattr(RealRig, "_pace", lambda self: None)
    monkeypatch.setattr(real_rig.devices, "disable_all", lambda buses: setattr(state, "disabled", state.disabled + 1))
    ask, ask_record = approval.ask, approval.ask_record

    def recorded_ask(kind, title, text, details=None, **options):
        state.asked.append({"kind": kind, "title": title, "text": text, "details": details})
        return ask(kind, title, text, details, **options)

    def recorded_ask_record(kind, title, text, details=None, **options):
        state.asked.append({"kind": kind, "title": title, "text": text, "details": details})
        return ask_record(kind, title, text, details, **options)

    monkeypatch.setattr(approval, "ask", recorded_ask)
    monkeypatch.setattr(approval, "ask_record", recorded_ask_record)
    previous = signal.getsignal(signal.SIGUSR1)
    yield state
    signal.signal(signal.SIGUSR1, previous)


def _rig(tmp_path, *, gripper_limits: bool = True, **runtime: Any) -> RealRig:
    path = write_config(tmp_path, CONFIG)
    limits = {arm: [0.0, 1.0] for arm in Y.ARMS} if gripper_limits else {}
    cell_calibration.save(tmp_path / "calibration.json",
                          {**cell_calibration.empty(), "rest_reference": REST, "gripper_limits": limits})
    return RealRig({"rig_mode": "on", "cell_config": str(path), **runtime})


def _operator(*answers: Callable[[dict[str, Any]], tuple[bool, str | None] | None]) -> threading.Thread:
    """Answers each question as it appears, in order; ``None`` from an answer leaves that one unanswered."""

    def run() -> None:
        seen: set[str] = set()
        for respond in answers:
            for _ in range(1000):
                waiting = [r for r in approval.pending() if r["id"] not in seen]
                if waiting:
                    break
                time.sleep(0.01)
            else:
                return
            request = waiting[0]
            seen.add(request["id"])
            decision = respond(request)
            if decision is not None:
                try:
                    approval.answer(request["id"], decision[0], decision[1], by="test")
                except Exception:  # noqa: BLE001 - a question withdrawn meanwhile
                    pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _approve(_request):
    return True, None


def _started(tmp_path, cell, **runtime: Any) -> RealRig:
    thread = _operator(_approve)
    rig = _rig(tmp_path, **runtime)
    thread.join(timeout=5)
    cell.asked.clear()
    return rig


# ------------------------------------------------------------------------------------- enabling
def test_a_declined_enable_enables_nothing_and_frees_the_cell(tmp_path, cell) -> None:
    thread = _operator(lambda request: (False, "the table is not clear"))
    with pytest.raises(YamError, match="the arms were not enabled: the table is not clear"):
        _rig(tmp_path, gripper_limits=False)
    thread.join(timeout=5)

    assert cell.arms == {}, "i2rt never took the motors"
    assert holder() is None
    [question] = cell.asked
    assert (question["kind"], question["title"]) == ("enable", "要给两条手臂通电吗？")
    assert "夹爪限位还没标定" in question["text"] and "不在记录的静止位" in question["text"]
    assert question["details"]["gripper_limits_known"] is False


def test_an_approved_enable_holds_the_arms(tmp_path, cell) -> None:
    thread = _operator(_approve)
    rig = _rig(tmp_path)
    thread.join(timeout=5)
    assert set(cell.arms) == set(Y.ARMS) and not cell.released()
    assert "夹爪限位" not in cell.asked[0]["text"]
    rig.close(ask=False)


def test_an_unanswered_enable_is_a_decline(tmp_path, cell, monkeypatch) -> None:
    monkeypatch.setattr(real_rig, "ENABLE_ANSWER_S", 0.2)
    with pytest.raises(YamError, match="the arms were not enabled: no answer in 0.2 s"):
        _rig(tmp_path)
    assert cell.arms == {}


def test_auto_approve_enables_without_a_question(tmp_path, cell) -> None:
    rig = _rig(tmp_path, auto_approve=True)
    assert set(cell.arms) == set(Y.ARMS) and cell.asked == []
    assert rig.operator_approval({"arms": ["left"]}) == (True, None)
    rig.close(ask=False)


# ------------------------------------------------------------------------------------- motions
PLAN = {"schema": "ora.yam-plan.v1", "arms": ["left"], "ticks": 75, "duration_s": 1.5,
        "min_inter_arm_distance_m": 0.31, "min_table_distance_m": {"left": 0.12},
        "max_joint_travel_rad": {"left": np.float64(0.5)}, "waypoints": {"left": [np.zeros(7)]}}


def test_each_motion_waits_for_its_own_approval(tmp_path, cell, monkeypatch) -> None:
    rig = _started(tmp_path, cell)

    thread = _operator(_approve)
    assert rig.operator_approval(PLAN) == (True, None)
    thread.join(timeout=5)
    [question] = cell.asked
    assert (question["kind"], question["title"]) == ("motion", "左臂动作等待批准")
    assert "左臂移动到智能体选的位置，约 1.5 秒；左臂离桌面最近 12 厘米，两臂之间最近 31 厘米，关节最多转 29°。" in question["text"]
    assert "计划：" in question["text"]

    thread = _operator(lambda request: (False, "too close to the cup"))
    assert rig.operator_approval({"schema": "ora.yam-plan.v1", "gripper": {"right": "close"}}) == (
        False, "too close to the cup")
    thread.join(timeout=5)
    assert cell.asked[-1]["title"] == "夹爪动作等待批准" and "右臂夹爪闭合（夹住）" in cell.asked[-1]["text"]

    monkeypatch.setattr(real_rig, "MOTION_ANSWER_S", 0.2)
    assert rig.operator_approval(PLAN) == (False, "no answer in 0.2 s")
    assert not cell.released()
    rig.close(ask=False)


def test_a_home_while_a_plan_waits_voids_the_plan(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)

    def home_then_approve(request):
        os.kill(os.getpid(), signal.SIGUSR1)       # the board's 回到起始位: the worker drives home itself
        time.sleep(0.3)
        return True, None

    thread = _operator(home_then_approve)
    approved, reason = rig.operator_approval(PLAN)
    thread.join(timeout=5)
    assert approved is False and reason.startswith("the operator sent the arms home while this plan waited")
    assert all(np.allclose(cell.arms[arm].q, REST[arm], atol=0.011) for arm in Y.ARMS)
    assert rig.at_rest and not cell.released()
    rig.close(ask=False)


# ------------------------------------------------------------------------------------- closing
def test_closing_at_rest_releases_without_a_question(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    rig.close()

    assert cell.asked == []
    assert all(np.allclose(cell.arms[arm].q, REST[arm], atol=0.011) for arm in Y.ARMS), "home first"
    assert all(arm.closed for arm in cell.arms.values()) and cell.disabled == 1
    assert holder() is None


def test_a_blocked_way_home_asks_and_a_decline_keeps_holding(tmp_path, cell, monkeypatch) -> None:
    rig = _started(tmp_path, cell)
    cell.arms["left"].blocked = True
    notes = []

    def decline_while_held(request):
        notes.append(cell.released())
        return False, "not supported yet"

    def free_the_arm_and_decline(request):
        notes.append(cell.released())
        cell.arms["left"].blocked = False             # someone moved the obstacle
        return False, "try again"

    thread = _operator(decline_while_held, free_the_arm_and_decline)
    rig.close()
    thread.join(timeout=5)

    assert notes == [False, False], "nothing was released while the operator declined"
    assert [q["kind"] for q in cell.asked] == ["release", "release"]
    first = cell.asked[0]
    assert first["title"] == "手臂没能回到起始位" and "左臂跟不上指令" in first["text"]
    assert "批准 = 就地释放" in first["text"] and "第 1/3 次" in first["text"]
    assert first["details"]["blocked_arms"] == ["left"]
    assert np.allclose(cell.arms["left"].q, REST["left"], atol=0.011), "released only once home"
    assert cell.released()


def test_an_approved_release_lets_go_where_the_arms_are(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    cell.arms["right"].blocked = True
    thread = _operator(_approve)
    rig.close()
    thread.join(timeout=5)

    assert [q["kind"] for q in cell.asked] == ["release"]
    assert np.allclose(cell.arms["right"].q, START["right"]) and cell.released()


def test_a_way_home_through_the_table_asks_with_that_reason_and_nothing_moves(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    cell.kinematics.table_m = -0.01
    before = {arm: a.q.copy() for arm, a in cell.arms.items()}
    thread = _operator(_approve)
    rig.close()
    thread.join(timeout=5)

    assert [q["kind"] for q in cell.asked] == ["release"]
    assert "直线路径会碰到桌面或另一条手臂" in cell.asked[0]["text"]
    assert cell.asked[0]["details"]["home"].startswith(real_rig.PATH_BLOCKED)
    assert all(np.allclose(cell.arms[arm].q, q) for arm, q in before.items()) and cell.released()


def test_at_the_bound_the_arms_are_released_in_place_and_the_last_question_says_so(tmp_path, cell,
                                                                                   monkeypatch) -> None:
    monkeypatch.setattr(real_rig, "RELEASE_ANSWER_S", 0.2)
    monkeypatch.setattr(real_rig, "MIN_ANSWER_S", 0.05)
    rig = _started(tmp_path, cell)
    cell.arms["left"].blocked = True
    rig.close()                                      # nobody answers

    assert [q["kind"] for q in cell.asked] == ["release"] * real_rig.CLOSE_HOME_ATTEMPTS
    assert "最后一次" not in cell.asked[0]["text"] and "最后一次" in cell.asked[-1]["text"]
    assert cell.released() and holder() is None


def test_the_close_budget_bounds_the_questions(tmp_path, cell, monkeypatch) -> None:
    monkeypatch.setattr(real_rig, "RELEASE_ANSWER_S", 5.0)
    monkeypatch.setattr(real_rig, "MIN_ANSWER_S", 0.05)
    rig = _started(tmp_path, cell, close_timeout_s=real_rig.CLOSE_MARGIN_S + 0.3)
    cell.arms["left"].blocked = True
    started = time.monotonic()
    rig.close()

    assert time.monotonic() - started < 3.0
    assert [q["kind"] for q in cell.asked] == ["release"] and "最后一次" in cell.asked[0]["text"]
    assert cell.released()


def test_a_home_that_reaches_rest_while_the_release_question_waits_ends_it(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    cell.arms["left"].blocked = True

    def unblock_and_home(request):
        cell.arms["left"].blocked = False
        os.kill(os.getpid(), signal.SIGUSR1)
        return None

    thread = _operator(unblock_and_home)
    rig.close()
    thread.join(timeout=5)
    assert [q["kind"] for q in cell.asked] == ["release"]
    assert np.allclose(cell.arms["left"].q, REST["left"], atol=0.011) and cell.released()
    assert approval.pending() == []


def test_auto_approve_closes_without_a_question(tmp_path, cell) -> None:
    rig = _rig(tmp_path, auto_approve=True)
    cell.arms["left"].blocked = True
    rig.close()
    assert cell.asked == [] and cell.released()


def test_a_start_up_failure_releases_without_a_question(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    rig.close(ask=False)
    assert cell.asked == [] and cell.released()
    assert all(np.allclose(cell.arms[arm].q, START[arm]) for arm in Y.ARMS), "already at the start pose"


def test_the_wrist_table_check_takes_the_owners_widened_limits(monkeypatch) -> None:
    plane = {"tilt_deg": 3.4, "height_m": -0.115, "inlier_fraction": 0.8, "points": 900}
    monkeypatch.setattr(real_rig.cell_geometry, "table_plane", lambda *a, **k: dict(plane))

    class Kinematics:
        camera_corrections: dict = {}

        def __init__(self, *_: Any) -> None:
            pass

        def wrist_camera(self, arm: str) -> np.ndarray:
            return np.eye(4)

        def set_state(self, *_: Any) -> None:
            pass

    monkeypatch.setattr(real_rig, "StationKinematics", Kinematics)
    monkeypatch.setattr(real_rig, "build_cell_model", lambda table: None)

    def decided(widened: dict) -> dict:
        rig = object.__new__(RealRig)
        rig.calibration = {"table": {"center_xy_m": [0.4, 0.0], "size_xy_m": [0.6, 0.9], "top_z_m": -0.09}}
        rig.cell = types.SimpleNamespace(wrist_table_check=widened)
        rig.cameras_by_role = {f"wrist_{arm}": types.SimpleNamespace(capture=lambda: {"depth": None, "intrinsic_cv": None})
                               for arm in Y.ARMS}
        rig.kinematics, rig.mode, rig.startup_report = Kinematics(), "on", {}
        rig._q, rig._finger = {}, {}
        rig._decide_table()
        return rig.startup_report

    strict = decided({})
    assert all(check["passed"] is False for check in strict["wrist_table"].values())
    assert "wrist_table_widened" not in strict
    widened = {"max_tilt_deg": 4.5, "max_height_gap_m": 0.035, "reason": "recalibrating"}
    report = decided(widened)
    assert all(check["passed"] is True for check in report["wrist_table"].values())
    assert report["wrist_table_widened"] == widened


def test_home_counts_a_sagging_arm_at_rest_and_says_so_in_the_lock(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    for arm in cell.arms.values():
        arm.sag = 0.02                          # the real station: 0.026 to 0.037 rad below rest when held

    assert rig.home() == "at rest, holding"
    assert holder().endswith(AT_REST) or AT_REST in holder()
    for arm in cell.arms.values():
        arm.sag = 0.06                          # too far from rest: holding, the homing mark gone
    rig.at_rest = False
    outcome = rig.home()
    assert outcome.startswith("not at rest")
    assert HOMING not in holder() and AT_REST not in holder()
    rig.close(ask=False)


# ------------------------------------------------------------------------------------- watching
def _alerts_until(predicate, timeout_s: float = 3.0) -> list[dict[str, Any]]:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        found = cell_alerts.current()["alerts"]
        if predicate(found):
            return found
        time.sleep(0.02)
    raise AssertionError(f"alerts never matched: {cell_alerts.current()}")


def test_enabled_jaws_are_held_off_the_stops(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)

    for arm in cell.arms.values():
        assert arm.jaws == 1.0 and arm.last_command[-1] == Y.GRIPPER_HOLD_RANGE[1]
    rig.step({}, {"left": Y.FINGER_OPEN_M})          # an "open" target is held just inside the stop
    assert cell.arms["left"].last_command[-1] == Y.GRIPPER_HOLD_RANGE[1]
    rig.close(ask=False)


def test_a_stopped_control_loop_is_an_alert_and_marks_the_lock(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    cell.arms["right"].motor_chain.running = False    # a motor protection stopped i2rt's loop

    found = _alerts_until(lambda a: any(x["code"] == "arms_not_controlled" for x in a))
    assert found[0]["level"] == "critical" and found[0]["arm"] == "right"
    assert NOT_CONTROLLED in holder()
    events = rig.step({}, {}).events
    assert {"rig_alert", "motor_fault"} <= {event["code"] for event in events}
    rig.close(ask=False)
    assert cell_alerts.current()["alerts"] == []


def test_a_warming_motor_and_a_pushing_gripper_are_said_before_the_protection(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    cell.arms["left"].temperature_c = 52.0
    cell.arms["left"].gripper_effort_nm = 5.0
    found = _alerts_until(lambda a: {"motor_warm", "gripper_pushing"} <= {x["code"] for x in a})
    assert all(x["level"] == "warning" for x in found)
    assert any("夹爪" in x["text"] for x in found)

    rig.step({}, {"left": 0.0})                         # a gripper command: the left gripper works
    assert grippers.in_force_control(cell.arms["left"]) and cell.arms["left"].gripper_torque < 0.0
    cell.arms["left"].temperature_c = 62.0             # hot and still pushing: relaxed, so it stops
    _alerts_until(lambda a: any(x["code"] == "motor_hot" for x in a))
    end = time.monotonic() + 2.0
    while time.monotonic() < end and grippers.in_force_control(cell.arms["left"]):
        time.sleep(0.02)
    assert not grippers.in_force_control(cell.arms["left"])
    assert cell.arms["left"].gripper_gain == 0.0 and cell.arms["left"].gripper_torque == 0.0
    rig.close(ask=False)


def test_a_gripper_works_only_for_a_gripper_command_and_rests_relaxed(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    assert all(arm.gripper_gain == 0.0 for arm in cell.arms.values())    # enabled: no gripper gives torque

    rig.step({"left": rig.arm_q("left")}, {})          # an arm motion leaves the jaws relaxed
    assert cell.arms["left"].gripper_gain == 0.0
    rig.step({}, {"right": 0.0})
    # Force control: no position gain, a torque toward closed within the cap; the left gripper untouched.
    right = cell.arms["right"]
    assert grippers.in_force_control(right) and right.gripper_gain == 0.0
    assert -Y.GRIPPER_FORCE_LIMIT_N * Y.GRIPPER_FORCE_TO_TORQUE - 1e-12 <= right.gripper_torque < 0.0
    assert not grippers.in_force_control(cell.arms["left"]) and cell.arms["left"].gripper_torque == 0.0
    assert rig.home() == "at rest, holding"            # at rest nothing is gripped
    assert all(arm.gripper_gain == 0.0 for arm in cell.arms.values())
    rig.close(ask=False)


def test_home_with_every_control_loop_stopped_drives_nothing(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    for arm in cell.arms.values():
        arm.motor_chain.running = False         # both motor chains stopped: nothing left to command
    before = {arm: a.q.copy() for arm, a in cell.arms.items()}

    assert rig.home() == "no control loop running: nothing driven"
    assert all(np.array_equal(cell.arms[arm].q, q) for arm, q in before.items())
    assert HOMING not in (holder() or "")
    rig.close(ask=False)


def test_home_through_the_table_moves_nothing_and_holds_where_the_arms_are(tmp_path, cell) -> None:
    """The episode's home checks the straight line home against the table and between the arms first
    (``ora.cell.homing``), as the operator tools' home always has."""
    rig = _started(tmp_path, cell)
    cell.kinematics.table_m = -0.01
    before = {arm: a.q.copy() for arm, a in cell.arms.items()}

    outcome = rig.home()
    assert outcome.startswith("path home not clear (the left arm passes through the table at 0%)")
    assert all(np.allclose(cell.arms[arm].q, q) for arm, q in before.items())
    assert not rig.at_rest and AT_REST not in (holder() or "")
    cell.kinematics.table_m = 0.2
    assert rig.home() == "at rest, holding"
    rig.close(ask=False)


def test_the_operators_stop_here_ends_a_motion_where_the_arms_are(tmp_path, cell) -> None:
    from ora.cell.operator import approval

    rig = _started(tmp_path, cell)
    start = rig.arm_q("left").copy()
    target = start + np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    rig.step({"left": start + (target - start) * 0.1}, {})
    assert rig.step({"left": target}, {}).events == []
    record = approval.request_stop_motion(by="test")
    assert record["pid"] == os.getpid()
    events = rig.step({"left": target + 0.01}, {}).events
    assert [e["code"] for e in events] == ["operator_hold"] and "test" in events[0]["detail"]
    assert approval.take_stop_request() is None              # consumed
    assert rig.step({}, {}).events == []                     # the next motion is not stopped
    rig.close(ask=False)


def test_a_stop_sent_while_the_agent_plans_refuses_that_motion_before_it_moves(tmp_path, cell) -> None:
    from ora.cell.operator import approval

    rig = _started(tmp_path, cell, auto_approve=True)
    approval.request_stop_motion(by="web")
    approved, reason = rig.operator_approval({"arms": ["left"]})
    assert approved is False and "stopped the robot while this motion was being planned (web)" in reason
    assert rig.operator_approval({"arms": ["left"]}) == (True, None)        # consumed: the next one runs
    approval.request_stop_motion(by="web")
    assert approval.withdraw_stop_request() is True and approval.withdraw_stop_request() is False
    assert rig.operator_approval({"arms": ["left"]}) == (True, None)
    rig.close(ask=False)


def test_an_arm_pushed_on_from_outside_stops_and_holds_where_it_is(tmp_path, cell) -> None:
    """RealRig's guards: measured joint torque over i2rt's model torque for the ticks is external_load_stop."""
    rig = _started(tmp_path, cell)
    arm = cell.arms["right"]
    arm.kdl = types.SimpleNamespace(compute_inverse_dynamics=lambda q, qd, qdd: np.zeros(6))
    pushed = np.zeros(6)
    pushed[2] = Y.EXTERNAL_LOAD_STOP_NM[2] + 2.0
    original = arm.get_observations
    arm.get_observations = lambda: {**original(), "joint_eff": pushed.copy()}
    codes = []
    for _ in range(Y.EXTERNAL_LOAD_STOP_TICKS):
        codes = [event["code"] for event in rig.step({}, {}).events]
    assert "external_load_stop" in codes
    assert rig.loads.take_peaks()["right"][2] == round(pushed[2], 3)
    rig.close(ask=False)


def test_without_smooth_commands_a_tick_is_one_position_step(tmp_path, cell) -> None:
    rig = _started(tmp_path, cell)
    target = np.asarray(START["left"], float) + 0.01
    rig.step({"left": target}, {})
    time.sleep(3 * Y.CONTROL_PERIOD_S)
    assert cell.arms["left"].states == [] and rig._streamer is None
    assert np.allclose(cell.arms["left"].last_command[:6], target)
    rig.close(ask=False)


def test_smooth_commands_stream_each_tick_along_its_line_with_its_velocity(tmp_path, cell) -> None:
    """runtime.smooth_arm_commands: between ticks the target moves along the line from the last one, with that
    line's velocity as the feed-forward, and ends on the tick's target with none."""
    rig = _started(tmp_path, cell, smooth_arm_commands=True)
    arm = cell.arms["left"]
    start = rig._commanded["left"].copy()
    target = start + 0.012                                     # 0.6 rad/s for one tick
    arm.states = []
    rig.step({"left": target}, {})
    time.sleep(2.5 * Y.CONTROL_PERIOD_S)
    streamed = [(pos[:6], vel[:6]) for _, pos, vel in arm.states]
    assert len(streamed) >= 3, "several commands within one tick"
    expected_velocity = (target - start) / Y.CONTROL_PERIOD_S
    assert all(np.allclose(vel, expected_velocity) for _, vel in streamed)
    shares = [float(np.mean((pos - start) / (target - start))) for pos, _ in streamed]
    assert shares == sorted(shares) and 0.0 <= shares[0] < 1.0
    assert np.allclose(arm.last_command[:6], target), "the tick ends on its target"
    assert rig._segments == {}
    # A hold is commanded at once, never streamed.
    rig.step({"left": target + 0.012}, {})
    rig._hold_measured()
    assert rig._segments == {}
    rig.close(ask=False)
