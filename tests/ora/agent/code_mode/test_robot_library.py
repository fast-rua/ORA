"""The ``robot`` library's ``library: extended`` half (``ora/agent/code_mode/robot.py``): typed results and filled
tool parameters, read off what scripts read on recorded runs (``tests/data/code_mode/script-answers.json``).

The answers were rebuilt with ORA's own receipt projectors, delivery rules and causal filter from code-mode runs of
2026-10-27/28 on LIBERO-Pro, RoboTwin, RoboCasa365, RoboLab (legacy tools: the effect is ``receipt``) and RoboDojo
(synchronous toolkit tools: ``result.receipt``, a multi-stage ``grasp_execute`` adds ``result.stages``). Without
``library: extended`` a call returns the answer unchanged and sends its arguments as given.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.code_mode import docs as code_docs, robot
from ora.agent.code_mode.ora_lib import Direction, Point

ANSWERS = json.loads((Path(__file__).parents[4] / "tests/data/code_mode/script-answers.json").read_text())["answers"]

#: Every recorded answer's (backend, tool, status, failure_code, failure_stage, ok).
OUTCOMES = [
    ("libero", "geometry_query", None, None, None, True),
    ("libero", "move_pose", "reached", None, None, True),
    ("libero", "move_pose", "not_reached", "execution_fault", "execution", False),
    ("libero", "move_to", "reached", None, None, True),
    ("libero", "observe", None, None, None, True),
    ("libero", "rotate_pitch", "reached", None, None, True),
    ("libero", "rotate_wrist", "not_reached", "execution_fault", "execution", False),
    ("libero", "set_gripper", "not_reached", "endpoint_residual_exceeded", "endpoint_verification", False),
    ("libero", "set_gripper", "not_reached", "gripper_stalled_before_target", "execution", False),
    ("libero", "wait_ticks", "reached", None, None, True),
    ("robocasa365", "move_base", "not_reached", "blocked", "execution", False),
    ("robocasa365", "move_pose", "not_reached", "endpoint_residual_exceeded", "endpoint_verification", False),
    ("robocasa365", "move_pose", "plan_rejected", "target_unreachable_at_base_pose", "plan_validation", False),
    ("robocasa365", "move_to", "plan_rejected", "target_unreachable_at_base_pose", "plan_validation", False),
    ("robocasa365", "recover_pose", "reached", None, None, True),
    ("robocasa365", "rotate_wrist", "reached", None, None, True),
    ("robocasa365", "set_gripper", "reached", None, None, True),
    ("robocasa365", "set_gripper", "not_reached", "gripper_stalled_before_target", "execution", False),
    ("robocasa365", "wait_ticks", "reached", None, None, True),
    ("robodojo", "grasp_execute", "executed", None, None, True),
    ("robodojo", "grasp_execute", "stopped", "controller_stalled", "approach", False),
    ("robodojo", "move_pose", "reached", None, None, True),
    ("robodojo", "move_pose", "planner_failed", "planner_failed", "planner", False),
    ("robodojo", "move_to", "reached", None, None, True),
    ("robodojo", "observe", None, None, None, True),
    ("robodojo", "recover_pose", "reached", None, None, True),
    ("robodojo", "return_home", "not_reached", "episode_terminal_during_execution", "execution", False),
    ("robodojo", "rotate_pitch", "planner_failed", "planner_failed", "planner", False),
    ("robodojo", "rotate_wrist", "reached", None, None, True),
    ("robodojo", "set_gripper", "reached", None, None, True),
    ("robodojo", "set_gripper", "not_reached", "gripper_stalled_before_target", "execution", False),
    ("robodojo", "wait_ticks", "reached", None, None, True),
    ("robolab", "move_pose", "reached", None, None, True),
    ("robolab", "move_pose", "not_reached", "controller_stalled", "execution", False),
    ("robolab", "move_to", "reached", None, None, True),
    ("robolab", "set_gripper", "not_reached", "gripper_stalled_before_target", "execution", False),
    ("robolab", "set_gripper", "reached", None, None, True),
    ("robotwin", "grasp_execute", "completed", None, None, True),
    ("robotwin", "mark_point", "solved", None, None, True),
    ("robotwin", "move_pose", "reached", None, None, True),
    ("robotwin", "move_pose", "executed", None, None, True),
    ("robotwin", "move_to", "reached", None, None, True),
    ("robotwin", "recover_pose", "reached", None, None, True),
    ("robotwin", "set_gripper", "executed", None, None, True),
]


def _recorded(backend: str, tool: str, status: str | None = None) -> robot.Result:
    for item in ANSWERS:
        answer = item["answer"]
        result = robot.typed(answer["tool"], answer)
        if item["backend"] == backend and answer["tool"] == tool and (status is None or result.status == status):
            return result
    raise LookupError((backend, tool, status))


@pytest.mark.parametrize("item,expected", list(zip(ANSWERS, OUTCOMES)),
                         ids=[f"{row[0]}-{row[1]}-{row[2]}" for row in OUTCOMES])
def test_each_recorded_answer_reads_into_its_outcome_and_stays_the_same_dict(item, expected) -> None:
    answer = item["answer"]
    result = robot.typed(answer["tool"], answer)
    assert (item["backend"], answer["tool"], result.status, result.failure_code, result.failure_stage,
            result.ok) == expected
    assert dict(result) == answer and result.raw is answer and isinstance(result, dict)
    assert json.dumps(result, sort_keys=True) == json.dumps(answer, sort_keys=True), "json.dumps(r) still works"
    assert result["ok"] is True, "the answer's own flag: the call was not refused"
    acted = "receipt" in answer or "receipt" in answer["result"] or "stages" in answer["result"]
    if acted:  # every recorded backend states the pose after an action in the next public robot facts, not here
        assert result.pose_after is None
    assert repr(result).startswith(f"Result({answer['tool']}: ok={expected[-1]}")


def test_target_errors_come_from_the_targeted_arm_in_radians() -> None:
    moved = _recorded("libero", "move_pose", "reached")  # orientation_deg 0.117309 (policy_angles_degrees_v1)
    assert moved.target_error_m == pytest.approx(0.0010389587677125725)
    assert moved.target_error_deg == pytest.approx(0.117309)
    assert moved.arms["right"].target_error_m == moved.target_error_m and list(moved.arms) == ["right"]
    rejected = _recorded("robocasa365", "move_pose", "plan_rejected")  # not moved: the whole distance remains
    assert rejected.target_error_m == pytest.approx(0.5196, abs=1e-4) and rejected.ok is False
    assert _recorded("robodojo", "move_pose", "reached").target_error_m == pytest.approx(0.0018827147649033164)
    assert _recorded("robotwin", "move_pose", "reached").target_error_m == pytest.approx(0.002258)


def _cut_moves() -> list[dict]:
    path = Path(__file__).parents[3] / "data/code_mode/cut-moves.json"
    return json.loads(path.read_text(encoding="utf-8"))["receipts"]


@pytest.mark.parametrize("status, ok", [("reached", True), ("not_reached", False)])
def test_a_move_that_reached_before_its_joints_settled_is_ok_as_the_prompt_says(status, ok) -> None:
    receipt = {"status": status, "failure_code": "arm_not_settled", "failure_stage": "settle"}
    result = robot.typed("move_pose", {"tool": "move_pose", "ok": True, "result": {"receipt": receipt}})
    assert (result.status, result.failure_code, result.ok) == (status, "arm_not_settled", ok)


@pytest.mark.parametrize("item", _cut_moves(), ids=lambda item: item["recorded_summary"].split(",")[-1].strip())
def test_a_cut_move_reads_as_the_distance_left_to_its_target(item) -> None:
    receipt = item["receipt"]
    arm = receipt["detail"]["target_arms"][0]
    result = robot.typed("move_pose", {"tool": "move_pose", "ok": True, "result": {"receipt": receipt}})
    left = receipt["detail"]["arms"][arm]["motion_progress"]["remaining_to_goal_m"]
    assert receipt["detail"]["arms"][arm]["pose_error"]["translation_m"] < 0.001, "the cut point's residual"
    assert (result.status, result.failure_code, result.ok) == ("not_reached", "controller_timeout", False)
    assert result.target_error_m == left == result.arms[arm].target_error_m
    assert result.target_error_deg is None and result.cut_short is True and result.arms[arm].cut_short is True


def test_a_cut_stage_of_a_grasp_reads_as_a_cut_move_after_earlier_errors() -> None:
    cut = _cut_moves()[0]["receipt"]
    arm = cut["detail"]["target_arms"][0]
    earlier = {"status": "reached", "detail": {"target_arms": [arm], "arms": {arm: {
        "targeted": True, "pose_error": {"translation_m": 0.002, "orientation_deg": 1.5}}}}}
    answer = {"tool": "grasp_execute", "ok": True, "result": {"status": "aborted", "stages": [
        {"stage": "open", "status": "executed", "receipt": earlier},
        {"stage": "pregrasp", "status": "not_reached", "receipt": cut}]}}
    result = robot.typed("grasp_execute", answer)
    assert result.cut_short is True and result.target_error_deg is None
    assert result.target_error_m == cut["detail"]["arms"][arm]["motion_progress"]["remaining_to_goal_m"]


def test_a_move_that_was_not_cut_has_no_cut_short_field_value() -> None:
    assert _recorded("robodojo", "move_pose", "reached").cut_short is None


def test_gripper_signals_per_backend() -> None:
    libero = _recorded("libero", "set_gripper", "not_reached")  # detail.gripper_outcome, its one effective target
    assert (libero.measured_aperture_mm, libero.commanded_opening) == (pytest.approx(25.2644, abs=1e-4), 0.0)
    assert libero.measured_opening == pytest.approx(0.3158, abs=1e-4) and libero.arms["right"].commanded_opening == 0.0
    robolab = _recorded("robolab", "set_gripper", "reached")  # detail.arms.<arm>.gripper_outcome
    assert robolab.measured_aperture_mm == pytest.approx(84.7525, abs=1e-4) and robolab.commanded_opening == 1.0
    robodojo = _recorded("robodojo", "set_gripper", "reached")  # normalized opening only: no calibrated aperture
    assert robodojo.measured_aperture_mm is None and robodojo.measured_opening == pytest.approx(0.9756, abs=1e-4)
    assert _recorded("robotwin", "set_gripper").gripper_command == "close"  # the command target, nothing measured
    assert _recorded("robotwin", "set_gripper").measured_opening is None
    grasp = _recorded("robotwin", "grasp_execute")  # detail.verification.gripper_opening_m, in metres
    assert grasp.measured_aperture_mm == pytest.approx(49.5256, abs=1e-4)


def test_a_multi_stage_tool_reads_its_own_status_its_first_failed_stage_and_each_latest_signal() -> None:
    executed = _recorded("robodojo", "grasp_execute", "executed")
    assert executed.ok and executed.failure_code is None, "a closing jaw that stalls on the object passes its stage"
    assert executed.commanded_opening == 0.0 and executed.measured_opening == pytest.approx(0.7316, abs=1e-4)
    stopped = _recorded("robodojo", "grasp_execute", "stopped")
    assert (stopped.failure_stage, stopped.failure_code) == ("approach", "controller_stalled")
    assert stopped.target_error_m == pytest.approx(0.0216, abs=1e-4)


def test_contact_is_only_a_public_estimate() -> None:
    assert _recorded("libero", "set_gripper", "not_reached").contact is True  # contact_motion.contact_estimated
    assert _recorded("libero", "move_pose", "reached").contact is False
    assert _recorded("robolab", "move_pose", "reached").contact is None, "no estimate: physical_cause unknown"
    pushed = robot.typed("move_pose", {"ok": True, "result": {}, "receipt": {
        "status": "not_reached", "failure_code": "blocked", "physical_cause": "blocked_by_contact", "detail": {}}})
    assert pushed.contact is True


@pytest.mark.parametrize("tool, detail", [
    ("move_pose", {"endpoint": {"stop_reason": "contact_detected"}}),  # the gripper contact stop: reached
    ("move_base", {"base": {"stop_reason": "base_contact"}}),
])
def test_a_measured_contact_stop_is_a_contact(tool, detail) -> None:
    stopped = robot.typed(tool, {"ok": True, "result": {}, "receipt": {
        "status": "reached", "failure_code": None, "physical_cause": "unknown", "detail": detail}})
    assert stopped.contact is True
    other = robot.typed(tool, {"ok": True, "result": {}, "receipt": {
        "status": "not_reached", "failure_code": "blocked", "physical_cause": "unknown",
        "detail": {"endpoint": {"stop_reason": "joint_lag"}, "base": {"stop_reason": "no_progress"}}}})
    assert other.contact is None, "a stop no contact made is no estimate"


def test_observe_gives_each_arm_s_measured_pose_and_gripper() -> None:
    single = _recorded("libero", "observe")
    assert len(single.pose_after) == 7 and single.gripper_command == "close"
    assert single.measured_aperture_mm == pytest.approx(3.7613, abs=1e-4)
    dual = _recorded("robodojo", "observe")
    assert dual.pose_after is None and dual.measured_opening is None, "two arms: read result.arms"
    assert set(dual.arms) == {"left", "right"} and len(dual.arms["left"].pose_after) == 7
    assert dual.arms["right"]["gripper_command"] == "open"


def test_an_unprojected_receipt_is_read_inside_and_gives_the_pose_after() -> None:
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    result = robot.typed("move_pose", {"ok": True, "result": {"status": "submitted"}, "receipt": {"receipt": {
        "status": "reached", "failure_code": None, "after": {"left": {"pose_xyz_xyzw": pose}},
        "detail": {"target_arms": ["left"], "arms": {"left": {"pose_error": {"translation_m": 0.002,
                                                                           "orientation_rad": 0.01}}}}}}})
    assert result.ok and result.pose_after == pose and result.target_error_deg == pytest.approx(math.degrees(0.01))


# --- pose_after: the measured pose in the observation the action published ---------------------------------------

LEFT = [-0.2995290756225586, -0.35229986906051636, 0.921500027179718, -4.327419134142474e-08,
        -6.000607299938565e-08, 0.7072135210037231, 0.7069999575614929]
RIGHT = [0.3004709482192993, -0.35229986906051636, 0.921500027179718, -4.327419134142474e-08,
         -6.000607299938565e-08, 0.7072135210037231, 0.7069999575614929]


def mount_at(root, monkeypatch, *, arms: dict | None = None, error: dict | None = None) -> None:
    """A mounted observation index in ``root`` (``library.py``'s layout) with the measured end effectors ``arms``."""

    (root / "0001").mkdir(parents=True)
    (root / "0001" / "proprioception.json").write_text(json.dumps({"end_effectors": {
        arm: {"pose_xyz_xyzw": pose} for arm, pose in (arms or {}).items()}}))
    index = {"schema": "ora.code-mode-observation.v1", "observation_id": "observation-0002", "cameras": {}}
    index.update({"error": error} if error else {"proprioception": "0001/proprioception.json"})
    (root / "current.json").write_text(json.dumps(index))
    monkeypatch.setattr(robot, "OBSERVATION_MOUNT", str(root))


def acted(backend: str, tool: str, status: str | None = None, *, action: int = 1) -> dict:
    """A recorded action answer as the host sends it: with the number it gives the action in the script (RoboDojo's
    recorded answers were pruned of it)."""

    for item in ANSWERS:
        if item["backend"] == backend and item["answer"]["tool"] == tool and (
                status is None or robot.typed(tool, item["answer"]).status == status):
            return {**item["answer"], "action": action}
    raise LookupError((backend, tool, status))


def test_an_action_s_pose_after_is_the_measured_pose_in_the_observation_it_published(tmp_path, monkeypatch) -> None:
    """None of the five backends puts the pose after an action in the answer a script receives; each publishes it in
    the next observation, which the mount follows. Recorded move answers of the three dual-arm backends and the
    single-arm LIBERO one."""
    mount_at(tmp_path, monkeypatch, arms={"left": LEFT, "right": RIGHT})
    for backend, tool, status, arm in (("robodojo", "move_pose", "reached", "left"), ("robodojo", "move_to", "reached", "left"),
                                       ("robotwin", "move_pose", "reached", "left"),
                                       ("robodojo", "set_gripper", "reached", "left"), ("libero", "move_pose", "reached", "right")):
        answer = acted(backend, tool, status)
        result = robot.typed(tool, answer)
        assert result.pose_after == {"left": LEFT, "right": RIGHT}[arm], (backend, tool)
        assert result.arms["left"].pose_after == LEFT and result.arms["right"].pose_after == RIGHT
        assert result.target_error_m is not None or tool == "set_gripper"
        assert dict(result) == answer, "the answer itself is unchanged"
    # Without the number the host gives an action, the answer is a read: nothing is added.
    plain = robot.typed("move_pose", {k: v for k, v in acted("robodojo", "move_pose", "reached").items() if k != "action"})
    assert plain.pose_after is None and set(plain.arms) == {"left"} and plain.arms["left"].pose_after is None


def test_a_receipt_s_own_pose_wins_and_an_unclear_arm_gives_no_single_pose(tmp_path, monkeypatch) -> None:
    mount_at(tmp_path, monkeypatch, arms={"left": LEFT, "right": RIGHT})
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    receipt = {"receipt": {"status": "reached", "failure_code": None, "after": {"left": {"pose_xyz_xyzw": pose}},
                           "detail": {"target_arms": ["left"]}}}
    result = robot.typed("move_pose", {"ok": True, "result": {"status": "submitted"}, "receipt": receipt, "action": 2})
    assert result.pose_after == pose and result.arms["right"].pose_after == RIGHT
    named = robot.typed("move_pose", {"ok": True, "result": {}, "action": 2,
                                      "receipt": {"status": "reached", "detail": {"target_arms": ["right"]}}})
    assert named.pose_after == RIGHT, "the arm the receipt names, though it states nothing else of it"
    # An action whose answer names no arm, on a two-arm mount: each arm's pose, no single one.
    nameless = {"ok": True, "result": {}, "receipt": {"status": "reached"}, "action": 3}
    unclear = robot.typed("wait_ticks", nameless)
    assert unclear.pose_after is None and (unclear.arms["left"].pose_after, unclear.arms["right"].pose_after) == (LEFT, RIGHT)
    gripper = robot.typed("set_gripper", acted("libero", "set_gripper", "not_reached"))  # it names its arm (right)
    assert gripper.pose_after == RIGHT


def test_a_single_arm_mount_gives_the_one_pose_to_an_action_that_names_no_arm(tmp_path, monkeypatch) -> None:
    mount_at(tmp_path, monkeypatch, arms={"right": RIGHT})
    result = robot.typed("wait_ticks", {"ok": True, "result": {}, "receipt": {"status": "reached"}, "action": 3})
    assert result.pose_after == RIGHT and list(result.arms) == ["right"]


def test_no_pose_after_where_no_observation_is_mounted_or_it_is_closed(tmp_path, monkeypatch) -> None:
    answer = acted("robodojo", "move_pose", "reached")
    monkeypatch.setattr(robot, "OBSERVATION_MOUNT", str(tmp_path / "nothing"))
    assert robot.typed("move_pose", answer).pose_after is None, "no mount"
    mount_at(tmp_path / "closed", monkeypatch, arms={"left": LEFT},
             error={"code": "action_pending_fresh_observation", "message": "closed"})
    assert robot.typed("move_pose", answer).pose_after is None, "the observation is closed: no guess"


def test_the_pose_after_of_a_real_mount_is_what_observe_reports(published, tmp_path, monkeypatch) -> None:
    from ora.agent.code_mode.library import ObservationMount

    mount = ObservationMount(published.control_root, tmp_path / "mount")
    mount.root.mkdir()
    mount.refresh()
    monkeypatch.setattr(robot, "OBSERVATION_MOUNT", str(mount.root))
    manifest = json.loads((published.control_root.parent / "policy-observations" / "observation-0001"
                           / "observation.json").read_text())
    measured = {arm: record["pose_xyz_xyzw"] for arm, record in manifest["proprioception"]["end_effectors"].items()}
    result = robot.typed("move_pose", acted("robodojo", "move_pose", "reached"))
    assert result.pose_after == measured["left"] and result.arms["right"].pose_after == measured["right"]
    observed = robot.typed("observe", _recorded("robodojo", "observe"))  # observe reports the same fact the same way
    assert observed.arms["left"].pose_after is not None


def test_a_reused_answer_is_marked() -> None:
    answer = {"ok": True, "tool": "vision_ground", "result": {"status": "candidates"}}
    fresh, reused = robot.typed("vision_ground", answer), robot.typed("vision_ground", {**answer, "reused": True})
    assert (fresh.reused, reused.reused) == (False, True) and reused["reused"] is True
    assert "reused" not in repr(fresh) and repr(reused).endswith("reused=True)")
    assert robot.typed("vision_ground", {**answer, "reused": "yes"}).reused is False


def test_two_targeted_arms_give_the_larger_error_and_no_single_pose() -> None:
    errors = {arm: {"targeted": True, "pose_error": {"translation_m": value, "orientation_deg": 1.0}}
              for arm, value in (("left", 0.004), ("right", 0.009))}
    result = robot.typed("move_pose", {"ok": True, "result": {"status": "executed", "receipt": {
        "status": "reached", "detail": {"target_arms": ["left", "right"], "arms": errors}}}})
    assert result.target_error_m == 0.009 and result.arms["left"].target_error_m == 0.004


def test_a_recovery_cut_by_its_budget_reads_as_cut_with_no_distance_to_a_pose() -> None:
    effect = {"status": "not_reached", "failure_code": "controller_timeout", "detail": {
        "target_arms": ["left"], "arms": {"left": {
            "targeted": True, "pose_error": {"translation_m": 0.0, "orientation_rad": 0.0},
            "recovery_progress": {"outcome": "cut_short", "joint_distance_end_rad": 0.28}}}}}
    cut = robot.typed("return_home", {"ok": True, "receipt": effect})
    assert cut.cut_short is True and cut.target_error_m is None and cut.target_error_deg is None


def test_the_summary_is_the_action_s_line_or_the_refusal_s_words() -> None:
    line = "left arm 7.7 mm from target, not reached: controller tracking error, TCP moved 37 mm"
    moved = robot.typed("move_pose", {"ok": True, "receipt": {"status": "not_reached", "action_summary": line}})
    assert moved.summary == line and f"summary={line!r}" in repr(moved)
    why = "no joint configuration reaches this pose even with nothing in the way; nothing was executed"
    refused = robot.typed("move_pose", {"ok": False, "error": {"code": "goal_unreachable", "message": why}})
    assert refused.summary == why and refused.failure_code == "goal_unreachable"
    assert robot.typed("mark_point", {"ok": True, "result": {"marks": []}}).summary is None


def test_a_refused_or_failed_read_only_call() -> None:
    refused = robot.typed("move_pose", {"ok": False, "error": {"code": "script_action_limit", "message": "limit"}})
    assert (refused.ok, refused.status, refused.failure_code) == (False, "refused", "script_action_limit")
    failed = robot.typed("mark_point", {"ok": True, "result": {"status": "failed", "marks": []}})
    assert failed.ok is False and failed.status == "failed"


# --- the library fills tool parameters ---------------------------------------------------------------------------

class Value:
    """A stand-in for ``ora_lib``'s typed values: NumPy arrays and a ``frame``."""

    def __init__(self, frame: str, **fields) -> None:
        self.frame = frame
        for name, value in fields.items():
            setattr(self, name, np.asarray(value, float))


class Jsonable:
    def __init__(self, data: dict) -> None:
        self.data = data

    def to_json(self) -> dict:
        return self.data


def test_an_offset_in_a_tcp_frame_is_a_delta_in_the_end_effector_frame() -> None:
    assert robot.target(Value("tcp:right", xyz=[0, 0, -0.05]), arm="right") == {
        "position_mode": "delta", "translation_m": [0.0, 0.0, -0.05], "position_frame": "end_effector",
        "orientation_mode": "hold_current"}


def test_a_world_position_is_absolute_and_a_world_pose_brings_its_quaternion() -> None:
    assert robot.target({"frame": "world", "xyz": [0.4, 0.1, 0.9]}, interaction="precision") == {
        "position_mode": "absolute", "position_m": [0.4, 0.1, 0.9], "orientation_mode": "hold_current",
        "interaction": "precision"}
    pose = Jsonable({"frame": "world", "xyz": [0.4, 0.1, 0.9], "xyzw": [0, 1, 0, 0]})
    assert robot.target(pose) == {"position_mode": "absolute", "position_m": [0.4, 0.1, 0.9],
                                  "orientation_mode": "absolute_quaternion", "orientation_xyzw": [0.0, 1.0, 0.0, 0.0]}


def test_approach_and_jaw_line_world_directions_are_world_axes() -> None:
    down, across = Value("world", v=[0, 0, -1]), {"frame": "world", "v": [0, 1, 0]}
    assert robot.target(None, approach=down, jaw_line=across) == {
        "position_mode": "hold_current", "orientation_mode": "world_axes",
        "approach_world": [0.0, 0.0, -1.0], "jaw_line_world": [0.0, 1.0, 0.0]}


@pytest.mark.parametrize("call,message", [
    (lambda: robot.target(Value("camera:agentview", xyz=[0, 0, 0.5])), "position.to('world')"),
    (lambda: robot.target(Value("tcp:left", xyz=[0, 0, 0.01]), arm="right"), "moves the right arm"),
    (lambda: robot.target([0.4, 0.1, 0.9]), "needs a frame"),
    (lambda: robot.target(Value("world", xyz=[[0, 0, 1], [0, 0, 2]])), "one 3-vector"),
    (lambda: robot.target(approach=Value("world", v=[0, 0, -1])), "go together"),
    (lambda: robot.target(approach=Value("tcp:right", v=[0, 0, 1]), jaw_line=Value("world", v=[0, 1, 0])),
     "approach.to('world')"),
    (lambda: robot.target(Value("world", xyz=[0, 0, 1], xyzw=[0, 0, 0, 1]), approach=Value("world", v=[0, 0, -1]),
                          jaw_line=Value("world", v=[0, 1, 0])), "give the orientation once"),
    (lambda: robot.target(Value("tcp:right", xyz=[0, 0, 0], xyzw=[0, 0, 0.38, 0.92])), "convert the pose"),
    (lambda: robot.target(Value("world", v=[0, 0, 1])), "not a direction"),
])
def test_a_value_the_tool_cannot_take_is_refused_with_the_conversion_to_make(call, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message.replace("(", r"\(").replace(")", r"\)")):
        call()


def test_move_pose_and_move_to_fill_typed_targets_and_leave_the_tool_s_own_form() -> None:
    plain = {"position_mode": "hold_current", "orientation_mode": "axis_angle_delta", "rotation_axis": "z",
             "rotation_angle_deg": 30, "orientation_frame": "end_effector"}
    filled = robot._fill_arguments("move_pose", {"targets": {
        "right": Value("tcp:right", xyz=[0, 0, 0.02]), "left": plain}, "speed": "careful"})
    assert filled == {"targets": {"right": robot.target(Value("tcp:right", xyz=[0, 0, 0.02])), "left": plain},
                      "speed": "careful"}
    library_form = robot._fill_arguments("move_pose", {"targets": {"left": {
        "position": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}, "approach": Value("world", v=[0, 0, -1]),
        "jaw_line": Value("world", v=[1, 0, 0])}}})
    assert library_form["targets"]["left"]["orientation_mode"] == "world_axes"
    assert robot._fill_arguments("move_to", {"targets": {"left": Value("world", xyz=[0.1, 0.2, 0.3])},
                                            "expect_contact": True}) == {
        "targets": {"left": {"position_m": [0.1, 0.2, 0.3]}}, "expect_contact": True}
    assert robot._fill_arguments("move_to", {"targets": {"left": {"position_m": [1, 2, 3]}}}) == {
        "targets": {"left": {"position_m": [1, 2, 3]}}}
    with pytest.raises(ValueError, match="use move_pose"):
        robot._fill_arguments("move_to", {"targets": {"right": Value("tcp:right", xyz=[0, 0, 0.02])}})
    with pytest.raises(ValueError, match="use one form"):
        robot._fill_arguments("move_pose", {"targets": {"right": {"position": Value("world", xyz=[0, 0, 1]),
                                                                 "orientation_mode": "hold_current"}}})
    untouched = {"targets": {"left": Value("world", xyz=[0, 0, 1])}}
    assert robot._fill_arguments("rotate_wrist", untouched) == untouched, "other tools are left as they are"
    # a single-arm surface's move_to takes the point as its own position_m
    assert robot._fill_arguments("move_to", {"position_m": Value("world", xyz=[0.1, 0.2, 0.3]), "speed": "normal"}) == {
        "position_m": [0.1, 0.2, 0.3], "speed": "normal"}
    assert robot._fill_arguments("move_to", {"position_m": [1, 2, 3]}) == {"position_m": [1, 2, 3]}
    with pytest.raises(ValueError, match="world position"):
        robot._fill_arguments("move_to", {"position_m": Value("tcp:right", xyz=[0, 0, 0.02])})


# --- base contracts: answers and arguments exactly as before ----------------------------------------------------

ANSWER = _recorded("libero", "move_pose", "reached").raw
REFUSAL = {"ok": False, "error": {"code": "script_action_limit", "message": "limit"}}


@pytest.mark.parametrize("extended", [False, True])
def test_the_bash_library_types_results_only_when_extended(monkeypatch, extended) -> None:
    sent: list[dict] = []
    replies = iter([ANSWER, REFUSAL])
    monkeypatch.setattr(robot, "_channel", lambda: SimpleNamespace(
        exchange=lambda message: sent.append(message) or next(replies)))
    if extended:
        monkeypatch.setenv(robot.LIBRARY_ENV, "extended")
    else:
        monkeypatch.delenv(robot.LIBRARY_ENV, raising=False)
    offset = {"frame": "tcp:right", "xyz": [0, 0, -0.05]}
    answer = robot.tools.move_pose(targets={"right": offset})
    with pytest.raises(robot.ToolRefused) as refused:
        robot.tools.move_pose(targets={"right": offset})
    assert sent[0]["arguments"] == ({"targets": {"right": robot.target(offset)}} if extended
                                    else {"targets": {"right": offset}})
    assert isinstance(answer, robot.Result) is extended and answer == ANSWER
    assert type(answer) is (robot.Result if extended else dict)
    assert (refused.value.result is not None) is extended
    if extended:
        assert refused.value.result.failure_code == "script_action_limit"


def test_a_result_carries_the_id_of_its_call_and_the_explanation_of_a_failed_one() -> None:
    answer = {"ok": True, "tool": "geometry_query", "result": {}, "call_id": "g0#004", "explain": "geometry_query(head)"}
    typed = robot.typed("geometry_query", answer)
    assert typed.call_id == "g0#004" and typed.explain == "geometry_query(head)" and typed["call_id"] == "g0#004"
    plain = robot.typed("observe", {"ok": True, "tool": "observe", "result": {}})
    assert plain.call_id is None and plain.explain is None and "call_id" not in repr(plain)
    assert robot._call_id(typed) == "g0#004" and robot._call_id("g0#005") == "g0#005"
    assert robot._call_id({"call_id": "g0#006"}) == "g0#006"
    refused = robot.ToolRefused("geometry_query", {"ok": False, "error": {"code": "x"}})
    refused.result = robot.typed("geometry_query", {"ok": False, "call_id": "g0#007", "error": {"code": "x"}})
    assert robot._call_id(refused) == "g0#007"
    for nothing in (None, 3, plain, robot.ToolRefused("observe", {"ok": False})):
        with pytest.raises(ValueError, match="explain needs the result of a call"):
            robot._call_id(nothing)


def test_explain_sends_the_op_once_per_call_and_a_refusal_raises(monkeypatch) -> None:
    sent = []

    class Channel:
        def exchange(self, message, *, answer=True):
            sent.append(message)
            if message["call_id"] == "g0#404":
                return {"ok": False, "error": {"code": "unknown_call", "message": "no such call"}}
            return {"ok": True, "text": "what it did\n"}

    monkeypatch.setattr(robot, "_channel", lambda: Channel())
    monkeypatch.setattr(robot, "_EXPLAIN", {})
    result = robot.typed("geometry_query", {"ok": True, "call_id": "g0#001"})
    assert robot.explain(result) == robot.explain(result) == robot.explain("g0#001") == "what it did\n"
    assert sent == [{"op": "explain", "call_id": "g0#001"}]
    with pytest.raises(robot.ToolRefused) as error:
        robot.explain("g0#404")
    assert error.value.code == "unknown_call" and "explain" in robot.__all__


def test_a_query_takes_typed_poses_in_world_as_lists_and_refuses_the_rest_with_the_conversion_to_make() -> None:
    world = SimpleNamespace(frame="world", xyz=np.array([0.1, 0.2, 0.3]), xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    arguments = robot.query_arguments({"arm": "left", "poses": world, "seed_joints_deg": np.zeros(6)})
    assert arguments["poses"] == [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] and arguments["arm"] == "left"
    assert isinstance(arguments["seed_joints_deg"], np.ndarray), "the channel writes arrays as lists"
    as_dict = {"frame": "world", "xyz": [0.1, 0.2, 0.3], "xyzw": [0, 0, 0, 1]}
    plain = [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    assert robot.query_arguments({"poses": [as_dict, plain, world]})["poses"] == [
        [0.1, 0.2, 0.3, 0, 0, 0, 1], plain, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]
    assert robot.query_arguments({"poses": np.array([plain, plain])})["poses"] == [plain, plain]
    assert robot.query_arguments({"joints_deg": [0] * 6}) == {"joints_deg": [0] * 6}, "no poses: unchanged"
    tcp = SimpleNamespace(frame="tcp:left", xyz=[0, 0, 0.1], xyzw=[0, 0, 0, 1])
    with pytest.raises(ValueError, match=r"poses\[0\] is in frame 'tcp:left'.*convert it with poses\[0\].to\('world'\)"):
        robot.query_arguments({"poses": [tcp]})
    with pytest.raises(TypeError, match="must be a pose"):
        robot.query_arguments({"poses": [SimpleNamespace(frame="world", xyz=[0, 0, 0])]})
    assert "query" in robot.__all__ and "query_arguments" in robot.__all__


# --- a dry run (``dry_run=True`` on a motion tool): the plan, typed ---------------------------------------------

def _dry_answer(feasible: bool = True, arms: tuple[str, ...] = ("left",), explain: str | None = "move_to(left)  ...") -> dict:
    poses = {"left": [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "right": [0.3, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]}
    record = ({"dry_run": True, "action": "move_to", "status": "planned", "feasible": True, "failure_code": None,
               "failure_stage": None, "route": "line", "screening": {"status": "clear", "segments_screened": 16},
               "arms": {arm: {"predicted_pose": poses[arm], "end_joints_deg": [10.0 * (i + 1) for i in range(6)]}
                        for arm in arms}} if feasible else
              {"dry_run": True, "action": "move_to", "status": "infeasible", "feasible": False,
               "failure_code": "paced_path_unreachable", "failure_stage": "ik", "refusal": {"code": "paced_path_unreachable"}})
    return {"ok": True, "tool": "move_to", "dry_run": True, "call_id": "s0#003", "result": record,
            **({"explain": explain} if explain else {})}


def test_a_dry_run_reads_into_the_plan_and_has_no_executed_outcome() -> None:
    result = robot.typed("move_to", _dry_answer())
    assert isinstance(result, robot.Plan) and isinstance(result, robot.Result)
    assert (result.dry_run, result.feasible) == (True, True)
    assert result.predicted_pose == [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0] and result.predicted_poses == {
        "left": result.predicted_pose}
    assert result.end_joints_deg == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert result.route == "line" and result.screening["segments_screened"] == 16
    assert (result.failure_code, result.failure_stage) == (None, None)
    assert result.explain == "move_to(left)  ..." and result.call_id == "s0#003"
    assert result["result"]["arms"]["left"]["predicted_pose"] == result.predicted_pose, "the dict is unchanged"
    assert result["result"]["status"] == "planned" and result.get("ok") is True, "dict access is the answer's own"
    assert repr(result) == "Plan(move_to: feasible=True, predicted_pose=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], route='line')"


@pytest.mark.parametrize("field", robot.RUN_FIELDS)
def test_a_plan_has_none_of_a_run_s_fields_and_says_the_call_runs_it(field) -> None:
    """A plan read as a run did nothing silently: ``if r.ok`` on a feasible plan was True and moved nothing."""
    plan = robot.typed("move_to", _dry_answer())
    with pytest.raises(robot.PlanNotRun, match=r"move_to returned a plan and nothing has run: .*tools\.move_to\(\.\.\.\) "
                                               "without dry_run=True runs it"):
        getattr(plan, field)


def test_an_infeasible_plan_says_why() -> None:
    result = robot.typed("move_to", _dry_answer(feasible=False))
    assert (result.dry_run, result.feasible) == (True, False)
    assert (result.failure_code, result.failure_stage) == ("paced_path_unreachable", "ik")
    assert result.predicted_pose is None and result.predicted_poses == {} and result.route is None
    assert result.screening is None and result.end_joints_deg is None
    with pytest.raises(robot.PlanNotRun):
        result.ok


def test_the_dry_run_part_lists_what_a_plan_reads_and_says_what_raises() -> None:
    """``robot.docs(name, "dry_run")`` said a plan has ``.ok`` (``.feasible (also .ok)``), which raises ``PlanNotRun``
    (a plan read as a run, ``if r.ok``, did nothing silently: ``test_sandbox.py``): 7 scripts ended on it. Every field
    the part lists reads on a plan, and every one it says raises does."""
    listed = code_docs.DRY_RUN_RESULT.split("As a Result")[1].split(";")[0]
    fields = re.findall(r"\.(\w+)", listed)
    assert {"feasible", "predicted_pose", "summary", "explain"} <= set(fields) and "ok" not in fields
    for feasible in (True, False):
        plan = robot.typed("move_to", _dry_answer(feasible=feasible))
        assert isinstance(plan, robot.Plan) and plan.feasible is feasible
        for field in fields:
            getattr(plan, field)
    named = re.findall(r"\.(\w+)", code_docs.DRY_RUN_RESULT.split("; a run's fields")[1].split(":")[0])
    assert {"ok", "status"} <= set(named) and set(named) <= set(robot.RUN_FIELDS), "the fields it says raise, raise"


def test_two_arms_give_each_prediction_and_no_single_one() -> None:
    result = robot.typed("move_to", _dry_answer(arms=("left", "right")))
    assert result.predicted_pose is None and result.end_joints_deg is None
    assert set(result.predicted_poses) == {"left", "right"} and result.predicted_poses["right"][1] == -0.4


def test_a_call_that_is_not_a_dry_run_has_none_of_it() -> None:
    plain = robot.typed("move_to", {"ok": True, "result": {"status": "executed", "arms": {"left": {"predicted_pose": [0] * 7}}}})
    assert (plain.dry_run, plain.feasible, plain.predicted_pose, plain.route, plain.screening) == (False, None, None, None, None)
    assert plain.predicted_poses == {} and "feasible" not in repr(plain)
    refused = robot.typed("move_to", {"ok": False, "dry_run": True, "error": {"code": "dry_run_unavailable"}})
    assert (refused.dry_run, refused.ok, refused.status, refused.failure_code, refused.feasible) == (
        True, False, "refused", "dry_run_unavailable", None)


@pytest.mark.parametrize("extended", [False, True])
def test_dry_run_goes_to_the_host_as_a_plain_argument_of_the_call(monkeypatch, extended) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(robot, "_channel", lambda: SimpleNamespace(
        exchange=lambda message: sent.append(message) or _dry_answer()))
    if extended:
        monkeypatch.setenv(robot.LIBRARY_ENV, "extended")
    else:
        monkeypatch.delenv(robot.LIBRARY_ENV, raising=False)
    result = robot.tools.move_to(targets={"left": {"position_m": [0.4, 0.0, 0.2]}}, dry_run=True)
    assert sent == [{"op": "call", "name": "move_to", "arguments": {
        "targets": {"left": {"position_m": [0.4, 0.0, 0.2]}}, "dry_run": True}}]
    assert isinstance(result, robot.Result) is extended


def test_dry_run_survives_the_filling_of_typed_targets(monkeypatch) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(robot, "_channel", lambda: SimpleNamespace(
        exchange=lambda message: sent.append(message) or _dry_answer()))
    monkeypatch.setenv(robot.LIBRARY_ENV, "extended")
    offset = {"frame": "tcp:left", "xyz": [0, 0, -0.05]}
    robot.tools.move_pose(targets={"left": offset}, dry_run=True)
    assert sent[0]["arguments"] == {"targets": {"left": robot.target(offset, arm="left")}, "dry_run": True}


# --- through ``Tools`` to the channel: what a ``bash`` program sends -----------------------------------------------

TOOLKIT = Path(robot.__file__).parents[1] / "toolkit/tools"
#: Definitions as the policy session has them; ``hold`` takes no argument.
DEFINITIONS = {**{name: json.loads((TOOLKIT / name / "description.json").read_text())
                  for name in ("wait_ticks", "move_to", "move_pose")},
               "hold": {"name": "hold", "description": "Hold.", "inputSchema": {"type": "object", "properties": {}}}}


class Host:
    """The far end of ``robot.py``'s channel (``exchange``). ``docs`` answers as the policy session does, from the
    generated entry; every call is answered."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def answer(self, message: dict) -> dict:
        if message["op"] != "docs":
            return {"ok": True, "tool": message["name"], "result": {}}
        if message["name"] not in DEFINITIONS:
            return {"ok": False, "error": {"code": "unknown_function", "message": f"no function {message['name']!r}"}}
        parts = code_docs.entry_parts(DEFINITIONS[message["name"]])
        return {"ok": True, "text": parts[message["part"]] + "\n"}

    def exchange(self, message: dict, *, answer: bool = True) -> dict:
        self.sent.append(json.loads(json.dumps(robot._json_safe(message))))
        return self.answer(self.sent[-1])

    def of(self, op: str) -> list[dict]:
        return [message for message in self.sent if message["op"] == op]


def tools_of(host: Host, monkeypatch, *, extended: bool = True):
    """``tools`` talking to ``host``."""

    monkeypatch.setattr(robot, "_DOCS", {})
    if extended:
        monkeypatch.setenv(robot.LIBRARY_ENV, "extended")
    else:
        monkeypatch.delenv(robot.LIBRARY_ENV, raising=False)
    monkeypatch.setattr(robot, "_channel", lambda: host)
    return robot.tools


@pytest.fixture
def host(monkeypatch) -> Host:
    found = Host()
    found.tools = tools_of(found, monkeypatch)
    return found


POINT = Point([0.4, 0.1, 0.9], frame="world")
PLAIN = {"position_mode": "absolute", "position_m": [0.4, 0.1, 0.9], "orientation_mode": "hold_current",
         "interaction": "precision"}


@pytest.mark.parametrize("dry", [{}, {"dry_run": True}], ids=["call", "dry_run"])
@pytest.mark.parametrize("given", [
    {"position": POINT, "interaction": "precision"},
    {"position": {"frame": "world", "xyz": [0.4, 0.1, 0.9]}, "interaction": "precision"},
    {"frame": "world", "xyz": [0.4, 0.1, 0.9], "interaction": "precision"},
    robot.target(POINT, interaction="precision"),
    PLAIN,
], ids=["typed-value", "typed-dict", "flat-typed-dict", "target-output", "tool-form"])
def test_interaction_reaches_the_channel_from_either_form(host, given, dry) -> None:
    """``interaction`` is a key of the tool's target and of the library's: 0 of 201 moves carried it, because any
    target in the tool's form with it (``target()``'s own output) was refused as a mix."""
    host.tools.move_pose(targets={"right": given}, **dry)
    assert host.sent == [{"op": "call", "name": "move_pose", "arguments": {"targets": {"right": PLAIN}, **dry}}]


@pytest.mark.parametrize("dry", [{}, {"dry_run": True}], ids=["call", "dry_run"])
@pytest.mark.parametrize("extra", [{}, {"interaction": "precision"}], ids=["plain", "with-interaction"])
def test_a_real_mix_is_still_refused_and_nothing_is_sent(host, extra, dry) -> None:
    with pytest.raises(ValueError, match="use one form") as refused:
        host.tools.move_pose(targets={"right": {"position": POINT, "position_m": [0.4, 0.1, 0.9], **extra}}, **dry)
    assert "(position_m)" in str(refused.value) and "interaction" not in str(refused.value)
    assert host.sent == []


def test_interaction_alone_does_not_make_a_target_the_library_s(host) -> None:
    host.tools.move_pose(targets={"right": {"interaction": "precision"}})
    assert host.of("call")[0]["arguments"] == {"targets": {"right": {"interaction": "precision"}}}, "the tool's to refuse"


def test_positional_arguments_bind_to_the_signature_in_its_order(host) -> None:
    """The names come from the ``docs`` op, once per function; only calls count as calls."""
    host.tools.wait_ticks(15)
    host.tools.wait_ticks(16)
    host.tools.move_to({"left": POINT}, expect_contact=True)
    host.tools.move_pose({"left": {"position": POINT}}, "careful", dry_run=True)
    calls = [(message["name"], message["arguments"]) for message in host.of("call")]
    assert calls == [("wait_ticks", {"ticks": 15}), ("wait_ticks", {"ticks": 16}),
                     ("move_to", {"targets": {"left": {"position_m": [0.4, 0.1, 0.9]}}, "expect_contact": True}),
                     ("move_pose", {"targets": {"left": robot.target(POINT)}, "speed": "careful", "dry_run": True})]
    assert [message["name"] for message in host.of("docs")] == ["wait_ticks", "move_to", "move_pose"]


def test_an_arm_moving_tool_s_last_position_is_dry_run_as_docs_prints_it(host) -> None:
    assert code_docs.signature(DEFINITIONS["move_to"]) == "tools.move_to(targets, expect_contact?, dry_run?)"
    host.tools.move_to({"left": POINT}, False, True)
    assert host.of("call")[0]["arguments"] == {"targets": {"left": {"position_m": [0.4, 0.1, 0.9]}},
                                               "expect_contact": False, "dry_run": True}


def test_keywords_alone_ask_for_no_documentation(host) -> None:
    host.tools.wait_ticks(ticks=15)
    assert [message["op"] for message in host.sent] == ["call"]


@pytest.mark.parametrize("call,message", [
    (lambda tools: tools.wait_ticks(1, 2), "tools.wait_ticks(ticks) takes 1 argument(s), not 2"),
    (lambda tools: tools.hold(1), "tools.hold() takes 0 argument(s), not 1"),
    (lambda tools: tools.wait_ticks(1, ticks=2), "tools.wait_ticks(ticks) got ticks twice, by position and by name"),
])
def test_too_many_or_a_name_given_twice_is_a_type_error_with_the_signature(host, call, message) -> None:
    with pytest.raises(TypeError, match=re.escape(message)):
        call(host.tools)
    assert host.of("call") == []


def test_a_positional_call_of_no_function_says_so(host) -> None:
    with pytest.raises(Exception, match="no function 'nope'") as refused:
        host.tools.nope(1)
    assert type(refused.value).__name__ == "ToolRefused" and host.of("call") == []


def test_a_base_call_stays_keywords_only_and_asks_for_nothing(monkeypatch) -> None:
    found = Host()
    tools = tools_of(found, monkeypatch, extended=False)
    with pytest.raises(TypeError, match=re.escape("ToolFunction.__call__() takes 1 positional argument but 2 were given")):
        tools.wait_ticks(15)
    assert found.sent == []
    assert type(tools.wait_ticks).__name__ == "ToolFunction"
    tools.wait_ticks(ticks=15)
    assert found.sent == [{"op": "call", "name": "wait_ticks", "arguments": {"ticks": 15}}]


def test_a_missing_key_says_where_it_is_or_which_keys_there_are() -> None:
    answer = {"ok": True, "tool": "vision_ground", "result": {"status": "candidates", "candidates": [{"score": 1}]}}
    result = robot.typed("vision_ground", answer)
    with pytest.raises(KeyError) as nested:
        result["candidates"]
    assert nested.value.args == ('candidates is under result: r["result"]["candidates"]',)
    assert str(nested.value) == """'candidates is under result: r["result"]["candidates"]'""", "no escapes in a traceback"
    with pytest.raises(KeyError, match='r\\["result"\\]\\["status"\\]'):
        result["status"]
    with pytest.raises(KeyError) as other:
        result["pose"]
    assert other.value.args == ("no pose in this answer; its keys are ok, tool, result",)
    refused = robot.typed("move_pose", {"ok": False, "error": {"code": "x"}})
    with pytest.raises(KeyError, match="no result in this answer; its keys are ok, error"):
        refused["result"]
    assert "candidates" not in result and result.get("candidates") is None and result.get("candidates", 3) == 3
    assert result["result"]["candidates"] == [{"score": 1}] and result.status == "candidates"
    assert dict(result) == answer and json.dumps(result) == json.dumps(answer)


def test_a_real_direction_is_no_position_though_it_has_xyz() -> None:
    down = Direction([0, 0, -1], "world")
    with pytest.raises(TypeError, match="not a direction"):
        robot.target(down)
    with pytest.raises(TypeError, match="must be a point"):
        robot._fill_arguments("move_to", {"targets": {"left": down}})
    assert robot.target(POINT, approach=down, jaw_line=Direction([0, 1, 0], "world"))["approach_world"] == [0, 0, -1]
