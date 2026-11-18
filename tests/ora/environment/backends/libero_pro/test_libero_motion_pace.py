"""LIBERO's global route is paced at the grip site and its wrist checked before anything moves.

The planner's joint path is retimed by the shared ``motion_pace`` (one knot
per 50 ms control tick, joints at most the executor's per-tick step), sent
timed, and reported as ``pacing``; an unrequested wrist turn or a path over
the tick budget is refused with no controller write; the sparse receipt
carries ``motion``; a local OSC segment caps its commanded step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ora.environment.backends.libero_pro import planned_motion
from ora.environment.backends.libero_pro.common import LiberoProError
from ora.environment.backends.libero_pro.session import (
    LiberoProSession, libero_environment_init_params, libero_public_environment_context,
)
from ora.environment.controllers.jaw_flip_retry import jaw_twin
from tests.support.libero.correctness_repair import START, TARGET, _session
from tests.support.orchestrator.runtime_v1 import FakeLiberoEnvironment, _fake_libero_metadata

ROOT = Path(__file__).resolve().parents[5]


def _free_space() -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v3", "action_id": "action-0", "observation_id": "observation-0",
        "bound_observation_sha256": "0" * 64, "action": "move_free_space",
        "arguments": {"arm": "right", "target_pose": {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}},
    }


def _route(session: Any, path: list[list[float]]) -> None:
    planner = session.motion_planner
    original = planner.plan

    def plan(request: dict[str, Any]) -> dict[str, Any]:
        answer = original(request)
        answer["trajectory"]["positions"] = [request["joint_positions"], *path]
        return answer

    planner.plan = plan


def test_the_global_route_is_retimed_and_sent_one_knot_per_tick(tmp_path, monkeypatch):
    session, plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # 6 cm of grip site at the stand-in chain
    sent: list[dict[str, Any]] = []
    setter = session.environment.set_env_attr
    session.environment.set_env_attr = lambda name, value: (
        sent.append(value) if name == "harness_joint_trajectory_command" else None, setter(name, value))
    receipt = session._execute_planned_motion(_free_space())
    command = sent[0]
    assert command["timed"] is True and command["dt_s"] == 0.05
    steps = np.abs(np.diff(np.asarray([[0.0] * 7, *command["positions"]]), axis=0)).max(axis=1)
    # Fast: no Cartesian cap; every joint at most the executor's 0.05 rad per 50 ms tick.
    # 0.6 rad at 1 rad/s, reached at 40 rad/s^2: 13 ticks of 50 ms.
    assert steps.max() <= 0.05 + 1e-9 and len(command["positions"]) == 13
    pacing = receipt["pacing"]
    assert pacing["planned_ticks"] == 13 and pacing["duration_s"] == pytest.approx(0.65)
    assert pacing["speed"]["speed"] == "fast" and pacing["speed"]["linear_m_s"] is None
    assert pacing["speed"]["joint_speed_deg_s"] == pytest.approx(math.degrees(1.0), abs=1e-3)
    assert pacing["speed"]["planned_peak_linear_m_s"] == pytest.approx(0.1, abs=1e-3)
    assert pacing["wrist"]["travel_deg"] == 0.0 and pacing["wrist"]["requested_turn_deg"] == 0.0
    assert len(plant.commands) >= 12


def test_a_move_says_which_timing_ran_and_passes_the_panda_to_it(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert session._execute_planned_motion(_free_space())["pacing"]["timing"]["implementation"] == "motion_pace"
    seen: list[dict[str, Any]] = []

    def pace(*arguments: Any, **options: Any) -> Any:
        seen.append(options)
        raise planned_motion.PaceRefusal("refused here", code="paced_motion_budget")

    monkeypatch.setattr(planned_motion, "pace_path", pace)
    session._execute_planned_motion(_free_space())
    assert seen[0]["robot"] == "franka_panda"


def test_the_next_observation_embeds_the_policy_s_copy_and_the_sealed_receipt_keeps_its_timing(
        tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    receipt = session._execute_planned_motion(_free_space())
    assert receipt["pacing"]["timing"]["implementation"] == "motion_pace"
    sealed = {"path": "receipts/action-0000.json", "sha256": "0" * 64,
              "receipt": {"motion": receipt["pacing"], "segment": receipt}}

    def factory(**_kwargs):
        context = libero_public_environment_context(
            {"problem": "Libero", "problem_folder": "libero_goal_task"},
            init_params=libero_environment_init_params(max_sim_steps=100))
        return FakeLiberoEnvironment(), _fake_libero_metadata(tmp_path, context)

    observer = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO", data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate", required_libero_pro_revision="libero-test",
        required_data_revision="data-test", workspace_bounds_m={"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3]},
        max_sim_steps=100, environment_factory=factory)
    observer.start()
    observer.last_receipt = sealed
    observation, _images = observer.observe(tmp_path / "policy-observations")
    embedded = observation["previous_execution_receipt"]["receipt"]
    assert "timing" not in embedded["motion"] and "timing" not in embedded["segment"]["pacing"]
    assert embedded["segment"]["pacing"]["planned_ticks"] == receipt["pacing"]["planned_ticks"]
    written = (tmp_path / "policy-observations" / observation["observation_id"] / "observation.json").read_text()
    assert "motion_pace" not in written and '"timing":' not in written
    assert "timing" in observer.last_receipt["receipt"]["motion"]


def test_the_session_times_with_the_interpreter_preflight_bound(tmp_path, monkeypatch):
    from ora.environment.controllers import motion_timing

    session, _plant = _session(tmp_path, monkeypatch)
    session.metadata = {}
    timing = {"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}
    chosen: list[Any] = []
    monkeypatch.setattr(motion_timing, "configure", lambda python, **_: chosen.append(python))
    monkeypatch.setattr(motion_timing, "binding", lambda: {"implementation": "toppra_ruckig", "python_sha256": "a" * 64})
    monkeypatch.setenv("ORA_TIMING_PYTHON", "/from/the/environment/python")
    session.qualified_controller_binding = {"motion_timing": {"python": "/bound/python", "python_sha256": "a" * 64}}
    session._configure_motion_timing(timing)
    assert chosen[-1] == "/bound/python" and session.metadata["motion_timing_binding"]["python_sha256"] == "a" * 64
    session.qualified_controller_binding = {"motion_timing": {"python": "/bound/python", "python_sha256": "b" * 64}}
    with pytest.raises(LiberoProError, match="changed after preflight"):
        session._configure_motion_timing(timing)
    # A direct session without a preflight binding reads the variable; a baseline without timing keeps the fallback.
    session.qualified_controller_binding = None
    session._configure_motion_timing(timing)
    assert chosen[-1] == "/from/the/environment/python"
    session._configure_motion_timing(None)
    assert chosen[-1] is None
    monkeypatch.delenv("ORA_TIMING_PYTHON")
    with pytest.raises(LiberoProError, match="unbound"):
        session._configure_motion_timing(timing)


def test_a_requested_speed_caps_the_grip_site(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    session._active_motion_speed = {"speed": "custom", "speed_m_s": 0.02, "slow_approaching": False}
    receipt = session._execute_planned_motion(_free_space())
    pacing = receipt["pacing"]
    # 6 cm at 2 cm/s, reached in 0.25 s: at least 3 s.
    assert pacing["planned_ticks"] >= 60
    assert pacing["speed"]["speed"] == "custom" and pacing["speed"]["speed_m_s"] == 0.02
    assert pacing["speed"]["planned_peak_linear_m_s"] == pytest.approx(0.02, abs=1e-4)


def test_an_unrequested_wrist_turn_is_refused_before_any_controller_write(tmp_path, monkeypatch):
    session, plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.0] * 6 + [1.6], [0.0] * 6 + [3.1], [0.0] * 6 + [1.6], [0.0] * 7])
    receipt = session._execute_planned_motion(_free_space())
    assert receipt["motion_status"] == "refused" and receipt["steps_used"] == 0
    assert receipt["feedback"]["planning_failure_reason"] == "paced_path_wrist_unwind"
    refused = receipt["pacing"]["refused"]
    assert refused["code"] == "paced_path_wrist_unwind"
    assert refused["path_turn_deg"] == pytest.approx(355.2, abs=0.1)
    assert refused["requested_turn_deg"] == pytest.approx(0.0, abs=0.01)
    assert refused["wrist"]["max_deg"] == pytest.approx(177.6, abs=2.0)  # read at the knots
    assert plant.commands == []


def test_a_route_over_the_tick_budget_is_refused_whole(tmp_path, monkeypatch):
    session, plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    session._active_motion_speed = {"speed": "custom", "speed_m_s": 0.0005, "slow_approaching": False}
    receipt = session._execute_planned_motion(_free_space())
    assert receipt["feedback"]["planning_failure_reason"] == "paced_motion_budget"
    assert receipt["pacing"]["refused"]["tick_budget"] == 2000
    assert receipt["pacing"]["refused"]["needed_ticks"] > 2000
    assert plant.commands == []


def test_the_sparse_receipt_reports_motion_and_names_the_pacing_refusal(tmp_path, monkeypatch):
    session, _plant = _session(tmp_path, monkeypatch)
    _route(session, [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    action = {
        "schema": "ora-motion-action.v3", "action_id": "action-0", "observation_id": "observation-0",
        "bound_observation_sha256": "0" * 64, "action": "move_pose",
        "arguments": {"targets": {"right": {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}},
                      "interactions": {"right": "free_space"},
                      "speed": {"speed": "normal", "slow_approaching": True}},
    }
    receipt = session._execute_sparse_motion_v3(action)
    motion = receipt["motion"]
    assert motion["speed"]["speed"] == "normal" and motion["speed"]["slow_approaching"] is True
    assert motion["speed"]["slow"]["linear_m_s"] == 0.1 and motion["speed"]["slow_approach_m"] == 0.02
    assert "wrist" in motion and motion["planned_ticks"] > 0
    assert getattr(session, "_active_motion_speed") is None
    _route(session, [[0.0] * 6 + [1.6], [0.0] * 6 + [3.1], [0.0] * 6 + [1.6], [0.0] * 7])
    refused = session._execute_sparse_motion_v3({**action, "action_id": "action-1"})
    assert (refused["execution_status"], refused["failure_stage"], refused["failure_code"]) == (
        "plan_rejected", "pacing", "paced_path_wrist_unwind")
    assert refused["motion"]["refused"]["unrequested_turn_deg"] > 45


#: A path that turns the hand 355 degrees for none asked (the pacing refuses it), and 6 cm with no turn.
UNWINDS = [[0.0] * 6 + [1.6], [0.0] * 6 + [3.1], [0.0] * 6 + [1.6], [0.0] * 7]
SHORT = [[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def _twin_routes(session: Any) -> list[bool]:
    """The planner unwinds the hand on the way to the requested orientation and goes straight to its jaw twin.

    Returns, per pose plan asked, whether it asked for the twin (the hand's target turned about z by pi)."""
    planner = session.motion_planner
    original, asked = planner.plan, []
    planner.client = None  # nothing kept: a dry run asks the planner as the real call does

    def plan(request: dict[str, Any]) -> dict[str, Any]:
        answer = original(request)
        twin = abs((request.get("target_pose_base_xyzw") or [0.0] * 7)[5]) > 0.5  # a recovery asks joints
        asked.append(twin)
        answer["trajectory"]["positions"] = [request["joint_positions"], *(SHORT if twin else UNWINDS)]
        return answer

    planner.plan = plan
    return asked


def _move_pose() -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v3", "action_id": "action-0", "observation_id": "observation-0",
        "bound_observation_sha256": "0" * 64, "action": "move_pose",
        "arguments": {"targets": {"right": {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}},
                      "interactions": {"right": "free_space"}},
    }


def test_a_wrist_refusal_with_open_jaws_runs_the_jaw_twin_and_says_so(tmp_path, monkeypatch):
    session, plant = _session(tmp_path, monkeypatch)
    asked = _twin_routes(session)
    twin = jaw_twin(TARGET)
    plant.endpoint = list(twin)
    receipt = session._execute_sparse_motion_v3(_move_pose())
    assert asked == [False, True] and plant.commands
    retry = receipt["motion"]["wrist"]["jaw_flip_retry"]
    assert retry["applied"] is True and retry["reason"] == "paced_path_wrist_unwind" and retry["turn_deg"] == 180.0
    assert retry["requested_orientation_xyzw"] == TARGET[3:] and retry["executed_orientation_xyzw"] == twin[3:]
    assert retry["refused_unrequested_turn_deg"] > 45
    # The twin's 6 cm path ran; the arrival is judged at the twin, and the arguments keep the request.
    assert receipt["motion"]["wrist"]["path_turn_deg"] == 0.0 and receipt["execution_status"] == "reached"
    assert receipt["arms"]["right"]["pose_error"]["orientation_rad"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["segment"]["feedback"]["endpoint_error"]["orientation_rad"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["arguments"]["targets"]["right"]["orientation_xyzw"] == TARGET[3:]
    # Both plans are evidence; the receipt names the one that ran.
    assert receipt["segment"]["planner"]["canonical_artifact"]["path"] == "curobo/plans/action-0000-jaw-twin.json"
    assert (tmp_path / "curobo/plans/action-0000.json").is_file()


def test_held_jaws_a_recovery_and_a_refused_twin_keep_the_wrist_refusal(tmp_path, monkeypatch):
    session, plant = _session(tmp_path, monkeypatch)
    asked = _twin_routes(session)
    plant.robot.gripper.current_action = [-1.0, 1.0]  # the directive closes: the jaws may hold something
    refused = session._execute_sparse_motion_v3(_move_pose())
    assert (refused["failure_stage"], refused["failure_code"]) == ("pacing", "paced_path_wrist_unwind")
    assert asked == [False] and "jaw_flip_retry" not in refused["motion"]["refused"] and not plant.commands
    # A close just accepted holds too, while the ramping directive may still read open.
    plant.robot.gripper.current_action = [1.0, -1.0]
    session._gripper_command_target = "close"
    assert session._execute_sparse_motion_v3(_move_pose())["failure_code"] == "paced_path_wrist_unwind"
    assert asked == [False, False]
    session._gripper_command_target = "open"
    # A recovery's joint goal is never retried, whatever the jaws.
    _route(session, UNWINDS)
    recovered = session._execute_planned_motion({**_free_space(), "action": "recover_motion",
                                                  "arguments": {"arm": "right", "strategy": "home"}})
    assert recovered["feedback"]["planning_failure_reason"] == "paced_path_wrist_unwind"
    assert len(session.motion_planner.requests) == 3 and not plant.commands
    # An open jaw whose twin unwinds too keeps the refusal and says the twin was tried.
    tried = session._execute_planned_motion(_free_space())
    assert tried["pacing"]["refused"]["jaw_flip_retry"] == {
        "applied": False, "reason": "paced_path_wrist_unwind", "turn_deg": 180.0,
        "refused_unrequested_turn_deg": tried["pacing"]["refused"]["unrequested_turn_deg"],
        "twin_refused": "paced_path_wrist_unwind"}
    assert len(session.motion_planner.requests) == 5 and not plant.commands


@pytest.mark.parametrize("jaws", ["open", "closed"])
def test_a_dry_run_answers_what_the_real_call_does(tmp_path, monkeypatch, jaws):
    session, plant = _session(tmp_path, monkeypatch)
    asked = _twin_routes(session)
    if jaws == "closed":
        plant.robot.gripper.current_action = [-1.0, 1.0]
    plan = session.plan_action(_move_pose())
    assert asked == ([False, True] if jaws == "open" else [False]) and not plant.commands
    if jaws == "closed":
        assert (plan["status"], plan["failure_code"], plan["failure_stage"]) == (
            "refused", "paced_path_wrist_unwind", "pacing")
        assert "jaw_flip_retry" not in plan["constraints"]
    else:
        path, settle = plan["segments"]
        assert plan["status"] == "planned" and path["jaw_flip_retry"]["applied"] is True
        assert settle["poses"][-1] == pytest.approx(jaw_twin(TARGET))
    plant.endpoint = list(jaw_twin(TARGET))
    receipt = session._execute_sparse_motion_v3(_move_pose())
    assert receipt["execution_status"] == ("reached" if jaws == "open" else "plan_rejected")
    if jaws == "open":
        assert receipt["motion"]["wrist"]["jaw_flip_retry"] == path["jaw_flip_retry"]


def _first_local_command(tmp_path: Path, speed: dict[str, Any] | None) -> list[float]:
    """The first OSC command of a 3 cm servo segment, normalized on the 50 mm controller range."""

    environment = FakeLiberoEnvironment()
    context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100))
    session = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO", data_root=tmp_path / "data", output_root=tmp_path / "run",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate", required_libero_pro_revision="libero-test",
        required_data_revision="data-test", workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]},
        max_sim_steps=100, environment_factory=lambda **_kwargs: (environment, _fake_libero_metadata(tmp_path, context)),
    )
    session.start()
    session.motion_profile = json.loads((ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json")
                                        .read_text(encoding="utf-8"))
    session._active_motion_speed = speed
    try:
        session.execute({
            "schema": "ora-motion-action.v1", "observation_id": "observation-0", "action_id": "servo",
            "action": "servo_delta", "bound_observation_sha256": "bound",
            "arguments": {"arm": "right", "target_pose": {"position_m": [0.0, 0.0, 0.53],
                                                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                          "delta_pose": {"translation_m": [0.0, 0.0, 0.03], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                          "frame": "world", "max_iterations": 5},
        })
        return environment.first_action[0, :3].tolist()
    finally:
        session.close()


@pytest.mark.parametrize("speed,normalized", [
    # The default caps nothing: the per-axis 25 mm clip, 0.5 of the 50 mm controller range.
    (None, 0.5),
    ({"speed": "normal", "slow_approaching": False}, 0.3),  # 0.3 m/s x 50 ms = 15 mm
    ({"speed": "careful", "slow_approaching": False}, 0.1),  # 5 mm
    ({"speed": "custom", "speed_m_s": 0.2, "slow_approaching": False}, 0.2),  # 10 mm
])
def test_a_local_segment_caps_its_commanded_step(tmp_path, monkeypatch, speed, normalized):
    original = FakeLiberoEnvironment.step

    def step(self, action):
        if not hasattr(self, "first_action"):
            self.first_action = np.asarray(action)
        return original(self, action)

    monkeypatch.setattr(FakeLiberoEnvironment, "step", step)
    assert _first_local_command(tmp_path, speed)[2] == pytest.approx(normalized, abs=1e-6)
