"""The global route's JOINT_POSITION controller: motion profile v3's gains, lead and replay.

v3 binds a reach of 0.2 rad, kp 50 and a velocity lead on every knot, and paces
the route at what that controller follows; v2 keeps RoboSuite's defaults (kp 50,
0.05 rad). A replay loads exactly the gains its run executed with, so a sealed v2
run replays on the defaults and a v3 run on its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment import replay_worker
from ora.environment.backends.libero_pro import planned_motion, worker_environment
from ora.environment.backends.libero_pro.session import _CameraMetadataEnvironment
from ora.environment.controllers.hybrid_motion import (
    HybridMotionProfileError,
    lead_joint_targets,
    libero_global_joint_ceilings,
    libero_joint_controller,
    libero_settle_at_rest,
    precision_force_stop_enabled,
    validate_hybrid_motion_profile,
)
from tests.support.libero.correctness_repair import ROOT, TARGET, _session

PROFILES = ROOT / "ora/policy/embodiments/franka-panda"
V2 = json.loads((PROFILES / "motion-profile-v2.json").read_text(encoding="utf-8"))
V3 = json.loads((PROFILES / "motion-profile-v3.json").read_text(encoding="utf-8"))
# The fixture planner's goal is the start plus 0.05 rad on every joint; joint 1 is held 0.01 rad short of it.
START_JOINTS_BLOCK = 0.04


def _action() -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v3", "action_id": "action-0",
        "observation_id": "observation-0", "bound_observation_sha256": "0" * 64,
        "action": "move_free_space",
        "arguments": {"arm": "right", "target_pose": {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}},
    }


def test_v3_binds_the_controller_it_paces_for_and_v2_keeps_robosuite_defaults() -> None:
    assert validate_hybrid_motion_profile(V3) == V3
    gains = libero_joint_controller(V3)
    assert gains == {"mode": "JOINT_POSITION", "kp": 50.0, "damping_ratio": 1.0, "output_max_rad": 0.2,
                     "lead_s": pytest.approx(2.0 / math.sqrt(50.0))}
    assert libero_joint_controller(V2) is None
    assert libero_global_joint_ceilings(V3, tick_s=0.05) == (0.55, 15.0)
    # v2's historical ceiling: one 0.05 rad command step per 50 ms tick, no acceleration cap.
    assert libero_global_joint_ceilings(V2, tick_s=0.05) == (pytest.approx(1.0), None)
    assert precision_force_stop_enabled(V2) and precision_force_stop_enabled(V3)


@pytest.mark.parametrize("change,message", [
    (lambda p: p["global_motion"].pop("joint_controller_kp"), "fields drifted"),
    (lambda p: p["global_motion"].update(joint_controller_kp=400), "joint_controller_kp"),
    # The lead at 0.75 rad/s, 2 / sqrt(50) x 0.75 = 0.212 rad, passes the 0.2 rad reach.
    (lambda p: p["global_motion"].update(joint_follow_speed_rad_s=0.75), "past its reach"),
    (lambda p: p["global_motion"].update(joint_follow_accel_rad_s2=0.0), "joint_follow_accel_rad_s2"),
])
def test_v3_refuses_a_controller_it_cannot_follow(change: Any, message: str) -> None:
    profile = json.loads(json.dumps(V3))
    change(profile)
    with pytest.raises(HybridMotionProfileError, match=message):
        validate_hybrid_motion_profile(profile)


def test_v3_requires_its_rest_step() -> None:
    profile = json.loads(json.dumps(V3))
    profile["global_motion"].pop("terminal_settle_rest_step_rad")
    with pytest.raises(HybridMotionProfileError, match="fields drifted"):
        validate_hybrid_motion_profile(profile)


@pytest.mark.parametrize("rest,step,position,orientation,settled", [
    (0.002, 0.0019, 0.005, 0.05, True),      # at rest, on the local bound (compared <=, as the hand-off is)
    (0.002, 0.0021, 0.001, 0.001, False),    # still moving
    (0.002, 0.001, 0.0051, 0.0, False),      # outside the local 5 mm
    (0.002, 0.001, 0.0, 0.051, False),       # outside the local 0.05 rad
    (None, 0.0, 0.0, 0.0, False),            # v1, v2: no rest rule
    (0.002, None, 0.0, 0.0, False),          # no tick measured yet
])
def test_a_v3_settle_ends_at_rest_inside_the_local_bound(rest, step, position, orientation, settled) -> None:
    assert libero_settle_at_rest(rest_step_rad=rest, last_step_rad=step, position_error_m=position,
                                 orientation_error_rad=orientation) is settled


@pytest.mark.parametrize("profile_name,status,settled_by", [
    ("motion-profile-v2.json", "stopped", None),
    ("motion-profile-v3.json", "reached", "at_rest_in_local_bound"),
])
def test_an_arm_at_rest_short_of_its_joint_goal(tmp_path, monkeypatch, profile_name, status, settled_by) -> None:
    """Joint 1 stops 0.01 rad short of its goal while the grip site sits on the target: v2 waits out its
    40 settle ticks for the 0.005 rad joint tolerance, v3 ends the settle once the arm is at rest."""

    session, plant = _session(tmp_path, monkeypatch, profile_name=profile_name)
    step = plant.step

    def blocked_step(command: Any) -> Any:
        result = step(command)
        plant.robot._joint_positions[0] = min(plant.robot._joint_positions[0], START_JOINTS_BLOCK)
        return result

    plant.step = blocked_step  # type: ignore[method-assign]
    receipt = session._execute_planned_motion(_action())
    settle = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())["terminal_settle"]
    assert receipt["motion_status"] == status
    assert settle["settled_by"] == settled_by
    # One check before the first settle tick and one after each.
    assert len(settle["trace"]) == settle["ticks"] + 1
    assert settle["ticks"] == 40 if settled_by is None else settle["ticks"] <= 1


def test_v2_admits_no_controller_fields_of_its_own() -> None:
    profile = json.loads(json.dumps(V2))
    profile["global_motion"]["joint_controller_kp"] = 150
    with pytest.raises(HybridMotionProfileError, match="fields drifted"):
        validate_hybrid_motion_profile(profile)


def test_each_knot_is_aimed_ahead_by_its_own_speed_and_the_last_is_the_goal() -> None:
    start = [0.0, 1.0]
    knots = [[0.01, 1.0], [0.03, 1.0], [0.06, 1.0], [0.08, 1.0], [0.09, 1.0]]
    lead = 0.2
    targets = lead_joint_targets(knots, start, tick_s=0.05, lead_s=lead)
    # Central differences, the start before the first knot: (0.03 - 0.0) / 0.1 = 0.3 rad/s.
    assert targets[0] == pytest.approx([0.01 + lead * 0.3, 1.0])
    assert targets[2] == pytest.approx([0.06 + lead * (0.08 - 0.03) / 0.1, 1.0])
    assert targets[-1] == knots[-1]
    assert lead_joint_targets([[0.5, 0.5]], start, tick_s=0.05, lead_s=lead) == [[0.5, 0.5]]
    with pytest.raises(HybridMotionProfileError):
        lead_joint_targets(knots, [0.0], tick_s=0.05, lead_s=lead)


def test_the_route_is_paced_within_what_v3_s_controller_follows(tmp_path, monkeypatch) -> None:
    session, _plant = _session(tmp_path, monkeypatch, profile_name="motion-profile-v3.json")
    seen: dict[str, Any] = {}
    real = planned_motion.pace_path

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(planned_motion, "pace_path", spy)
    session._execute_planned_motion(_action())
    assert (seen["joint_speed_ceiling_rad_s"], seen["joint_accel_ceiling_rad_s2"]) == (0.55, 15.0)


def test_v3_execution_loads_its_gains_and_aims_every_knot_ahead(tmp_path, monkeypatch) -> None:
    session, plant = _session(tmp_path, monkeypatch, profile_name="motion-profile-v3.json")
    loaded: list[dict[str, Any]] = []
    load = plant.load_controller

    def record_load() -> None:
        loaded.append(dict(plant.robot.controller_config))
        load()

    plant.robot._load_controller = record_load
    aimed: dict[str, Any] = {}
    real_lead = worker_environment.lead_joint_targets

    def spy(knots: Any, start: Any, **kwargs: Any) -> Any:
        aimed.update(knots=np.asarray(knots).tolist(), start=np.asarray(start).tolist(), **kwargs)
        aimed["targets"] = real_lead(knots, start, **kwargs)
        return aimed["targets"]

    monkeypatch.setattr(worker_environment, "lead_joint_targets", spy)
    receipt = session._execute_planned_motion(_action())

    gains = libero_joint_controller(V3)
    assert receipt["motion_status"] == "reached"
    assert loaded[0] == {"type": "JOINT_POSITION", "kp": 50.0, "damping_ratio": 1.0,
                         "output_max": 0.2, "output_min": -0.2}
    assert loaded[-1] == {"type": "OSC_POSE"}
    assert aimed["lead_s"] == pytest.approx(gains["lead_s"]) and aimed["tick_s"] == 0.05
    assert aimed["start"] == [0.0] * 7
    first = np.clip((np.asarray(aimed["targets"][0]) - np.zeros(7)) / 0.2, -1.0, 1.0)
    np.testing.assert_allclose(plant.commands[0][:7], first)
    tracking = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert tracking["controller_gains"] == gains
    assert session.replay_recorder.entries[0]["metadata"]["joint_controller"] == gains


def test_v2_execution_keeps_robosuite_s_joint_controller(tmp_path, monkeypatch) -> None:
    session, plant = _session(tmp_path, monkeypatch)
    loaded: list[dict[str, Any]] = []
    load = plant.load_controller

    def record_load() -> None:
        loaded.append(dict(plant.robot.controller_config))
        load()

    plant.robot._load_controller = record_load
    session._execute_planned_motion(_action())
    assert loaded[0] == {"type": "JOINT_POSITION"}
    tracking = json.loads((tmp_path / "tracking/joint-action-0000.json").read_text())
    assert tracking["controller_gains"] is None
    assert "joint_controller" not in session.replay_recorder.entries[0]["metadata"]


def _replay(entry: dict[str, Any], commands: np.ndarray, plant: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    """Replay one recorded joint entry through the Viewer/program replay code on a fresh worker."""

    plant.robot._joint_positions = np.zeros(7)
    plant.robot.controller_config = {"type": "OSC_POSE"}
    plant.robot.action_dim = plant._action_dim = 7
    worker = _CameraMetadataEnvironment(plant)
    modes: list[Any] = []
    loaded: list[dict[str, Any]] = []
    load = plant.load_controller

    def record_load() -> None:
        loaded.append(dict(plant.robot.controller_config))
        load()

    plant.robot._load_controller = record_load

    def set_attribute(name: str, value: Any) -> None:
        if name == "harness_replay_controller_mode":
            modes.append(value)
        setattr(worker, name, value)

    environment = SimpleNamespace(current_raw_obs=None, env=SimpleNamespace(set_env_attr=set_attribute))
    session = SimpleNamespace(_require_environment=lambda: environment,
                              _worker_attr=lambda name: getattr(worker, name))
    replay_worker._replay_libero_entry(session, entry, {"commands": commands}, capture=lambda *_args: None)
    return modes, loaded


@pytest.mark.parametrize("profile_name", ["motion-profile-v2.json", "motion-profile-v3.json"])
def test_a_replay_loads_the_gains_its_run_executed_with(tmp_path, monkeypatch, profile_name) -> None:
    """Sealed v2 runs replay bit-exact on RoboSuite's defaults; a v3 run on its recorded gains."""

    session, plant = _session(tmp_path, monkeypatch, profile_name=profile_name)
    live: list[dict[str, Any]] = []
    load = plant.load_controller

    def record_load() -> None:
        live.append(dict(plant.robot.controller_config))
        load()

    plant.robot._load_controller = record_load
    session._execute_planned_motion(_action())
    executed = plant.robot._joint_positions.copy()
    entry = dict(session.replay_recorder.entries[0])
    commands = np.asarray(plant.commands, dtype=np.float64)

    modes, loaded = _replay(entry, commands, plant)

    gains = libero_joint_controller(json.loads((PROFILES / profile_name).read_text(encoding="utf-8")))
    assert modes == [gains or "JOINT_POSITION", "OSC_POSE"]
    assert loaded[0] == live[0]
    assert plant.robot._joint_positions.tolist() == executed.tolist()


def test_a_replay_on_the_wrong_gains_would_not_reproduce_the_run(tmp_path, monkeypatch) -> None:
    """The regression the recorded gains guard: a v3 run replayed on RoboSuite's defaults ends elsewhere."""

    session, plant = _session(tmp_path, monkeypatch, profile_name="motion-profile-v3.json")
    session._execute_planned_motion(_action())
    executed = plant.robot._joint_positions.copy()
    entry = dict(session.replay_recorder.entries[0])
    entry["metadata"] = {key: value for key, value in entry["metadata"].items() if key != "joint_controller"}
    _replay(entry, np.asarray(plant.commands, dtype=np.float64), plant)
    assert not np.allclose(plant.robot._joint_positions, executed)


def test_a_replay_refuses_gains_for_another_controller() -> None:
    worker = _CameraMetadataEnvironment(SimpleNamespace(robots=[SimpleNamespace()]))
    with pytest.raises(Exception, match="replay controller mode is invalid"):
        worker.harness_replay_controller_mode = {**libero_joint_controller(V3), "mode": "OSC_POSE"}


def test_the_torque_monitor_reports_unclipped_peaks_against_the_actuator_range() -> None:
    torques = iter([np.array([10.0, -85.0, 0.0]), np.array([-20.0, 40.0, 13.0])])
    controller = SimpleNamespace(run_controller=lambda: next(torques))
    robot = SimpleNamespace(controller=controller,
                            torque_limits=(np.array([-80.0, -80.0, -12.0]), np.array([80.0, 80.0, 12.0])))
    monitor = worker_environment._monitor_joint_torques(robot)
    controller.run_controller()
    controller.run_controller()
    assert monitor.report() == {
        "source": "robosuite_joint_position_unclipped", "peak_abs_nm": [20.0, 85.0, 13.0],
        "limit_nm": [80.0, 80.0, 12.0], "substeps": 2, "clipped_substeps": 2,
    }
    assert worker_environment._monitor_joint_torques(SimpleNamespace()) is None


def test_the_default_libero_baseline_binds_v4() -> None:
    catalog = json.loads((ROOT / "ora/policy/baselines/native-full-explicit-motion-direct-v4.json").read_text())
    assert catalog["instances"]["libero_pro"]["runtime"]["motion_profile"] == (
        "ora/policy/embodiments/franka-panda/motion-profile-v4.json")
