"""The station's arm paths planned by the shared cuRobo V2 planner (``runtime.motion_planner``, off by default).

The switch and binding cases run anywhere. The planning cases build the station model (MuJoCo and i2rt) and answer
for the planner with a stand-in that solves on a second station model, so they need no GPU; the live case runs the
real worker and skips without ``CUROBO_V2_PYTHON`` and ``CUROBO_V2_ROOT``, as in CI.
"""

from __future__ import annotations

import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pytest

from ora.contracts.policy_receipt import policy_receipt
from ora.environment.backends.yam import motion, shared_planner
from ora.environment.benchmarks.yam import MOTION_PLANNER, YamBenchmarkError, validate_runtime_extension
from ora.environment.controllers import yam_station as Y
from ora.environment.controllers.depth_world import ROBOT_WORLDS, tool_from_hand
from ora.environment.controllers.motion_worker import MotionPlannerError

ON = {"motion_planner": dict(MOTION_PLANNER)}
DOWN = [1.0, 0.0, 0.0, 0.0]


def test_the_switch_is_exactly_the_shared_planner_and_off_by_default():
    runtime = {"task": "sim_pick_place", "seed": 0}
    assert validate_runtime_extension(runtime)["motion_planner"] is None
    assert validate_runtime_extension({**runtime, **ON})["motion_planner"] == MOTION_PLANNER
    with pytest.raises(YamBenchmarkError, match="motion_planner is absent or exactly"):
        validate_runtime_extension({**runtime, "motion_planner": {**MOTION_PLANNER, "world_model": "none"}})
    assert not shared_planner.enabled(types.SimpleNamespace(config={}))
    assert shared_planner.enabled(types.SimpleNamespace(config=ON))


def test_the_yam_plans_in_the_public_depth_world_with_its_own_hand():
    parameters, _turn = ROBOT_WORLDS[shared_planner.ROBOT]
    # The hand convention's jaw line (+y) is the TCP's closing line (+x); its approach is the TCP's.
    turn = tool_from_hand(shared_planner.ROBOT)[:3, :3]
    assert np.allclose(turn @ [0, 1, 0], [1, 0, 0]) and np.allclose(turn @ [0, 0, 1], [0, 0, 1])
    # The box reaches the other arm's base 0.61 m to either side, the 0.62 m reach forward, and below the table.
    low, high = parameters.effective_low_m, parameters.effective_high_m
    assert low[1] < -0.61 < 0.61 < high[1] and high[0] >= 0.62 and low[2] < 0.0
    assert parameters.held_max_gap_m < Y.MAXIMUM_APERTURE_M


def test_a_preflight_binding_or_the_environment_locates_the_planner(tmp_path, monkeypatch):
    native = tmp_path / "run" / "native"
    native.mkdir(parents=True)
    config = ON["motion_planner"]
    monkeypatch.delenv("CUROBO_V2_PYTHON", raising=False)
    monkeypatch.delenv("CUROBO_V2_ROOT", raising=False)
    assert shared_planner.located(config, native) is None
    monkeypatch.setenv("CUROBO_V2_PYTHON", "/envs/curobo/bin/python")
    monkeypatch.setenv("CUROBO_V2_ROOT", "/src/curobo")
    assert shared_planner.located(config, native) == ("/envs/curobo/bin/python", "/src/curobo")
    check = {"check": shared_planner.PREFLIGHT_CHECK, "passed": True, "detail": {"python": "/p", "root": "/r"}}
    (native.parent / "preflight.json").write_text(json.dumps({"checks": [check]}))
    assert shared_planner.located(config, native) == ("/p", "/r")
    (native.parent / "preflight.json").write_text(json.dumps({"checks": [{**check, "passed": False}]}))
    with pytest.raises(Exception, match="preflight binding is absent or invalid"):
        shared_planner.located(config, native)


def test_without_a_planner_the_session_says_the_station_plans(tmp_path, monkeypatch):
    monkeypatch.delenv("CUROBO_V2_PYTHON", raising=False)
    monkeypatch.delenv("CUROBO_V2_ROOT", raising=False)
    session = types.SimpleNamespace(config=ON, output_root=tmp_path / "native")
    bound = shared_planner.start(session)
    assert bound["implementation"] == shared_planner.STATION_IMPLEMENTATION and session.shared_planner is None
    assert json.loads((tmp_path / "native/motion-planner/binding.json").read_text()) == bound
    assert shared_planner.start(types.SimpleNamespace(config={}, output_root=tmp_path / "off")) is None
    assert not (tmp_path / "off").exists()


def test_the_policy_copy_of_a_receipt_leaves_the_planner_out():
    receipt = {"motion": {"planned_ticks": 3, "planned_by": {"left": {"implementation": "curobo_v2"}}},
               "failure_code": None}
    assert policy_receipt(receipt) == {"motion": {"planned_ticks": 3}, "failure_code": None}


# ---------------------------------------------------------------- the station model and a stand-in planner
class StandIn:
    """A planner answering on a second station model: the left chain in its base frame, as the worker plans."""

    def __init__(self, kinematics_type, pose_matrix) -> None:
        self.kin = kinematics_type()
        self.pose_matrix = pose_matrix
        self.requests: list[tuple[str, dict]] = []
        self.plan_reason: str | None = None
        self.unsolved: set[int] = set()
        self.blocked_row: int | None = None
        self.fail = False
        base = np.eye(4)
        base[:3, 3] = Y.BASE_IN_WORLD_XYZ_M["left"]
        self.base = base

    def _ik(self, start, pose):
        world = self.base @ self.pose_matrix(pose[:3], pose[3:])
        solved = self.kin.ik("left", world, np.asarray(start, float), max_iterations=400)
        if not solved.ok:      # a folded start: the ready pose's seed, as the worker's many seeds would find it
            solved = self.kin.ik("left", world, np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), max_iterations=400)
        return solved

    def plan(self, request):
        self.requests.append(("plan", request))
        if self.fail:
            raise MotionPlannerError("the worker died")
        start = request["start"]["joint_positions"]
        base = {"request": request, "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": 0.01},
                "world": {"sha256": "0" * 64}}
        if self.plan_reason is not None:
            return {**base, "status": "refused", "reason": self.plan_reason, "trajectory": None}
        solved = self._ik(start, request["goal"]["pose_base_xyzw"])
        assert solved.ok
        rows = [((1 - s) * np.asarray(start) + s * solved.q).tolist() for s in np.linspace(0.0, 1.0, 40)]
        return {**base, "status": "planned", "reason": None, "trajectory": {"positions": rows, "dt_s": 0.02}}

    def solve_ik(self, request):
        self.requests.append(("ik", request))
        results, ends = [], []
        for index, query in enumerate(request["queries"]):
            seed = request["start"]["joint_positions"] if query["from_index"] is None else ends[query["from_index"]]
            if seed is None:
                results.append({"solved": False, "joint_positions": None, "position_error_m": None})
                ends.append(None)
                continue
            solved = self._ik(seed, query["pose_base_xyzw"])
            ok = solved.ok and index not in self.unsolved
            results.append({"solved": ok, "joint_positions": solved.q.tolist() if ok else None,
                            "position_error_m": solved.position_error_m})
            ends.append(solved.q.tolist() if ok else None)
        return {"status": "solved", "reason": None, "results": results, "planner": {"implementation": "curobo_v2"}}

    def check_path(self, request):
        self.requests.append(("check", request))
        blocked = self.blocked_row is not None
        return {"status": "blocked" if blocked else "clear", "first_blocked_row": self.blocked_row,
                "link": "left_link5" if blocked else None, "penetration_m": 0.02 if blocked else None}

    def forward_kinematics(self, rows):
        out = []
        for row in rows:
            pose = np.linalg.inv(self.base) @ self.kin.fk("left", np.asarray(row, float))
            out.append(motion.pose_xyz_xyzw(pose))
        return out

    def close(self):
        pass


@pytest.fixture
def station(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    from ora.environment.backends.yam.kinematics import StationKinematics, pose_matrix
    from ora.environment.backends.yam.rig import ready_configuration
    from ora.environment.backends.yam.scene import DEFAULT_TABLE, build_cell_model

    kin = StationKinematics(build_cell_model(table=DEFAULT_TABLE))
    q = ready_configuration(kin)
    camera = {"depth": np.zeros((4, 4), np.float32), "intrinsic_cv": np.eye(3), "cam2world_cv": np.eye(4)}
    rig = types.SimpleNamespace(arm_q=lambda arm: np.asarray(q[arm], float), finger_m=lambda arm: Y.FINGER_OPEN_M,
                                table_top_z_m=float(DEFAULT_TABLE["top_z_m"]),
                                cameras=lambda: {"front": dict(camera), "wrist_left": {**camera, "cam2world_cv": None}})
    session = types.SimpleNamespace(
        kinematics=kin, rig=rig, home_q={}, last_safe_q=None, config=dict(ON), output_root=tmp_path / "native",
        round_index=0, control_ticks=0, finger_targets={arm: Y.FINGER_OPEN_M for arm in Y.ARMS},
        finger_state=lambda: {arm: Y.FINGER_OPEN_M for arm in Y.ARMS})
    stand_in = StandIn(StationKinematics, pose_matrix)
    root = session.output_root / "motion-planner"
    root.mkdir(parents=True)
    session.shared_planner = shared_planner.SharedPlanner(client=stand_in, root=root, binding={})
    return types.SimpleNamespace(session=session, kin=kin, q=q, planner=stand_in,
                                 at=lambda xyz: pose_matrix(xyz, DOWN))


def test_the_planner_s_forward_kinematics_must_land_on_the_station_model(station):
    report = shared_planner.fk_parity(station.kin, station.planner)
    assert report["passed"] and report["samples"] == 2 * 13
    station.kin.joint_offsets = {"left": np.full(6, 0.05)}      # the planner told readings, not model angles
    station.planner.forward_kinematics = lambda rows: StandIn.forward_kinematics(
        station.planner, [np.asarray(row) - 0.05 for row in rows])
    assert not shared_planner.fk_parity(station.kin, station.planner)["passed"]


def test_a_free_space_move_is_one_plan_in_the_depth_world_and_the_station_checks_its_ticks(station):
    session, at = station.session, station.at
    plans, _targets = motion.move_plans(session, {"targets": {"left": {"position_m": [0.40, 0.12, 0.10],
                                                                       "orientation_xyzw": DOWN}},
                                                  "interactions": {"left": "free_space"}})
    assert plans["left"][0][0].shape == (4, 4) and len(plans["left"][0]) == 1    # no lift searched
    schedule = motion.plan_motion(session, plans)
    kinds = [kind for kind, _ in station.planner.requests]
    assert kinds == ["ik", "plan"]                        # the admission's reach question, then the plan
    plan = station.planner.requests[-1][1]
    assert plan["world"]["model"] == "public_depth_voxel_v1" and plan["held_object"] == {"source": "world"}
    arrays = np.load(plan["world"]["input"]["path"])
    assert [str(name) for name in arrays["camera_names"]] == ["front"]      # the uncalibrated wrist left out
    assert np.allclose(arrays["other_arm_joint_positions"], station.q["right"])
    entry = schedule.motion["planned_by"]["left"]
    assert entry["implementation"] == "curobo_v2" and entry["route"] == "plan"
    final = session.kinematics.fk("left", schedule.finals["left"])
    assert np.linalg.norm(final[:3, 3] - at([0.40, 0.12, 0.10])[:3, 3]) < 2e-3
    assert schedule.summary["min_table_distance_m"]["left"] > 0.0 and len(schedule.ticks) > 1


def test_the_calibration_goes_on_every_request_and_comes_off_every_row(station):
    session, kin = station.session, station.kin
    offsets = np.array([0.01, -0.02, 0.015, 0.0, 0.02, -0.01])
    kin.joint_offsets = {"right": offsets}
    correction = np.eye(4)
    correction[:3, 3] = [0.004, -0.003, 0.002]
    kin.set_base_corrections({"right": correction})
    target = station.at([0.40, -0.12, 0.10])
    samples, _record = shared_planner.plan_pose(session, "right", session.rig.arm_q("right"), target)
    plan = station.planner.requests[-1][1]
    assert np.allclose(plan["start"]["joint_positions"], station.q["right"] + offsets)
    # The goal is in the corrected base frame of the moving arm.
    base = shared_planner.base_world(kin, "right")
    assert np.allclose(base[:3, 3], np.asarray(Y.BASE_IN_WORLD_XYZ_M["right"]) + correction[:3, 3], atol=1e-9)
    assert np.allclose(plan["goal"]["pose_base_xyzw"][:3], (np.linalg.inv(base) @ target)[:3, 3], atol=1e-9)
    rows = np.asarray(station.planner.plan(plan)["trajectory"]["positions"])
    assert np.allclose(samples[-1], rows[-1] - offsets) and np.allclose(samples[0], station.q["right"])


def test_a_start_just_past_a_limit_plans_from_inside_it_and_further_is_refused(station):
    session, kin = station.session, station.kin
    rest = np.zeros(6)
    station.q["left"] = rest
    kin.joint_offsets = {"left": np.array([0.0, -math.radians(1.0), -math.radians(0.5), 0.0, 0.0, 0.0])}
    samples, record = shared_planner.plan_pose(session, "left", rest, station.at([0.30, 0.25, 0.12]))
    start = np.asarray(station.planner.requests[-1][1]["start"]["joint_positions"])
    assert start[1] == pytest.approx(shared_planner.START_LIMIT_INSET_RAD)
    assert start[2] == pytest.approx(shared_planner.START_LIMIT_INSET_RAD)
    assert [item["joint"] for item in record["start_inside_limits"]] == [2, 3]
    # The path starts at the measured start, then steps inward: the station's widened limit check passes it.
    assert np.allclose(samples[0], rest)
    assert all(kin.within_limits("left", row, start=rest) for row in samples)
    kin.joint_offsets = {"left": np.array([0.0, -math.radians(3.0), 0.0, 0.0, 0.0, 0.0])}
    with pytest.raises(shared_planner.Refused) as refused:
        shared_planner.plan_pose(session, "left", rest, station.at([0.30, 0.25, 0.12]))
    assert refused.value.code == "joint_limit"


@pytest.mark.parametrize("reason, code", sorted(shared_planner.CODES.items()))
def test_a_planner_refusal_keeps_the_station_s_words(station, reason, code):
    station.planner.plan_reason = reason
    with pytest.raises(motion.Refusal) as refused:
        motion.plan_motion(station.session, {"left": ([station.at([0.40, 0.12, 0.10])], "free_space")})
    assert refused.value.code == code and refused.value.detail["arm"] == "left"
    assert "record" not in refused.value.detail and "curobo" not in json.dumps(refused.value.detail)


def test_a_planner_that_fails_leaves_the_move_to_the_station_and_says_so(station):
    station.planner.fail = True
    low = station.kin.ik("left", station.at([0.28, 0.22, 0.05]), station.q["left"], max_iterations=400)
    assert low.ok
    station.q["left"] = low.q
    schedule = motion.plan_motion(station.session, {"left": ([station.at([0.40, 0.15, 0.05])], "free_space")})
    entry = schedule.motion["planned_by"]["left"]
    assert entry["implementation"] == shared_planner.STATION_IMPLEMENTATION and "worker died" in entry["reason"]
    assert entry["route"] == "lift_traverse_descend"
    log = (station.session.output_root / "motion-planner/plans.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["error"] == "the worker died"


def test_a_precision_line_is_chained_shared_ik_checked_in_the_world(station):
    session, planner = station.session, station.planner
    here = session.kinematics.fk("left", station.q["left"])
    target = here.copy()
    target[2, 3] -= 0.06
    schedule = motion.plan_motion(session, {"left": ([target], "precision")})
    (_, ik), (_, check) = planner.requests[-2:]
    assert [query["from_index"] for query in ik["queries"]] == [None, *range(len(ik["queries"]) - 1)]
    assert len(ik["queries"]) == 6                                # every 1 cm along the 6 cm descent
    assert ik["world"]["input"]["path"].endswith("-blank.npz")
    assert check["allowed_links"] == list(shared_planner.GRIPPER_LINKS)
    assert not check["world"]["input"]["path"].endswith("-blank.npz")
    assert schedule.motion["planned_by"]["left"]["route"] == "line"
    # Every planned TCP stays on the straight line.
    for tick in schedule.ticks:
        point = session.kinematics.fk("left", tick["left"])[:3, 3]
        assert abs(point[0] - here[0, 3]) < 1.5e-3 and abs(point[1] - here[1, 3]) < 1.5e-3
    planner.blocked_row = 3
    with pytest.raises(motion.Refusal) as refused:
        motion.plan_motion(session, {"left": ([target], "precision")})
    assert refused.value.code == "path_blocked"
    planner.blocked_row, planner.unsolved = None, {2}
    with pytest.raises(motion.Refusal) as refused:
        motion.plan_motion(session, {"left": ([target], "precision")})
    assert refused.value.code == "ik_unreachable"


def test_an_unreachable_target_names_the_height_the_shared_ik_reaches(station):
    session, planner = station.session, station.planner
    target = station.at([0.30, 0.25, 0.30])
    reach = lambda request: [q["pose_base_xyzw"][2] for q in request["queries"]]
    original = planner.solve_ik

    def solve(request):
        answer = original(request)
        for result, z in zip(answer["results"], reach(request)):
            if z > 0.15:
                result.update(solved=False, joint_positions=None)
        return answer

    planner.solve_ik = solve
    with pytest.raises(motion.Refusal) as refused:
        motion.admit_target(session, "left", target, "free_space")
    assert refused.value.code == "target_ik_unreachable"
    assert refused.value.detail["highest_reachable_z_m"] == pytest.approx(0.15, abs=1e-6)
    heights = planner.requests[-1][1]
    assert len(heights["queries"]) == 28             # one request: 0.29 down to the free-space clearance


@pytest.mark.skipif(not (os.environ.get("CUROBO_V2_PYTHON") and os.environ.get("CUROBO_V2_ROOT")),
                    reason="no cuRobo V2 interpreter (CUROBO_V2_PYTHON, CUROBO_V2_ROOT)")
def test_the_live_planner_plans_the_simulated_station_from_rest_past_its_limits(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    from ora.environment.backends.yam.kinematics import pose_matrix
    from ora.environment.backends.yam.session import YamSession

    session = YamSession(config={"rig_mode": "sim", "task_instruction": "live", **ON}, output_root=tmp_path / "native")
    session.start()
    try:
        assert session.shared_planner.binding["fk_parity"]["passed"]
        session.rig._q = {arm: np.zeros(6) for arm in Y.ARMS}
        session.kinematics.joint_offsets = {arm: np.array([0.0, -0.015, -0.01, 0.0, 0.0, 0.0]) for arm in Y.ARMS}
        session.rig._sync()
        target = pose_matrix([0.38, 0.12, 0.12], DOWN)
        motion.admit_target(session, "left", target, "free_space")
        schedule = motion.plan_motion(session, {"left": ([target], "free_space")})
        entry = schedule.motion["planned_by"]["left"]
        assert entry["implementation"] == "curobo_v2" and [item["joint"] for item in entry["start_inside_limits"]] == [2, 3]
        final = session.kinematics.fk("left", schedule.finals["left"])
        assert np.linalg.norm(final[:3, 3] - target[:3, 3]) < 3e-3
        assert Path(tmp_path / "native/motion-planner/binding.json").is_file()
    finally:
        session.close()
