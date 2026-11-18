"""The TOPP-RA + Ruckig worker end to end, under its own interpreter (``ORA_TIMING_PYTHON``).

Nothing here imports the libraries: every test talks to the real worker through ``TimingClient``, so this suite
runs from the project's interpreter and skips where no timing interpreter exists, as in CI. There the
contract (``tests/ora/environment/controllers/test_motion_planner.py``), the delegation and the fallback
(``test_motion_timing.py``) are exercised with stand-ins.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ora.contracts.motion_speed import SLOW_APPROACH_M, speed_caps
from ora.environment.controllers import motion_pace as mp
from ora.environment.controllers import motion_timing
from ora.environment.controllers.motion_planner import PROTOCOL, MotionPlannerError
from ora.environment.controllers.motion_retime import RETIME_REQUEST_SCHEMA, TimingClient
from ora.environment.planners.timing.requirements import PINS

PYTHON = os.environ.get("ORA_TEST_TIMING_PYTHON")
pytestmark = pytest.mark.skipif(not PYTHON, reason="no timing interpreter (ORA_TIMING_PYTHON)")
ROOT = Path(__file__).resolve().parents[5]
TICK = 0.05
CEILING = 1.0
WRIST = 6
LIMITS = (-2.8973, 2.8973)
#: cuRobo's franka_panda.urdf: parent-to-child origin and roll of panda_joint1..7.
PANDA = (((0, 0, 0.333), 0.0), ((0, 0, 0), -math.pi / 2), ((0, -0.316, 0), math.pi / 2),
         ((0.0825, 0, 0), math.pi / 2), ((-0.0825, 0.384, 0), -math.pi / 2), ((0, 0, 0), math.pi / 2),
         ((0.088, 0, 0), math.pi / 2))
#: The grip site along the hand axis, past link 7, and the hand's yaw.
GRIP_M, HAND_YAW = 0.107 + 0.1035, -math.pi / 4
SPEEDS = {"fast": None, "normal": {"speed": "normal", "slow_approaching": False},
          "careful": {"speed": "careful", "slow_approaching": False},
          "normal+slow": {"speed": "normal", "slow_approaching": True},
          "custom": {"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False}}


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def fk(rows: Any) -> list[list[float]]:
    poses = []
    for q in rows:
        transform = np.eye(4)
        for angle, (origin, roll) in zip(q, PANDA):
            shift = np.eye(4)
            shift[:3, 3] = origin
            transform = transform @ shift @ _rx(roll) @ _rz(angle)
        shift = np.eye(4)
        shift[2, 3] = GRIP_M
        transform = transform @ shift @ _rz(HAND_YAW)
        R = transform[:3, :3]
        w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
        y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
        z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        poses.append([*transform[:3, 3].tolist(), x / norm, y / norm, z / norm, w / norm])
    return poses


FIXTURE = json.loads((ROOT / "tests/data/motion-pace-paths.json").read_text())["paths"]
PATHS = [path["joints"] for path in FIXTURE]


@pytest.fixture(scope="module")
def timing() -> Any:
    client = TimingClient(Path(PYTHON))
    client.start()
    yield client
    client.close()


def request_for(path: Any, speed: Any = None, **overrides: Any) -> dict[str, Any]:
    grid = mp.joint_path_grid(path, fk, wrist_joint_index=WRIST)
    jerk = motion_timing.jerk_limits("franka_panda", 7)
    return {"schema": RETIME_REQUEST_SCHEMA, "joint_names": [f"j{index}" for index in range(7)],
            "path": {"joint_positions": grid.grid.joints["arm"].tolist(),
                     "tool_poses_xyzw": [list(pose) for pose in grid.poses]},
            "speed": speed, "slow_approach": True,
            "limits": {"joint_velocity_rad_s": [CEILING] * 7, "joint_acceleration_rad_s2": None,
                       "joint_jerk_rad_s3": jerk["rad_s3"], "jerk_source": jerk["source"]},
            "tick_s": TICK, "budget_ticks": 2000, "cut_to_budget": False,
            "wrist": {"joint_index": WRIST, "limits_rad": list(LIMITS)}, **overrides}


@pytest.mark.parametrize("path", PATHS, ids=[path["label"] for path in FIXTURE])
@pytest.mark.parametrize("label", list(SPEEDS))
def test_every_cap_holds_at_every_tick(timing, path, label):
    answer = timing.retime(request_for(path, SPEEDS[label]))
    assert answer["status"] == "timed", answer["detail"]
    rows = np.asarray(answer["trajectory"]["positions"])
    assert rows[0].tolist() == pytest.approx(path[0]) and rows[-1].tolist() == pytest.approx(path[-1], abs=1e-12)
    assert answer["trajectory"]["velocities"][0] == [0.0] * 7 and answer["trajectory"]["velocities"][-1] == [0.0] * 7
    assert answer["detail"]["max_deviation_m"] <= 0.001
    poses = np.asarray(fk(rows))
    steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    caps, slow = speed_caps(SPEEDS[label])
    cap, ramp = caps["linear_m_s"], caps["linear_accel_m_s2"]
    speeds = steps / TICK
    assert np.abs(np.diff(rows, axis=0)).max() / TICK <= CEILING
    if cap is not None:
        limit = np.full(len(speeds), cap)
        accel = np.full(len(speeds), ramp)
        if slow is not None:
            zone = np.cumsum(np.concatenate([[0.0], steps]))[:-1] >= steps.sum() - SLOW_APPROACH_M - 1e-9
            limit[zone], accel[zone] = slow["linear_m_s"], slow["linear_accel_m_s2"]
        assert np.all(speeds <= limit)
        assert np.all(np.abs(np.diff(speeds)) / TICK <= accel[1:])


def test_an_answer_repeats_bit_for_bit(timing):
    request = request_for(PATHS[1], SPEEDS["normal+slow"])
    first, second = timing.retime(request), timing.retime(request)
    assert first["trajectory"] == second["trajectory"] and first["report"] == second["report"]


def test_repeated_rows_are_the_same_path(timing):
    path = PATHS[0]
    plain = timing.retime(request_for(path, SPEEDS["normal"]))
    held = timing.retime(request_for([path[0], path[0], *path, path[-1]], SPEEDS["normal"]))
    assert plain["trajectory"] == held["trajectory"]
    near = request_for(path, SPEEDS["normal"])
    near["path"]["joint_positions"].insert(1, [value + 1e-8 for value in near["path"]["joint_positions"][0]])
    near["path"]["tool_poses_xyzw"].insert(1, near["path"]["tool_poses_xyzw"][0])
    assert timing.retime(near)["trajectory"]["positions"] == plain["trajectory"]["positions"]


def test_a_path_that_does_not_move_is_one_tick_at_rest(timing):
    answer = timing.retime(request_for([PATHS[0][0]]))
    assert answer["status"] == "timed" and answer["report"]["ticks"] == 1
    assert answer["trajectory"]["positions"][0] == answer["trajectory"]["positions"][1]


def test_a_corner_stops_the_move(timing):
    start = np.asarray(PATHS[0][0])
    corner = start + [0.3, 0, 0, 0, 0, 0, 0]
    path = [start.tolist(), corner.tolist(), (corner + [0, 0.3, 0, 0, 0, 0, 0]).tolist()]
    answer = timing.retime(request_for(path, SPEEDS["normal"]))
    rows = np.asarray(answer["trajectory"]["positions"])
    assert answer["status"] == "timed" and answer["detail"]["pieces"] == 2
    # The move slows to rest at the corner: the step at the corner is far below the cruise.
    steps = np.abs(np.diff(rows, axis=0)).max(axis=1)
    closest = int(np.argmin(np.abs(rows - corner).max(axis=1)))
    assert min(steps[closest - 1], steps[closest]) < 0.3 * steps.max()


def test_the_wrist_unwinds_refused_with_the_shared_message(timing):
    path = [[0.0] * 6 + [angle] for angle in np.linspace(-1.5, 1.5, 150)]
    request = request_for(path)
    # The hand turns the long way round: 172 degrees along the path, none from start to end.
    poses = request["path"]["tool_poses_xyzw"]
    poses[-1] = poses[0]
    answer = timing.retime(request)
    assert answer["status"] == "refused" and answer["reason"] == "paced_path_wrist_unwind"
    refusal = answer["detail"]["refusal"]
    assert "the wrist would unwind" in refusal["message"]
    assert refusal["constraints"]["unrequested_turn_deg"] > 45 and "wrist" in refusal["constraints"]


def test_the_budget_refuses_or_cuts_and_never_inside_a_large_wrist_turn(timing):
    path = PATHS[1]
    whole = timing.retime(request_for(path, SPEEDS["careful"]))
    needed = whole["report"]["ticks"]
    refused = timing.retime(request_for(path, SPEEDS["careful"], budget_ticks=needed - 1))
    assert refused["reason"] == "paced_motion_budget"
    assert refused["detail"]["refusal"]["constraints"] == {"tick_budget": needed - 1, "needed_ticks": needed}
    cut = timing.retime(request_for(path, SPEEDS["careful"], budget_ticks=needed // 2, cut_to_budget=True))
    assert cut["status"] == "timed" and cut["report"]["ticks"] <= needed // 2
    assert 0 < cut["report"]["fraction_of_move"] < 1 and cut["report"]["needed_ticks"] == needed
    assert timing.retime(request_for(path, SPEEDS["careful"], budget_ticks=1, cut_to_budget=True))["reason"] \
        == "paced_motion_budget"
    turn = [[0.0] * 6 + [angle] for angle in np.linspace(-1.0, 1.0, 120)]
    start = timing.retime(request_for(turn, SPEEDS["careful"], budget_ticks=5, cut_to_budget=True))
    assert start["reason"] == "paced_motion_budget"
    assert "unsplit_span" in start["detail"]["refusal"]["constraints"]


def test_a_lower_jerk_limit_certifies_at_a_lower_acceleration(timing):
    path = PATHS[0]
    free = timing.retime(request_for(path))
    tight = request_for(path)
    tight["limits"]["joint_jerk_rad_s3"] = [400.0] * 7
    tight["limits"]["jerk_source"] = "a test's"
    answer = timing.retime(tight)
    assert answer["status"] == "timed" and answer["detail"]["jerk_source"] == "a test's"
    assert answer["report"]["ticks"] > free["report"]["ticks"]


def test_a_jerk_limit_nothing_can_meet_is_timing_infeasible(timing):
    request = request_for(PATHS[0])
    request["limits"]["joint_jerk_rad_s3"] = [1e-3] * 7
    answer = timing.retime(request)
    assert answer["status"] == "refused" and answer["reason"] == "timing_infeasible"
    assert answer["detail"]["refusal"]["constraints"]["attempts"] >= 1


def test_the_worker_times_paths_and_plans_nothing(timing):
    binding = timing.start()
    versions = binding["worker"]["ready"]["versions"]
    pins = dict(pin.split("==") for pin in PINS)
    assert versions == {name: pins[name] for name in ("toppra", "ruckig", "numpy", "scipy")}
    client = TimingClient(Path(PYTHON))
    client.start()
    with pytest.raises(MotionPlannerError, match="refused the plan command"):
        client._exchange("plan", {})
    assert client._process is None
    assert PROTOCOL.endswith(".v2")


def test_pace_path_runs_through_the_worker_and_keeps_its_refusals():
    motion_timing.configure(PYTHON)
    arguments = dict(tick_s=TICK, joint_speed_ceiling_rad_s=CEILING, wrist_joint_index=WRIST,
                     wrist_limits_rad=LIMITS, robot="franka_panda")
    result = mp.pace_path(PATHS[1], fk, speed=SPEEDS["normal+slow"], budget_ticks=2000, **arguments)
    report = result.report()
    assert report["timing"]["implementation"] == "toppra_ruckig"
    assert report["timing"]["versions"]["toppra"] == "0.6.10" and report["timing"]["versions"]["ruckig"] == "0.19.4"
    assert "support.franka.de" in report["timing"]["jerk_source"]
    assert result.rows()[-1] == pytest.approx(PATHS[1][-1]) and result.slow_from_knot is not None
    assert report["speed"]["planned_peak_linear_m_s"] <= 0.3
    with pytest.raises(mp.PaceRefusal) as error:
        mp.pace_path(PATHS[1], fk, speed=SPEEDS["careful"], budget_ticks=2, **arguments)
    assert error.value.code == "paced_motion_budget"


def test_no_ruckig_input_ever_holds_waypoints():
    """Every Ruckig input a move uses still has empty ``intermediate_positions`` when it is done."""

    script = r'''
import json, sys
from ora.environment.planners.timing import retime as core
inputs = []
real = core.ruckig_input
def spy(dof):
    made = real(dof)
    inputs.append(made)
    return made
core.ruckig_input = spy
answer = core.retime(json.loads(sys.stdin.read()))
assert answer["status"] == "timed", answer
assert inputs and all(len(item.intermediate_positions) == 0 for item in inputs)
print(len(inputs))
'''
    done = subprocess.run([PYTHON, "-c", script], input=json.dumps(request_for(PATHS[0])), text=True,
                          capture_output=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT),
                                                              "MPLBACKEND": "Agg"}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert int(done.stdout.strip().splitlines()[-1]) >= 1
