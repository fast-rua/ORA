"""RoboDojo's side of the shared motion base, over a stand-in session and planner client."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.toolkit.lib.motion_plan.checks import planner_refusal
from ora.environment.backends.robodojo import reach_check, shared_motion
from ora.environment.controllers.depth_world import read_world_inputs
from ora.environment.controllers.motion_worker import MotionPlannerError

JOINTS = [0.1, 1.0, 1.0, 0.0, 0.0, 0.0]
OTHER = [-0.1, 1.0, 1.0, 0.0, 0.0, 0.0]


class Client:
    def __init__(self, answer=None):
        self.answer, self.requests = answer, []

    def plan(self, request):
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    check_path = plan


def _session(tmp_path, client=None, *, clock=(5, 5)):
    root = tmp_path / "obs"
    for camera in ("head", "left_wrist", "right_wrist"):
        (root / camera).mkdir(parents=True)
        np.save(root / camera / "depth-m.npy", np.ones((48, 64), dtype=np.float32), allow_pickle=False)
        (root / camera / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": [[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1.0]],
            "cam2world_cv": np.eye(4).tolist()}), encoding="utf-8")
    robots = {"left": SimpleNamespace(entity_origin_pose=[-0.3, -0.45, 0.765, 0.707107, 0.0, 0.0, 0.707107]),
              "right": SimpleNamespace(entity_origin_pose=[0.3, -0.45, 0.765, 0.707107, 0.0, 0.0, 0.707107])}
    manager = SimpleNamespace(get_end_effector_real_val=lambda robot, env_idx_list: np.array([[0.02, 0.021]]))
    return SimpleNamespace(
        output_root=tmp_path / "native", robots=robots, gripper_directives={"left": 0.0}, motion_planner=client,
        environment=SimpleNamespace(robot_manager=manager), last_observation_root=root, last_observation_clock=(5, 5),
        last_observation_id="robodojo:t:layout-0:observation-0001", _clock=lambda: clock,
        _proprioception=lambda: {"end_effectors": {"left": {"arm_joint_qpos": JOINTS},
                                                   "right": {"arm_joint_qpos": OTHER}}})


def _planned():
    return {"status": "planned", "reason": None, "trajectory": {"positions": [JOINTS, OTHER]},
            "world": {"sha256": "a" * 64, "summary": {"held_voxels": 3, "goal_refusal": None}}, "planner": {},
            "timing_s": {}, "start_tool_pose_base_xyzw": [0, 0, 0, 0, 0, 0, 1]}


def _plan_request(**fields):
    return {"arm": "left", "goal_pose_xyz_xyzw": [0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 1.0], "goal_joints": None, **fields}


def test_a_plan_asks_in_the_arms_base_frame_from_the_measured_joints_with_the_other_arm_in_the_snapshot(tmp_path):
    client = Client(_planned())
    answer = shared_motion.plan_path(_session(tmp_path, client), _plan_request())
    assert answer["status"] == "planned" and answer["path_joints"] == [JOINTS, OTHER]
    assert answer["start_joints"] == JOINTS and answer["world_sha256"] == "a" * 64
    assert "planner" not in answer and "world" not in answer
    request = client.requests[0]
    assert request["robot"] == "arx_x5" and request["start"]["joint_positions"] == JOINTS
    # World (0, -0.1, 1.0) from the left base at (-0.3, -0.45, 0.765), turned +90 degrees about z.
    assert request["goal"]["pose_base_xyzw"][:3] == pytest.approx([0.35, -0.3, 0.235], abs=1e-5)
    inputs = read_world_inputs(Path(request["world"]["input"]["path"]), request["world"]["input"]["sha256"])
    assert [camera.name for camera in inputs.cameras] == ["head", "left_wrist", "right_wrist"]
    assert inputs.other_arm_joint_positions.tolist() == OTHER and inputs.gripper_directive == 1.0
    assert inputs.gripper_finger_qpos.tolist() == [0.02, 0.021]


def test_one_observation_is_one_snapshot_per_arm(tmp_path):
    client = Client(_planned())
    session = _session(tmp_path, client)
    shared_motion.plan_path(session, _plan_request())
    shared_motion.plan_path(session, _plan_request())
    assert client.requests[0]["world"] == client.requests[1]["world"]


@pytest.mark.parametrize("kind", ["plan_path", "contact_path"])
def test_no_planner_or_a_stale_observation_is_unavailable(tmp_path, kind):
    request = _plan_request() if kind == "plan_path" else {"arm": "left", "joints_rad": [JOINTS]}
    run = getattr(shared_motion, kind)
    assert run(_session(tmp_path), request) == {"arm": "left", "status": "unavailable", "reason": "no_planner"}
    (tmp_path / "s").mkdir()
    answer = run(_session(tmp_path / "s", Client(_planned()), clock=(6, 5)), request)
    assert answer["reason"] == "stale_observation"


def test_a_refusal_carries_why_to_the_tool_that_says_it(tmp_path):
    why = "the arm reaches this pose with nothing in the way, but the hand would be 1.0 cm inside what the depth shows"
    refused = {**_planned(), "status": "refused", "reason": "goal_unreachable", "trajectory": None,
               "detail": {"goal_diagnosis": {"message": why}}}
    answer = shared_motion.plan_path(_session(tmp_path, Client(refused)), _plan_request())
    assert answer["reason"] == "goal_unreachable" and answer["goal_refusal"] == {"message": why}
    assert why in str(planner_refusal(answer, arm="left"))


def test_a_failed_planner_never_stops_a_move(tmp_path):
    session = _session(tmp_path, Client(MotionPlannerError("timed out")))
    assert shared_motion.plan_path(session, _plan_request())["reason"] == "planner_error"
    (tmp_path / "e").mkdir()
    session = _session(tmp_path / "e", Client({"status": "error", "reason": "timeout"}))
    assert shared_motion.plan_path(session, _plan_request()) == {"arm": "left", "status": "unavailable",
                                                                 "reason": "timeout"}


def test_a_contact_path_exempts_the_fingers_and_names_what_it_met(tmp_path):
    client = Client({"status": "blocked", "reason": None, "first_blocked_row": 2, "link": "link6", "penetration_m": 0.02})
    answer = shared_motion.contact_path(_session(tmp_path, client), {"arm": "left", "joints_rad": [JOINTS]})
    assert answer["status"] == "blocked" and answer["limiting_link"] == "link6"
    assert client.requests[0]["allowed_links"] == ["link7", "link8"]
    assert client.requests[0]["tolerance_m"] == shared_motion.CONTACT_TOLERANCE_M


def test_a_refused_pace_is_an_answer_and_a_timed_one_carries_no_library_name(tmp_path, monkeypatch):
    from ora.environment.backends.robodojo import reachability
    from ora.environment.controllers import motion_pace, motion_timing

    motion_timing.configure(None)
    monkeypatch.setattr(reachability, "fk_link6", lambda session, arm, row: [*row[:3], 0.0, 0.0, 0.0, 1.0])
    rows = np.linspace([0.0] * 6, [0.2] + [0.0] * 5, 12).tolist()
    request = {"arm": "left", "joints_rad": rows, "speed": {"speed": "normal", "slow_approaching": False},
               "slow_approach": False, "budget_ticks": 100, "cut_to_budget": True, "unwind_check": True}
    session = _session(tmp_path)
    timed = shared_motion.pace_path(session, request)
    assert timed["status"] == "timed" and timed["ticks"] == len(timed["path_joints"]) and "timing" not in timed
    assert (session.output_root / "motion-planner/timing.jsonl").is_file()

    def refuse(*args, **kwargs):
        raise motion_pace.PaceRefusal("the wrist unwinds", code="paced_path_wrist_unwind", constraints={"turn": 1})

    monkeypatch.setattr(motion_pace, "pace_path", refuse)
    refused = shared_motion.pace_path(session, request)
    assert refused == {"arm": "left", "status": "refused", "reason": "paced_path_wrist_unwind",
                       "message": "the wrist unwinds", "constraints": {"turn": 1}}


def test_the_reach_check_is_a_reference_with_or_without_a_planner(monkeypatch):
    monkeypatch.setattr(reach_check, "fk_calibrated", lambda *arguments: True)
    session = SimpleNamespace(_proprioception=lambda: {"end_effectors": {"left": {}}}, _clock=lambda: 0,
                              motion_planner=None)
    ask = {"position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.1}
    query = {"pose_xyzw": [0.0] * 7, "from_index": None}
    answer = reach_check.reach_check_batch(session, "left", [query], **ask)
    assert answer["matches_move_pose_admission"] is False and answer["results"][0]["reason"] == "no_planner"



class Preparing:
    """A running planner client that records the worlds it was asked to prepare, one exchange each."""

    _process = object()

    def __init__(self, answer=None, during=None):
        self.answer, self.prepared, self.during = answer, [], during

    def prepare(self, worlds):
        self.prepared.append(worlds)
        if self.during is not None:
            self.during()
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer or {"status": "prepared", "reason": None, "prepared": len(worlds), "timing_s": {"total": 1.5}}


def _prebuild_lines(session):
    path = session.output_root / "motion-planner/prebuild.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def _starts(client):
    return [[item["start"]["joint_positions"] for item in worlds] for worlds in client.prepared]


def test_a_published_observation_has_both_arms_scenes_prepared_off_the_simulator_thread(tmp_path):
    client = Preparing()
    session = _session(tmp_path, client)
    shared_motion.prebuild(session)
    shared_motion.join_prebuild(session)
    assert _starts(client) == [[JOINTS], [OTHER]], "one arm per exchange"
    assert all(worlds[0]["world"]["model"] == "public_depth_voxel_v1" for worlds in client.prepared)
    # The snapshots a request then reads are the ones prepared: no second snapshot of the observation.
    snapshots = sorted((session.output_root / "motion-planner/world-inputs").iterdir())
    assert len(snapshots) == 2
    record, _ = shared_motion.world_input(session, "left")
    assert record == client.prepared[0][0]["world"]["input"]
    (line,) = _prebuild_lines(session)
    assert line["status"] == "prepared" and [arm["arm"] for arm in line["arms"]] == ["left", "right"]
    assert line["observation_id"] == session.last_observation_id and session._planner_prebuilds == []


def test_the_arm_that_just_moved_is_prepared_first(tmp_path):
    client = Preparing()
    session = _session(tmp_path, client)
    session._prebuild_joints = {"left": JOINTS, "right": [0.0] * 6}
    shared_motion.prebuild(session)
    shared_motion.join_prebuild(session)
    assert _starts(client) == [[OTHER], [JOINTS]]


def test_a_newer_observation_supersedes_the_arms_not_yet_asked(tmp_path):
    session = _session(tmp_path, None)
    client = Preparing(during=lambda: setattr(session, "last_observation_id", "robodojo:t:layout-0:observation-0002"))
    session.motion_planner = client
    shared_motion.prebuild(session)
    shared_motion.join_prebuild(session)
    (line,) = _prebuild_lines(session)
    assert len(client.prepared) == 1 and line["status"] == "superseded" and len(line["arms"]) == 1


@pytest.mark.parametrize("answer", [MotionPlannerError("motion planner request timed out"), RuntimeError("broken")])
def test_a_failed_prebuild_is_a_line_of_evidence_and_nothing_else(tmp_path, answer):
    session = _session(tmp_path, Preparing(answer))
    shared_motion.prebuild(session)
    shared_motion.join_prebuild(session)
    (line,) = _prebuild_lines(session)
    assert line["status"] == "error" and line["arms"][0]["reason"] == "prepare_failed"
    assert answer.args[0] in line["arms"][0]["error"] and len(line["arms"]) == 1, "the next arm is not asked"


def test_no_running_planner_or_a_stale_observation_prepares_nothing(tmp_path):
    session = _session(tmp_path, None)
    shared_motion.prebuild(session)
    assert getattr(session, "_planner_prebuilds", []) == []
    stopped = Preparing()
    stopped._process = None
    shared_motion.prebuild(_session(tmp_path / "stopped", stopped))
    stale = Preparing()
    shared_motion.prebuild(_session(tmp_path / "stale", stale, clock=(6, 6)))
    assert stopped.prepared == [] and stale.prepared == []
