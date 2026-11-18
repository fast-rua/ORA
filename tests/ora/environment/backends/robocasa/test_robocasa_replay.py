"""RoboCasa365 replay: what a run records, the program built from it and its re-execution, on a stand-in kitchen."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment import replay_program as rp
from ora.environment import replay_program_worker as program_worker
from ora.environment import replay_worker as worker
from ora.environment.backends.robocasa.common import RoboCasaError
from ora.environment.backends.robocasa.construction import region_order
from ora.environment.backends.robocasa.robot import OmronIO, StepResult
from ora.environment.backends.robocasa.session import REPLAY_CONTROLLER, RoboCasaSession
from ora.environment.controller_replay import load_controller_replay_arrays, load_controller_replay_manifest
from ora.environment.controllers import robocasa_omron as C

GRIP = [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
BASE = [1.0, -2.0, 0.5]


class Kitchen:
    """A deterministic stand-in: each action vector moves one state, and the task is won at ``win_at``."""

    def __init__(self, win_at: float = 1e9) -> None:
        self.data = SimpleNamespace(time=0.0, qpos=np.zeros(3), qvel=np.zeros(3))
        self.sim = SimpleNamespace(data=self.data, model=SimpleNamespace())
        self.win_at = win_at
        self.received: list[Any] = []

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.received.append(np.array(action))
        self.data.time += 0.05
        self.data.qpos[0] += float(np.sum(action))
        self.data.qvel[1] = float(action[0])
        action[:] = 0.0  # a controller free to reuse its input must not reach the record
        return {}, 0.0, False, {}

    def _check_success(self) -> bool:
        return bool(self.data.qpos[0] >= self.win_at)


def _io(kitchen: Kitchen, monkeypatch: pytest.MonkeyPatch) -> OmronIO:
    monkeypatch.setattr(OmronIO, "bind", lambda _self: None)
    monkeypatch.setattr(OmronIO, "arm_qpos", lambda _self: [0.0] * 7)
    monkeypatch.setattr(OmronIO, "base_pose", lambda _self: [0.0, 0.0, 0.0])
    return OmronIO(kitchen)


def test_every_control_step_is_kept_as_sent_and_recording_touches_no_random_state(monkeypatch) -> None:
    kitchen = Kitchen()
    io = _io(kitchen, monkeypatch)
    state = (random.getstate(), np.random.get_state()[1].tobytes())

    io.step(arm_joints=[0.1] * 7)
    io.step(base_velocity_world=(0.2, 0.0, 0.1))
    io.step(gripper=C.GRIPPER_CLOSE_COMMAND, torso_delta_m=0.01)
    commands = io.take_commands()

    assert (random.getstate(), np.random.get_state()[1].tobytes()) == state
    assert len(commands) == 3 and io.take_commands() == []
    for sent, kept in zip(kitchen.received, commands):
        assert kept.shape == (C.ACTION_DIM,) and np.array_equal(sent, kept)  # kept before the step could change it
    assert commands[1][C.MODE_INDEX] == C.MODE_BASE and commands[0][C.MODE_INDEX] == C.MODE_ARM
    assert commands[2][C.GRIPPER_INDEX] == C.GRIPPER_CLOSE_COMMAND


def test_a_control_step_replayed_sends_the_vector_and_reads_the_predicate_as_the_run_did(monkeypatch) -> None:
    kitchen = Kitchen(win_at=1.0)
    io = _io(kitchen, monkeypatch)

    assert io.apply(np.full(C.ACTION_DIM, 1.0)) == StepResult(reward=0.0, official_success=True)
    assert io.control_steps == 1 and io.take_commands() == []  # a replay records nothing


def test_the_state_hash_holds_the_whole_scene_not_only_the_robot(monkeypatch) -> None:
    kitchen = Kitchen()
    io = _io(kitchen, monkeypatch)
    before = io.state_sha256()

    kitchen.data.qvel[2] = 5e-324  # the smallest difference an object's velocity can show

    assert io.state_sha256() != before
    assert len(before) == 64


def _region(x: float) -> dict[str, Any]:
    return {"offset": [x, 0.0, 0.5], "size": [0.8, 1.5]}


@pytest.fixture()
def counter(monkeypatch: pytest.MonkeyPatch) -> Any:
    """RoboCasa's ``Counter``: its regions in the order ``answers`` gives, as ``list(set(geoms))`` hands them out."""

    class Counter:
        answers: list[list[float]] = []

        def get_reset_regions(self, env: Any = None) -> dict[str, Any]:
            return {f"geom_{index}": _region(x) for index, x in enumerate(self.answers.pop(0))}

    module = SimpleNamespace(Counter=Counter)
    for name in ("robocasa", "robocasa.models", "robocasa.models.fixtures"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robocasa.models.fixtures.counter", module)
    return Counter


def test_a_run_only_reads_a_counter_s_regions_and_keeps_their_order(counter: Any) -> None:
    original = counter.get_reset_regions
    state = (random.getstate(), np.random.get_state()[1].tobytes())
    counter.answers = [[-0.8, 0.8]]

    with region_order() as kept:
        answer = counter().get_reset_regions()

    assert answer == {"geom_0": _region(-0.8), "geom_1": _region(0.8)}  # as RoboCasa answered
    assert len(kept) == 1 and json.loads(kept[0][0])["offset"][0] == -0.8
    assert counter.get_reset_regions is original
    assert (random.getstate(), np.random.get_state()[1].tobytes()) == state


def test_a_replay_gets_the_run_s_order_under_the_same_names(counter: Any) -> None:
    counter.answers = [[-0.8, 0.8]]
    with region_order() as kept:
        counter().get_reset_regions()

    counter.answers = [[0.8, -0.8]]  # this process's set came out the other way round
    with region_order(kept) as replayed:
        answer = counter().get_reset_regions()

    assert list(answer) == ["geom_0", "geom_1"]
    assert answer["geom_0"] == _region(-0.8) and answer["geom_1"] == _region(0.8)
    assert replayed == kept


def test_a_replay_refuses_a_counter_that_differs_from_the_run_s(counter: Any) -> None:
    counter.answers = [[-0.8, 0.8]]
    with region_order() as kept:
        counter().get_reset_regions()

    counter.answers = [[0.8, 0.5]]
    with pytest.raises(RoboCasaError, match="other regions"):
        with region_order(kept):
            counter().get_reset_regions()
    counter.answers = [[-0.8, 0.8], [-0.8, 0.8]]
    with pytest.raises(RoboCasaError, match="other regions"):
        with region_order(kept):
            counter().get_reset_regions(), counter().get_reset_regions()
    with pytest.raises(RoboCasaError, match="listed regions 0 times, the run's 1"):
        with region_order(kept):
            pass


def test_a_kitchen_without_robocasa_is_left_alone() -> None:
    with region_order() as kept:
        pass

    assert kept == []


def _built_kitchen(built: list[Any], order: list[float]) -> Any:
    """A kitchen whose stand mixer stands in whichever region its seeded draw lands on, RoboCasa's way."""

    from robocasa.models.fixtures.counter import Counter

    class Built:
        horizon = 750

        def __init__(self) -> None:
            Counter.answers = [list(order)]
            regions = Counter().get_reset_regions()
            self.mixer_x = list(regions.values())[0]["offset"][0]  # the same index, whichever process
            built.append(self)

        def reset(self) -> None:
            pass

        def get_ep_meta(self) -> dict[str, Any]:
            return {"lang": "Open the right drawer."}

        def _check_success(self) -> bool:
            return False

    return lambda *, task, seed: Built()


def test_a_kitchen_rebuilt_in_a_run_s_order_is_the_run_s_and_the_policy_sees_none_of_it(tmp_path, monkeypatch, counter) -> None:
    monkeypatch.setattr(OmronIO, "__post_init__", lambda _self: None)
    for name, value in (("grip_pose", list(GRIP)), ("arm_qpos", [0.0] * 7), ("base_pose", list(BASE)),
                        ("public_gripper_geometry", {}), ("finger_qpos", [0.0, 0.0])):
        monkeypatch.setattr(OmronIO, name, lambda _self, value=value: value)
    monkeypatch.setattr("ora.environment.backends.robocasa.session.shared_motion.start", lambda _session: None)
    built: list[Any] = []
    run = RoboCasaSession(output_root=tmp_path / "run", config={
        "task": "OpenDrawer", "seed": 3, "environment_factory": _built_kitchen(built, [-0.8, 0.8])})

    reset = run.start()

    record = json.loads((tmp_path / "run/controller-replay/region-order-round-0000.json").read_text())
    assert record["round_index"] == 0 and len(record["regions"]) == 1
    assert "regions" not in reset  # the Supervisor's copy, and round-resets/, are as before
    assert "regions" not in json.loads((tmp_path / "run/round-resets/round-0000.json").read_text())
    replay = RoboCasaSession(output_root=tmp_path / "replay", config={
        "task": "OpenDrawer", "seed": 3, "environment_factory": _built_kitchen(built, [0.8, -0.8]),
        "region_order": {0: record["regions"]}})
    replay.start()
    assert built[1].mixer_x == built[0].mixer_x == -0.8


def _action(sequence: int, name: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {"ticks": 2}
    return {"schema": "ora-motion-action.v3", "observation_id": "o", "action_id": f"a{sequence}",
            "bound_observation_sha256": "0" * 64, "action": name, "arguments": arguments}


def _session(tmp_path: Path, kitchen: Kitchen, monkeypatch: pytest.MonkeyPatch, *, ticks: list[int],
             status: str = "reached") -> RoboCasaSession:
    """A session whose motion is a list of tick counts: each action sends that many vectors."""
    session = RoboCasaSession(config={"task": "OpenDrawer", "seed": 3}, output_root=tmp_path / "native")
    session.io = _io(kitchen, monkeypatch)
    session.horizon = 750
    plan = iter(ticks)

    def motion(_session: Any, action: Any) -> dict[str, Any]:
        for _ in range(next(plan)):
            session.after_control_step(session.io.step(base_velocity_world=(0.1, 0.0, 0.0)))
        return {"execution_status": status, "action_id": action["action_id"]}

    monkeypatch.setattr("ora.environment.backends.robocasa.session.execute_motion_v3", motion)
    monkeypatch.setattr("ora.environment.backends.robocasa.session.validate_runtime_action", lambda action: action)
    return session


def test_each_action_records_its_vectors_and_the_scene_it_started_and_ended_in(tmp_path, monkeypatch) -> None:
    kitchen = Kitchen()
    session = _session(tmp_path, kitchen, monkeypatch, ticks=[3, 0, 2], status="not_reached")

    hashes = [session.io.state_sha256()]
    for sequence in range(3):
        session.execute(_action(sequence, "move_base"))
        hashes.append(session.io.state_sha256())
    manifest_path = session.output_root / "controller-replay" / "manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)

    assert manifest["backend"] == "robocasa"
    assert [entry["physics_steps"] for entry in manifest["entries"]] == [3, 0, 2]
    assert [entry["replayable"] for entry in manifest["entries"]] == [True, False, True]
    assert {entry["controller"] for entry in manifest["entries"]} == {REPLAY_CONTROLLER}
    assert manifest["entries"][0]["execution"]["status"] == "not_reached"
    assert manifest["entries"][2]["metadata"]["state_sha256"] == {"before": hashes[2], "after": hashes[3]}
    assert manifest["entries"][1]["artifact"] is None
    commands = load_controller_replay_arrays(manifest_path, manifest["entries"][2])["commands"]
    assert commands.dtype == np.float64 and commands.shape == (2, C.ACTION_DIM)
    assert np.array_equal(commands, np.stack(kitchen.received[3:5]))


def test_a_round_is_recorded_on_its_entries(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, Kitchen(), monkeypatch, ticks=[1])
    session.replay_recorder.begin_round(2)

    session.execute(_action(0, "wait"))

    assert session.replay_recorder.entries[0]["round_index"] == 2


# ---------------------------------------------------------------------------- the program


def _run(tmp_path: Path, *, ticks: list[int] = (2, 0, 3), success_action: int | None = 2,
         round_indexes: list[int] | None = None, final_round: int | None = None) -> Path:
    """A run directory as a RoboCasa365 episode leaves it: the kitchen's own record, receipts and verdict."""
    run = tmp_path / "runs" / "ora-test-robocasa-0001"
    native = run / "native"
    (native / "receipts").mkdir(parents=True)
    (native / "round-resets").mkdir()
    rounds = round_indexes or [0] * len(ticks)
    session = RoboCasaSession(config={"task": "OpenDrawer", "seed": 3}, output_root=native)
    kitchen = Kitchen()
    state = lambda: hashlib.sha256(np.concatenate(([kitchen.data.time], kitchen.data.qpos, kitchen.data.qvel)).tobytes()).hexdigest()
    control_step = 0
    current = 0
    first_success = None
    for sequence, count in enumerate(ticks):
        if rounds[sequence] != current:
            current = rounds[sequence]
            kitchen = Kitchen()
            session.replay_recorder.begin_round(current)
        before = state()
        commands = np.ones((count, C.ACTION_DIM)) * (sequence + 1.0)
        for command in commands:
            kitchen.step(command.copy())
            control_step += 1
            if sequence == success_action:
                first_success = first_success or {"action_sequence": sequence, "control_step": control_step}
        session.replay_recorder.record(
            action_sequence=sequence, action={"action_id": f"a{sequence}", "action": "move_base"},
            controller=REPLAY_CONTROLLER, arrays={"commands": commands} if count else {}, physics_steps=count,
            replayable=bool(count), metadata={"state_sha256": {"before": before, "after": state()}})
        session.replay_recorder.update(sequence, execution_status="reached" if count else "plan_rejected")
        (native / "receipts" / f"action-{sequence:04d}.json").write_text(json.dumps({
            "action_id": f"a{sequence}", "action": "move_base", "after": {"right": {"pose_xyz_xyzw": GRIP}},
            "base_after": {"pose_xy_yaw": BASE}}))
    last_round = current if final_round is None else final_round
    for index in range(last_round + 1):
        (native / "round-resets" / f"round-{index:04d}.json").write_text(json.dumps({"seed": 3 + index}))
    for index in range(last_round + 1):
        (native / "controller-replay" / f"region-order-round-{index:04d}.json").write_text(json.dumps(
            {"round_index": index, "regions": [[str(index)]]}))
    (native / "official-verdict.json").write_text(json.dumps({
        "official_success": first_success is not None, "first_success": first_success, "round_index": last_round,
        "control_steps": control_step}))
    (run / "plan.json").write_text(json.dumps({
        "task": "OpenDrawer", "seed": 3, "source": {"commit": "c0ffee"}, "policy": {"action_codec": "motion_action_v3"},
        "runtime": {"kind": "robocasa", "python_env": "ROBOCASA_PYTHON"}}))
    return run


def test_a_program_is_one_step_per_action_with_the_base_and_scene_it_must_reach(tmp_path) -> None:
    program = rp.build_program(_run(tmp_path))

    assert program["backend"] == "robocasa" and program["status"] == "replayable"
    assert [(step["kind"], step["physics_steps"]) for step in program["steps"]] == [
        ("controls", 2), ("skip", 0), ("controls", 3)]
    expect = program["steps"][0]["expect"]
    assert expect["right"] == GRIP and expect["base"] == BASE
    assert len(expect["state_before"]) == len(expect["state_after"]) == 64
    assert program["expect"] == {"official_success": True, "first_success_action": 2,
                                 "first_success_control_step": 3, "round_index": 0}
    assert "program.controls(2)  # move_base, robocasa_composite_action, 3 control steps" in rp.render_program(program)


def _measured(**fields: Any) -> dict[str, Any]:
    arms = {"right": {"bit_identical": True, "position_mm": 0.0}, "base": {"bit_identical": True, "position_mm": 0.0},
            "state_after": {"bit_identical": True}}
    return {"steps": [{"action_sequence": 0, "arms": arms}], "official_success": True, "first_success_action": 1,
            "first_success_control_step": 9, **fields}


def test_the_check_fails_on_a_base_a_scene_or_the_control_step_of_the_verdict() -> None:
    program = {"backend": "robocasa", "recorded": True, "expect": {
        "official_success": True, "first_success_action": 1, "first_success_control_step": 9}}

    assert rp._compare(program, _measured())["failures"] == []
    for arm in ("base", "state_after"):
        moved = _measured()
        moved["steps"][0]["arms"][arm] = {"bit_identical": False, "position_mm": None}
        assert rp._compare(program, moved)["pose_mismatches"][0]["arm"] == arm
    assert rp._compare(program, _measured(first_success_control_step=10))["failures"] == [
        "the replay's verdict differs from the run's"]


# ---------------------------------------------------------------------------- the worker


class Session:
    """The session as the replay drives it: rebuilt per round from a seed, with one latch."""

    physics_steps_per_tick = 25

    def __init__(self, *, wins: bool = True, drift_at: int | None = None) -> None:
        self.kitchen = Kitchen()
        self.seeds: list[tuple[int, int]] = []
        self.wins, self.drift_at = wins, drift_at
        self.action_sequence = 0
        self.first_success: dict[str, Any] | None = None
        self.control_steps = 0
        self.closed = False
        self.io = self

    @property
    def data(self) -> Any:
        return self.kitchen.data

    def start(self) -> None:
        pass

    def restart_round(self, round_index: int, *, seed: int) -> None:
        self.kitchen = Kitchen()
        self.seeds.append((round_index, seed))

    state_sha256 = OmronIO.state_sha256

    def apply(self, command: Any) -> StepResult:
        self.kitchen.step(np.array(command))
        self.control_steps += 1
        if self.drift_at == self.action_sequence:
            self.kitchen.data.qvel[2] += 1e-12  # an object nudged: the robot's end pose below does not move
        return StepResult(reward=0.0, official_success=self.wins and self.action_sequence == 2)

    def after_control_step(self, result: StepResult) -> None:
        if result.official_success and self.first_success is None:
            self.first_success = {"action_sequence": self.action_sequence, "control_step": self.control_steps}

    def grip_pose(self) -> list[float]:
        return list(GRIP)

    def base_pose(self) -> list[float]:
        return list(BASE)

    def persist_official_verdict(self) -> dict[str, Any]:
        return {"official_success": self.first_success is not None, "first_success": self.first_success,
                "control_steps": self.control_steps}

    def close(self) -> None:
        self.closed = True


def _rebuilt(session: Session) -> Any:
    def build(_plan: Any, _output: Any, regions: Any) -> Session:
        session.regions = regions
        return session
    return build


def _replay(run: Path, session: Session, monkeypatch: pytest.MonkeyPatch, *, scene_trace: Any = None) -> dict[str, Any]:
    monkeypatch.setattr(worker, "_robocasa_replay_session", _rebuilt(session))
    manifest_path = run / "native/controller-replay/manifest.json"
    program = rp.build_program(run)
    return program_worker._run_robocasa_program(
        plan={}, manifest=load_controller_replay_manifest(manifest_path), manifest_path=manifest_path,
        output_root=run.parent / "out", program=program, scene_trace=scene_trace), program


def test_a_replay_feeds_the_recorded_vectors_and_passes_only_on_identical_bits(tmp_path, monkeypatch) -> None:
    session = Session()

    measured, program = _replay(_run(tmp_path), session, monkeypatch)

    assert len(session.kitchen.received) == 5 and session.closed
    assert [set(row["arms"]) for row in measured["steps"]] == [{"state_before", "right", "base", "state_after"}] * 3
    assert all(arm["bit_identical"] for row in measured["steps"] for arm in row["arms"].values())
    assert rp._compare({**program, "recorded": True}, measured)["failures"] == []
    assert measured["first_success_action"] == 2 and measured["first_success_control_step"] == 3


def test_a_nudged_object_fails_the_check_though_the_robot_ended_on_the_same_pose(tmp_path, monkeypatch) -> None:
    measured, program = _replay(_run(tmp_path), Session(drift_at=0), monkeypatch)

    comparison = rp._compare({**program, "recorded": True}, measured)

    assert {row["arm"] for row in comparison["pose_mismatches"]} == {"state_after", "state_before"}
    assert comparison["failures"][0].endswith("action end poses differ from their receipts")
    assert all(row["arms"]["right"]["bit_identical"] for row in measured["steps"])


def test_a_verdict_the_replay_does_not_repeat_fails_the_check(tmp_path, monkeypatch) -> None:
    measured, program = _replay(_run(tmp_path), Session(wins=False), monkeypatch)

    assert rp._compare({**program, "recorded": True}, measured)["failures"] == ["the replay's verdict differs from the run's"]


def test_a_step_that_disagrees_with_its_record_is_refused_and_the_session_closed(tmp_path, monkeypatch) -> None:
    run = _run(tmp_path)
    session = Session()
    monkeypatch.setattr(worker, "_robocasa_replay_session", _rebuilt(session))
    manifest_path = run / "native/controller-replay/manifest.json"
    program = rp.build_program(run)
    program["steps"][0]["kind"] = "skip"

    with pytest.raises(worker.ReplayWorkerError, match="program step 0 is skip but its record replays as controls"):
        program_worker._run_robocasa_program(plan={}, manifest=load_controller_replay_manifest(manifest_path),
                                             manifest_path=manifest_path, output_root=tmp_path / "out", program=program)
    assert session.closed


class Trace:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def begin_round(self, _session: Any, round_index: int) -> None:
        self.events.append(("round", round_index))

    def begin_action(self, entry: Any) -> None:
        self.events.append(("action", entry["action_sequence"]))

    def after_tick(self) -> None:
        self.events.append("tick")

    def end_action(self) -> None:
        self.events.append("end")

    def close(self) -> None:
        self.events.append("close")


def test_a_new_round_rebuilds_from_the_seed_its_reset_recorded_and_the_trace_sees_each_tick(tmp_path, monkeypatch) -> None:
    run = _run(tmp_path, ticks=[1, 2], round_indexes=[0, 1], success_action=None)
    session, trace = Session(wins=False), Trace()

    _replay(run, session, monkeypatch, scene_trace=trace)

    assert session.seeds == [(1, 4)] and session.regions == {0: [["0"]], 1: [["1"]]}
    assert trace.events == [("round", 0), ("action", 0), "tick", "end", ("round", 1), ("action", 1), "tick", "tick", "end",
                            "close"]


def test_a_last_round_with_no_action_in_it_is_still_entered(tmp_path, monkeypatch) -> None:
    run = _run(tmp_path, ticks=[1], success_action=None, final_round=2)
    session = Session(wins=False)

    _replay(run, session, monkeypatch)

    assert session.seeds == [(2, 5)]


def test_the_job_routes_a_robocasa_run_through_its_scene_trace(tmp_path, monkeypatch) -> None:
    run = _run(tmp_path)
    session, traces = Session(), []

    def tracer(directory: Path, run_id: str) -> Any:
        trace = Trace()
        trace.finish = lambda: {"shape": [1, 1, 7]}
        traces.append((directory, run_id, trace))
        return trace

    monkeypatch.setattr(worker, "_robocasa_replay_session", _rebuilt(session))
    monkeypatch.setattr(worker, "_robocasa_scene_trace", tracer)
    result = program_worker.run_program({
        "run_root": str(run), "output_root": str(tmp_path / "out"), "project_root": str(tmp_path),
        "program": rp.build_program(run), "scene_trace": {"directory": str(tmp_path / "trace")}})

    assert result["backend"] == "robocasa" and result["scene_trace"] == {"shape": [1, 1, 7]}
    assert (tmp_path / "out" / "replay-program-result.json").is_file()
    assert traces[0][1] == run.name and "tick" in traces[0][2].events


# ---------------------------------------------------------------------------- a run without controls


def _old_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "ora-old-robocasa-0001"
    (run / "native" / "receipts").mkdir(parents=True)
    (run / "native" / "receipts" / "action-0000.json").write_text(json.dumps(
        {"action_id": "a0", "action": "move_base", "steps_used": 40, "after": {"right": {"pose_xyz_xyzw": GRIP}}}))
    (run / "plan.json").write_text(json.dumps({"task": "OpenDrawer", "seed": 3, "runtime": {"kind": "robocasa"}}))
    return run


def test_a_run_sealed_without_controls_says_so_and_gets_no_program(tmp_path) -> None:
    run = _old_run(tmp_path)

    written = rp.write_program_at_seal(run)

    assert written["status"] == "unavailable" and "only receipts" in written["reason"]
    record = json.loads((run / "replay-program" / rp.UNAVAILABLE_JSON).read_text())
    assert record["schema"] == rp.UNAVAILABLE_SCHEMA and "never rebuilt from receipts" in record["reason"]
    assert not (run / "replay-program" / rp.PROGRAM_JSON).exists()


def test_an_already_sealed_run_is_refused_and_stays_as_it_was(tmp_path) -> None:
    run = _old_run(tmp_path)
    (run / "seal.json").write_text("{}")
    before = sorted(path for path in run.rglob("*"))

    with pytest.raises(rp.ReplayProgramUnavailable, match="no controller replay"):
        rp.write_program(run, tmp_path / "exported")
    with pytest.raises(rp.ReplayProgramUnavailable, match="no controller replay"):
        rp.prepare_check(run, tmp_path / "check")

    assert not (tmp_path / "exported" / rp.PROGRAM_PY).exists()
    assert sorted(path for path in run.rglob("*")) == before
