"""The worker side of RoboLab's contract witness, on a scripted scene (no simulator import).

Each case resets the seed except the first (the session's own start reset it), lets the arm hold
past RoboLab's two-step artifact window, runs its ops and reads the official verdict over its ticks.
The reset case also records the visibility probe.
"""

from __future__ import annotations

import json

from ora.environment.backends.robolab import contract_witness


class _Bridge:
    def take_ticks(self):
        return []


class _Session:
    task = "AnimalsInBinTask"

    def __init__(self):
        self.bridge, self.restarts = _Bridge(), []

    def restart_round(self, index, *, seed):
        self.restarts.append((index, seed))


class _Scene:
    """Success appears once both lizards were placed into the bin."""

    log: list = []

    def __init__(self, session, keep=()):
        self.results, self.placed, self.parked, self.ignore_until = [], set(), [], 0

    def hold(self, ticks):
        self.results += [len(self.placed) == 2] * ticks

    def visibility(self, names):
        return {name: {"visible": True} for name in names}

    def place_over(self, name, reference, index, count, mode="into"):
        self.placed.add(name)
        _Scene.log.append(("into", name, reference))

    def beside(self, name, reference):
        _Scene.log.append(("beside", name, reference))

    def box(self, name):
        return None, None, [0.0, 0.0, 0.0]


def test_each_case_runs_from_its_own_reset_and_the_record_compares_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(contract_witness, "Scene", _Scene)
    plan = {"seed": 0, "visibility_objects": ["lizard_figurine", "grey_bin"], "cases": [
        {"case": "reset_state", "expected": False, "ops": [], "ticks": 3},
        {"case": "beside", "expected": False, "ops": [{"op": "place", "object": "lizard_figurine",
                                                       "beside": "grey_bin"}], "ticks": 5},
        {"case": "all_in", "expected": True, "ops": [
            {"op": "place", "object": "lizard_figurine", "into": "grey_bin", "slot": 0, "slots": 2},
            {"op": "place", "object": "lizard_figurine_01", "into": "grey_bin", "slot": 1, "slots": 2}], "ticks": 5},
        {"case": "unknown", "expected": True, "ops": [{"op": "fly"}], "ticks": 1}]}
    session = _Session()
    job = {"schema": contract_witness.JOB_SCHEMA, "task": session.task, "plan": plan,
           "output": str(tmp_path / "witness.json")}
    record = contract_witness.run_witness(job, session=session)
    assert session.restarts == [(1, 0), (2, 0), (3, 0)]
    cases = {case["case"]: case for case in record["cases"]}
    assert cases["reset_state"]["matches"] and cases["beside"]["matches"] and cases["all_in"]["matches"]
    assert cases["all_in"]["official_success"] is True
    assert cases["unknown"]["matches"] is False and "unknown witness op" in cases["unknown"]["error"]
    assert record["all_cases_match"] is False and record["visibility_at_reset"]["grey_bin"]["visible"]
    assert json.loads((tmp_path / "witness.json").read_text())["schema"] == contract_witness.WITNESS_SCHEMA


class _LandingScene(_Scene):
    """A verdict that flickers on while the objects land, then stays off."""

    def hold(self, ticks):
        for _ in range(ticks):
            self.results.append(len(self.results) < 4)


def test_an_exact_count_negative_does_not_count_the_landing_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr(contract_witness, "Scene", _LandingScene)
    ops = [{"op": "place", "object": "lizard_figurine", "into": "grey_bin", "settle": False}]
    plan = {"seed": 0, "cases": [
        {"case": "one_too_many", "expected": False, "ops": ops, "ticks": 20, "ignore_ticks": 10},
        {"case": "counted", "expected": False, "ops": ops, "ticks": 20}]}
    session = _Session()
    job = {"schema": contract_witness.JOB_SCHEMA, "task": session.task, "plan": plan,
           "output": str(tmp_path / "witness.json")}
    cases = {case["case"]: case for case in contract_witness.run_witness(job, session=session)["cases"]}
    assert cases["one_too_many"]["matches"] and cases["one_too_many"]["ignored_ticks"] == 10
    assert cases["counted"]["official_success"] is True and not cases["counted"]["matches"]


def test_the_plan_names_every_object_its_ops_touch():
    plan = {"visibility_objects": ["grey_bin"], "cases": [{"ops": [
        {"op": "place", "object": "keyboard", "beside": "grey_bin"},
        {"op": "count_in", "objects": ["mouse", "phone"], "container": "grey_bin", "k": 1}]}]}
    assert contract_witness._named(plan) == {"grey_bin", "keyboard", "mouse", "phone"}


def test_a_job_for_another_task_is_refused(tmp_path):
    session = _Session()
    job = {"schema": contract_witness.JOB_SCHEMA, "task": "OtherTask", "plan": {"cases": []},
           "output": str(tmp_path / "w.json")}
    try:
        contract_witness.run_witness(job, session=session)
    except contract_witness.RobolabError as exc:
        assert "does not name this session's task" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a job for another task must be refused")


class _SetDownScene(_Scene):
    """Records the stated placements, parking and grasp options; success once the hammer is set down."""

    def set_down(self, name, **spec):
        self.placed.add(name)
        _Scene.log.append(("put", name, spec))

    def park(self, name, index):
        _Scene.log.append(("park", name, index))

    def grasp(self, name, order=None, depth=None, flip=False, shift=0.0):
        _Scene.log.append(("grasp", name, order, depth, flip, shift))

    def hold(self, ticks):
        self.results += [bool(self.placed)] * ticks


def test_a_plan_may_state_the_spot_the_turn_and_the_grasp(tmp_path, monkeypatch):
    monkeypatch.setattr(contract_witness, "Scene", _SetDownScene)
    _Scene.log = []
    plan = {"seed": 0, "cases": [
        {"case": "out", "expected": True, "ticks": 2, "ops": [
            {"op": "park", "object": "center_bin", "index": 1},
            {"op": "put", "object": "red_hammer", "center": [0.3, -0.2], "rotation": "flat:90"},
            {"op": "put", "object": "wood_hammer", "reference": "left_bin", "dx": 0.1,
             "bottom": "floor:left_bin+0.02"}]},
        {"case": "held", "expected": False, "ticks": 2, "ops": [
            {"op": "grasp", "object": "pitcher", "order": [1, 0, 2], "depth": 0.0, "flip": True, "shift": 0.08}]}]}
    session = _Session()
    job = {"schema": contract_witness.JOB_SCHEMA, "task": session.task, "plan": plan,
           "output": str(tmp_path / "witness.json")}
    cases = {case["case"]: case for case in contract_witness.run_witness(job, session=session)["cases"]}
    assert cases["out"]["matches"] and cases["held"]["matches"]
    assert _Scene.log == [
        ("park", "center_bin", 1),
        ("put", "red_hammer", {"reference": None, "center": [0.3, -0.2], "dx": 0.0, "dy": 0.0, "bottom": "table",
                               "rotation": "flat:90"}),
        ("put", "wood_hammer", {"reference": "left_bin", "center": None, "dx": 0.1, "dy": 0.0,
                                "bottom": "floor:left_bin+0.02", "rotation": "own"}),
        ("grasp", "pitcher", [1, 0, 2], 0.0, True, 0.08)]
    assert contract_witness._named(plan) >= {"center_bin", "red_hammer", "wood_hammer", "left_bin", "pitcher"}


def test_a_stated_turn_lays_the_longest_axis_level_or_stands_it_up():
    import numpy as np

    scene = contract_witness.Scene.__new__(contract_witness.Scene)
    corners = np.array([[x, y, z] for x in (-0.2, 0.2) for y in (-0.05, 0.05) for z in (-0.01, 0.01)])
    scene.pose = lambda name: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    scene.local_corners = lambda name: corners
    flat = scene.turned("hammer", "flat:90")
    assert np.allclose(flat[:, 0], [0.0, 1.0, 0.0]) and np.allclose(np.abs(flat[:, 2]), [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.det(flat), 1.0)
    assert np.allclose(scene.turned("hammer", "yaw:90")[:, 0], [0.0, 1.0, 0.0])
    assert np.allclose(scene.turned("hammer", "long_up")[:, 0], [0.0, 0.0, 1.0])
    tilted = scene.turned("hammer", "tilt:30:0")
    assert np.isclose(tilted[2, 0], np.cos(np.radians(30))) and np.isclose(tilted[0, 0], np.sin(np.radians(30)))
    try:
        scene.turned("hammer", "sideways")
    except contract_witness.RobolabError as exc:
        assert "unknown witness rotation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown turn must be refused")
