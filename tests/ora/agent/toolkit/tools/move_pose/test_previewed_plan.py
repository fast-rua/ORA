"""Executing what was previewed (``lib/motion_plan/kept.py``): a sampling planner answers differently each time, so a real
call that replanned would run another path than the one shown. The same request on the same observation runs the
previewed trajectory, byte for byte, and says so; any change of observation, request or budget plans as ever.
"""
import json
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.lib.motion_plan import kept
from ora.agent.toolkit.tools.move_pose import tool
from tests.support.agent.toolkit_dry_run import GOAL, Fixture, compiled, measured

CONFIG = {"synchronous_execution": True,
          "motion": {"joint_trajectory": True, "capabilities": [], "examples": [], "global_planner": True}}


@pytest.fixture(autouse=True)
def current(monkeypatch):
    return measured(monkeypatch)


def _sampling():
    """A planner whose every answer takes another way round."""
    calls = []

    def answer():
        calls.append(1)
        bend = 0.01 * len(calls)
        path = [[0.3, 0.0, 0.2, 0, 0, 0], [0.35, bend, 0.2, 0, 0, 0], [*GOAL[:3], 0, 0, 0]]
        return {"status": "planned", "reason": None, "start_joints": path[0], "path_joints": path}

    return answer


def _knots(fixture, index=0):
    return json.dumps(fixture.executed[index][0]["arguments"]["arms"]["left"], sort_keys=True)


def _dry(api, arguments):
    with dry_run.dry():
        return tool.run(api, arguments)


def test_the_real_call_runs_the_trajectory_the_dry_run_planned_byte_for_byte():
    fixture = Fixture(config=CONFIG, planned=_sampling())
    record = _dry(fixture.api, {"targets": {}})
    planned = fixture.queries.count("plan_path")
    result = tool.run(fixture.api, {"targets": {}})
    assert fixture.queries.count("plan_path") == planned, "nothing is planned again"
    kept_plan = kept.move_plan(json.loads(fixture.state[kept.STATE_KEY])["plans"].popitem()[1])
    knots = fixture.executed[0][0]["arguments"]["arms"]["left"]
    assert knots["knots_rad"] == kept_plan.knots["joints"]["left"]
    assert knots["expected_link6_xyz_xyzw"] == kept_plan.expected["left"]
    assert result["receipt"]["ran_previewed_plan"] is True
    assert result["motion"]["route"] == record["route"]


def _labelled(monkeypatch, label):
    action, compilation = compiled([0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    action["arguments"]["interactions"] = {"left": label}
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kwargs: (action, compilation))


def test_a_preview_keeps_its_label_a_call_with_another_plans_as_ever(monkeypatch):
    fixture = Fixture()
    _labelled(monkeypatch, "contact_approach")
    record = _dry(fixture.api, {"targets": {}})
    paced = fixture.queries.count("pace_path")
    result = tool.run(fixture.api, {"targets": {}})
    assert result["receipt"]["ran_previewed_plan"] is True and fixture.queries.count("pace_path") == paced
    assert record["route"] == result["motion"]["route"] == "line"
    _labelled(monkeypatch, "precision")
    other = tool.run(fixture.api, {"targets": {}})
    assert "ran_previewed_plan" not in other["receipt"] and fixture.queries.count("pace_path") > paced


@pytest.mark.parametrize("label, flag", [("contact_approach", False), ("contact_approach", True), ("free_space", True)])
def test_a_dry_run_reports_the_expect_contact_the_real_call_uses(monkeypatch, label, flag):
    fixture = Fixture()
    _labelled(monkeypatch, label)
    arguments = {"targets": {}, **({"expect_contact": True} if flag else {})}
    record = _dry(fixture.api, arguments)
    result = tool.run(fixture.api, arguments)
    assert record["motion"]["expect_contact"] is flag and result["motion"]["expect_contact"] is flag
    assert fixture.executed[0][0]["arguments"].get("expect_contact", False) is flag
    assert record["route"] == result["motion"]["route"]
    assert result["receipt"]["ran_previewed_plan"] is True


def test_a_primitive_s_dry_run_plans_its_expect_contact_as_the_real_call_does():
    fixture = Fixture()
    arguments = {"targets": {}, "expect_contact": True}
    with dry_run.dry():
        record = tool.compile_and_submit(fixture.api, arguments, config=fixture.config, primitive="move_to")
    paced = fixture.queries.count("pace_path")
    tool.compile_and_submit(fixture.api, arguments, config=fixture.config, primitive="move_to")
    assert fixture.queries.count("pace_path") == paced, "the real call runs the plan the dry run kept"
    assert record["motion"]["expect_contact"] is True and record["route"] == "line"
    assert fixture.executed[0][0]["arguments"]["expect_contact"] is True


def test_a_dry_run_and_a_replan_differ_where_the_planner_does():
    """The control of the test above: without the kept plan the sampler's second answer runs."""
    fixture, plain = Fixture(config=CONFIG, planned=_sampling()), Fixture(config=CONFIG, planned=_sampling())
    tool.run(plain.api, {"targets": {}})
    tool.run(plain.api, {"targets": {}})
    assert _knots(plain, 0) != _knots(plain, 1)
    assert "ran_previewed_plan" not in tool.run(fixture.api, {"targets": {}})["receipt"]


def test_a_changed_observation_replans(current):
    fixture = Fixture(config=CONFIG, planned=_sampling())
    _dry(fixture.api, {"targets": {}})
    planned = fixture.queries.count("plan_path")
    current.identity = SimpleNamespace(json=lambda: {"observation_id": "next", "observation_sha256": "other"})
    result = tool.run(fixture.api, {"targets": {}})
    assert fixture.queries.count("plan_path") == planned + 1 and "ran_previewed_plan" not in result["receipt"]


def test_a_changed_request_replans():
    fixture = Fixture(config=CONFIG, planned=_sampling())
    _dry(fixture.api, {"targets": {}})
    planned = fixture.queries.count("plan_path")
    result = tool.run(fixture.api, {"targets": {}, "speed": "normal"})
    assert fixture.queries.count("plan_path") == planned + 1 and "ran_previewed_plan" not in result["receipt"]


def test_a_budget_that_changed_replans():
    fixture = Fixture(config=dict(CONFIG, episode_native_actions_remaining=40), planned=_sampling())
    _dry(fixture.api, {"targets": {}})
    fixture.config = dict(CONFIG, episode_native_actions_remaining=30)
    planned = fixture.queries.count("plan_path")
    tool.run(fixture.api, {"targets": {}})
    assert fixture.queries.count("plan_path") == planned + 1


def test_two_previews_each_run_their_own_plan():
    fixture = Fixture(config=CONFIG, planned=_sampling())
    _dry(fixture.api, {"targets": {}, "speed": "normal"})
    _dry(fixture.api, {"targets": {}})
    planned = fixture.queries.count("plan_path")
    assert tool.run(fixture.api, {"targets": {}, "speed": "normal"})["receipt"]["ran_previewed_plan"] is True
    assert fixture.queries.count("plan_path") == planned


def test_a_run_without_the_state_plans_as_ever():
    fixture = Fixture(config=CONFIG, planned=_sampling())
    api = SimpleNamespace(**{key: value for key, value in vars(fixture.api).items() if key != "state"})
    assert "ran_previewed_plan" not in tool.run(api, {"targets": {}})["receipt"]
    assert _dry(api, {"targets": {}})["feasible"] is True, "a state that cannot keep it only loses the reuse"


def test_only_one_observation_s_plans_are_kept():
    fixture = Fixture(config=CONFIG, planned=_sampling())
    identity = {"observation_id": "o", "observation_sha256": "s"}
    for index in range(kept.LIMIT + 2):
        kept.keep(fixture.api, identity, str(index), {"n": index})
    assert len(json.loads(fixture.state[kept.STATE_KEY])["plans"]) == kept.LIMIT
    assert kept.recall(fixture.api, {"observation_id": "p", "observation_sha256": "s"}, str(kept.LIMIT)) is None
