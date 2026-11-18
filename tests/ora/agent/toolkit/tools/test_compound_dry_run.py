"""A dry run of the jaws and of a committed grasp: what the policy side can say of them without the world."""
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.tools import _dry_run
from ora.agent.toolkit.tools.grasp_execute import _preview
from tests.support.agent.toolkit_dry_run import GOAL, START, Fixture, measured

HANDS = {"left": {"contact_center_local": [0, 0, 0], "opening_axis_local": [0, 1, 0], "maximum_aperture_m": 0.09}}
CONFIG = {"synchronous_execution": True,
          "motion": {"joint_trajectory": True, "capabilities": [], "examples": [], "preview_hands": HANDS}}
SEQUENCE = ["open", "pregrasp", "approach", "close", "lift", "verify"]


@pytest.fixture(autouse=True)
def current(monkeypatch):
    return measured(monkeypatch)


def _target(pose):
    return {"position_m": pose[:3], "orientation_xyzw": pose[3:]}


def _commitment():
    contact, lift = [0.4, 0.0, 0.15, 0, 0, 0, 1], [0.4, 0.0, 0.25, 0, 0, 0, 1]
    return {"arm": "left", "poses": {"pregrasp": _target(GOAL), "approach": _target(contact), "lift": _target(lift)},
            "grippers": {"open": 1.0, "close": 0.0}}


def _segments(record):
    return [(item["kind"], item["planned"], item["not_predicted"]) for item in record["segments"]]


def test_a_grasp_shows_every_planned_segment_up_to_contact_and_nothing_after_it_as_known(current):
    fixture = Fixture(config=CONFIG)
    record = _preview.record(fixture.api, CONFIG, _commitment(), SEQUENCE, current)
    assert fixture.executed == []
    assert _segments(record) == [("gripper", True, False), ("arm", True, False), ("arm", True, False),
                                 ("gripper", True, False), ("arm", False, True)]
    assert record["segments"][3]["label"] == _dry_run.JAWS_CLOSE and "not predicted" in record["segments"][-1]["label"]
    assert record["feasible"] is True and record["route"] == "grasp_sequence" and record["stages"] == SEQUENCE
    planned = sum(item["native_actions"] for item in record["segments"] if item["kind"] == "arm" and item["planned"])
    assert record["arms"]["left"]["native_actions"] == planned, "the lift is not in it"


def test_the_approach_starts_where_the_pregrasp_ends(current):
    fixture = Fixture(config=CONFIG)
    _preview.record(fixture.api, CONFIG, _commitment(), SEQUENCE, current)
    assert fixture.queries.count("ik_chain") >= 2, "the approach is solved from the pregrasp's end, not from here"


def test_a_grasp_whose_path_cannot_be_planned_says_which_stage(current):
    screen = {"status": "table_collision", "arm": "left", "first_blocked_segment": 1}
    fixture = Fixture(config=CONFIG, screens=[screen])
    record = _preview.record(fixture.api, CONFIG, _commitment(), SEQUENCE, current)
    assert record["feasible"] is False and record["failure_code"] == "table_collision"
    assert record["refusal"]["constraints"]["stage"] == "pregrasp"


def test_the_jaws_show_their_width_and_never_call_a_closed_jaw_closed():
    view = _dry_run.View({"proprioception": {"end_effectors": {"left": {"pose_xyz_xyzw": START, "gripper": 0.5}}}},
                         lambda: [], HANDS, None, None)
    record = _dry_run.jaws("set_gripper", {"left": "close"}, view)
    arm = record["arms"]["left"]
    assert (arm["jaw_width_from_m"], arm["jaw_width_to_m"]) == (0.045, 0.0)
    assert "not predicted" in record["segments"][0]["label"] and record["feasible"] is True
    opened = _dry_run.jaws("set_gripper", {"left": "open"}, view)
    assert opened["segments"][0]["label"] == _dry_run.JAWS_OPEN and opened["arms"]["left"]["jaw_width_to_m"] == 0.09


def test_set_gripper_stops_before_it_commands_anything(monkeypatch, current):
    from ora.agent.toolkit.tools.set_gripper import tool

    monkeypatch.setattr(tool, "observe", lambda api: current)
    monkeypatch.setattr(tool, "measured_states", lambda manifest, arms: {"left": START})
    fixture = Fixture(config={**CONFIG, "execution": {"generation": 3},
                              "motion": {**CONFIG["motion"], "mutation_refusals": {}}})
    api = SimpleNamespace(**{**vars(fixture.api), "act": dry_run.NoActions()})
    with dry_run.dry():
        record = tool.run(api, {"targets": {"left": "close"}})
    assert record["dry_run"] is True and fixture.executed == [] and fixture.state == {}


def test_a_refused_previews_legend_repeats_the_refusals_own_reason_within_bounds():
    from ora.agent.toolkit.tools import _dry_run

    short = _dry_run._headline({"failure_code": "goal_unreachable", "refusal": {"message": "no joint configuration"}})
    assert short == "refused: goal_unreachable; nothing moves. no joint configuration"
    assert _dry_run._headline({"failure_code": "x"}) == "refused: x; nothing moves"
    long = _dry_run._headline({"failure_code": "x", "refusal": {"message": "word " * 200}})
    assert long.endswith("...") and len(long) < 40 + _dry_run.HEADLINE_REASON_CHARS


def _seen_view():
    import numpy as np
    from ora.agent.toolkit.lib import action_preview

    extrinsic = np.eye(4)
    extrinsic[2, 3] = 1.0
    camera = action_preview.Camera("any", np.full((240, 320, 3), 90, dtype=np.uint8), None,
                                   np.asarray([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]]), extrinsic)
    return _dry_run.View({}, lambda: [camera], HANDS, None, None)


def test_a_real_call_refused_by_its_plan_carries_the_picture_of_what_was_asked():
    from ora.agent.toolkit.tools.errors import ToolRefusal

    error = _dry_run.pictured("move_pose", ToolRefusal("no joint configuration", code="goal_unreachable"),
                              {"left": START}, {"left": GOAL}, _seen_view)
    assert len(error.images_base64) == 1 and error.fields["code"] == "goal_unreachable"
    assert "images_base64" not in error.fields, "the picture is not part of the public envelope"


def test_a_malformed_request_or_a_dry_run_gets_no_picture_on_its_refusal():
    from ora.agent.toolkit.tools.errors import ToolRefusal

    malformed = _dry_run.pictured("move_pose", ToolRefusal("bad", code="invalid_motion_arguments"),
                                  {"left": START}, {"left": GOAL}, _seen_view)
    assert not malformed.images_base64
    with dry_run.dry():
        asked = _dry_run.pictured("move_pose", ToolRefusal("x", code="goal_unreachable"), {"left": START},
                                  {"left": GOAL}, _seen_view)
    assert not asked.images_base64 and asked.requested == {"left": [START, GOAL]}, "the dry run's answer draws it"


def test_a_picture_that_cannot_be_drawn_leaves_the_refusal_as_it_was():
    from ora.agent.toolkit.tools.errors import ToolRefusal

    def broken():
        raise RuntimeError("no cameras")

    error = _dry_run.pictured("move_pose", ToolRefusal("x", code="goal_unreachable"), {"left": START},
                              {"left": GOAL}, broken)
    assert not error.images_base64 and str(error) == "x"


def test_the_refusals_picture_reaches_the_failed_result_as_its_images():
    from ora.agent.session_core.tool_arguments import _tool_error
    from ora.agent.toolkit.host.failures import invoke
    from ora.agent.toolkit.tools.errors import ToolRefusal
    from ora.agent.error_classes import ToolError

    def tool(api, arguments):
        error = ToolRefusal("refused", code="goal_unreachable")
        error.images_base64 = ["cGljdHVyZQ=="]
        raise error

    with pytest.raises(ToolError) as raised:
        invoke(tool, None, {})
    result = _tool_error(raised.value)
    assert result["images_base64"] == ["cGljdHVyZQ=="] and result["error"]["code"] == "goal_unreachable"
    assert "images_base64" not in result["error"]
    assert "images_base64" not in _tool_error(ToolError("plain")), "a refusal without a picture is as before"
