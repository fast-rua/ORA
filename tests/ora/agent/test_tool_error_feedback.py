"""Argument refusals say what was wrong, and identical repeats are counted.

The RoboDojo g016 ``classify_objects`` cell sent ``geometry_query`` the same
arguments fourteen times in five minutes -- ``points``, ``region`` and
``world_points_m`` together -- and every result was the same bare refusal:
``invalid_arguments``, ``retryable: true``, ``field``/``received`` null and
``constraints`` empty. Three things were missing, and each is checked here for
every tool rather than for that one call:

* an argument refusal states that the unchanged request is refused again
  (``retry_requires_changed_arguments``) while ``retryable`` keeps its single
  meaning -- the policy can still recover through a later call;
* a mutually exclusive argument rule names the rule and the argument names
  that were received, never their values;
* the gateway counts byte-identical consecutive failures per policy session
  and appends one factual note to the new result, never to history.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.agent.surface_catalog import catalog_grants, surface_of
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.toolkit.host.runtime import base as runtime_base
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.session_core.repeated_failures import RepeatedFailureTracker
from ora.agent.error_classes import (
    ARGUMENT_REFUSAL_CODES,
    ToolError,
    require_at_most_one_of,
    require_exactly_one_of,
)
from tests.support.orchestrator.runtime_v1 import publish_synthetic

#: The arguments the g016 cell sent, verbatim.
G016_ARGUMENTS = {
    "camera": "head", "points": [[224, 177], [148, 184], [246, 146]],
    "region": [0, 0, 1, 1], "step": 48, "window_px": 2,
    "world_points_m": [[0, 0, 0]],
}
OPERATIONS = ["points", "region", "world_points_m"]
NOTE = (
    "this identical request has failed {count} times in a row with {code}; "
    "sending it unchanged will fail again"
)


# --- the envelope -----------------------------------------------------------

def test_robot_module_still_exports_the_one_tool_error_class():
    assert runtime_base.ToolError is ToolError


@pytest.mark.parametrize("code", sorted(ARGUMENT_REFUSAL_CODES))
def test_an_argument_refusal_says_the_unchanged_request_is_refused_again(code):
    reported = ToolError("bad request", code=code).as_dict()
    assert reported["retry_requires_changed_arguments"] is True
    # retryable keeps its one meaning across every consumer: a later call
    # through this tool can still succeed. It is not a resend instruction.
    assert reported["retryable"] is True


@pytest.mark.parametrize("code,retryable", [
    ("stale_observation", True), ("target_geometry_unavailable", True),
    ("internal_tool_error", False), ("episode_terminal", False),
])
def test_other_codes_make_no_claim_about_an_unchanged_repeat(code, retryable):
    reported = ToolError("refused", code=code, retryable=retryable).as_dict()
    assert "retry_requires_changed_arguments" not in reported
    assert reported["retryable"] is retryable


def test_a_raising_site_can_state_the_flag_explicitly():
    assert ToolError(
        "x", code="invalid_arguments", retry_requires_changed_arguments=False,
    ).as_dict()["retry_requires_changed_arguments"] is False
    assert ToolError(
        "x", code="target_ref_unavailable", retry_requires_changed_arguments=True,
    ).as_dict()["retry_requires_changed_arguments"] is True


def test_a_non_tool_exception_becomes_an_internal_error_without_the_flag():
    error = _tool_error(RuntimeError("boom"))["error"]
    assert error["code"] == "internal_tool_error" and error["retryable"] is False
    assert "retry_requires_changed_arguments" not in error


# --- mutually exclusive argument rules --------------------------------------

def _refusal(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


def test_exactly_one_of_names_the_rule_and_the_received_names_only():
    error = _refusal(
        require_exactly_one_of, G016_ARGUMENTS, OPERATIONS,
        message="provide exactly one of points, region or world_points_m",
    )
    assert error == {
        "code": "invalid_arguments",
        "message": "provide exactly one of points, region or world_points_m",
        "retryable": True,
        "retry_requires_changed_arguments": True,
        "field": None,
        "received": ["camera", "points", "region", "step", "window_px", "world_points_m"],
        "constraints": {"exactly_one_of": OPERATIONS, "received": OPERATIONS},
        "infrastructure": False,
    }
    none = _refusal(require_exactly_one_of, {"camera": "head"}, OPERATIONS, message="m")
    assert none["constraints"] == {"exactly_one_of": OPERATIONS, "received": []}
    assert require_exactly_one_of({"camera": "head", "region": [0, 0, 1, 1]}, OPERATIONS, message="m") == "region"


def test_at_most_one_of_treats_an_explicit_null_as_omitted():
    assert require_at_most_one_of({"xyz_m": None, "delta_mm": [1, 0, 0]}, ("xyz_m", "delta_mm"), message="m") == "delta_mm"
    assert require_at_most_one_of({}, ("xyz_m", "delta_mm"), message="m") is None
    error = _refusal(
        require_at_most_one_of, {"xyz_m": [0.1, 0, 0.2], "delta_mm": [5, 0, 0], "gripper": "open"},
        ("xyz_m", "delta_mm"), message="provide xyz_m or delta_mm, not both",
    )
    assert error["message"] == "provide xyz_m or delta_mm, not both"
    assert error["received"] == ["delta_mm", "gripper", "xyz_m"]
    assert error["constraints"] == {"at_most_one_of": ["xyz_m", "delta_mm"], "received": ["xyz_m", "delta_mm"]}


def test_refusals_carry_argument_names_never_argument_values():
    error = _refusal(require_exactly_one_of, G016_ARGUMENTS, OPERATIONS, message="m")
    encoded = json.dumps(error)
    for value in ("224", "177", "[0, 0, 1, 1]", "48", "head"):
        assert value not in encoded, value


def test_geometry_query_reports_the_g016_violation(tmp_path):
    _observation, _pointer, _workspace, _control, tools, _store = publish_synthetic(tmp_path)
    error = _refusal(tools._geometry, {**G016_ARGUMENTS, "camera": "head_camera"})
    assert error["message"] == "provide exactly one of points, region or world_points_m"
    assert error["retry_requires_changed_arguments"] is True
    assert error["received"] == ["camera", "points", "region", "step", "window_px", "world_points_m"]
    assert error["constraints"] == {"exactly_one_of": OPERATIONS, "received": OPERATIONS}
    unknown = _refusal(tools._geometry, {"camera": "head_camera", "points": [[1, 1]], "scale": 2})
    assert unknown["received"] == ["camera", "points", "scale"]
    assert unknown["constraints"]["unknown"] == ["scale"]
    assert unknown["constraints"]["required"] == ["camera"]


# --- report_issue (refusal census E7) ---------------------------------------

def _report_issue_runtime(tmp_path, *, advisory: bool):
    """The Dojo default surface, served as a run serves it; without the advisory capability only the tool checks."""
    from dataclasses import replace

    from tests.support.agent.toolkit_run import served

    from tests.support.robodojo.perception import BOUNDS, FakeHeadGrounding, FakeX5Proposer

    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control,
        manual_path=Path(__file__).resolve().parents[5]
        / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile="robodojo_placement_toolkit_v5", toolkit_grants=catalog_grants("robodojo_placement_toolkit_v5"),
        controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=FakeX5Proposer(),
    )
    if not advisory:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            name for name in runtime.surface.capabilities if name != "public_rgbd_retention_advisory_v1"))
    return served(runtime)


@pytest.mark.parametrize("advisory", [True, False])
@pytest.mark.parametrize("message", ["", " ", "\n\t  "])
def test_a_blank_report_issue_message_names_the_field_and_rule(tmp_path, advisory, message):
    runtime = _report_issue_runtime(tmp_path, advisory=advisory)
    error = _refusal(runtime.invoke, "report_issue", {"message": message})
    assert error["code"] == "invalid_arguments" and error["field"] == "message"
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True
    assert error["received"] == {"type": "string", "length": len(message)}
    if message:
        # Whitespace passes the declared minLength; the handler states the rule.
        assert error["constraints"] == {"minLength": 1, "rule": "non_whitespace"}
    else:
        # Every surface refuses the empty string with the declared schema's rule.
        assert error["constraints"] == {"violated": "minLength", "minLength": 1}
    assert not (runtime.control_root / "openeta-reported-issues.jsonl").exists()


@pytest.mark.parametrize("arguments,field", [
    ({}, None),
    ({"message": 3}, "message"),
    ({"message": ["broken"]}, "message"),
    ({"message": "grasp_plan returns no candidates", "severity": "high"}, None),
])
def test_a_malformed_report_issue_names_what_was_wrong_without_values(tmp_path, arguments, field):
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    error = _refusal(runtime.invoke, "report_issue", arguments)
    assert error["code"] == "invalid_arguments" and error["field"] == field
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True
    assert error["constraints"]["violated"] in {"required", "type", "additionalProperties"}
    assert "broken" not in json.dumps(error) and "high" not in json.dumps(error)


@pytest.mark.parametrize("arguments", [["message"], "message", None])
def test_a_non_object_report_issue_is_a_policy_refusal_never_an_internal_error(tmp_path, arguments):
    """An internal_tool_error would count as infrastructure evidence for abort."""
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    error = _refusal(runtime._report_issue, arguments)
    assert error["code"] == "invalid_arguments" and error["constraints"]["violated"] == "type"
    assert error["infrastructure"] is False


#: A surface of each dispatch branch: synchronous execution (RoboDojo), proposals (LIBERO) and the five robot tools.
TOOLKIT_SURFACES = ("robodojo_placement_toolkit_v5", "libero_placement_direct_overhead_v1", "robot_runtime_v2")


def _toolkit_runtime(tmp_path, profile):
    """``profile`` served from the toolkit over fake services; a synchronous surface gets an offline executor."""
    from ora.agent.toolkit.host.runtime.dispatch import install
    from tests.support.agent.toolkit_legacy import Executor
    from tests.support.orchestrator.runtime_v1 import MANUAL as ROBOT_MANUAL
    from tests.support.robodojo.perception import BOUNDS, FakeHeadGrounding, FakeX5Proposer

    control = tmp_path / "control"
    control.mkdir()
    if profile == "robot_runtime_v2":
        runtime = runtime_base.RobotToolRuntime(
            control_root=control, manual_path=ROBOT_MANUAL, event_path=tmp_path / "events.jsonl",
            workspace_bounds_m=BOUNDS, state_root=None, tool_surface_profile=profile)
    else:
        runtime = ManipulationToolRuntime(
            control_root=control,
            manual_path=Path(__file__).resolve().parents[3] / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
            event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
            tool_surface_profile=profile, toolkit_grants=catalog_grants(profile), controlled_end_effector="right",
            segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
            grasp_backend=FakeX5Proposer())
    if runtime.surface.supports("toolkit_synchronous_execution_v1"):
        runtime.toolkit_executor = Executor("tool_output")
    return install(runtime)


@pytest.mark.parametrize("profile", TOOLKIT_SURFACES)
@pytest.mark.parametrize("arguments", [["message"], "message", None, 3])
def test_every_toolkit_tool_refuses_a_non_object_call_as_its_schema_does(tmp_path, profile, arguments):
    """A malformed call is the policy's, refused before anything reads it: an internal_tool_error would count as
    infrastructure evidence for abort."""
    runtime = _toolkit_runtime(tmp_path, profile)
    handlers = [runtime._report_issue] if "report_issue" in runtime.allowed_tools else []
    calls = [(name, lambda name=name: runtime.invoke(name, arguments)) for name in runtime.allowed_tools]
    for name, call in [*calls, *(("report_issue handler", lambda: handler(arguments)) for handler in handlers)]:
        error = _refusal(call)
        assert error["code"] == "invalid_arguments", name
        assert error["message"] == "$input has the wrong input type; expected object", name
        assert error["constraints"] == {"violated": "type", "type": ["object"]}, name
        assert error["field"] is None and error["infrastructure"] is False, name
        assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True, name
    assert not (runtime.control_root / "openeta-reported-issues.jsonl").exists()


def test_a_report_issue_with_text_is_recorded_stripped(tmp_path):
    runtime = _report_issue_runtime(tmp_path, advisory=False)
    assert runtime.invoke("report_issue", {"message": "  grasp_plan returns no candidates \n"}) == {"success": True}
    recorded = json.loads((runtime.control_root / "openeta-reported-issues.jsonl").read_text())
    assert recorded["message"] == "grasp_plan returns no candidates"


# --- repeated identical failures --------------------------------------------

def _failure(code="invalid_arguments", **kwargs):
    return {"ok": False, "error": ToolError("refused", code=code, **kwargs).as_dict()}


def test_identical_consecutive_failures_are_counted_on_the_new_result_only():
    tracker = RepeatedFailureTracker()
    first = _failure()
    assert tracker.record("geometry_query", G016_ARGUMENTS, first) is first
    second = tracker.record("geometry_query", dict(reversed(list(G016_ARGUMENTS.items()))), _failure())
    third_input = _failure()
    third = tracker.record("geometry_query", G016_ARGUMENTS, third_input)
    assert second["error"]["repetition"] == {
        "identical_failures_in_a_row": 2,
        "note": NOTE.format(count=2, code="invalid_arguments"),
    }
    assert third["error"]["repetition"] == {
        "identical_failures_in_a_row": 3,
        "note": NOTE.format(count=3, code="invalid_arguments"),
    }
    assert "repetition" not in first["error"] and "repetition" not in third_input["error"]


@pytest.mark.parametrize("interruption", [
    ("geometry_query", {**G016_ARGUMENTS, "step": 49}, _failure()),
    ("observe", {}, _failure()),
    ("geometry_query", G016_ARGUMENTS, {"ok": True, "points": []}),
    ("geometry_query", G016_ARGUMENTS, _failure("stale_observation")),
    ("geometry_query", None, _failure()),
])
def test_any_different_call_or_outcome_resets_the_count(interruption):
    tracker = RepeatedFailureTracker()
    tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    name, arguments, result = interruption
    tracker.record(name, arguments, result)
    after = tracker.record("geometry_query", G016_ARGUMENTS, _failure())
    assert "repetition" not in after["error"]


def test_the_note_claims_a_certain_repeat_failure_only_for_argument_refusals():
    tracker = RepeatedFailureTracker()
    for _ in range(2):
        repeated = tracker.record("place_execute", {"plan_ref": "p"}, _failure("stale_plan_ref"))
    assert repeated["error"]["repetition"] == {
        "identical_failures_in_a_row": 2,
        "note": "this identical request has failed 2 times in a row with stale_plan_ref",
    }


def test_two_policy_sessions_never_share_a_count():
    left, right = RepeatedFailureTracker(), RepeatedFailureTracker()
    left.record("geometry_query", G016_ARGUMENTS, _failure())
    assert "repetition" not in right.record("geometry_query", G016_ARGUMENTS, _failure())["error"]


# --- the v3 causal projection carries the new public fields -----------------

def test_native_causal_projection_keeps_the_refusal_and_its_count():
    planner = object.__new__(PolicyDomainMixin)
    planner.tools = SimpleNamespace(surface=surface_of("robodojo_placement_toolkit_v5"))
    tracker = RepeatedFailureTracker()
    for _ in range(3):
        try:
            require_exactly_one_of(G016_ARGUMENTS, OPERATIONS, message="provide exactly one of points, region or world_points_m")
        except ToolError as exc:
            result = tracker.record("geometry_query", G016_ARGUMENTS, _tool_error(exc))
    visible = planner._causal_tool_result("geometry_query", result)["error"]
    assert visible["retry_requires_changed_arguments"] is True
    assert visible["constraints"] == {"exactly_one_of": OPERATIONS, "received": OPERATIONS}
    assert visible["received"] == sorted(G016_ARGUMENTS)
    assert visible["repetition"]["note"] == NOTE.format(count=3, code="invalid_arguments")


# --- the agent core: one count per policy session, history untouched --------

def test_the_agent_core_appends_the_count_without_editing_history(tmp_path: Path):
    """A Synthetic episode on the Pi core (its RepeatedFailureTracker runs in
    ora-pi/src/core/dispatch.ts): identical refusals in a row are counted on the
    new result only, any other call resets the count, and every request is an
    exact extension of the previous one."""
    from tests.support.agent_core import episodes
    from tests.support.agent_core.episodes import tool_reply
    from tests.support.agent_core.preconditions import require_pi_runtime

    require_pi_runtime()
    outside = {"point_id": "P0", "view": "agentview", "x": 99999, "y": 1}
    reordered = {"y": 1, "x": 99999, "view": "agentview", "point_id": "P0"}
    script = [
        tool_reply(("m1", "mark_point", outside)),
        tool_reply(("m2", "mark_point", reordered)),
        tool_reply(("m3", "mark_point", outside)),
        tool_reply(("view", "observe", {"views": ["not-a-view"]})),
        tool_reply(("m4", "mark_point", outside)),
        tool_reply(("act", "close_gripper", {})),
        tool_reply(("end", "finish_episode", {"outcome": "failure", "reason": "probe", "attempts": ["scripted"]})),
    ]
    run = episodes.run_core("pi", tmp_path, responses=script, actions=1)
    assert run["core"] == "pi" and run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    requests = [json.loads(request) for request in run["requests"]]
    results = {message["tool_call_id"]: json.loads(message["content"])
               for message in requests[-1]["messages"] if message.get("role") == "tool"}
    repetition = {call: result.get("error", {}).get("repetition") for call, result in results.items()}
    assert repetition["m1"] is None and repetition["view"] is None and repetition["m4"] is None
    assert repetition["m2"]["identical_failures_in_a_row"] == 2
    assert repetition["m3"] == {
        "identical_failures_in_a_row": 3, "note": NOTE.format(count=3, code="invalid_arguments"),
    }
    assert results["m1"]["error"]["retry_requires_changed_arguments"] is True
    assert results["act"]["status"] == "submitted"
    # Each result is appended once; nothing already delivered is rewritten.
    for earlier, later in zip(requests, requests[1:]):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
