"""``explain`` renders what one perception-tool call did from the tool's stage template, deterministically.

``ora/agent/toolkit/explain.py`` with ``tools/<name>/explain.json`` for ``vision_ground``, ``vision_segment``,
``geometry_query`` and ``grasp_plan`` (the perception tools) and the arm-moving tools ``move_pose``, ``move_to``,
``rotate_wrist``, ``rotate_pitch``, ``recover_pose`` and ``return_home``, whose templates also render a dry run
(``interface/dry_run.py``). The templates are held to the tools in three ways: every field they read is a
field of the tool's input or result schema, or of a dry run's record (``problems``), every stage event they read is one
the tool's source records with ``stage()`` (an AST scan of its call sites), and every recorded answer of the tool
(``perception-answers.json``, ``motion-answers.json``: what scripts read on sealed runs of 2026-10-27/31) renders with
every field present. Renderings are held to a golden text (``perception-explanations.json``,
``motion-explanations.json``); after a deliberate change of a template or the renderer, rebuild it with

    python - <<'E'  # from the repository root
    import json; from ora.agent.toolkit import explain
    answers = json.load(open("tests/data/code_mode/perception-answers.json"))["answers"]
    golden = {f"{i:02d} {a['backend']} {a['tool']}": explain.render(a["tool"], arguments=a["arguments"], result=a["result"]).text for i, a in enumerate(answers)}
    open("tests/data/code_mode/perception-explanations.json", "w").write("{\\n" + ",\\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in golden.items()) + "\\n}\\n")
    E

and the same over ``motion-answers.json`` for ``motion-explanations.json`` (``strict`` off for a refused answer).
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from ora.agent.composition.surface.schema_validation import validate_schema_instance
from ora.agent.toolkit import explain
from ora.agent.toolkit.interface import dry_run

ROOT = Path(__file__).resolve().parents[4]
TOOLS_ROOT = ROOT / "ora/agent/toolkit/tools"
DATA = ROOT / "tests/data/code_mode"
ANSWERS = json.loads((DATA / "perception-answers.json").read_text(encoding="utf-8"))["answers"]
GOLDEN = json.loads((DATA / "perception-explanations.json").read_text(encoding="utf-8"))
PERCEPTION = ("geometry_query", "grasp_plan", "vision_ground", "vision_segment")
MOTION = ("move_pose", "move_to", "recover_pose", "return_home", "rotate_pitch", "rotate_wrist")
MOTION_ANSWERS = json.loads((DATA / "motion-answers.json").read_text(encoding="utf-8"))["answers"]
MOTION_GOLDEN = json.loads((DATA / "motion-explanations.json").read_text(encoding="utf-8"))
#: Where a tool's stage() calls are: move_to and the rotations compile through move_pose, return_home through recover_pose.
SOURCES = {"move_to": "move_pose", "rotate_wrist": "move_pose", "rotate_pitch": "move_pose", "return_home": "recover_pose"}


def _description(tool: str) -> dict[str, Any]:
    return json.loads((TOOLS_ROOT / tool / "description.json").read_text(encoding="utf-8"))


def _answer(tool: str, **where: Any) -> dict[str, Any]:
    """The first recorded answer of ``tool`` whose result has the given top-level fields (``key=None``: lacks it)."""

    for item in ANSWERS:
        if item["tool"] == tool and all((key in item["result"]) == (value is not None) for key, value in where.items()):
            return item
    raise LookupError((tool, where))


def _render(item: dict[str, Any], **kwargs: Any) -> explain.Explanation:
    explained = explain.render(item["tool"], arguments=item["arguments"], result=item["result"], **kwargs)
    assert explained is not None
    return explained


def test_the_perception_and_the_motion_tools_have_a_template_and_no_other_tool_does() -> None:
    assert explain.tools() == tuple(sorted(PERCEPTION + MOTION))
    assert explain.template("set_gripper") is None and explain.template("../x") is None
    assert explain.render("set_gripper", arguments={}, result={}) is None


@pytest.mark.parametrize("tool", PERCEPTION + MOTION)
def test_a_template_reads_only_fields_that_exist(tool: str) -> None:
    description = _description(tool)
    spec = explain.template(tool)
    assert explain.problems(spec, description["inputSchema"], description["resultSchema"]) == []
    assert explain.reads(spec), "a template reads something"


@pytest.mark.parametrize("tool", PERCEPTION + MOTION)
def test_the_defaults_a_template_assumes_are_the_input_schemas(tool: str) -> None:
    properties = _description(tool)["inputSchema"]["properties"]
    spec = json.dumps(explain.template(tool))
    for name, value in re.findall(r"args\.(\w+)\|or:(\d+)", spec):  # a number stood in for an argument's default
        assert str(properties[name]["default"]) == value, name


def test_problems_names_a_field_that_does_not_exist() -> None:
    description = _description("geometry_query")
    spec = copy.deepcopy(explain.template("geometry_query"))
    spec["stages"][0]["line"] = ["{args.camera}", "{args.pixels|count}", "{result.points[].depth_mm}",
                                 "{events.depth_window[0].radius}", "{events.window[0].window_px}",
                                 "{result.points[?visible=true]|count}", "{error.reason}"]
    issues = explain.problems(spec, description["inputSchema"], description["resultSchema"])
    assert sorted(issues) == sorted([
        "args.pixels is not a field of the call's args",
        "result.points[].depth_mm is not a field of the call's result",
        "events.depth_window[0].radius is not a field of the call's events",
        "events.window[0].window_px is not a field of the call's events",
        "result.points[?visible=true] filters on visible, which the elements do not have",
        "error.reason is not a field of the call's error"])


def _recorded_events(tool: str) -> dict[str, set[str]]:
    """The stage events a tool's source records: ``stage("name", field=...)`` call sites, by name."""

    found: dict[str, set[str]] = {}
    for path in sorted((TOOLS_ROOT / SOURCES.get(tool, tool)).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "stage":
                assert isinstance(node.args[0], ast.Constant), path
                found.setdefault(node.args[0].value, set()).update(keyword.arg for keyword in node.keywords)
    return found


@pytest.mark.parametrize("tool", PERCEPTION + MOTION)
def test_the_stage_events_a_template_declares_are_the_ones_the_tool_records(tool: str) -> None:
    declared = {name: set(fields) for name, fields in (explain.template(tool).get("events") or {}).items()}
    assert declared == _recorded_events(tool)


def test_every_recorded_answer_is_covered() -> None:
    assert {item["tool"] for item in ANSWERS} == set(PERCEPTION)
    assert len(ANSWERS) >= 30 and len(ANSWERS) == len(GOLDEN)
    kinds = {(item["tool"], "error" in item["result"]) for item in ANSWERS}
    assert kinds == {(tool, refused) for tool in PERCEPTION for refused in (False, True)}


@pytest.mark.parametrize("index", range(len(ANSWERS)))
def test_a_recorded_answer_renders_with_every_field_it_reads_and_stays_as_it_was(index: int) -> None:
    item = ANSWERS[index]
    refused = "error" in item["result"]
    # a refused call may have lacked the arguments the header names; every other field is held to the template
    explained = _render(item, strict=not refused)
    assert explained.text == GOLDEN[f"{index:02d} {item['backend']} {item['tool']}"]
    assert explained.text.startswith(f"{item['tool']}(") and explained.text.endswith("\n")
    assert explained.failed or not refused
    again = explain.render(item["tool"], arguments=copy.deepcopy(item["arguments"]),
                           result=json.loads(json.dumps(item["result"], sort_keys=True)))
    assert again.text == explained.text, "the same inputs give the same text, whatever the key order"


def test_a_refused_call_says_who_refused_it_and_what_to_do() -> None:
    limit = explain.render("geometry_query", arguments={"camera": "head", "points": [[1, 2]]}, result={
        "error": {"code": "script_call_limit", "message": "this script already made 24 other call(s), its limit",
                  "constraints": {"max_calls": 24}}})
    assert limit.failed and limit.text.splitlines() == [
        "geometry_query(head, 1 pixel)  refused: script_call_limit",
        "1. request: lift 1 pixel of the head depth image to world points",
        "2. refusal: the Supervisor refused the call: this script already made 24 other call(s), its limit  -> not run",
        "next: end this script and make the remaining calls in the next one, which starts with a fresh allowance; "
        "batch what can be batched"]
    stale = _render(_answer("grasp_plan", error=True, camera=None))
    assert "the tool refused the call" in stale.text and stale.text.splitlines()[-1].startswith("next: ")
    assert explain.decision_of({"code": "invalid_arguments"}) == {"by": "tool", "code": "invalid_arguments"}
    assert explain.decision_of({"code": "turn_ended"}) == {"by": "supervisor", "code": "turn_ended"}
    assert explain.decision_of(None) is None


def test_a_vision_ground_without_a_region_is_a_failed_call_with_the_way_out() -> None:
    base = _answer("vision_ground", error=None, candidates=True)
    result = {**base["result"], "status": "no_candidate", "candidates": [],
              "diagnostics": {**base["result"]["diagnostics"], "returned_candidate_count": 0}}
    explained = explain.render("vision_ground", arguments={"camera": "head", "text": "blue cup"}, result=result,
                               events=[{"stage": "query_cache", "fields": {"hit": False}}], strict=True)
    assert explained.failed and explained.text.splitlines() == [
        'vision_ground(head, "blue cup")  no candidate',
        '1. request: find regions matching "blue cup" in the head camera, at most 3',
        "2. service: asked the grounding service; 0 regions returned  -> no region",
        'conclusion: no region matched "blue cup" in the head view',
        "next: rephrase it as a short visible description, ground it in another camera, or use vision_segment with a "
        "point you can place inside the object"]
    reused = explain.render("vision_ground", arguments={"camera": "head", "text": "blue cup"}, result=base["result"],
                            events=[{"stage": "query_cache", "fields": {"hit": True}}], strict=True)
    assert "2. service: reused the kept result of this query on this observation (no new request)" in reused.text


def test_a_geometry_query_names_the_window_it_used_and_the_pixels_it_could_not_lift() -> None:
    item = next(item for item in ANSWERS if item["tool"] == "geometry_query"
                and any(not point["valid"] for point in item["result"].get("points", ())))
    result = item["result"]
    count, holes = len(result["points"]), [point["pixel_uv"] for point in result["points"] if not point["valid"]]
    explained = explain.render("geometry_query", arguments=item["arguments"], result=result,
                               events=[{"stage": "depth_window", "fields": {"window_px": 2}}], strict=True)
    lines = explained.text.splitlines()
    assert lines[0] == (f"geometry_query({item['arguments']['camera']}, {count} pixels)  "
                        f"{count - len(holes)} of {count} pixels lifted to world points")
    assert lines[2] == ("2. depth window: each pixel takes the median of its valid depths within 2 px; "
                        "a pixel with none is invalid")
    assert lines[3].endswith(f"  -> {len(holes)} without depth")
    assert lines[4:4 + len(holes)] == [f"   pixel ({u}, {v}): no valid depth in its window "
                                       "(a hole, a mask or out of the sensor's range)" for u, v in holes]
    nothing = {**result, "points": [{"pixel_uv": [1, 2], "valid": False}]}
    failed = explain.render("geometry_query", arguments={"camera": "head", "points": [[1, 2]]}, result=nothing)
    assert failed.failed and failed.text.splitlines()[0].endswith("no pixel has valid depth")
    assert failed.text.splitlines()[-1].startswith("next: move the pixel farther inside the intended surface")


def test_a_grasp_plan_shows_the_candidates_the_cap_cut_and_why_the_ik_check_was_unavailable() -> None:
    item = _answer("grasp_plan", error=None, candidates=True)
    cut = [{"stage": "display_cap", "fields": {"compiled": 5, "shown": 3}}] + [
        {"stage": "cut_candidate", "fields": {"candidate_id": f"grasp-{n}", "backend_rank": 3 + n, "reachable": flag}}
        for n, flag in enumerate((True, None))]
    explained = _render(item, events=cut)
    lines = explained.text.splitlines()
    (stage,) = [index for index, line in enumerate(lines) if ". display cap:" in line]
    assert lines[stage].endswith("3 of 5 compiled candidates shown  -> 2 candidates cut")
    assert lines[stage + 1:stage + 3] == [
        "   grasp-0 (backend rank 3, reachable yes) cut by max_candidates",
        "   grasp-1 (backend rank 4, reachable unknown) cut by max_candidates"]
    unavailable = copy.deepcopy(item["result"])
    unavailable["ik_reachability"] = {"status": "unavailable", "ik_model": None, "matches_move_pose_admission": None,
                                      "order": "backend_order", "checked_candidate_count": 0,
                                      "reachable_candidate_count": 0}
    for candidate in unavailable["candidates"]:
        candidate["ik_reachability"] = {"reachable": None, "member": None, "pregrasp": None, "approach": None}
    text = explain.render("grasp_plan", arguments=item["arguments"], result=unavailable,
                          events=[{"stage": "ik_unavailable", "fields": {"failure": "tool_unavailable"}}]).text
    assert ("the IK check was not available, so the backend's order was kept and no candidate carries a verdict; "
            "failure: tool_unavailable") in text and "reachability unknown" in text


def test_a_vision_segment_lists_the_masks_the_limits_left_out() -> None:
    item = _answer("vision_segment", error=None, candidates=True)
    events = [{"stage": "mask_omitted", "fields": {"query_index": 1, "reason": "call_cap", "backend_score": 0.31}}]
    text = _render(item, events=events, strict=True).text
    assert "1 mask did not make the delivery" in text
    assert "   point 1: a mask scored 0.31 was left out (call_cap)" in text


# --- the arm-moving tools --------------------------------------------------------------------------------------------

COMPILED = {"arm": "left", "interaction": "free_space", "translation_distance_m": 0.0394, "rotation_angle_deg": 0.0,
            "jaw_symmetry": "requested", "target_reachable": False, "reachable_fraction": 0.625,
            "reachable_translation_m": 0.0246, "nearest_turn_deg": 15, "nearest_tilt_deg": 15.0,
            "clearance_status": "measured", "clearance_m": 0.031, "clearance_observed_fraction": 0.7}


def test_every_recorded_motion_answer_is_covered() -> None:
    assert {item["tool"] for item in MOTION_ANSWERS} == {"move_pose", "move_to", "rotate_wrist", "return_home", "recover_pose"}
    assert len(MOTION_ANSWERS) >= 10 and len(MOTION_ANSWERS) == len(MOTION_GOLDEN)
    assert {item["tool"] for item in MOTION_ANSWERS if "error" in item["result"]} == {"move_pose"}
    assert any(item["result"].get("receipt", {}).get("status") == "not_reached" for item in MOTION_ANSWERS)


@pytest.mark.parametrize("index", range(len(MOTION_ANSWERS)))
def test_a_recorded_motion_answer_renders_with_every_field_it_reads_and_stays_as_it_was(index: int) -> None:
    item = MOTION_ANSWERS[index]
    refused = "error" in item["result"]
    explained = explain.render(item["tool"], arguments=item["arguments"], result=item["result"], strict=not refused)
    assert explained.text == MOTION_GOLDEN[f"{index:02d} {item['backend']} {item['tool']}"]
    assert explained.text.startswith(f"{item['tool']}(") and explained.text.endswith("\n")
    assert explained.failed == (refused or item["result"]["receipt"]["status"] != "reached")
    assert explained.text == explain.render(item["tool"], arguments=item["arguments"],
                                            result=json.loads(json.dumps(item["result"], sort_keys=True))).text


def test_a_refused_move_names_who_refused_it_and_the_way_out() -> None:
    bounds = next(item for item in MOTION_ANSWERS if item["result"].get("error", {}).get("code") == "workspace_bounds")
    text = explain.render("move_pose", arguments=bounds["arguments"], result=bounds["result"]).text
    assert "2. refusal" not in text and "1. refusal: the tool refused the call: target pose is outside" in text
    assert text.splitlines()[-1].startswith("next: choose a target inside the action workspace")
    limit = next(item for item in MOTION_ANSWERS if item["result"].get("error", {}).get("code") == "script_action_refused")
    shown = explain.render("move_pose", arguments=limit["arguments"], result=limit["result"])
    assert shown.failed and "the Supervisor refused the call" in shown.text and "turn_must_end" in shown.text


def _dry(tool: str, **fields: Any) -> dict[str, Any]:
    """A dry run's record, held to the shape ``dry_run.RESULT_SCHEMA`` declares."""

    record = {"dry_run": True, "action": tool, "status": "planned", "feasible": True, "failure_code": None,
              "failure_stage": None, "claim_boundary": "x", **fields}
    validate_schema_instance(record, dry_run.RESULT_SCHEMA)
    return record


def test_a_dry_run_that_works_reads_as_a_plan_with_what_compiling_found() -> None:
    record = _dry("move_to", route="joint", arms={"left": {
        "target_pose": [0.4, 0, 0.2, 0, 0, 0, 1], "predicted_pose": [0.4, 0, 0.2, 0, 0, 0, 1],
        "end_joints_deg": [10, 0, 0, 0, 0, 0], "joint6": {"min_limit_margin_deg": 171.0}}},
        screening={"status": "clear", "segments_screened": 12, "min_inter_arm_clearance_m": 0.0825, "against": "the arm"},
        motion={"native_actions": 12, "whole_native_actions": 12, "settle_native_actions": 3, "duration_s": 0.48,
                "fraction_of_move": 1.0, "native_action_budget": 20})
    explained = explain.render("move_to", arguments={"targets": {"left": {"position_m": [0.4, 0, 0.2]}}}, result=record,
                               events=[{"stage": "compiled", "fields": COMPILED}], strict=True)
    assert not explained.failed and explained.text.splitlines() == [
        "move_to(left)  dry run: feasible",
        "1. request: compiled 1 target from the measured pose",
        "   left: free_space; translate 3.9 cm; orientation held",
        "2. reach check: read-only IK of the target pose of 1 arm  -> did not solve",
        "   left: the target pose does not solve; the straight line to it is reachable to 62% (2.5 cm); alternative "
        "found: turning the orientation 15.0 deg (tilt 15.0 deg from straight down) makes the endpoint solvable",
        "3. observed clearance: depth below the fingertips at the target; a measurement, not a collision check  -> did not block",
        "   left: observed clearance 3.1 cm; 70% of the footprint observed",
        "4. plan: joint route, 12 native actions (0.48 s)  -> solved",
        "5. screening: 12 segments of the planned path screened against the arm; least inter-arm clearance 8.2 cm  -> clear",
        "6. budget: 12 native actions for the whole move, 20 may be spent; 12 planned with 3 for the settle  -> fits",
        "7. prediction: left ends at (0.400, 0.000, 0.200, 0.000, 0.000, 0.000, 1.000); left wrist roll stays "
        "171.0 deg from its limit",
        "conclusion: the call would run as planned; the same call without dry_run executes it, and the world's response "
        "(contact, a held object) is not predicted"]


def test_a_move_cut_to_its_budget_is_feasible_but_says_it_is_cut() -> None:
    record = _dry("move_pose", route="line", arms={"left": {"target_pose": [1.5, 0, 0.2, 0, 0, 0, 1],
                                                              "predicted_pose": [0.9, 0, 0.2, 0, 0, 0, 1]}},
                  screening={"status": "clear", "segments_screened": 40, "against": "the arm"},
                  motion={"native_actions": 47, "whole_native_actions": 90, "settle_native_actions": 3, "duration_s": 1.9,
                          "fraction_of_move": 0.52, "native_action_budget": 50})
    text = explain.render("move_pose", arguments={"targets": {}}, result=record, strict=False).text
    assert text.splitlines()[0] == "move_pose()  dry run: feasible but cut short, 52% of the move"
    assert ("budget: 90 native actions for the whole move, 50 may be spent; 47 planned with 3 for the settle  "
            "-> cut short: covers 52% of the move") in text


@pytest.mark.parametrize("code,stage,needle", [
    ("paced_path_unreachable", "ik", "the target is out of reach with this orientation or route"),
    ("paced_path_wrist_unwind", "ik", "reach the goal by another route"),
    ("inter_arm_collision", "screening", "choose targets that keep the arm clear"),
    ("paced_motion_budget", "budget", "split the move into shorter ones"),
    ("route_contact_approach_exceeds_local_bound", "route", "use a free_space or precision interaction"),
    ("workspace_bounds", "request", "choose a target inside the action workspace"),
])
def test_an_infeasible_dry_run_names_the_stage_that_refused_it_and_what_to_change(code: str, stage: str, needle: str) -> None:
    record = _dry("move_pose", status="infeasible", feasible=False, failure_code=code, failure_stage=stage,
                  refusal={"code": code, "message": f"{code} message", "constraints": {"fraction_of_move": 0.4,
                                                                                         "native_action": 3, "arm": "left"}})
    explained = explain.render("move_pose", arguments={"targets": {}}, result=record)
    assert explained.failed and explained.text.splitlines()[0] == f"move_pose()  dry run: would be refused, {code}"
    assert f"{code} at the {stage} stage; nothing was executed" in explained.text
    assert needle in explained.text.splitlines()[-1] and explained.text.splitlines()[-1].startswith("next: ")


def test_a_recovery_dry_run_says_what_it_would_undo_or_how_it_would_unwind() -> None:
    undo = {"undone_call": {"tool_call_id": "call-3", "tool": "move_pose", "ordinal": 2}, "complete": True,
            "undoable_calls": []}
    record = _dry("recover_pose", route="measured_retrace", arms={"left": {
        "target_pose": [0.3, 0, 0.2, 0, 0, 0, 1], "predicted_pose": [0.3, 0, 0.2, 0, 0, 0, 1],
        "end_joints_deg": [0, 0, 0, 0, 0, 0], "method": "measured_retrace", "undo": undo}},
        screening={"status": "clear", "candidates_screened": 1, "against": "the arm"},
        motion={"native_actions": 8, "whole_native_actions": 8, "settle_native_actions": 2, "duration_s": 0.32,
                "fraction_of_move": 1.0, "native_action_budget": 50})
    text = explain.render("recover_pose", arguments={"arms": ["left"]}, result=record, strict=True).text
    assert text.splitlines()[:4] == [
        "recover_pose(left)  dry run: feasible",
        "1. request: 1 arm to undo the latest arm-moving call of: left",
        "2. plan: measured_retrace, 8 native actions (0.32 s); left: measured_retrace; left would undo its move_pose "
        "(call call-3)  -> planned",
        "3. screening: 1 candidate screened, the last against the arm  -> clear"]
    home = _dry("return_home", route="retract_then_unwind", arms={"left": {
        "target_pose": [0, 0, 0.3, 0, 0, 0, 1], "predicted_pose": [0, 0, 0.3, 0, 0, 0, 1], "method": "joint_interpolation",
        "unwind_order": "fold_then_yaw", "lift": "planned"}},
        screening={"status": "clear", "candidates_screened": 2, "against": "the arm"},
        motion={"native_actions": 14, "whole_native_actions": 14, "settle_native_actions": 2, "duration_s": 0.56,
                "fraction_of_move": 1.0, "native_action_budget": 50})
    shown = explain.render("return_home", arguments={"arms": ["left"]}, result=home, strict=True).text
    assert "left unwinds fold_then_yaw; left lift planned" in shown and "2 candidates screened" in shown
    nothing = _dry("recover_pose", status="infeasible", feasible=False, failure_code="retrace_nothing_to_undo",
                   failure_stage="plan", refusal={"code": "plan_rejected", "message": "nothing is left to undo",
                                                  "constraints": {"failure_code": "retrace_nothing_to_undo"}})
    said = explain.render("recover_pose", arguments={"arms": ["left"]}, result=nothing)
    assert said.failed and said.text.splitlines()[-1].startswith("next: return_home goes straight back")


def test_a_dry_run_the_surface_cannot_answer_says_to_make_the_call_itself() -> None:
    error = {"code": "dry_run_unavailable", "message": "a dry run of this call is unavailable: the environment plans it",
             "retryable": False, "constraints": {"reason": "the environment plans this surface's moves when they execute"}}
    for tool, arguments in (("move_to", {"targets": {}}), ("recover_pose", {"arms": ["left"]})):
        explained = explain.render(tool, arguments=arguments, result={"error": error}, strict=False)
        assert explained.failed and explained.text.splitlines()[0].endswith("refused: dry_run_unavailable")
        assert explained.text.splitlines()[-1] == (
            "next: make the call itself: the environment plans this surface's moves when they execute; "
            "a call the environment refuses before it moves costs nothing")


def test_an_executed_move_that_stopped_short_is_a_failed_call_with_its_receipt_summary() -> None:
    result = {"status": "executed", "action": "move_pose", "receipt": {
        "status": "not_reached", "failure_code": "controller_tracking_error", "failure_stage": "execution",
        "action_summary": "left arm 34 mm / 9 deg from target, not reached"}}
    explained = explain.render("move_pose", arguments={"targets": {}}, result=result, strict=True)
    assert explained.failed and explained.text.splitlines()[0] == "move_pose()  executed, not reached"
    assert "execution: left arm 34 mm / 9 deg from target, not reached  -> not reached: controller_tracking_error at execution" in explained.text
    assert explained.text.splitlines()[-1].startswith("next: observe, then decide: recover_pose undoes")


# --- the template language ------------------------------------------------------------------------------------------

SPEC: dict[str, Any] = {
    "schema": explain.TEMPLATE_SCHEMA, "tool": "probe", "events": {"pick": ["name"]},
    "call": ["{args.who}", {"if": {"has": "args.n"}, "text": "{args.n|n:item}"}],
    "outcomes": [{"if": {"has": "error"}, "text": "refused", "failed": True},
                 {"if": {"lt": ["result.score", 0.5]}, "text": "weak", "failed": True}, {"text": "fine"}],
    "stages": [
        {"name": "measure", "doc": "d", "line": ["{result.score|f2} ({result.score|pct})",
                                                   {"if": {"has": "result.size"}, "text": "{result.size|cm} / {result.size|mm}"}],
         "verdict": [{"if": {"gt": ["result.score", 0.9]}, "text": "high"}, {"text": "ok"}],
         "each": {"over": "result.rows[?ok=true]", "limit": 2, "line": "{index}: {item.id} at {item.uv|uv}"}},
        {"name": "skipped", "doc": "d", "when": {"has": "result.nothing"}, "line": "never"},
        {"name": "events", "doc": "d", "when": {"not": {"empty": "events.pick"}},
         "line": "{events.pick|count|n:pick}: {events.pick[0].name}"},
    ],
    "conclusion": [{"if": {"any": [{"eq": ["result.mode", "a"]}, {"in": ["result.mode", ["b", "c"]]}]}, "text": "mode {result.mode}"},
                   {"text": "no mode"}],
    "next": [{"if": {"ne": ["result.mode", "a"]}, "text": "try again"}],
}


def _probe(**kwargs: Any) -> explain.Explanation:
    return explain.render_spec(SPEC, tool="probe", **kwargs)


def test_a_template_fills_formats_filters_and_stage_events() -> None:
    result = {"score": 0.9714, "size": 0.0394, "mode": "b",
              "rows": [{"id": "r1", "ok": True, "uv": [3, 4.4]}, {"id": "r2", "ok": False, "uv": [1, 1]},
                       {"id": "r3", "ok": True, "uv": [5, 6]}, {"id": "r4", "ok": True, "uv": [7, 8]}]}
    explained = _probe(arguments={"who": "ann", "n": 1}, result=result,
                       events=[{"stage": "pick", "fields": {"name": "x"}}, {"stage": "pick", "fields": {"name": "y"}}],
                       strict=True)
    assert explained.text.splitlines() == [
        "probe(ann, 1 item)  fine",
        "1. measure: 0.97 (97%); 3.9 cm / 39 mm  -> high",
        "   0: r1 at (3, 4)",
        "   1: r3 at (5, 6)",
        "   ... and 1 more",
        "2. events: 2 picks: x",
        "conclusion: mode b"] and not explained.failed


def test_a_failed_outcome_adds_what_to_do_next_and_a_missing_field_is_left_out_or_raises() -> None:
    weak = _probe(arguments={"who": "ann"}, result={"score": 0.2, "mode": "z", "rows": []})
    assert weak.failed and weak.text.splitlines() == [
        "probe(ann)  weak", "1. measure: 0.20 (20%)  -> ok", "conclusion: no mode", "next: try again"]
    lenient = _probe(arguments={}, result={"score": 0.7, "mode": "a"})  # no who, no rows: clauses are left out
    assert lenient.text.splitlines()[:2] == ["probe()  fine", "1. measure: 0.70 (70%)  -> ok"]
    with pytest.raises(explain.ExplainMiss):
        _probe(arguments={}, result={"score": 0.7, "mode": "a"}, strict=True)


def test_paths_conditions_and_ops_are_checked() -> None:
    scope = {"a": {"b": [{"c": 1}, {"c": 2}], "d": None}}
    assert explain._resolve("a.b[1].c", scope) == 2 and explain._resolve("a.b[].c", scope) == [1, 2]
    assert explain._resolve("a.b[?c!=1].c", scope) == [2] and explain._resolve("a.d.x", scope) is explain._MISSING
    assert explain._resolve("a.b[5]", scope) is explain._MISSING
    with pytest.raises(ValueError):
        explain._resolve("a..b", scope)
    with pytest.raises(ValueError):
        explain._holds({"has": "a", "empty": "a"}, scope)
    with pytest.raises(ValueError):
        explain._holds({"between": ["a", 1]}, scope)
    with pytest.raises(ValueError):
        explain._placeholder("a.b|loud", scope)
    assert explain._placeholder("a.d|or:none given", scope) == "none given"
    assert explain._placeholder("a.d|or:3|n:mask", scope) == "3 masks"
    assert explain._placeholder("a.b[].c|max|f1", scope) == "2.0"
    assert explain._placeholder("a.b[].c|join", scope) == "1, 2"
    assert explain._fill("{{literal}} {a.b[0].c}", scope) == "{literal} 1"


def test_the_documentation_names_every_stage_and_event() -> None:
    for tool in PERCEPTION + MOTION:
        spec = explain.template(tool)
        text = explain.documentation(spec)
        for number, stage in enumerate(spec["stages"], start=1):
            assert f"{number}. {stage['name']}: {stage['doc']}" in text
        for name, fields in (spec.get("events") or {}).items():
            assert f"- {name}({', '.join(fields)})" in text
        assert text.startswith("Explain (robot.explain(r); a failed call carries it as r.explain)")
