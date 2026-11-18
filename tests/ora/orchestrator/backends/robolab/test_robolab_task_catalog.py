"""RoboLab-120 admission: the generated roster, contracts, pending reviews and the generator's templates.

All 120 tasks are admitted (decision 2026-11-05); three contracts were reviewed with a seed-0 witness
on 2026-10-18 and the other 117 were generated per predicate family and promoted once theirs matched
(97 on 2026-11-05, the last 20 on 2026-11-16). A pending
contract is hash-bound and rendered from its review like a reviewed one, but its review says
``needs_review`` and the resolver withholds it. The templates are checked on synthetic task facts,
so these tests need no RoboLab checkout; the regeneration check runs where ``ROBOLAB_ROOT`` is set.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from ora.agent.composition.task_contract.delivery import PENDING_REASON, resolve_task_contract_layer
from ora.agent.composition.task_contract.review import audit_registry, render_public_contract, review_errors
from ora.environment.benchmarks.robolab import REVISION, TASKS
from ora.orchestrator.backends.robolab import task_catalog, task_review, task_wording
from ora.orchestrator.backends.robolab.task_source import Facts, Term

ROOT = Path(__file__).resolve().parents[5]
REGISTRY = json.loads((ROOT / task_catalog.REGISTRY).read_text(encoding="utf-8"))
REVIEWED = {"BananaInBowlTask", "RubiksCubeAndBananaTask", "StackYellowOnRedTask"}
#: Promoted after their seed-0 witness matched every case (``ora dev robolab-tasks promote``; 2026-11-05 and -16).
WITNESSED = sorted(set(REGISTRY["tasks"]) - REVIEWED)
PENDING = sorted(REGISTRY["pending_review"])
FORBIDDEN_WORDS = re.compile(r"\b(within|at least|at most|no more than|tolerance|threshold|epsilon|predicate"
                             r"|check_success|eval_success|quaternion|qpos)\b", re.IGNORECASE)


def _review(task: str) -> dict:
    entry = REGISTRY["tasks"].get(task) or REGISTRY["pending_review"][task]
    return json.loads((ROOT / entry["semantic_review"]).read_text(encoding="utf-8"))


def test_the_roster_admits_all_one_hundred_twenty_tasks_at_fifteen_hertz():
    assert len(TASKS) == 120
    for task, entry in TASKS.items():
        assert entry["native_action_limit"] == entry["episode_length_s"] * 15, task
        assert entry["instruction"] and entry["official_success"] and entry["skill"], task


def test_the_three_tasks_admitted_first_keep_their_reviewed_roster_entries():
    assert TASKS["BananaInBowlTask"] == {
        "episode_length_s": 50, "native_action_limit": 750, "skill": "pick_and_place",
        "instruction": "Pick up the banana and place it in the bowl",
        "official_success": "the banana rests in the bowl, touching it, with the gripper detached"}
    assert TASKS["RubiksCubeAndBananaTask"]["official_success"] == (
        "both the Rubik's cube and the banana rest in the bowl, touching it, with the gripper detached")
    assert TASKS["StackYellowOnRedTask"]["official_success"] == "the yellow block is stacked on top of the red block"


def test_every_admitted_task_is_reviewed_or_pending_review_and_never_both():
    reviewed, pending = set(REGISTRY["tasks"]), set(REGISTRY["pending_review"])
    assert REVIEWED <= reviewed and len(reviewed) == 120 and len(pending) == 0
    assert not reviewed & pending and reviewed | pending == set(TASKS)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_a_contract_renders_its_review_and_states_no_private_detail(task):
    entry = REGISTRY["tasks"].get(task) or REGISTRY["pending_review"][task]
    text = (ROOT / entry["path"]).read_text(encoding="utf-8")
    review = _review(task)
    assert render_public_contract(review) == text and len(review["public_clauses"]) == entry["clause_count"]
    assert not re.search(r"\d", text), "a contract carries no number"
    assert "_" not in text, "a contract names no identifier or predicate"
    assert not FORBIDDEN_WORDS.search(text)
    assert entry["verified_revisions"] == review["verified_revisions"] == [REVISION]


@pytest.mark.parametrize("task", WITNESSED)
def test_a_witnessed_contract_passed_every_check_and_is_delivered(task):
    review = _review(task)
    assert review["status"] == "reviewed" and review_errors(review) == []
    assert all(check["status"] == "passed" for check in review["checks"].values())
    assert review["witness_result"]["all_cases_match"] is True and review["compatibility"]["status"] != "blocked"
    assert REGISTRY["tasks"][task]["verdict"] == task_catalog.REVIEWED_VERDICT
    layer, disclosure = resolve_task_contract_layer({"kind": "robolab", "task": task, "required_revision": REVISION},
                                                    root=ROOT)
    assert layer is not None and disclosure["delivered"] is True


@pytest.mark.parametrize("task", PENDING)
def test_a_pending_review_is_never_treated_as_reviewed(task):
    review = _review(task)
    assert review["status"] == "needs_review"
    assert "semantic_review_incomplete" in review_errors(review)
    assert REGISTRY["pending_review"][task]["verdict"] == task_catalog.PENDING_VERDICT
    # A pending review either waits for its witness or holds the witness case that did not match.
    witness = [review["checks"][check]["status"] for check in ("success_witness", "counterexamples")]
    assert set(witness) <= {"pending", "passed", "failed"} and ("failed" in witness or "pending" in witness)
    for check in ("predicate_and_helpers", "reset_and_randomization", "success_timing", "no_expert_strategy",
                  "privacy"):
        assert review["checks"][check]["status"] == "passed", check
    assert review["witness_plan"]["cases"][0] == {"case": "reset_state", "expected": False, "ops": [], "ticks": 3}
    assert review["compatibility"]["status"] in ("ready", "ready_with_note", "blocked")
    layer, disclosure = resolve_task_contract_layer({"kind": "robolab", "task": task, "required_revision": REVISION},
                                                    root=ROOT)
    assert layer is None and disclosure["delivered"] is False and disclosure["reason"] == PENDING_REASON
    assert disclosure["semantic_review"] == REGISTRY["pending_review"][task]["semantic_review"]


def test_a_task_moved_back_to_pending_review_is_withheld(tmp_path):
    """No task is pending now (2026-11-16); a registry that lists one under pending_review still withholds it."""
    staged = json.loads(json.dumps(REGISTRY))
    staged["pending_review"]["GrabAFruitTask"] = staged["tasks"].pop("GrabAFruitTask")
    path = tmp_path / task_catalog.REGISTRY
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(staged), encoding="utf-8")
    layer, disclosure = resolve_task_contract_layer(
        {"kind": "robolab", "task": "GrabAFruitTask", "required_revision": REVISION}, root=tmp_path)
    assert layer is None and disclosure["delivered"] is False and disclosure["reason"] == PENDING_REASON


def test_the_descriptor_rule_selects_exactly_the_tasks_whose_instruction_leaves_objects_to_infer():
    flagged = {task for task, entry in TASKS.items() if task_wording.refers_by_descriptor(entry["instruction"])}
    assert flagged == set(task_wording.INSTRUCTION_DESCRIPTORS) and len(flagged) == 39
    assert not flagged & REVIEWED
    for instruction in ("Pick up the banana and place it in the bowl", "Stack the yellow block on the red block",
                        "Put the red mug upright so that the opening is facing upwards.", "Put the lizards in the bin"):
        assert not task_wording.refers_by_descriptor(instruction), instruction
    for instruction in ("Put all the red things in the grey bin", "Place the larger object in the grey bin.",
                        "Put the green spoons in the pot", "Put all of the serving spoons with no holes in the pot"):
        assert task_wording.refers_by_descriptor(instruction), instruction


def _target_phrases(review: dict) -> set[str]:
    basis = next(item for item in review["conditions"] if item["id"] == "object_identity")["source_basis"]
    names = re.findall(r"'(\w+)'", basis.split("termination names", 1)[1])
    phrases = set()
    for name in names:
        base = re.sub(r"_?\d+$", "", name)
        text = task_wording.OBJECT_PHRASES.get(name) or task_wording.OBJECT_PHRASES.get(base)
        if text:
            phrases.add(text.lower())
    return phrases


@pytest.mark.parametrize("task", sorted(task_wording.INSTRUCTION_DESCRIPTORS))
def test_a_descriptor_contract_never_names_the_objects_it_leaves_to_infer(task):
    """Naming them would answer the task: the contract keeps the instruction's own words."""

    review = _review(task)
    text = render_public_contract(review).lower()
    instruction = TASKS[task]["instruction"].lower()
    named = sorted(phrase for phrase in _target_phrases(review)
                   if re.search(rf"\b{re.escape(phrase)}s?\b", text) and phrase not in instruction)
    assert named == [], f"{task} names {named}"
    assert "either one" not in text and not re.search(r"\b(both|all (two|three|four|five))\b", text)
    identity = next(item for item in review["conditions"] if item["id"] == "object_identity")
    assert identity["disposition"] == "withheld_private" and identity["public_clause_indices"] == []


def test_the_audit_lists_pending_reviews_apart_from_reviewed_ones():
    report = audit_registry(ROOT, "robolab")
    assert report["counts"] == {"review_record_valid": 120}


def test_the_roster_listing_reports_contract_and_compatibility_state():
    rows = {row["task"]: row for row in task_catalog.roster_rows(ROOT)}
    assert len(rows) == 120 and sum(row["contract"] == "reviewed" for row in rows.values()) == 120
    assert rows["BananasInCrateTask"]["compatibility"] == "ready_with_note"
    assert "remote_scene_geometry" in rows["BananasInCrateTask"]["notes"]


@pytest.mark.skipif(not os.environ.get("ROBOLAB_ROOT"), reason="needs the pinned RoboLab checkout")
def test_the_committed_roster_contracts_and_reviews_are_what_the_generator_writes():
    output = task_catalog.generate(Path(os.environ["ROBOLAB_ROOT"]), ROOT)
    drift = [path for path, text in output.files.items() if (ROOT / path).read_text(encoding="utf-8") != text]
    assert drift == []
    for task, report in task_catalog.regression_report(output, ROOT).items():
        assert report["contract_identical"] and report["roster_identical"], task


# ---------------------------------------------------------------------------------------------- templates
def _facts(task: str, func: str, params: dict, *, prims: dict | None = None, scene: str = "scene.usda",
           episode_s: int = 60, subtasks: list | None = None, terms: list[Term] | None = None) -> Facts:
    return Facts(task=task, file=f"robolab/tasks/benchmark/{task}.py", instruction="do it", episode_s=episode_s,
                 scene=scene, contact_objects=["table"], attributes=[],
                 terms=terms or [Term("success", func, params)], subtasks_text="", subtask_calls=subtasks or [],
                 metadata={}, prims=prims or {})


def test_one_object_in_a_container_with_contact_reproduces_the_reviewed_wording():
    facts = _facts("BananaInBowlTask", "object_in_container",
                   {"object": "banana", "container": "bowl", "require_contact_with": True,
                    "require_gripper_detached": True})
    projection = task_wording.project(facts)
    assert projection.clauses == [
        "The banana must end up inside the bowl and touching it.",
        "Let go of the banana there: the task is not complete while any part of the gripper still touches it."]
    assert projection.official_success == "the banana rests in the bowl, touching it, with the gripper detached"
    assert [item["disposition"] for item in projection.conditions] == ["included", "included", "withheld_private"]


def test_exactly_k_and_one_of_say_how_many_count():
    choose = _facts("Crate", "object_in_container", {"object": ["banana", "banana_01", "banana_02"],
                                                     "container": "purple_crate", "logical": "choose", "K": 2,
                                                     "require_gripper_detached": True})
    clauses = task_wording.project(choose).clauses
    assert clauses[0].startswith("Exactly two of these must be in the purple crate at the end")
    assert "three or more there does not complete the task" in clauses[0]
    either = _facts("Either", "object_in_container", {"object": ["smartphone", "remote_control"],
                                                      "container": "grey_bin", "logical": "any",
                                                      "require_gripper_detached": True})
    assert task_wording.project(either).clauses[0] == (
        "Either the smartphone or the remote control must end up inside the grey bin; one of them is enough.")


def test_the_stack_term_states_no_release_and_keeps_that_gap_private():
    facts = _facts("Tower", "stacked", {"objects": ["red_block", "blue_block", "green_block"],
                                        "order": "bottom_to_top"})
    projection = task_wording.project(facts)
    assert len(projection.clauses) == 1 and "from bottom to top" in projection.clauses[0]
    gap = next(item for item in projection.conditions if item["id"] == "scoring_gap")
    assert gap["disposition"] == "withheld_private" and gap["public_clause_indices"] == []


def test_a_pick_up_completes_while_held_and_a_direction_is_the_robots():
    held = task_wording.project(_facts("Glasses", "object_picked_up",
                                       {"object": "glasses", "surface": "table", "distance": 0.05}))
    assert held.clauses[1].startswith("The task completes while they are held")
    left = task_wording.project(_facts("Left", "object_left_of",
                                       {"object": "rubiks_cube", "reference_object": "bowl",
                                        "require_gripper_detached": True},
                                       prims={"rubiks_cube": {"asset": "../objects/hot3d/rubiks_cube.usd"}}))
    assert left.clauses[0] == ("The Rubik's cube must end up to the left of the bowl as seen from the robot, rather "
                               "than diagonally beside it.")


def test_the_subtask_comparison_names_what_the_termination_adds():
    facts = _facts("Cube", "object_in_container",
                   {"object": "rubiks_cube", "container": "bowl", "require_contact_with": True,
                    "require_gripper_detached": True},
                   subtasks=[("pick_and_place", {"object": ["rubiks_cube"], "container": "bowl", "logical": "all"})])
    assert task_review.termination_vs_subtasks(facts)["notes"] == [
        "contact: the termination also requires contact with bowl"]
    grab = _facts("Grab", "object_picked_up", {}, terms=[
        Term("success_a", "object_picked_up", {"object": "bagel_01", "surface": "table", "distance": 0.05})],
        subtasks=[("object_grabbed", {"object": "bagel_01"})])
    assert task_review.termination_vs_subtasks(grab)["differs"] is True


def test_compatibility_blocks_a_missing_object_and_notes_long_horizons():
    scan = {"scenes": {"scene.usda": {"objects": {"banana": {}, "bowl": {}}, "remote_geometry": [],
                                      "local_missing_geometry": []}}}
    ok = _facts("Long", "object_in_container", {"object": "banana", "container": "bowl"}, episode_s=300)
    result = task_review.compatibility(ok, scan, {})
    assert result["status"] == "ready_with_note" and result["notes"][0]["code"] == "long_horizon"
    missing = _facts("Missing", "object_in_container", {"object": "apple_01", "container": "bowl"})
    assert task_review.compatibility(missing, scan, {})["status"] == "blocked"


def test_a_witness_plan_checks_one_too_many_before_exactly_k():
    facts = _facts("Crate", "object_in_container", {"object": ["banana", "banana_01", "banana_02"],
                                                    "container": "purple_crate", "logical": "choose", "K": 2,
                                                    "require_gripper_detached": True})
    cases = task_review.witness_plan(facts)["cases"]
    assert [(case["case"], case["expected"]) for case in cases] == [
        ("reset_state", False), ("beside", False), ("one_too_many", False), ("exactly_k", True)]


def _promotable(tmp_path: Path, *, pending_check: bool = False, blocked: bool = False) -> Path:
    contracts = tmp_path / task_catalog.CONTRACTS
    (contracts / "reviews").mkdir(parents=True)
    review = _review("BananaInBowlTask")
    review.update(task="X", status="needs_review")
    if pending_check:
        review["checks"]["success_witness"] = {"status": "pending", "basis": "not run"}
    if blocked:
        review["compatibility"] = {"status": "blocked", "blocked": [{"code": "solved_at_reset"}], "notes": []}
    text = render_public_contract(review)
    (contracts / "X-completion-v1.md").write_text(text, encoding="utf-8")
    (contracts / "reviews/X-review-v1.json").write_text(json.dumps(review), encoding="utf-8")
    entry = {"path": f"{task_catalog.CONTRACTS}/X-completion-v1.md", "semantic_review":
             f"{task_catalog.CONTRACTS}/reviews/X-review-v1.json", "verified_revisions": [REVISION],
             "verdict": task_catalog.PENDING_VERDICT, "family": "in_container", "compatibility": "ready"}
    (tmp_path / task_catalog.REGISTRY).write_text(json.dumps({"tasks": {}, "pending_review": {"X": entry}}),
                                                  encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("pending_check, blocked, promoted", [(False, False, ["X"]), (True, False, []),
                                                              (False, True, [])])
def test_promotion_takes_only_a_fully_checked_unblocked_review(tmp_path, pending_check, blocked, promoted):
    root = _promotable(tmp_path, pending_check=pending_check, blocked=blocked)
    assert task_catalog.promote(None, root=root, reviewer="r", today="2026-11-06") == promoted
    registry = json.loads((root / task_catalog.REGISTRY).read_text(encoding="utf-8"))
    review = json.loads((root / task_catalog.CONTRACTS / "reviews/X-review-v1.json").read_text(encoding="utf-8"))
    if promoted:
        assert set(registry["tasks"]) == {"X"} and registry["pending_review"] == {}
        assert registry["tasks"]["X"]["verdict"] == task_catalog.REVIEWED_VERDICT
        assert "family" not in registry["tasks"]["X"] and review["status"] == "reviewed"
    else:
        assert registry["tasks"] == {} and review["status"] == "needs_review"
