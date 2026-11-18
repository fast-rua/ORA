"""The private review of a generated RoboLab-120 contract: bindings, checks, compatibility and witness plan.

``build_review`` fills every check a source reading settles (predicate and helpers, reset and
randomization, assets and frames except the upright family's, success timing, no expert strategy,
privacy) and leaves ``success_witness``, ``counterexamples`` and ``public_observability`` pending for
the seed-0 witness, whose plan it writes. ``termination_vs_subtasks`` records where the official
termination differs from the partial-score subtasks; ``compatibility`` classifies the task as
ready, ready with note or blocked.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.environment.benchmarks.robolab import REVISION
from ora.environment.controllers.robolab_droid import MAXIMUM_APERTURE_M

from .task_source import HELPERS, Facts, _as_list, _asset_key, family, logical, references, targets
from .task_wording import FRAME_CONTAINERS, TASK_REVIEW, Projection, _call, phrase

REVIEW_SCHEMA = "ora.task-contract-review.v1"
WITNESS_PLAN_SCHEMA = "ora.robolab-witness-plan.v1"
GENERATED_REVIEWER = "Generated source review (ora dev robolab-tasks generate); seed-0 witness pending"
GENERATED_AT = "2026-11-05"
#: Today's longest admitted official limit before RoboLab-120 (60 s at 15 Hz).
PREVIOUS_LONGEST_LIMIT = 900
#: The default per-episode robot-decision budget of a composed cell (spec.benchmark.DEFAULT_BUDGETS).
DEFAULT_DECISIONS = 80

# ------------------------------------------------------------------------------- termination vs subtasks
_SUBTASK_EQUIVALENT = {"pick_and_place": "object_in_container", "pick_and_place_on_surface": "object_on_top",
                       "pick_and_place_grouped": "object_in_container", "object_inside": "object_in_container"}
_PROGRESS = {"object_grabbed", "object_dropped", "object_above_bottom"}


def termination_vs_subtasks(facts: Facts) -> dict[str, Any]:
    """Where the termination (ORA's official success) differs from what the partial-score subtasks suggest."""
    notes: list[str] = []
    final = [(name, params) for name, params in facts.subtask_calls if name not in _PROGRESS]
    funcs = {_SUBTASK_EQUIVALENT.get(name, name) for name, _ in final}
    term_funcs = {_SUBTASK_EQUIVALENT.get(term.func, term.func) for term in facts.terms}
    sub_objects: set[str] = set()
    for name, params in final:
        sub_objects |= set(_as_list(params.get("object") or params.get("objects")))
        for group in params.get("groups") or []:
            sub_objects |= set(_as_list(group["object"]))
    grabbed_only = not final and any(name == "object_grabbed" for name, _ in facts.subtask_calls)
    if grabbed_only:
        notes.append("the subtasks score a grasp only; the termination also requires the lift")
    elif funcs and not funcs & term_funcs and not (term_funcs == {"object_picked_up"} and "object_above" in funcs):
        notes.append(f"relation: subtasks score {sorted(funcs)}, the termination checks {sorted(term_funcs)}")
    term_objects = set(targets(facts))
    if final and sub_objects and sub_objects != term_objects and "stacked" not in term_funcs:
        extra, missing = sorted(term_objects - sub_objects), sorted(sub_objects - term_objects)
        notes.append("objects: the termination " + "; ".join(
            part for part in (f"adds {extra}" if extra else "", f"omits {missing}" if missing else "") if part))
    mode, count = logical(facts)
    for name, params in final:
        sub_mode = params.get("logical", "all")
        if name in _SUBTASK_EQUIVALENT and (sub_mode != mode or (mode == "choose" and int(params.get("K", 1)) != count)):
            notes.append(f"logical: subtasks {sub_mode}{'/K=' + str(params.get('K')) if params.get('K') else ''}, "
                         f"termination {mode}{'/K=' + str(count) if mode == 'choose' else ''}")
            break
    term = facts.terms[0].params
    if term.get("require_contact_with") and all(
            name in _SUBTASK_EQUIVALENT or not params.get("require_contact_with") for name, params in final):
        partner = term["require_contact_with"] if isinstance(term["require_contact_with"], str) else (
            term.get("container") or term.get("reference_object"))
        notes.append(f"contact: the termination also requires contact with {partner}")
    if any(t.params.get("require_gripper_detached") for t in facts.terms) and final and not any(
            name in _SUBTASK_EQUIVALENT or params.get("require_gripper_detached") for name, params in final) and not any(
            name == "object_dropped" for name, _ in facts.subtask_calls):
        notes.append("release: the termination requires release, the subtasks do not")
    if "stacked" in term_funcs and any(name == "object_grabbed" for name, _ in facts.subtask_calls):
        pass  # a grasp before the stack is subtask progress, not a termination difference
    return {"differs": bool(notes), "subtasks": facts.subtasks_text, "notes": notes}


# ----------------------------------------------------------------------------------------- compatibility
def compatibility(facts: Facts, scan: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    notes: list[dict[str, str]] = []
    kind = family(facts)
    scene = scan["scenes"].get(facts.scene)
    blocked: list[dict[str, str]] = []
    if scene is None:
        blocked.append({"code": "scene_not_scanned", "detail": f"{facts.scene} is absent from the asset scan"})
    else:
        named = set(targets(facts)) | set(references(facts))
        missing = sorted(name for name in named if name not in scene["objects"])
        if missing:
            blocked.append({"code": "object_missing_from_scene", "detail": f"{missing} not in {facts.scene}"})
        if scene["local_missing_geometry"]:
            blocked.append({"code": "local_asset_missing", "detail": ", ".join(scene["local_missing_geometry"])})
        if scene["remote_geometry"]:
            notes.append({"code": "remote_scene_geometry", "detail": "the scene loads "
                          + ", ".join(sorted({url.rsplit('/', 1)[-1] for url in scene["remote_geometry"]}))
                          + " from the Omniverse content server at load time; the worker host needs that network "
                          "access (the purple crate's geometry)"})
        for name in sorted(named):
            item = scene["objects"].get(name) or {}
            if item.get("joints") or item.get("articulation_root"):
                blocked.append({"code": "articulated_object", "detail": f"{name} has joints the motion surface "
                                "cannot drive"})
    if facts.native_limit > PREVIOUS_LONGEST_LIMIT:
        long = facts.native_limit >= 2 * PREVIOUS_LONGEST_LIMIT
        notes.append({"code": "long_horizon" if long else "longer_horizon",
                      "detail": f"official limit {facts.native_limit} native actions ({facts.episode_s} s at 15 Hz; "
                                f"{facts.native_limit * 3} under the default extended budget), against "
                                f"{PREVIOUS_LONGEST_LIMIT} for the longest task admitted before"})
    moves = len(set(targets(facts)))
    if moves >= 5 and logical(facts)[0] == "all":
        notes.append({"code": "many_objects", "detail": f"{moves} objects to move; the default "
                      f"max_robot_decisions {DEFAULT_DECISIONS} may bind before the native limit"})
    if logical(facts)[0] == "choose":
        notes.append({"code": "exact_count", "detail": "exactly K objects must satisfy the relation; one more fails"})
    if kind == "upright":
        notes.append({"code": "reorientation", "detail": "needs the object turned upright (a regrasp or a "
                      "placement in a new orientation); expressible with move_pose orientations"})
    if kind == "ordered_in_container":
        notes.append({"code": "order_latched", "detail": "a wrong first placement can never succeed later in the "
                      "episode (first placements are latched)"})
    if "large_storage_rack" in references(facts):
        notes.append({"code": "floor_rack", "detail": "the storage rack stands on the floor beside the table; its "
                      "near face is the reachable part, unverified until a run"})
    wide = []
    for name in dict.fromkeys(targets(facts)):
        asset = _asset_key((facts.prims.get(name) or {}).get("asset"))
        dims = (catalog.get(f"assets/{asset}") or {}).get("dims") if asset else None
        if dims and min(dims) > MAXIMUM_APERTURE_M:
            wide.append(phrase(facts, name))
    if wide:
        notes.append({"code": "wide_object", "detail": "no face of " + " or ".join(f"the {text}" for text in sorted(
            set(wide))) + " fits the open gripper (85 mm); a grasp needs a rim, handle, stem or edge"})
    review = TASK_REVIEW.get(facts.task, {})
    if kind == "stacked" or review.get("gap"):
        notes.append({"code": "scoring_gap", "detail": review.get("gap") or "the stack term does not require "
                      "releasing the top object"})
    if review.get("reset"):
        notes.append({"code": "reset_relation", "detail": review["reset"]})
    status = "blocked" if blocked else ("ready_with_note" if any(
        note["code"] not in ("reset_relation",) for note in notes) else "ready")
    return {"status": status, "blocked": blocked, "notes": notes}


# ------------------------------------------------------------------------------------------ witness plan
#: An exact-count negative sets K+1 objects down at once; while they land the count passes through K, which a
#: robot setting them down one by one never shows past K. Its verdict counts from this tick on.
LANDING_TICKS = 15


def witness_plan(facts: Facts) -> dict[str, Any]:
    """The seed-0 witness the worker runs (``environment/backends/robolab/contract_witness.py``)."""
    kind = family(facts)
    objects = list(dict.fromkeys(targets(facts)))
    mode, count = logical(facts)
    params = facts.terms[0].params
    cases: list[dict[str, Any]] = []

    def case(name: str, expected: bool, ops: list[dict[str, Any]], ticks: int = 40, ignore_ticks: int = 0) -> None:
        cases.append({"case": name, "expected": expected, "ops": ops, "ticks": ticks,
                      **({"ignore_ticks": ignore_ticks} if ignore_ticks else {})})

    def into(names: Iterable[str], container: str, mode_: str = "into", settle: bool = True) -> list[dict[str, Any]]:
        names = list(names)
        return [{"op": "place", "object": name, mode_: container, "slot": index, "slots": len(names),
                 **({} if settle else {"settle": False})} for index, name in enumerate(names)]

    case("reset_state", False, [], ticks=3)
    if kind in ("in_container", "ordered_in_container"):
        container = params["container"]
        mode_ = "onto" if FRAME_CONTAINERS.get(container) == "box" else "into"
        case("beside", False, [{"op": "place", "object": objects[0], "beside": container}])
        if kind == "ordered_in_container":
            case("wrong_order", False, into(objects[::-1], container, mode_), ticks=60)
            case("in_order", True, into(objects, container, mode_), ticks=60)
        elif mode == "all":
            if len(objects) > 1:
                # Counted from the reset: an object that starts inside stays, so one that starts outside is left out.
                case("all_but_one", False, [{"op": "count_in", "objects": objects, "container": container,
                                             "k": len(objects) - 1, "mode": mode_}], ticks=60)
            case("all_in", True, into(objects, container, mode_), ticks=60)
        elif mode == "any":
            case("one_in", True, into(objects[:1], container, mode_))
        else:
            if count + 1 <= len(objects):
                case("one_too_many", False, into(objects[:count + 1], container, mode_, settle=False), ticks=60,
                     ignore_ticks=LANDING_TICKS)
            case("exactly_k", True, [{"op": "count_in", "objects": objects, "container": container, "k": count,
                                      "mode": mode_}], ticks=60)
    elif kind == "groups_in_containers":
        groups = params["groups"]
        swapped = []
        for group, other in zip(groups, groups[1:] + groups[:1]):
            swapped += into(_as_list(group["object"]), other["container"])
        case("swapped", False, swapped, ticks=60)
        sorted_ops = []
        for group in groups:
            sorted_ops += into(_as_list(group["object"]), group["container"])
        case("sorted", True, sorted_ops, ticks=60)
    elif kind == "on_surface":
        reference = params["reference_object"]
        case("beside", False, [{"op": "place", "object": objects[0], "beside": reference}])
        if mode == "choose":
            case("one_too_many", False, into(objects[:count + 1], reference, "onto", settle=False), ticks=60,
                 ignore_ticks=LANDING_TICKS)
            case("exactly_k", True, [{"op": "count_in", "objects": objects, "container": reference, "k": count,
                                      "mode": "onto"}], ticks=60)
        else:
            case("all_on", True, into(objects if mode == "all" else objects[:1], reference, "onto"), ticks=60)
    elif kind == "stacked":
        names = list(params["objects"])
        if params.get("order") == "top_to_bottom":
            names.reverse()
        case("beside", False, [{"op": "place", "object": names[1], "beside": names[0]}])
        if params.get("order") not in (None, "None") and len(names) > 2:
            wrong = [names[1], names[0], *names[2:]]
            case("wrong_order", False, [{"op": "stack", "objects": wrong}], ticks=60)
        case("stacked", True, [{"op": "stack", "objects": names}], ticks=60)
    elif kind in ("out_of_container", "out_onto_surface"):
        container = params["container"]
        case("out", True, [{"op": "place", "object": name, "beside": container} for name in objects], ticks=60)
    elif kind == "picked_up":
        case("lifted_not_held", False, [{"op": "lift", "object": objects[0], "dz": 0.15}], ticks=2)
        case("held_and_lifted", True, [{"op": "grasp", "object": objects[0]}], ticks=40)
    elif kind == "upright":
        case("inverted", False, [{"op": "orient", "object": name, "pose": "inverted"} for name in objects])
        case("upright", True, [{"op": "orient", "object": name, "pose": "upright"} for name in objects], ticks=60)
    elif kind == "relative_position":
        reference = params["reference_object"]
        direction = {"object_left_of": "left", "object_right_of": "right", "object_in_front_of": "front",
                     "object_behind": "behind"}[facts.terms[0].func]
        opposite = {"left": "right", "right": "left", "front": "behind", "behind": "front"}[direction]
        case("opposite", False, [{"op": "relative", "object": objects[0], "reference": reference,
                                  "direction": opposite}])
        case("placed", True, [{"op": "relative", "object": objects[0], "reference": reference,
                               "direction": direction}])
    elif kind == "centered":
        case("centered", True, [{"op": "table_center", "object": objects[0]}])
    return {"schema": WITNESS_PLAN_SCHEMA, "seed": 0, "cases": cases,
            "visibility_objects": list(dict.fromkeys(objects + [name for name in references(facts) if name != "table"])),
            "note": "each case starts from a fresh reset of seed 0; the robot holds its reset pose with the gripper "
                    "open unless a case grasps; success is any official result over the case's ticks"}


# ------------------------------------------------------------------------------------------------ review
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span_sha256(path: Path, function: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return hashlib.sha256("".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")).hexdigest()
    raise ValueError(f"{function} is not defined in {path}")


def predicate_binding(facts: Facts, source: Path) -> tuple[str, str]:
    func = facts.terms[0].func
    relative, function = (HELPERS["predicate_logic"], "check_stacked") if func == "stacked" else (
        HELPERS["conditionals"], func)
    method = (f"SHA256 of exact UTF-8 AST FunctionDef lineno..end_lineno span of {relative}:{function}, preserving "
              f"trailing newline.")
    return span_sha256(source / relative, function), method


def source_dependencies(facts: Facts, source: Path) -> list[dict[str, str]]:
    kind = family(facts)
    helpers = ["conditionals", "predicate_logic"]
    if kind in ("in_container", "groups_in_containers", "ordered_in_container", "out_of_container",
                "out_onto_surface"):
        helpers.append("hull_check")
    helpers += ["world_state", "env"]
    items = [{"role": "task_predicate_and_reset", "path": facts.file, "sha256": _sha256(source / facts.file)}]
    items += [{"role": "shared_semantic_helper" if name not in ("env",) else "official_verdict_boundary",
               "path": HELPERS[name], "sha256": _sha256(source / HELPERS[name])} for name in helpers]
    items.append({"role": "scene_reset_state", "path": f"assets/scenes/{facts.scene}",
                  "sha256": _sha256(source / "assets/scenes" / facts.scene)})
    return items


def _reset_basis(facts: Facts) -> str:
    named = [name for name in dict.fromkeys(targets(facts) + references(facts)) if name != "table"]
    review = TASK_REVIEW.get(facts.task, {})
    text = (f"Scene {facts.scene}: {', '.join(named)} among {len(facts.contact_objects)} contact objects. The env's "
            "only event is reset_scene_to_default and the task adds none, so every seed resets to the same authored "
            "layout; a seed varies physics only.")
    if review.get("reset"):
        text += f" At reset {review['reset']}."
    return text


ASSET_BASIS = {
    "in_container": "Containment tests the centroid of the object's convex hull in the container's local frame "
                    "against the container hull with its top faces removed (hull_check), so a tipped container fails "
                    "and the region above the open top counts.",
    "groups_in_containers": "Each group uses object_in_container: hull centroid in the container's open-top hull, "
                            "in the container's local frame.",
    "ordered_in_container": "Each placement uses the object_in_container hull test in the container's frame; the "
                            "latch reads env.episode_length_buf at the first true step.",
    "on_surface": "Support is the contact force from the surface on the object (upward, inside the support cone) "
                  "plus the object's box centroid inside the surface's world-AABB footprint; height is not bounded.",
    "stacked": "World-frame oriented boxes of each pair and their contact sensor (check_stacked).",
    "out_of_container": "Not in the container's open-top hull (centroid test in the container frame).",
    "out_onto_surface": "Not in the container's open-top hull and supported by the table's upward contact force.",
    "picked_up": "Gripper contact (any finger body) and every box corner above the table's top face by the lift "
                 "margin, with the centroid over the table's footprint.",
    "relative_position": "Vector between root positions in the robot root frame, projected on the table plane, "
                         "inside the direction cone; ORA's public poses use the same robot root frame.",
    "centered": "Box centroid xy against the table box centroid xy; contact with the table.",
}


def build_review(facts: Facts, source: Path, scan: Mapping[str, Any], catalog: Mapping[str, Any],
                 projection: Projection) -> dict[str, Any]:
    kind = family(facts)
    upright = kind == "upright"
    checks = {
        "predicate_and_helpers": {"status": "passed", "basis": (
            f"Read {facts.task} terminations: {'; '.join(t.name + ' = ' + _call(t) for t in facts.terms)}, and the "
            f"helpers in conditionals.py and predicate_logic.py; RobolabEnv records success as the "
            f"TerminationManager's terminated flag, the OR of every non-timeout term.")},
        "reset_and_randomization": {"status": "passed", "basis": _reset_basis(facts)},
        "assets_and_frames": ({"status": "pending", "basis": (
            "Upright is the object's local +z against world up; that +z is the asset's standing axis is confirmed by "
            "the phase-B witness setting the upright pose.")} if upright else
            {"status": "passed", "basis": ASSET_BASIS[kind]}),
        "success_timing": {"status": "passed", "basis": (
            "DoneTerm success is evaluated every control step; RobolabEnv freezes the env at the first true step "
            "(a termination within its first two steps is reset as a physics artifact) and the ORA session latches "
            "it." + (" Placement order is latched at each object's first placement." if kind == "ordered_in_container"
                     else ""))},
        "public_observability": {"status": "pending", "basis": (
            "The clauses name objects by kind and color as the instruction does; whether each is in view of the "
            "over-shoulder or wrist camera at reset is checked by the phase-B witness's visibility probe.")},
        "success_witness": {"status": "pending", "basis": "Phase-B seed-0 witness not yet run."},
        "counterexamples": {"status": "pending", "basis": "Phase-B seed-0 witness not yet run."},
        "no_expert_strategy": {"status": "passed", "basis": "Clauses state end conditions only; no approach, grasp, "
                               "route or motion advice."},
        "privacy": {"status": "passed", "basis": "No coordinates, thresholds, fixture or object identifiers, joint "
                    "values or predicate names enter the public text; numeric limits stay in this private record."},
    }
    return {
        "schema": REVIEW_SCHEMA, "backend": "robolab", "task": facts.task, "status": "needs_review",
        "reviewer": GENERATED_REVIEWER, "reviewed_at": GENERATED_AT,
        "scope": (f"robolab {facts.task} at the pinned revision; generated per predicate family ({kind}); seed-0 "
                  "witness pending. Semantic projection of the official termination, not physical qualification."),
        "verified_revisions": [REVISION], "task_language": facts.instruction, "family": kind,
        "source_dependencies": source_dependencies(facts, source), "evidence": [], "checks": checks,
        "conditions": projection.conditions, "public_clauses": projection.clauses,
        "counterexamples": projection.counterexamples, "termination_vs_subtasks": termination_vs_subtasks(facts),
        "compatibility": compatibility(facts, scan, catalog), "witness_plan": witness_plan(facts),
        "limitations": ("Generated from the pinned source; the witness (phase B) covers seed 0 only and uses direct "
                        "state edits or scripted moves, never a policy rollout. Source-derived negative cases marked "
                        "as such are not executed."),
        "forbidden_public_claims": projection.forbidden,
    }


#: Fields and checks the phase-B witness owns (``contract_witness.fill_review``); regeneration keeps them.
WITNESS_FIELDS = ("evidence", "counterexamples", "witness_result")
WITNESS_CHECKS = ("success_witness", "counterexamples", "public_observability", "assets_and_frames")


def merge_witness(generated: dict[str, Any], existing: Mapping[str, Any] | None) -> dict[str, Any]:
    """A regenerated draft keeps what its witness recorded; a witness's blocker also blocks the task."""
    if not existing or not existing.get("witness_result"):
        return generated
    merged = dict(generated)
    for key in WITNESS_FIELDS:
        if key in existing:
            merged[key] = existing[key]
    merged["checks"] = dict(generated["checks"])
    for key in WITNESS_CHECKS:
        if key in existing.get("checks", {}) and (key != "assets_and_frames" or generated.get("family") == "upright"):
            merged["checks"][key] = existing["checks"][key]
    blocked = existing["witness_result"].get("blocked") or []
    if blocked:
        compatibility = dict(merged["compatibility"])
        compatibility["blocked"] = [*compatibility["blocked"], *blocked]
        compatibility["status"] = "blocked"
        merged["compatibility"] = compatibility
    return merged



__all__ = ["REVIEW_SCHEMA", "WITNESS_PLAN_SCHEMA", "build_review", "compatibility", "merge_witness",
           "predicate_binding", "termination_vs_subtasks", "witness_plan"]
