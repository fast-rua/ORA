"""Fixed prose leaves a result only where a description states it, word for word.

Every entry of ``delivery.trim_prose.ENTRIES`` is the exact text its producer
writes, and the composed description of its tool on each listed backend's v2
surface states it once (in the degree form that surface delivers). A text that
differs by a character stays in the result, and so does a text the run's
delivered tool list does not state (``trim.stated_prose``): an entry with no
backends, the LIBERO robot-only texts, leaves only the results of a run whose
delivered help still states it.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.context.delivery import trim
from ora.agent.context.delivery.trim_prose import ENTRIES, PROSE_KEYS
from ora.agent.composition.surface.units import degrees_text

ROOT = Path(__file__).resolve().parents[5]
V2_SURFACES = {
    "robodojo": "robodojo_placement_toolkit_v5",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}


def _constant(reference: str) -> str:
    module, _, name = reference.partition(":")
    value = importlib.import_module(module)
    for step in name.split("."):
        value = value[step] if isinstance(value, dict) else getattr(value, step)
    return value


def _inline_literals(path: str) -> set[str]:
    tree = ast.parse((ROOT / "ora" / path).read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}


#: Where each text comes from, as the producer writes it.
PRODUCERS = {
    "OBSERVED_CLEARANCE": lambda: _constant("ora.agent.toolkit.host.motion_rules.clearance:CLAIM_BOUNDARY"),
    "ROBODOJO_ONE_ARM_IK_SCOPE": lambda: _constant(
        "ora.environment.backends.robodojo.ik:INTER_ARM_PATH_EVIDENCE_SCOPE"),
    "ROBODOJO_BOTH_ARMS_IK_SCOPE": lambda: _constant(
        "ora.environment.backends.robodojo.ik:PATH_EVIDENCE_SCOPE"),
    "ROBODOJO_REACH": lambda: _constant("ora.agent.toolkit.tools.move_pose._reach:REACH_CLAIM"),
    "ROBODOJO_NEAREST_ORIENTATION": lambda: _constant(
        "ora.agent.composition.schemas.reach_orientation:CLAIM"),
    "ROBOTWIN_REACH": lambda: _constant("ora.contracts.robotwin_reach:CLAIM_BOUNDARY"),
    "ROBOTWIN_REACH_UNSOLVED": lambda: _unsolved(_constant("ora.contracts.robotwin_reach:CLAIM_BOUNDARY")),
    "ROBOTWIN_ALIGN": lambda: _constant("ora.contracts.robotwin_reach:CLAIM_BOUNDARY"),
    "ROBODOJO_ALIGN": lambda: _constant("ora.contracts.reach_check:CLAIM_BOUNDARY"),
    # The default LIBERO planner plans in the public-depth world; a robot-only
    # (world_model none) run writes CLAIM_BOUNDARY_ROBOT_ONLY, which stays in its results.
    "LIBERO_REACH": lambda: _constant(
        "ora.environment.backends.libero_pro.reach_check:CLAIM_BOUNDARY_DEPTH_WORLD"),
    "LIBERO_REACH_UNSOLVED": lambda: _unsolved(
        _constant("ora.environment.backends.libero_pro.reach_check:CLAIM_BOUNDARY_DEPTH_WORLD")),
    "NEAREST_ORIENTATION": lambda: _constant("ora.agent.composition.schemas.reach_reference:NEAREST_CLAIM"),
    "GRIPPER_STALL": lambda: _constant(
        "ora.environment.controllers.gripper_stall:GRIPPER_STALL_CLAIM_BOUNDARY"),
    "GRASP_PLAN_IK_REFERENCE": lambda: _constant("ora.agent.toolkit.tools.grasp_plan._reach:GRASP_ANNOTATION_CLAIM"),
    "GRASP_PLAN_JAW_SYMMETRY": lambda: _constant(
        "ora.agent.composition.schemas.grasp_jaw_symmetry:CLAIM_BOUNDARY"),
    "GRASP_EXECUTE": lambda: _constant("ora.agent.toolkit.host.skills.driver:CLAIM_BOUNDARY"),
    "GRASP_JAW_HOLD": lambda: _constant("ora.agent.toolkit.host.skills.jaw_hold:CLAIM_BOUNDARY"),
}
#: Texts written inline: the literal must occur in that producer's source.
INLINE = {
    "SEGMENTATION": "agent/toolkit/tools/vision_segment/tool.py",
    "GRASP_EXECUTE_PRECOMMIT": "agent/toolkit/tools/grasp_execute/_precommit.py",
    "GRASP_VERIFICATION": "agent/toolkit/host/perception/localization.py",
    "GRASP_LIFT_PROBE": "agent/toolkit/host/skills/lift_evidence.py",
    "WAIT": "contracts/wait.py",
}


#: Texts no producer writes any more; recorded definitions still state them, so they stay.
RECORDED_ONLY = {"ROBOTWIN_REACH_IK", "ROBOTWIN_REACH_IK_UNSOLVED", "ROBOTWIN_ALIGN_IK", "ROBOTWIN_REACH_DESCENT", "ROBOTWIN_REACH_UNSOLVED_DESCENT", "ROBOTWIN_ALIGN_DESCENT",
                 "NEAREST_ORIENTATION_DISAGREEMENT", "LIBERO_REACH_ROBOT_ONLY", "LIBERO_REACH_UNSOLVED_ROBOT_ONLY",
                 "LIBERO_PLANNING_NOTE_ROBOT_ONLY", "ROBODOJO_REACH_SDK", "ROBODOJO_NEAREST_ORIENTATION_SDK",
                 "ROBODOJO_ALIGN_SDK"}
#: A text no producer writes any more that a current surface still states: its help is unchanged.
STATED_ONLY = {"LIBERO_PLANNING_NOTE"}


def _unsolved(claim: str) -> str:
    """``reach_reference.unsolved``: the sampling claim, a space, the backend's claim."""
    return f"{_constant('ora.agent.toolkit.tools.move_pose._reference:SAMPLING_CLAIM')} {claim}"


@pytest.fixture(scope="module")
def surfaces():
    return {backend: surface_of(profile) for backend, profile in V2_SURFACES.items()}


def test_every_entry_has_one_producer_and_a_known_key():
    names = [entry.name for entry in ENTRIES]
    assert len(names) == len(set(names))
    assert set(PRODUCERS) | set(INLINE) | RECORDED_ONLY | STATED_ONLY == set(names)
    assert not set(PRODUCERS) & set(INLINE)
    for entry in ENTRIES:
        assert entry.key in PROSE_KEYS, entry.name
        assert set(entry.backends) <= set(V2_SURFACES), entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_each_text_is_what_its_producer_writes(entry):
    if entry.name in RECORDED_ONLY:
        assert not entry.backends, entry.name
    elif entry.name in STATED_ONLY:
        assert entry.backends, entry.name
    elif entry.name in PRODUCERS:
        assert PRODUCERS[entry.name]() == entry.text
    else:
        assert entry.text in _inline_literals(INLINE[entry.name]), entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_each_text_is_stated_in_its_tool_description_on_every_listed_backend(entry, surfaces):
    for backend in entry.backends:
        surface = surfaces[backend]
        # The fixed-prose rule is in v1 and in v2, which the default surfaces declare.
        assert trim.RULE_FIXED_PROSE in trim.declared_rules(surface.capabilities), backend
        description = next(item["description"] for item in surface.definitions if item["name"] == entry.tool)
        assert entry.text in description or degrees_text(entry.text) in description, (backend, entry.name)


def test_only_an_exact_canonical_text_leaves():
    observed = next(entry for entry in ENTRIES if entry.name == "OBSERVED_CLEARANCE")
    result = {"compiled_pose": {"targets": {"right": {"observed_clearance": {
        "claim_boundary": observed.text, "clearance_m": -0.0126}}}},
        "other": {"claim_boundary": observed.text + " "}, "note": "free text the tool wrote now"}
    delivered = trim.drop_canonical_prose(result)
    assert delivered["compiled_pose"]["targets"]["right"]["observed_clearance"] == {"clearance_m": -0.0126}
    assert delivered["other"] == {"claim_boundary": observed.text + " "}
    assert delivered["note"] == "free text the tool wrote now"
    # The same text under a key no rule names stays.
    assert trim.drop_canonical_prose({"summary": observed.text}) == {"summary": observed.text}


def test_the_degree_form_a_surface_delivers_also_matches():
    libero = next(entry for entry in ENTRIES if entry.name == "LIBERO_REACH")
    assert "rad" in libero.text and degrees_text(libero.text) != libero.text
    assert trim.drop_canonical_prose({"claim_boundary": degrees_text(libero.text)}) == {}


def test_a_robot_only_libero_claim_stays_in_its_result():
    """The LIBERO help states the depth-world texts; a world_model none run keeps its own."""
    from ora.environment.backends.libero_pro.reach_check import (
        CLAIM_BOUNDARY_ROBOT_ONLY, IK_MODEL_ROBOT_ONLY, ik_model_and_claim,
    )

    assert ik_model_and_claim("none") == (IK_MODEL_ROBOT_ONLY, CLAIM_BOUNDARY_ROBOT_ONLY)
    assert trim.drop_canonical_prose({"claim_boundary": CLAIM_BOUNDARY_ROBOT_ONLY}) == {
        "claim_boundary": CLAIM_BOUNDARY_ROBOT_ONLY}
    robot_only = "Counts overlap; raw joint bounds do not isolate self-collision or other constraints. " \
                 "World collision model is absent."
    assert trim.drop_canonical_prose({"note": robot_only}) == {"note": robot_only}


def _delivered(surface):
    from ora.agent.composition.surface.null_optional_arguments import deliver_tool_definitions
    return deliver_tool_definitions(list(surface.definitions), tuple(surface.capabilities))


@pytest.mark.parametrize("backend", sorted(V2_SURFACES))
def test_a_v2_surface_states_its_listed_texts_and_no_text_without_backends(backend, surfaces):
    stated = trim.stated_prose(_delivered(surfaces[backend]))
    for entry in ENTRIES:
        if backend in entry.backends:
            assert entry.text in stated.get(entry.key, ()), entry.name
        if not entry.backends:
            assert entry.text not in stated.get(entry.key, ()), entry.name
    # Without a delivered tool list the trim takes the texts the current surfaces state.
    assert {key: set(texts) for key, texts in trim.CANONICAL_PROSE.items()} == {
        key: {text for entry in ENTRIES if entry.key == key and entry.backends
              for text in (entry.text, degrees_text(entry.text))}
        for key in {entry.key for entry in ENTRIES if entry.backends}}


def test_only_the_texts_the_delivered_help_states_leave():
    """A recorded LIBERO help that states the robot-only texts drops them; today's help keeps them."""
    from ora.environment.backends.libero_pro.reach_check import CLAIM_BOUNDARY_DEPTH_WORLD

    robot_only = next(entry for entry in ENTRIES if entry.name == "LIBERO_PLANNING_NOTE_ROBOT_ONLY").text
    claim = next(entry for entry in ENTRIES if entry.name == "LIBERO_REACH_ROBOT_ONLY").text
    result = {"claim_boundary": claim, "detail": {"note": robot_only},
              "other": {"claim_boundary": CLAIM_BOUNDARY_DEPTH_WORLD}}
    current = trim.stated_prose(_delivered(surface_of(V2_SURFACES["libero_pro"])))
    assert trim.drop_canonical_prose(result, current) == {"claim_boundary": claim,
                                                          "detail": {"note": robot_only}, "other": {}}
    recorded = [{"name": "move_pose", "inputSchema": {"type": "object"},
                 "description": f'The claim is "{claim}" and the note "{robot_only}".'}]
    assert trim.drop_canonical_prose(result, trim.stated_prose(recorded)) == {
        "detail": {}, "other": {"claim_boundary": CLAIM_BOUNDARY_DEPTH_WORLD}}
    # A canonical text the delivered list does not state stays, even on a declaring surface.
    assert trim.drop_canonical_prose(result, trim.stated_prose([])) == result


def test_the_delivery_sites_trim_by_the_prose_they_are_given():
    from ora.agent.context.delivery.tool_results import deliver_tool_result
    from ora.agent.session_core.effect_delivery import delivered_projector
    from ora.agent.session_core.receipt_projection import ReceiptProjectionContext
    # The trim alone, so the result and effect reach it unconverted.
    surface = dataclasses.replace(surface_of(V2_SURFACES["libero_pro"]),
                                  capabilities=(trim.CAPABILITY,))
    claim = next(entry for entry in ENTRIES if entry.name == "LIBERO_REACH_ROBOT_ONLY").text
    recorded = trim.stated_prose([{"name": "move_pose", "description": claim}])
    result = {"claim_boundary": claim}
    assert deliver_tool_result(surface, "move_pose", result) == result
    assert deliver_tool_result(surface, "move_pose", result, prose=recorded) == {}
    project = delivered_projector(lambda record, _context: dict(record))
    assert project(result, ReceiptProjectionContext(None, "move_pose", surface.supports)) == result
    assert project(result, ReceiptProjectionContext(None, "move_pose", surface.supports,
                                                    stated_prose=recorded)) == {}
