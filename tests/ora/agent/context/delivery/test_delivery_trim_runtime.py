"""``observation_delivery_trim_v2`` at the delivery sites, and what stays recoverable.

The direct_v4 surfaces declare it. An observation the Pi core delivers (its
Python draft with the trims applied as retention hints) shows the trimmed facts; the policy-state observation, the hash-bound
calibration and the canonical ``tool_result`` keep every byte. That an aged
trimmed observation reopens to exactly the text the policy was first shown,
and that the full-image marker is unchanged, is held by the recorded Context
fixtures ``tests/data/context_differential/captured_delivery_06..09``.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from ora.agent.composition.baseline import load_native_baseline
from tests.support.agent.surface_catalog import surface_of
from ora.agent.context.delivery import trim
from ora.agent.context.delivery import tool_results as policy_delivery
from ora.agent.context.observation.current_visual import camera_geometry
from tests.support.agent.current_visual_delivery import ROOT, delivery_for, published
from tests.support.agent.observation_delivery import observation_text

BACKENDS = ("libero_pro", "robotwin")


def _bound_context(tmp_path, backend):
    context, projector, store, surface, source = published(tmp_path, backend)
    observation = json.loads((source / "observation.json").read_text(encoding="utf-8"))
    bound = dataclasses.replace(projector, workspace=tmp_path / "bound-workspace",
                                control_root=tmp_path / "bound-control", capabilities=surface.capabilities)
    delivered = bound.publish(observation=observation, observation_root=source, task_instruction="Inspect",
                              remaining_budget={}, policy_state=store.steps[-1])
    return context, delivered, store, surface, source, observation


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_default_surface_declares_the_capability(backend):
    instance = load_native_baseline(ROOT)["instances"][backend]
    surface = surface_of(instance["tool_surface"])
    assert surface.supports(trim.CAPABILITY)
    assert trim.declared_rules(surface.capabilities) == trim.RULES


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_native_message_keeps_the_calibration_and_trims_the_facts(tmp_path, backend):
    """The small rules apply; the comparison sentence, both pose matrices, the
    exact calibration digits and the calibration owner reach the policy."""
    legacy, delivered, store, surface, source, observation = _bound_context(tmp_path, backend)
    message, _ = delivery_for(tmp_path, delivered, surface).observation(delivered)
    envelope = observation_text(message)
    current = envelope["current_actual"]
    assert isinstance(envelope["comparison"]["instruction"], str)
    assert "terminal_lifecycle" not in envelope
    facts = current["public_robot_facts"]
    for name, entry in facts["cameras"].items():
        assert "image_view" not in entry and "native_pixel_to_image_offset_xy" not in entry, name
    assert all("image_view" in entry for entry in legacy["public_robot_facts"]["cameras"].values())
    assert json.loads((source / "observation.json").read_text(encoding="utf-8")) == observation
    for name, geometry in current["camera_geometry"].items():
        full = camera_geometry(source, observation["cameras"][name])
        assert {key: geometry[key] for key in full if key in geometry} == {
            key: full[key] for key in full if key in geometry}, name
        assert "cam2world_cv" in geometry and len(set(geometry) - {"intrinsic_cv"}) > 1, name
    assert facts.get("calibration_owner") == legacy["public_robot_facts"].get("calibration_owner")


@pytest.mark.parametrize("backend", ("libero_pro", "robotwin"))
def test_a_surface_without_the_capability_keeps_the_legacy_message(tmp_path, backend):
    legacy, _, _, _, _, _ = _bound_context(tmp_path, backend)
    assert policy_delivery.deliver_public_facts(
        legacy["public_robot_facts"], capabilities=("observation_delivery_trim_v0",)) is legacy["public_robot_facts"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_canonical_tool_result_keeps_every_byte(backend):
    from ora.agent.session_core.gateway import NativeToolGateway

    instance = load_native_baseline(ROOT)["instances"][backend]
    surface = surface_of(instance["tool_surface"])
    canonical = {"status": "ok", "candidates": [{"position_m": [0.20466682314872742, -0.1, 0.9],
                                                 "orientation_error_deg": 2.864789}]}
    before = copy.deepcopy(canonical)

    class Runtime:
        def __init__(self):
            self.surface = surface

    delivered = NativeToolGateway(Runtime()).record_proposed_result("move_pose", {"payload": {"arguments": {}}},
                                                                    canonical)
    assert canonical == before
    # The delivered copy keeps the exact digits.
    assert delivered["candidates"] == canonical["candidates"]
    # A tool result the policy authored is never rewritten.
    authored = {"outputs": [{"name": "x", "value": 0.20466682314872742}]}
    assert NativeToolGateway(Runtime()).record_proposed_result(
        "compute", {"payload": {"arguments": {}}}, authored) == authored
