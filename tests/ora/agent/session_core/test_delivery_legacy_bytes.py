"""Surfaces that do not declare the 0059 delivery capabilities keep exact bytes.

D1 (``last_action_summary_v1``), D2 (``policy_angles_degrees_v1``) and D3a
(``within_payload_deduplication_v1``) activate on the next surface generation.
Every surface that does not declare them composes, projects and publishes the
bytes it did before them. The direct_v3 default surfaces have left this tree;
RoboTwin's current default with its predecessor's recorded capabilities stands
for a surface that does not declare them.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from tests.support.agent.delivery_lessons_fixtures import (
    LEGACY_STAND_INS, NEW_CAPABILITIES, legacy_surface, robotwin_receipt, skill_receipt,
)
from ora.agent.session_core.receipt_projection import (
    _RECEIPT_EFFECT_PROJECTORS, ReceiptProjectionContext, resolve_receipt_effect_projector,
)


def _legacy(backend):
    """A surface of ``backend`` that declares none of the delivery capabilities."""
    return legacy_surface(backend)


RECEIPTS = {
    "robotwin": (robotwin_receipt(), robotwin_receipt(status="reached", stage=None, code=None,
                                                      executed=True, pose_error=(0.002, 0.007)),
                 skill_receipt()),
}


@pytest.mark.parametrize("backend", sorted(LEGACY_STAND_INS))
def test_effects_on_a_surface_without_the_capabilities_are_the_legacy_bytes(backend):
    surface = _legacy(backend)
    assert not set(NEW_CAPABILITIES) & set(surface.capabilities)
    for receipt in RECEIPTS[backend]:
        context = ReceiptProjectionContext(last_action_id=receipt["action_id"],
                                           last_action_public_name=None,
                                           surface_supports=surface.supports)
        legacy = _RECEIPT_EFFECT_PROJECTORS[receipt["schema"]](receipt, context)
        delivered = resolve_receipt_effect_projector(receipt["schema"])(receipt, context)
        assert json.dumps(delivered, ensure_ascii=False, sort_keys=True) == json.dumps(
            legacy, ensure_ascii=False, sort_keys=True)


@pytest.mark.parametrize("backend", sorted(LEGACY_STAND_INS))
def test_published_context_is_the_legacy_bytes(tmp_path, backend):
    """Passing the surface's own capabilities to the projector changes nothing."""
    from tests.support.agent.current_visual_delivery import published

    context, projector, store, _surface, source = published(tmp_path, backend)
    observation = json.loads((source / "observation.json").read_text())
    # A surface that declares none of the delivery capabilities.
    frozen = _legacy(backend)
    bound = dataclasses.replace(projector, workspace=tmp_path / "bound-workspace",
                                control_root=tmp_path / "bound-control",
                                capabilities=frozen.capabilities)
    again = bound.publish(observation=observation, observation_root=source,
                          task_instruction="Inspect", remaining_budget={},
                          policy_state=store.steps[-1])
    assert json.dumps(again["public_robot_facts"], sort_keys=True) == json.dumps(
        context["public_robot_facts"], sort_keys=True)


@pytest.mark.parametrize("backend", sorted(LEGACY_STAND_INS))
def test_tool_results_pass_both_gateways_unchanged(backend):
    from ora.agent.session_core.gateway import NativeToolGateway

    surface = _legacy(backend)
    result = {"candidates": [{"orientation_error_rad": 0.5}], "status": "ok"}

    class Runtime:
        def __init__(self):
            self.surface = surface

    delivered = NativeToolGateway(Runtime()).record_proposed_result(
        "align_plan", {"payload": {"arguments": {}}}, result)
    assert delivered == {"candidates": [{"orientation_error_rad": 0.5}], "status": "ok"}
