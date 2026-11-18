"""Policy-visible field names never collide with the privileged-field denylist.

`agent/evidence/trace_privacy.py` rejects any trace whose JSON carries a key with
a `FORBIDDEN_FRAGMENTS` term on a word boundary. A genuinely public field whose
*name* collides is indistinguishable from a leak: a RoboDojo batch published
`jaw_contact_center_pixels` on every observation and its evaluation came back
invalid with ~100 `privileged_artifact_field` reasons. The field is now
`jaw_center_pixels`.

These tests compose realistic policy-visible payloads -- current public robot
facts on every surface that projects the controlled point, RoboDojo sparse
motion and jaw receipts through the receipt projector, and every result-schema
property the live Dojo surfaces declare -- and run the verifier's own
`_forbidden_path` on them, so the next colliding public field fails here
instead of in a sealed run. The denylist itself is not relaxed.
"""
from __future__ import annotations

import json

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.context.observation.contact_points import (
    JAW_CENTER_PIXELS_FIELD, jaw_contact_center_delivery, project_policy_public_facts,
)
from ora.agent.composition.surface.view_config import controlled_point_projection_delivery
from ora.agent.evidence.trace_privacy import (
    PUBLIC_CONTACT_GEOMETRY_VALUES, _forbidden_path, _matches_forbidden_fragment,
)
from ora.agent.session_core.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
# Only the staged session requests the fake SDK boundary (a root conftest
# fixture), so the facts tests keep the real selector untouched.
from tests.support.robodojo.backend import sparse_session, trajectory_to, v3

GRASP_CONTACT_POSE = "public_grasp_contact_pose_v1"
LIVE_DOJO = ("robodojo_placement_toolkit_v5", "robodojo_placement_toolkit_v6")


def _projection_surfaces() -> list[str]:
    selected = []
    for profile in sorted(PROFILE_CATALOG):
        try:
            surface = surface_of(profile)
        except Exception:  # retired or override-only profiles do not deliver anything
            continue
        if controlled_point_projection_delivery(surface.view_config):
            selected.append(profile)
    return selected


PROJECTION_SURFACES = _projection_surfaces()


def _backend(profile: str) -> str:
    return next(name for name in ("robodojo", "libero", "robotwin", "robocasa", "robolab", "yam") if profile.startswith(name))


def _as_delivered(result: dict, surface) -> dict:
    """The shape the verifier scans: a provider request carrying the tool result as text."""
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": json.dumps(result)}]}],
            "tools": [dict(item) for item in surface.definitions]}


def _scan(value, surface):
    return _forbidden_path(value, allow_public_grasp_contact_pose=surface.supports(GRASP_CONTACT_POSE))


def test_every_live_projection_surface_is_covered():
    assert set(LIVE_DOJO) <= set(PROJECTION_SURFACES)
    assert any(name.startswith("libero") for name in PROJECTION_SURFACES)
    assert any(name.startswith("robotwin") for name in PROJECTION_SURFACES)


@pytest.mark.parametrize("profile", PROJECTION_SURFACES)
def test_projected_public_robot_facts_pass_the_privileged_field_audit(tmp_path, profile):
    from tests.support.agent.controlled_point_projection import _dojo_observation
    from tests.support.agent.public_robot_facts import fixture

    surface = surface_of(profile)
    backend = _backend(profile)
    if backend == "robodojo":
        _, observation, source = _dojo_observation(tmp_path)
    else:
        _, observation, source = fixture(tmp_path, backend)
    views = set(surface.view_config["automatic_actual_views"])
    observation["cameras"] = {name: value for name, value in observation["cameras"].items() if name in views}
    observation["main_camera"] = next(iter(observation["cameras"]))
    observation["robot_geometry_verification"]["descriptor_sha256"] = (
        surface.view_config["public_geometry_descriptor"]["sha256"])
    (source / "observation.json").write_text(json.dumps(observation))
    facts = project_policy_public_facts(
        observation=observation, observation_root=source, view_config=surface.view_config,
        controlled_point_projection=True)
    cameras = facts["cameras"].values()
    assert all("controlled_point_pixels" in camera for camera in cameras)
    if jaw_contact_center_delivery(surface.view_config):
        assert all(JAW_CENTER_PIXELS_FIELD in camera for camera in cameras)
    delivered = _as_delivered({"current_actual": {"public_robot_facts": facts}}, surface)
    assert _scan(delivered, surface) is None


def test_the_retired_jaw_field_name_would_fail_the_audit():
    """Pin the reason for the rename, so the denylist is not quietly relaxed around it."""
    assert _forbidden_path({"cameras": {"head": {"jaw_contact_center_pixels": {}}}}) is not None
    assert _forbidden_path({"cameras": {"head": {JAW_CENTER_PIXELS_FIELD: {}}}}) is None


#: The verifier's own value-checked exemptions (`contact_pose` shape, pinned
#: compiler constants, the servo `contact_mode` enum); every other declared key
#: must be clean.
VERIFIER_EXEMPT_KEYS = {"contact_pose", "contact_mode", *PUBLIC_CONTACT_GEOMETRY_VALUES}


def _keys(node, found):
    if isinstance(node, dict):
        for key, child in node.items():
            if isinstance(key, str):
                found.add(key)
            _keys(child, found)
    elif isinstance(node, list):
        for child in node:
            _keys(child, found)
    return found


@pytest.mark.parametrize("profile", [name for name in PROJECTION_SURFACES if name not in LIVE_DOJO])
def test_declared_result_and_input_fields_do_not_collide(profile):
    surface = surface_of(profile)
    declared = _keys(dict(surface.result_schemas), set()) | _keys(list(surface.definitions), set())
    colliding = {key for key in declared if _matches_forbidden_fragment(key.lower())}
    assert colliding <= VERIFIER_EXEMPT_KEYS, sorted(colliding - VERIFIER_EXEMPT_KEYS)


@pytest.fixture
def staged(tmp_path, fake_sdk_candidate_boundary):
    instance = sparse_session(tmp_path, contact_approach_route="global_precontact_then_local")
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("profile", LIVE_DOJO)
def test_projected_dojo_receipts_pass_the_privileged_field_audit(staged, tmp_path, profile):
    surface = surface_of(profile)
    context = ReceiptProjectionContext(last_action_id=None, last_action_public_name=None,
                                       surface_supports=surface.supports)
    receipts = [
        trajectory_to(staged, tmp_path, {"left": [0, 0, 0.7]})["receipt"],
        staged.execute(v3(staged, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"],
        trajectory_to(staged, tmp_path, {"left": [0, 0.05, 0.75], "right": [0, -0.05, 0.75]})["receipt"],
    ]
    for receipt in receipts:
        projector = resolve_receipt_effect_projector(receipt["schema"])
        assert projector is not None
        projected = projector(receipt, context)
        # Native delivers the envelope as the next observation's last_action_effect.
        delivered = _as_delivered({"last_action_effect": projected}, surface)
        assert _scan(delivered, surface) is None, receipt["action"]
