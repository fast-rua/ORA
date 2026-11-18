"""The live RoboDojo default surface carries several pose-tool capabilities at once.

Revisions 14-17 of the retired ``robodojo_placement_closed_skills_v1`` were
developed on three parallel branches, each against revision 13:

- 14 ``policy_abort_requires_infrastructure_evidence_v1`` (finish_episode abort admission);
- 15 ``reach_nearest_orientation_v1`` (unsolved ``reach_along_request`` gains
  ``nearest_reachable_orientation``);
- 16 ``observed_clearance_reference_v1`` (``observed_clearance`` per pose target);
- 17 ``pose_orientation_from_world_axes_v1`` (``move_pose`` ``world_axes`` mode
  and ``target_axes_world`` per pose target).

Its successor ``robodojo_placement_toolkit_v5`` (the RoboDojo default)
carries 15 and 17 with the v2 observed clearance and the v2 finish contract.
Each capability's own tests pin it alone. This pins the combination: the
three ``move_pose`` help sentences coexist in revision order.
"""
from __future__ import annotations

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.schemas.world_axes import COMPANIONS, MODE

DOJO = "robodojo_placement_toolkit_v5"
HELP_MARKERS = (  # revision order: 15, 16, 17
    "reach_along_request also carries nearest_reachable_orientation",
    "compiled_pose.targets.<arm>.observed_clearance (also on move_to",
    "world_axes compiles approach_world",
)


@pytest.fixture(scope="module")
def dojo():
    return surface_of(DOJO)


def _definition(surface, name):
    return next(item for item in surface.definitions if item["name"] == name)


def test_move_pose_carries_the_world_axes_mode_and_three_help_sentences_in_revision_order(dojo):
    definition = _definition(dojo, "move_pose")
    for arm in ("left", "right"):
        target = definition["inputSchema"]["properties"]["targets"]["properties"][arm]["properties"]
        assert MODE in target["orientation_mode"]["enum"]
        assert all(name in target for name in COMPANIONS)
    description = definition["description"]
    positions = [description.find(marker) for marker in HELP_MARKERS]
    assert all(description.count(marker) == 1 for marker in HELP_MARKERS)
    assert -1 not in positions and positions == sorted(positions)
    assert f"orientation_mode (hold_current, absolute_quaternion, axis_angle_delta, {MODE})" in description
