"""move_to, rotate_wrist and rotate_pitch carry the move_pose reach reference on RoboDojo only.

RoboDojo g009 stack_bowls issued 37 move_to against 6 move_pose, and two refused
lifts in its last round reported only planner_failed (E11G9-001). Under
pose_primitive_reach_along_request_v1 each primitive target is checked with
the read-only reach check on its own route (``toolkit/tools/move_pose/_reach.py``).
"""
from __future__ import annotations

from tests.support.agent.surface_catalog import surface_of
from ora.agent.toolkit.tools.move_pose._reach import PRIMITIVE_CAPABILITY as CAPABILITY

DOJO = "robodojo_placement_toolkit_v5"
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v2",)


def test_the_live_dojo_surface_declares_it_and_robotwin_does_not():
    dojo = surface_of(DOJO)
    assert dojo.supports(CAPABILITY)
    for profile in ROBOTWIN_LIVE:
        assert not surface_of(profile).supports(CAPABILITY)
