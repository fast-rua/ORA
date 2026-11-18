"""move_pose takes an orientation as world approach and jaw-line directions on the RoboDojo surface.

RoboDojo g012: organize_table, stack_bowls and arrange_largest_number
each meant a downward approach and hand-computed ``[0, -0.7071, 0, 0.7071]``,
whose link6 local +X (the approach axis) points world +Z. Under
``pose_orientation_from_world_axes_v1`` the ``world_axes`` orientation mode
takes ``approach_world`` and ``jaw_line_world``; this module holds what the Dojo
surface advertises. The compilation is the toolkit's
(``toolkit/tools/move_pose/_axes.py``, ``tests/ora/agent/toolkit/tools/move_pose/test_tool.py``).
"""
from __future__ import annotations

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.schemas.world_axes import CAPABILITY, MODE

DOJO = "robodojo_placement_toolkit_v5"


# --------------------------------------------------------------------------
# Composition: the Dojo surface declares the mode and the echo.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dojo():
    return surface_of(DOJO)


def _arm_schemas(surface):
    move_pose = next(item for item in surface.definitions if item["name"] == "move_pose")
    return move_pose, [move_pose["inputSchema"]["properties"]["targets"]["properties"][arm]["properties"]
                       for arm in ("left", "right")]


def test_the_dojo_move_pose_advertises_world_axes_with_its_companion_fields(dojo):
    assert dojo.supports(CAPABILITY)
    move_pose, arms = _arm_schemas(dojo)
    for properties in arms:
        assert properties["orientation_mode"]["enum"][-1] == MODE
        assert "world_axes REQUIRES approach_world AND jaw_line_world" in properties["orientation_mode"]["description"]
        for name, companion in (("approach_world", "jaw_line_world"), ("jaw_line_world", "approach_world")):
            field = properties[name]
            # optional_argument_null_as_omitted_v1 (bound after this binder) adds null.
            assert (field["type"], field["minItems"], field["maxItems"]) == (["array", "null"], 3, 3)
            assert field["description"].startswith(f"REQUIRED with orientation_mode=world_axes, together with {companion}.")
    assert "orientation_mode (hold_current, absolute_quaternion, axis_angle_delta, world_axes)" in move_pose["description"]
    assert "world_axes takes approach_world and jaw_line_world;" in move_pose["description"]
    assert "target_axes_world" in move_pose["description"]


# --------------------------------------------------------------------------
# Runtime on the fake Dojo plant.
# --------------------------------------------------------------------------
