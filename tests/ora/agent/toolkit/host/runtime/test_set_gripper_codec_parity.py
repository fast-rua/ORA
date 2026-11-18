"""One run emits one action codec, whichever backend the surface is for.

`set_gripper` routed to its v3 implementation only when a surface declared
`dual_arm_sparse_targets_v1`. The active RoboTwin surface declares that; the
active LIBERO surface declares `single_arm_flat_targets_v1` instead. So on
LIBERO `set_gripper` fell through to the flat path and emitted
`ora-motion-action.v1` while `open_gripper` on the same surface
emitted v3 -- two codecs inside one run.

The two helpers that read the surface's declared topology already existed and
already handled both shapes. The routing gate was the only thing that did not.
"""
from __future__ import annotations


from ora.agent.composition.surface.tool_surface import resolve_tool_surface

ACTIVE = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}


def test_the_two_active_surfaces_declare_different_topology_capabilities():
    """The premise. Widening one gate is only safe because these differ."""

    robotwin = resolve_tool_surface(ACTIVE["robotwin"])
    libero = resolve_tool_surface(ACTIVE["libero_pro"])
    assert robotwin.supports("dual_arm_sparse_targets_v1")
    assert not robotwin.supports("single_arm_flat_targets_v1")
    assert libero.supports("single_arm_flat_targets_v1")
    assert not libero.supports("dual_arm_sparse_targets_v1")
