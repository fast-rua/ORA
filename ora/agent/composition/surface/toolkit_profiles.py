"""The toolkit surfaces: their tool cohorts and the baselines they default to.

Behavior never branches on these names: runtime code asks the resolved
surface for ``toolkit_only_v1``, ``toolkit_synchronous_execution_v1`` and
``toolkit_legacy_projection_v1``. This module is the record of which baseline
a RunSpec naming a toolkit surface gets. The surfaces v1, v3 and v4 were retired on 2026-10-31, v2 on 2026-11-01;
the remaining toolkit surfaces (v5, v6) are file-backed and list their tools in their own surface.json.
"""

#: The first migration wave: installed on every toolkit dispatch, including the
#: compatibility opt-in on the legacy surface (the tool cohort of the retired
#: robodojo_placement_toolkit_v1).
FIRST_WAVE_TOOLS = (
    "observe",
    "compute",
    "vision_ground",
    "grasp_plan",
    "grasp_execute",
    "move_to",
    "move_pose",
    "check_task",
    "finish_episode",
)
#: The baseline a toolkit-only RunSpec gets when it names none.
TOOLKIT_BASELINES = {
    "robodojo_placement_toolkit_v5": "robodojo_toolkit_v6",
    "robodojo_placement_toolkit_v6": "robodojo_toolkit_v7",
}
