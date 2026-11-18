"""Offline parent-call budgets for the toolkit's tools.

These limits are implementation configuration, never model-supplied arguments: a call that supplies one is refused.
RSI may change them offline in a new revision. They are large on purpose: a call finishes what it was asked, and what
ends a move early is what the episode has left (``move_pose._smooth.allowance``), a stop or a refusal, not its tool.

A move runs one joint trajectory at any speed, so its budget is what one trajectory may use
(250 native actions, 10 s: about 3 m at normal speed, 1 m at careful, 2.4 m of
retraced path). One call can therefore spend 250 of the episode; its plan and a dry run say what it will cost. The
budgets were 50 and 20 until 2026-11-02: in 18 RoboDojo runs 34 moves stopped there short of their target, the longest
needing 137, and the model called again for the rest. A prepared grasp runs at most ``GRASP_ARM_STAGES`` arm stages
and ``GRASP_JAW_STAGES`` jaw commands, so its budget covers each at its limit; recorded grasps spent 43 at the median
and 81 at most. set_gripper is a close from fully open at 2 openings/s with an arm settle and the stall window (20 at
most in 425 recorded commands), and wait_ticks waits the ticks its caller names (1..20).
"""

from types import MappingProxyType

#: The most one move may spend: one joint trajectory (``joint_trajectory.MAX_TRAJECTORY_NATIVE_ACTIONS``).
MOVE_BUDGET = 250
SET_GRIPPER_BUDGET = 30
GRASP_ARM_STAGES = 3
GRASP_JAW_STAGES = 2

NATIVE_ACTION_BUDGETS = MappingProxyType({
    "move_to": MOVE_BUDGET, "move_pose": MOVE_BUDGET, "set_gripper": SET_GRIPPER_BUDGET,
    "recover_pose": MOVE_BUDGET, "return_home": MOVE_BUDGET,
    "rotate_wrist": MOVE_BUDGET, "rotate_pitch": MOVE_BUDGET, "wait_ticks": 20,
    "grasp_execute": GRASP_ARM_STAGES * MOVE_BUDGET + GRASP_JAW_STAGES * SET_GRIPPER_BUDGET,
})


def native_action_budget(name: str) -> int:
    """Return this revision's fixed parent budget; read-only/lifecycle tools cost 0."""
    return NATIVE_ACTION_BUDGETS.get(name, 0)
