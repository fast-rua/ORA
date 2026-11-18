"""RoboTwin's reach-check vocabulary: which IK answered, why it refused a pose, what it claims."""

from __future__ import annotations

#: The IK of the planner a global ``move_pose`` route plans through, in its world
#: (``backends/robotwin/shared_motion.py``).
IK_MODEL = "robotwin_reach_ik_depth_world_v1"
#: The preflight check whose passing detail binds a run's planner, and the robot file it plans from.
SHARED_PLANNER_CHECK = "shared_motion_planner"
ROBOT_FILE = "assets/embodiments/aloha-agilex/curobo_left.yml"
#: No solution held the pose within the arrival tolerance.
IK_FAILED = "ik_failed"
REASONS = (IK_FAILED,)
CLAIM_BOUNDARY = (
    "One IK solve per pose by the IK solver of the global planner move_pose uses, seeded from the measured joints "
    "or a chained query's solution, with joint limits, self-collision and a collision world built from the current "
    "depth images: the robot removed, the other arm at its measured joints, the space around the goal hand cleared, "
    "a held object moving with the hand, and space no camera sees counted as free. The solution's grip centre is "
    "checked against the arrival bound. It plans no trajectory and runs no plan validation, so move_pose can still "
    "refuse a pose reported reachable, and a finite seed search can miss a solution."
)

__all__ = ["CLAIM_BOUNDARY", "IK_FAILED", "IK_MODEL", "REASONS", "ROBOT_FILE", "SHARED_PLANNER_CHECK"]
