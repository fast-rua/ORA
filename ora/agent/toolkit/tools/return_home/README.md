# return_home

Each selected arm goes straight back to the episode reset joints
(`exec.recovery.<arm>.home`), whatever path it came by: link6 lifts up to
0.10 m straight up with its orientation held, then the joints unwind in the
first order (straight, fold-then-yaw, yaw-then-fold) whose timed path the SDK
collision spheres keep clear of the arm itself, the table top and the other
arm; without the lift when only that is clear. Planned in lib
(`lib.motion_plan.recovery`) and run as one `follow_joint_trajectory` marked
`recovery` (the host checks the restated home anchors; the previous anchor and
the retrace stacks are kept) at move_pose's fast speed, budget 250. Grippers
are held. The shared code is `recover_pose/_trajectory.py` (`run_home`).
Only where joint trajectories run (v5 on RoboDojo); elsewhere
`tool_unavailable`. Open ideas (not built): R0022 F15.

`dry_run` (code mode, `library: extended`): see recover_pose; the record names the lift, the unwind order and the predicted end pose.
