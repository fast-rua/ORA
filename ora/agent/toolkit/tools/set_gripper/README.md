# set_gripper

Open or close either or both arms through `api.act.execute_action`. Where the
backend executes joint trajectories (`motion.joint_trajectory`), the directive is
sent in the rate-limited form of the call's motion profile (`move_pose/_profile.py`;
without one `tool.CLOSE` / `RELEASE` / `OPEN`: 2 openings/s, 5 mm lead and an arm
settle for a close; profile v2: full drive with a 5 mm hold lead, a velocity first
reading and a 2-tick stall window) with budget 30 (`budgets.py`);
otherwise the released open/close form with budget 20. Optional `target_ref` remains an agent annotation, never
proof of attachment. State holds dispatch-time intent; a zero-spend receipt
restores the previous intent. Replay needs recorded state/action evidence; it
never executes an online action.
