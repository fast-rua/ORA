# recover_pose

On the toolkit trajectory surfaces (v5 on RoboDojo, `move_pose._smooth.available`)
recover_pose takes `arms` and undoes each arm's latest not-yet-undone
arm-moving tool call: `_trajectory.run_retrace` reads the top of
`exec.recovery.<arm>.undo` (the last 5 calls: identity, start joints and pose,
measured path of all their trajectories), plans the measured path from where
the arm stands back to that call's start in lib (`lib.motion_plan.recovery`)
and runs it as one `follow_joint_trajectory` marked `recovery` (the host checks
the restated stack top; a retrace that runs every knot pops the entry, so the
next call undoes the one before; no redo). Jaws are held. Normal speed, budget
50, settle 2. The result's `recovery_progress.undo` names the undone call,
whether it is complete or `path_left_fraction`, and `undoable_calls`. No
fallback: no recorded call, an empty stack or an arm off the top entry's path is
a zero-spend refusal (`retrace_not_recorded`, `retrace_nothing_to_undo`,
`retrace_arm_off_path`) that names return_home and move_pose instead.

The released-primitive surface (v2) calls it with `targets` (`previous`/`home`): read
`exec.recovery.<arm>.previous/home` and submit that exact anchor as the
environment's `recover_motion` (20 native actions). The owner rechecks it
before any worker dispatch; a stale or fabricated target is a zero-spend
refusal. Replay requires recorded anchors; missing state cannot be invented.
RoboDojo's worker refuses `recover_motion` now; the form stays for tool-level
replay of runs recorded on it.

`_trajectory.py` also holds return_home's code (`run_home`).

`dry_run` (code mode, `library: extended`): the `arms` form plans its retrace exactly as the real call does and returns
`tools/_dry_run.recovered` (the call it would undo, the method, the predicted end pose) instead of executing; the
released `targets` form, which the environment executes, has none (`dry_run_unavailable`). `return_home` shares the
path.
