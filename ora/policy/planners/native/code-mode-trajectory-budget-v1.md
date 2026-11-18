# What a Program Spends on This Surface (code mode)

The robot functions here execute their actions synchronously inside your
program, and every one of them spends from the budgets below.

- **Native actions.** Each action a function runs asks for at most the native
  actions its description states for it, as when you call it directly: a move
  (`move_pose` at any speed, `move_to`, `rotate_wrist`, `rotate_pitch`,
  `recover_pose`, `return_home`) runs as one smooth joint trajectory, a jaw
  command as one rate-limited command, and each of `grasp_execute`'s stages
  keeps its own limit. Each action is charged only the native actions it used;
  a refused one consumes none.
- **One allowance per program run.** All functions of one script or command
  share one allowance: up to 250 native actions for each action it may run
  (its action limit, which the script tool's description states, or half the
  robot decisions left when that is fewer), and never more than the episode
  has left. The `remaining_budget` a function returns is what this allowance
  has left; a function's own total, such as `grasp_execute`'s, does not apply
  inside a program.
- **A move fits what is left.** A move is planned to what it can spend: its
  own budget, what the episode has left and what this program's allowance has
  left. A move that needs more covers the part of its path that fits, ends at
  rest and is reported as cut short (`r.cut_short`, with `r.target_error_m` the
  distance left to the target; `motion.budget_limited` names the limit), never
  as reached; call it again with the same target, in a later program if this
  one's allowance is spent.
- **Robot decisions.** A script or command that consumed any native action is
  one robot decision, however many functions and actions it ran. One that only
  read, or whose every action was refused, spends none.
- **Undo is per call.** `recover_pose` undoes each named arm's latest
  arm-moving function call, one call of your program and not the whole
  program, back to where that call started; call it again to undo the call
  before.
- **`grasp_execute` runs whole.** Each of its stages that moves the arm or the
  jaws counts as one of the program's actions (five for a prepared grasp). It
  runs all of them even when fewer of the program's actions are left, and the
  program can then call no other world-changing function. Never arrange for
  `grasp_execute` to stop partway -- for instance right after its jaws open, to
  get a standalone release -- or to imitate another function. Open or close
  the jaws on their own with `set_gripper`.
