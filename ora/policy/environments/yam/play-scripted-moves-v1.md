## Moves you can already plan: one script

Each robot tool call you make on your own costs a full turn: the move runs,
then a new observation arrives before your next reply. A pick and place done
that way takes fifteen or more turns. When current evidence already fixes the
next several moves (where the object is, how to grasp it, where it goes), run
them as one `codemode` script instead, and look again only when the script
has finished or stopped.

Perceive and decide first, in ordinary tool calls: locate the object
(`vision_ground`, `vision_segment`, `mark_point`), choose the grasp
(`grasp_plan`), compute the poses (`compute`). Then write the script with the
fixed sequence, for example approach above, descend, close, lift, carry,
descend, open, retreat.

Inside the script, every robot tool returns a string; parse it with
`JSON.parse`. A world-changing call (`move_pose`, `move_to`, `set_gripper`,
`recover_pose`) returns only after the move has run, with:

- `receipt.execution_status`: `reached`, `not_reached` or `plan_rejected`;
- `receipt.failure_code`, and for a refused plan `receipt.refusal`;
- after `set_gripper`, `receipt.arms.<arm>.gripper_outcome` with `state`
  and `measured_aperture_mm`;
- `observation.proprioception`, the arms' state after the move (no images).

Check every call before the next one, and stop the script at the first
surprise with `return`, naming the step and the receipt fields that show it:

- A move is fine only when `execution_status` is `reached`.
- Closing on an object is expected to end `not_reached` with `state`
  `stalled_before_target` and `measured_aperture_mm` near the object's width.
  A full close (aperture near 0) means the grasp missed: stop there.
- After the lift and after the carry, check the arm's `gripper_opening_m` in
  `observation.proprioception` still matches the object's width. If it has
  closed, the object slipped: stop there.

Print a short line per step with `text(...)`, not whole receipts. Do not put
a decision that needs an image inside the script. After a script has moved the
robot, a direct world-changing tool call in the same turn is refused: end the
turn to see the result, or continue through another script.
