# Reading the Robot Model (code mode)

The robot functions here run on the toolkit's data interface, so a program can
read it directly as well.

- **You are choosing among poses and ask which the arm can hold:**
  `robot.query("reach_check", arm=..., poses=[...])`, up to 16 at once: inverse
  kinematics per pose, no motion planned, so not whether a move there will run.
- **You need a lower-level answer about the robot:** `robot.query("ik", ...)`,
  also `"fk"`, `"screen_path"`, `"probe_motion"`, `"align_plan"` and
  `"robot_state"`; `robot.docs("query_ik")` gives each one's arguments. A query
  moves nothing.
