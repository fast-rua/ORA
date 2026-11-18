# robot_submit_action

`robot_runtime_v2`'s `robot.submit_action`: one `ora-action.v4` `target_state` for one arm, with an optional pose and gripper
opening, bound to the current observation. The tool checks the arguments and the workspace; the host validates the action,
persists it and closes the observation (`toolkit_robot.py`). A second call on one observation is refused `duplicate_action`.
A valid proposal is no receipt: the next observation reports what happened.
