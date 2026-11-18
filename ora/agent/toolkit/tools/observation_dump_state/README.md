# observation_dump_state

`robot_runtime_v2`'s `observation.dump_state`: the policy-state snapshot of `step`, or the latest, with an inventory of its
cameras and its main image as a 512 px thumbnail. The host reads the snapshot (`toolkit_robot.py`) and the tool only shapes
it. It needs no open observation, so it still answers between a submitted action and the next observation.
