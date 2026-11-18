# Franka Panda compiled embodiment v2

## Identity and topology

Fixed-base Franka Panda: right 7-DoF arm; parallel-jaw grippers.

## Controlled points

`right_grip_site` is the controlled reference, not a fingertip. Its local offset is [0,0,0.097] m from the declared end effector root; commands already target it. Do not add this offset again.

## Frames and units

Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.

## End-effector axes

`right_grip_site` local approach [0, 0, 1]; jaw-closing line [1, 0, 0]. Current world directions are supplied with measured state.
Local +approach points out of the gripper, from the palm toward the fingertips.

## Motion capabilities

The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.

## Gripper semantics

`right_gripper`: nominal maximum jaw separation 0.08 m.
At reference finger qpos [0, 0] m, the fingertips end 9.249 mm beyond the controlled point along +approach, and the fingers reach back 44.4683 mm from it toward the wrist.
The separate pads run from 11.6 mm behind to 4.4 mm beyond the controlled point along +approach; pad and whole-finger extrema differ. Each pad is 8 mm thick along the jaw-closing line and 16 mm wide along local Y.

## Sensors and cameras

`agentview`: fixed RGB-D; use calibration from the same observation.
`wrist`: arm-mounted moving RGB-D; use calibration from the same observation.

## Preservation and evidence limits

Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.

## Dynamic facts and provenance

Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only.
