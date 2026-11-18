# Franka Panda with Robotiq 2F-85 compiled embodiment v2

## Identity and topology

Fixed-base Franka Panda with Robotiq 2F-85: right 7-DoF arm; parallel-jaw grippers.

## Controlled points

`right_tcp` is the controlled reference, not a fingertip. Its local offset is [0,0,0.131] m from the declared end effector root; commands already target it. Do not add this offset again.

## Frames and units

Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.

## End-effector axes

`right_tcp` local approach [0, 0, 1]; jaw-closing line [1, 0, 0]. Current world directions are supplied with measured state.
Local +approach points out of the gripper, from the palm toward the fingertips.

## Motion capabilities

The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.

## Gripper semantics

`right_gripper`: nominal maximum jaw separation 0.085 m.
With the jaw fully open, the fingertips end 18.1 mm beyond the controlled point along +approach, and the fingers reach back 86.764 mm from it toward the wrist.
The separate pads run from 19.9 mm behind to 18.1 mm beyond the controlled point along +approach; pad and whole-finger extrema differ. Each pad is 19.159 mm thick along the jaw-closing line and 22 mm wide along local Y.

## Sensors and cameras

`agentview`: fixed RGB-D; use calibration from the same observation.
`wrist`: arm-mounted moving RGB-D; use calibration from the same observation.

## Preservation and evidence limits

Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.

## Dynamic facts and provenance

Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only.
