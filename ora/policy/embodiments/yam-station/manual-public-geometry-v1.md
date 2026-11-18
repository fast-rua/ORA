# I2RT YAM compiled embodiment v2

## Identity and topology

Fixed-base I2RT YAM: left 6-DoF arm, right 6-DoF arm; parallel-jaw grippers.

## Controlled points

`left_tcp` is the controlled reference, not a fingertip. Its local offset is [0.0,0.0,0.14465] m from the declared manipulator flange; commands already target it. Do not add this offset again.
`right_tcp` is the controlled reference, not a fingertip. Its local offset is [0.0,0.0,0.14465] m from the declared manipulator flange; commands already target it. Do not add this offset again.

## Frames and units

Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.

## End-effector axes

`left_tcp` local approach [0, 0, 1]; jaw-closing line [1, 0, 0]. Current world directions are supplied with measured state.
`right_tcp` local approach [0, 0, 1]; jaw-closing line [1, 0, 0]. Current world directions are supplied with measured state.
Local +approach points out of the gripper, from the palm toward the fingertips.

## Motion capabilities

The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.

## Gripper semantics

`left_gripper`: nominal maximum jaw separation 0.0939 m.
At reference finger qpos [0.04695, 0.04695] m, the fingertips end 2.2 mm beyond the controlled point along +approach, and the fingers reach back 58 mm from it toward the wrist.
The separate pads run from 31.9 mm behind to 2.2 mm beyond the controlled point along +approach; pad and whole-finger extrema differ. Each pad is 10 mm thick along the jaw-closing line and 40 mm wide along local Y.
`right_gripper`: nominal maximum jaw separation 0.0939 m.
At reference finger qpos [0.04695, 0.04695] m, the fingertips end 2.2 mm beyond the controlled point along +approach, and the fingers reach back 58 mm from it toward the wrist.
The separate pads run from 31.9 mm behind to 2.2 mm beyond the controlled point along +approach; pad and whole-finger extrema differ. Each pad is 10 mm thick along the jaw-closing line and 40 mm wide along local Y.

## Sensors and cameras

`overhead`: fixed RGB-D; use calibration from the same observation.
`front`: fixed RGB-D; use calibration from the same observation.
`wrist_left`: arm-mounted moving RGB-D; use calibration from the same observation.
`wrist_right`: arm-mounted moving RGB-D; use calibration from the same observation.

## Preservation and evidence limits

Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.

## Dynamic facts and provenance

Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only.
