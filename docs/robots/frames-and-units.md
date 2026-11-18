# Frames and units

For each environment ORA drives: the world frame, the point a pose command
moves, the gripper axes, quaternion order, units, cameras, depth and control
timing. The values match the robot manuals the model reads (for example
[ARX X5](https://fastrua.ai/ora/docs/robots/embodiments/arx-x5/)) and the default tool surfaces. A
value marked **not published** is one ORA does not give the policy. For the message shapes that carry these values, see
[Policy I/O schema](../reference/policy-io.md). For installation, see the
environment pages for [LIBERO](../environments/libero.md),
[RoboTwin](../environments/robotwin.md) and [RoboDojo](../environments/robodojo.md).

## Conventions shared by every environment

| Quantity | Convention |
|---|---|
| Positions | World frame, metres |
| Orientation | Quaternion `[x, y, z, w]`; normalized on input (norm 0.98–1.02 accepted, then renormalized) |
| Measured pose | `pose_xyz_xyzw`: 7 numbers, position then quaternion |
| Angles | Radians in canonical records, receipts and controllers. On the three default robot surfaces (`policy_angles_degrees_v1`) the copy the model reads states every angle in degrees. Inputs named `rotation_angle_deg` / `angle_deg` are degrees; `rotation_rotvec_rad` / `rotation_vector_rad` stay radians |
| Pixels | OpenCV, origin top-left, `u` right, `v` down, native RGB resolution |
| Camera frame | OpenCV: `x` right, `y` down, `z` forward |
| Intrinsics | `intrinsic_cv`, 3x3, maps camera coordinates to native pixels |
| Extrinsics | `extrinsic_cv`, 4x4, **world-to-camera**; `cam2world_cv` is its inverse, computed in plain Python so its digits do not depend on the host's BLAS |
| Depth | float32 array, metres, optical-axis `z` distance (not ray length); delivered on demand to tools, not inlined each step |
| Calibration delivery | Per observation, for every camera, fixed or moving |

Each controlled point also has two local axes: **approach** (the direction the
fingers point) and **jaw closing line**. The measured state carries their
current world directions (`axes.approach_world`, `axes.closing_world`), so a
policy does not need to rotate local axes itself.

## LIBERO / LIBERO-Pro (Franka Panda)

| Item | Value |
|---|---|
| Robot | Fixed-base Franka Panda (MountedPanda, PandaGripper), one 7-DoF arm, public arm id `right` |
| World frame | "LIBERO world": right-handed, `+Z` up, metres. Origin location and `X`/`Y` directions: **not published** |
| Robot base pose | Owned by the runtime and used by the planner; **not published** to the policy |
| Action envelope | x `[-0.60, 0.60]`, y `[-0.50, 0.50]`, z `[-0.10, 1.30]` m (rejection bound, not reach) |
| Controlled point | `right_grip_site` (robosuite `grip_site`), the grip centre, 0.097 m along local `+Z` from the end-effector root. Commands already target it |
| Local axes | Approach `[0, 0, 1]` (local `+Z`, out of the gripper); jaw closing line `[1, 0, 0]` (local `+X`). Top-down means local `+Z` toward world `-Z` |
| Gripper command | Named directive `open` / `close` on the default surface |
| Gripper measurement | Two finger displacements (m); `measured_aperture_mm` = sum of their magnitudes x 1000; `measured_opening_normalized` = mean magnitude / 0.04, clipped to `[0, 1]`, 0 closed, 1 open. Commanded opening is shown beside it |
| Maximum aperture | 0.08 m jaw separation |
| Cameras | `agentview`: fixed RGB-D. `wrist` (source `robot0_eye_in_hand`): mounted on the gripper, moves with it. Main camera `agentview` |
| Image size | 256 x 256 (rendered rows are flipped to top-left order before publication) |
| Depth | Metric depth converted from the renderer's z-buffer; `valid_depth_range_m` (near, far) is recorded per camera |
| Control rate | One control step = 25 physics steps of 2 ms = 50 ms (`control_freq` 20) |
| Per command | `wait_ticks`: 1–20 control steps (up to 1.0 s). Planned motion: up to 4000 execution ticks and 2000 waypoints |
| Arrival tolerance | 0.005 m, 0.05 rad |

## RoboTwin (ALOHA-AgileX)

| Item | Value |
|---|---|
| Robot | Fixed-base dual-arm ALOHA-AgileX, two 6-DoF arms `left`, `right` |
| World frame | World metres. Origin and axis directions: **not published** to the policy. ORA's own task reviews, which the policy does not see, take world `+Z` as up and negative `Y` as nearer the robot |
| Action envelope | x `[-0.6, 0.6]`, y `[-0.4, 0.4]`, z `[0.7, 1.45]` m |
| Controlled point | `left_tcp` / `right_tcp`: grip-centre TCP, 0.12 m along local `+X` from the `link6` flange. Commands already target it |
| Local axes | Approach `[1, 0, 0]` (local `+X`); jaw closing line `[0, 1, 0]` (local `+Y`) |
| Native quaternion | The simulator's `[w, x, y, z]` is converted to `[x, y, z, w]` before publication |
| Gripper command | `open` / `close` per arm; internally open maps to 1, close to 0 |
| Gripper measurement | `gripper_opening_m`: measured jaw separation in metres; `gripper_state` uses a normalized 0 (closed) to 1 (open) scale that is not an aperture. No single maximum aperture is asserted for the tapered fingers |
| Cameras | `head_camera`, `front_camera`: fixed RGB-D. `left_camera`, `right_camera`: mounted on the grippers, move with them. Main camera `head_camera` |
| Image size | Set by the upstream task configuration; **not published** in ORA |
| Depth | Simulator depth in millimetres, divided by 1000 and stored as metres |
| Control rate | Physics step 4 ms; one tick = 10 physics steps = 40 ms. A policy-level control rate is **not published** |
| Per command | `wait_ticks`: 1–20 ticks (up to 0.8 s). Planned motion: up to 20000 physics steps (global) or 1280 (local) |
| Arrival tolerance | 0.005 m, 0.05 rad, checked at the TCP |

## RoboDojo (ARX X5)

| Item | Value |
|---|---|
| Robot | Two fixed-base ARX X5 arms, 6-DoF each, `left`, `right` |
| World frame | Episode-local Isaac Sim frame, metres; `+X` red, `+Y` green, `+Z` blue and up. Robot on the table's `-Y` side, so `+Y` runs from the robot to the far edge |
| Scene | Table 1.4 m x 1.1 m, top near z = 0.76 m (0.765 m in the workspace profile), spanning x `[-0.7, 0.7]`, y `[-0.6, 0.5]`. Arm bases at (±0.3, −0.45, 0.765); `link6` reset poses at (±0.3, −0.352, 0.922); head camera at z = 1.308 m |
| Action envelope | x `[-0.75, 0.75]`, y `[-0.65, 0.55]`, z `[0.72, 1.40]` m |
| Controlled point | `left_link6` / `right_link6`: the wrist flange, offset 0. Not the fingertips: the nominal jaw contact centre is 0.1501 m along the approach axis |
| Local axes | Approach `[1, 0, 0]`; jaw closing line `[0, 1, 0]`; local `Z` perpendicular to both |
| Native quaternion | Simulator order is `[w, x, y, z]` and never policy-visible |
| Gripper command | Range `[0, 1]`, 0 closed, 1 open; default surface uses `open` / `close`. An explicit command moves the drive target at most 0.2 (normalized) from the current encoder reading |
| Gripper measurement | `measured_opening_normalized`: mean of both finger joints on the command scale (empty close about 0.185, full open 1). `measured_aperture_mm` is `null`: physical pad aperture is uncalibrated. Nominal full-open pad separation 88.8 mm |
| Cameras | `head` (source `cam_head`): fixed RGB-D. `left_wrist`, `right_wrist`: mounted on the arms, move with them. Main camera `head` |
| Image size | **Not published** in ORA |
| Depth | Metres, `distance_to_image_plane`; `finite_depth_fraction` recorded per camera |
| Control rate | Physics 4 ms; 25 Hz control; one native action = 10 physics steps |
| Per command | A move may use up to 250 native actions of the task's native-action allowance, other commands 20 or 30; the receipt reports how many were used, a refused command uses none |
| Arrival tolerance | 0.01 m, 0.10 rad, gripper 0.03 (normalized); published each step in `controller_feedback` |

## Synthetic

Synthetic has no physics and no official verdict (`official_success: null`). It
checks action submission and observation sequencing.

| Item | Value |
|---|---|
| World frame | World metres, quaternion `[x, y, z, w]`; origin and axes **not published** |
| End effectors | Observation carries `left` and `right`, each starting at pose `[0, 0, 1, 0, 0, 0, 1]` with gripper 1.0. The target-state manual treats `right` as the single controllable grip site |
| Action envelope | Quickstart: x `[-0.6, 0.6]`, y `[-0.4, 0.4]`, z `[0.7, 1.45]` m |
| Gripper | Number in `[0, 1]`, 1 open, 0 closed |
| Cameras | `runtime.policy_cameras`, default one fixed `head_camera`. The target-state manual describes fixed `agentview` and moving `wrist` |
| Images and depth | 4 x 3 RGB test pattern; constant depth 1.0 m; identity intrinsics and extrinsics |
| Control rate | **Not published** (no simulated time). The quickstart ends after one accepted action |

## Cross-environment comparison

| | LIBERO / LIBERO-Pro | RoboTwin | RoboDojo | Synthetic |
|---|---|---|---|---|
| Arms | 1 (`right`) | 2 | 2 | 2 in state, 1 controllable |
| Controlled point | grip site (grip centre) | TCP, flange + 0.12 m | `link6` flange | grip site |
| Approach / jaw line | `+Z` / `+X` | `+X` / `+Y` | `+X` / `+Y` | not published |
| World up | `+Z` | `+Z` (maintainer notes only) | `+Z` | not published |
| Quaternion (policy) | xyzw | xyzw | xyzw | xyzw |
| Gripper measurement | aperture mm and normalized opening | jaw separation m | normalized encoder, no aperture | normalized value |
| Main camera | `agentview` | `head_camera` | `head` | `head_camera` |
| Wrist cameras | `wrist` | `left_camera`, `right_camera` | `left_wrist`, `right_wrist` | none by default |
| Native depth source | z-buffer to metres | mm to metres | metres | constant |
| Time unit | 50 ms control step | 40 ms tick | 40 ms native action | none |
| Arrival tolerance | 5 mm / 0.05 rad | 5 mm / 0.05 rad | 10 mm / 0.10 rad | not published |

## Silent mistakes

!!! warning "Differences that break code written for one environment"
    - **The controlled point is not the same physical point.** A RoboDojo pose
      moves `link6`, about 150 mm behind the jaw contact centre. A RoboTwin pose
      moves a TCP already 0.12 m ahead of the flange; LIBERO moves the grip
      centre. Code that adds a fingertip offset on every robot double-counts it on
      two of them.
    - **Approach axes differ.** Panda approaches along local `+Z`; ALOHA and
      ARX approach along local `+X`. A "top-down" quaternion copied from LIBERO
      points the RoboTwin or RoboDojo gripper sideways. Use the published
      `approach_world` and `closing_world`.
    - **Gripper numbers are not apertures by default.** Only LIBERO
      (`measured_aperture_mm`) and RoboTwin (`gripper_opening_m`) publish a
      measured separation, in different units. RoboDojo's 0.185 is an encoder
      floor, not 0 mm.
    - **Internal gripper scales run in opposite directions.** The measured
      opening is 0 closed / 1 open everywhere it is published, but the Panda
      motion profile writes open = 0.0 and close = 1.0. Send the named `open` /
      `close` directives, not numbers.
    - **Angle units change between what you send and what you read.** Default
      surfaces read back degrees, accept `rotation_angle_deg` in degrees, and
      keep `rotation_rotvec_rad` in radians.
    - **`extrinsic_cv` is world-to-camera.** Back-projection needs
      `cam2world_cv`. RoboTwin and Synthetic calibration files also contain
      `cam2world_gl` (OpenGL axes); do not mix it with OpenCV pixels.
    - **A "tick" is not one duration.** 50 ms on LIBERO, 40 ms on RoboTwin and
      RoboDojo, and on RoboDojo each tick also spends the task's native-action
      allowance.
    - **Workspace envelopes differ in height.** LIBERO z starts at −0.10 m;
      RoboTwin and RoboDojo start near the table at 0.70–0.72 m.

## Related pages

- [Policy I/O schema](../reference/policy-io.md): the messages that carry these values.
- [How ORA works](../how-ora-works.md) and the [evaluation protocol](../evaluation/protocol.md).
- [Public interfaces](../reference/public-interfaces.md) and
  [runtime lifecycle](../reference/runtime-lifecycle.md).
