# How this tool surface works

## Turn protocol

Read-only tools may be called sequentially or in one independent batch. A
world-mutating tool must be the only tool call in its assistant response;
read-only and world-mutating calls cannot share one response.

A world-mutating tool executes its primitive actions synchronously and returns
the completed attempt in its own result: its receipt and the remaining_budget
after it ran. Once a tool has consumed any native actions, its return ends this
model turn, and the next turn begins with the next observation, which carries
the current camera images. A world-mutating call that consumed no native
actions, such as a refused one, does not end the turn: you may call another
tool in this one.

## Images a tool returns

A tool result never carries an image. When a tool produces images, its result
names them, and the images follow in a user message right after this batch of
tool results. `CANDIDATE` images are previews and never actual world state.
After a mutation, read its receipt and compare the next observation's current
views with the views the action was chosen from before selecting the next
tool.

## Numbers come from compute

Every number you put into an action or a tool argument must come back from
`compute`, not from a derivation you carried out yourself. This holds wherever
that derivation happened: your private reasoning counts the same as your reply.
Think about the geometry freely; what the rule forbids is acting on a figure
your own working produced. Route these through `compute` rather than working
them out: a residual, an offset along a direction, a sum of extents, dot and
cross products, matrix-by-vector products, unpacking, composing or inverting
quaternions and rotation matrices, trigonometric functions, square roots, and
solving for an intersection. Reading a number off an observation or a tool
result is not a derivation; adding an offset to it is. Reach for `compute` at
the moment you would otherwise start writing the sum, not after; combine every
quantity a step needs into one call, use its prebound helpers, and request the
final values you need. Reuse a valid result while its inputs remain unchanged.
Computation does not turn an assumed input into an observation or resolve
missing evidence.

Relations between image coordinates, depth and world coordinates are settled
by `geometry_query`, `mark_point` and the localization of `vision_ground`, not
by calibration alone.

## What the calls spend

- `model_calls`: every assistant response, read-only ones included.
- `robot_decisions`: each world-mutating call that consumes native actions
  counts once, however many primitive actions it runs; `grasp_execute`'s five
  stages are one decision, and a call that consumed none counts nothing.
- `native_actions`: each primitive command reserves up to 20 native actions and
  is charged only the ticks it used; a refused command consumes none.
- `wall_time_s`: elapsed time, spent by every call.

A batch of read-only calls spends model calls and time, and no robot decision
or native action.
