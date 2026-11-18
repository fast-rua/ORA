# grasp_plan

`tool.run(api, args)` plans for the surface's grasp compiler, per named arm on a
dual-arm surface. A `single_arm_flat_targets_v1` surface takes no `arm`, plans for
its right arm and publishes no arm.
It reads the published observation and granted target, requests the full CGN
pool, then applies the tool's topdown/hybrid preference, compiler admission,
lift bound and IK/jaw-headroom ordering. It returns the existing result schema,
contact sheet and global public-scene review. No action is submitted.

A grip-site compiler (the Panda's, the Robotiq TCP's) publishes its whole identity
and the grip-site handoff of `lib.compilation`. `grasp.selection` in the tool
context names a contact-first hybrid, the Panda's: top-down candidates only when the
learned pool is empty, and a target too sparse for that family costs only that family.
A context without the key, every RoboDojo and sealed one, is top-down first.

`_scene`, `_proposals`, `_topdown`, `_height`, `_learned`, `_reach`, `_jaw`,
`_selection`, `_lift`, `_materialize` and `_records` are local implementation
helpers. Task preferences stay here; lib holds explicit pure computations.
Dependencies are only toolkit interface/lib/tools and standard numeric Python.
No host, path, service client, simulator object or ground-truth method is passed in.

Replay recomputes perception and pure geometry. Each IK request must match the
next recorded simulator request; a changed target or missing request fails with
`NotRecorded`. Optional absent review pixels are reported as unavailable.
Query cache reuse is live-only; historical query manifests are still readable
as inputs to `grasp_execute`. Same process; this is not a security sandbox.
