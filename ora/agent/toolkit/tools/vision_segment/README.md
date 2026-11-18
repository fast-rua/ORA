# vision_segment

Independent positive pixel prompts on the current or a retained public image.
Selection stays tool-local: reserve one mask per answered point, allocate
surplus rank-major, merge near-identical masks by IoU, and bound inspection
images. Full service candidates remain at the data-interface boundary.

Runtime may reuse a verified immutable query artifact. Replay reads recorded
RGB-D and re-requests SAM3; historical responses are comparison evidence only.
Mask normalization, geometry and drawing are pure calculations. Missing
recorded images fail explicitly. This tool performs no physical action and has
zero native action budget. Public targets persist for later grasp/geometry
calls; there is no instance-local cache or semantic identity oracle.

The default toolkit_v2 surface keeps the released point-query description,
input fields and result shape, with only the common zero-budget input added.
