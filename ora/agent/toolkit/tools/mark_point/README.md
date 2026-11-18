# mark_point

Batch one to eight current RGB-D pixels. Named measurements persist in
`tool.marks` (a JSON name-to-record map); same-pixel reuse and cross-view 25 mm
verification match the original tool. Reset clears this map. Reads use only
public observation bytes, not simulator ground truth. Replay reads recorded
pixels and state; absent evidence fails. Output overlays are pure computation.
A view the publication lacks fails each entry, as an unmeasurable pixel does; a
point-cloud view is refused, since the batch form measures camera pixels only.
A camera with no extrinsic refuses the call at the first pixel its depth accepts.
Where the environment plans the moves, the annotated view and the sheet are also
published as `observation_overlay` artifacts, which the Viewer shows; a RoboDojo
context has no `motion.environment_plans`, and the sheet stays in the result alone.
