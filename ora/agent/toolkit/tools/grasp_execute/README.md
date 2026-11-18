# grasp_execute

`tool.run(api, args)` commits one selected candidate, derives public localization
and retention reference, asks the precommit motion probe, and submits one skill
sequence action. It preserves the executor's existing ledger, commitment and
action schemas. The executor and worker remain outside this package.

Local helpers resolve granted candidates, measure selected public pixels,
publish localization and choose a common admitted jaw representative. Inputs
are boundary-frozen configuration, public query/mask blobs and observations;
outputs are prepared blobs and the original tool result. Dependencies are only
toolkit interface/lib/tools and standard numeric Python, with no file/network API.

Replay reads the recorded probe, or fails `NotRecorded`. Submission compares the
original action **and** the ledger/commitment content hashes at the cursor;
a different preparation is `Divergence`. Replay has no live control publisher.
The trusted live adapter validates and writes the existing control files before
publishing the single action. Same-process execution is not process isolation.


On synchronous toolkit-v2, `_stages.py` instead performs five primitive action
calls, each followed by a published state. `_verify.py`, `_jaw_hold.py` and
`_lift_evidence.py` retain the released measurement predicates using public
state and RGB-D inputs. Verification is read-only and does not render. The
parent budget controls total spend: an arm stage may spend what it has left, and
one the budget cuts short stops the sequence as a refusal does. V1
continues to use the existing prepared composite path described above.

Synchronous results also carry private `_runtime.skill_feedback` metadata for
the host's optional next-observation projection. Live dispatch removes it from
the public result. The metadata supplies the committed sequence, selected arm
and the actual read-only verify measurement; it supplies no image path or
execution authority. The owner checks the action receipts and binds the
post-close image source. The observation receipt switch controls both the
aggregate summary and these historical images.
