# Recorded-provider tape replay

`ora toolkit provider-replay` serves a sealed run's recorded model replies as a
read-only tape and compares every request a candidate run sent with the source
run's. It never launches ORA, a simulator or a scheduler.

The old-tool treatment, which derived a candidate RunSpec for the tape
(`provider-replay spec`, the allocated `router-task` and
`scripts/dev/router-provider-replay`), was retired on 2026-11-01 with its replay
baseline `native_full_explicit_motion_direct_replay_v1`. The `toolkit` treatment
(`robodojo_toolkit_legacy_replay_v1` on `robodojo_placement_toolkit_v3`) was
retired on 2026-10-31. Sealed replays keep their recorded baselines and
definitions.

## Serve a tape

```bash
./scripts/ora toolkit provider-replay serve \
  "$SOURCE_RUN" --output "$TAPE_OUTPUT" --port 18080
```

It returns the original OpenAI response chunks, including tool-call IDs,
reasoning and usage, in their recorded order. It records incoming requests
outside the source run. A consumed reply is never reused after a disconnect.
Exhausted tapes and stream-mode mismatches are errors. Loopback is the default;
use `--listen` only when the operator deliberately needs another host to
connect. `--port 0` lets the OS choose a free port; `--ready-file` publishes the
bound endpoint atomically.

A fixed response tape does not rewrite old step or target references, so an
unchanged tape can fail to find a previously recorded reference; record that
divergence instead of mapping numbers.

## Compare two runs

```bash
./scripts/ora toolkit provider-replay compare \
  "$SOURCE_RUN" "$CANDIDATE_RUN" --output "$COMPARISON_OUTPUT"
```

This reads sealed or compressed request artifacts without changing either run.
Each request gets a JSON file with both model-call IDs, observation steps, raw
byte hashes and equality, every differing JSON field, image
count/order/label/reference checks and each paired image's PSNR and fraction of
pixels differing by more than 16. 48–52 dB is labelled as the M7.4b
rendering-noise reference; lower values are separate. That label is a measured
comparison range, not a causal proof or an exception to equality. Exact pixels
use `pixels_equal=true`, `psnr_db=null`. Unmatched requests and images are
explicit. Output directories must be new and outside both runs. Exit 1 means
any request bytes or count differ.

`comparison.json.policy_configuration.source/candidate` records each saved
RunPlan's full baseline reference (including its hash), `require_live_marker`,
the saved surface and `implementation_commit`. These are
comparison metadata, not a normalization or part of request equality. Missing
plans are reported as null. The report also carries each run's toolkit
protocol compatibility; a version difference is a warning, not a request
difference.
