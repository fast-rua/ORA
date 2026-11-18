# Data-interface protocol and replay compatibility (M7.6 / R0016 Q9)

`ora/contracts/toolkit_protocol.py` defines the single, manually numbered
`PROTOCOL_VERSION = 1`. The current readers declare support for `[1]`. This is
a semantic protocol number, not a source/content hash. Source hashes remain
evidence of what actually ran.

Every newly launched run has public `toolkit-protocol.json`, including runs
with `record_toolkit_io=false`. It is written before the Supervisor starts:

```json
{
  "schema": "ora.toolkit.protocol.v1",
  "protocol_version": 1,
  "recording_enabled": false,
  "methods": ["perception.*", "state.*", "planner.*", "act.*", "episode.*", "privileged.*", "artifacts.*", "compute.*", "context.*"],
  "implementation": {"harness_commit": "<run commit>"}
}
```

When recording is enabled, `native/toolkit-io/manifest.json` declares the same
protocol, and every `calls.jsonl` begin/end event carries `protocol_version`.
The request identity/hash and invocation ID formats do not change. Existing
logs and snapshots need no migration; source runs are never backfilled.

Replay results contain `protocol_compatibility`: `source_version`,
`current_version`, `supported_versions`, `status` (`matched`, `mismatch`,
`unknown`), all observed declarations in `evidence`, and `warnings` with the
potentially affected method families. Conflicting root/manifest/event versions
remain visible. Missing legacy versions produce `unknown`, not an assumed 1.
`protocol_matched` and `exact_replay_eligible` refer only to protocol agreement:
agreement is necessary, not sufficient, for an exact-replay claim. Physics,
source identity, service comparisons and sensor statistics have their own
evidence. Warnings are also copied to the result's top-level `warnings`.

A mismatch or missing version does **not** block execution with the current
decoder. A decodable `method_version` mismatch also warns: matching compares
all other semantic request fields exactly, while validating the stored
semantic hash against the original request, including its original version.
No action parameters, service identities, boundary identities or blob hashes
are normalized or waived. Unknown wire schemas without a decoder remain
unsupported. A protocol number alone cannot supply a missing decoder.

## Audit of replay gates

Paths below are relative to `ora/`. “Kept” checks are intentionally not version
admission. This audit includes the old diagnostic and new snapshot paths.

| Path / check | Before M7.6 | After M7.6 |
|---|---|---|
| `agent/toolkit/interface/journal.py`: request `method_version` | Only 1 accepted | Decodable requests accepted; warning in protocol audit |
| `interface/replay.py`, `experiment.py`, `snapshot_services.py`: request matching | Hash equality also rejected a changed method-version label | Only that label ignored for matching; original hash still validated |
| `orchestrator/toolkit/replay.py`: recorded/auto/live; `snapshot_replay.py`: snapshot/step/native-action | No common protocol declaration | Root/manifest/events audited, including failed attempts; result includes both versions |
| `orchestrator/backends/robodojo/motion/replay.py`: kernel admission in launch/report, launch commit vs report commit | Kernel rejection unless admitted or explicitly overridden; report failed for another launch commit | Warnings include both identities and kernel admission's differing paths; no version-based failure |
| `orchestrator/backends/robodojo/toolkit_live.py`: legacy live kernel equality | Required identical kernel (native-action already warned) | Both paths warn; host/lease/clean-checkout constraints unchanged |
| `environment/backends/robodojo/control_replay.py`: `_require_same_sources` | Revision, layout hash and reset fingerprint differences rejected | Returns source warnings, persisted in worker and host reports |
| `environment/backends/robodojo/toolkit_restore.py` / `toolkit_native_state.py` | Native-action warned; old live restore rejected source drift | Shared warning audit for both; includes bootstrap warnings |
| `environment/backends/robodojo/bridge.py` / `source_identity.py`: required source pins, tracked modifications, layout hash, derived CuRobo template, interpreter/package version | Strict bootstrap rejection | Diagnostic worker config `replay_identity_warnings=true` warns with recorded/current values; production bootstrap remains strict |
| `environment/backends/robodojo/layout_rounds.py`: scheduled file digest and planned digest at round start | Rejected changed layout bytes | Diagnostic-only warning for planned identity; current file must still match bootstrap manifest and actual loaded scene |
| `orchestrator/evaluation/provider_replay.py`: tape comparison | No version-based gate; raw request comparison | Adds source/current and candidate protocol reports and implementation-commit warning; raw request differences unchanged |
| `orchestrator/toolkit/surface_audit.py` | Current vs frozen surface already warned | Unchanged; malformed frozen evidence and its own hash mismatch still reject |
| `agent/toolkit/lib/manifest.py`, toolkit tool/script implementation hashes | Evidence only; no identity gate | Unchanged |
| Q32 qualification/revalidation | Missing/stale qualification and kernel already warnings | Unchanged; this task does not loosen production launch |

The original `python -m ora.environment.backends.robodojo.control_replay`
entry now explicitly enables the diagnostic source-warning bootstrap, even
with an older worker config. A production `qualification_binding` in that
config is recorded as warning evidence and omitted from diagnostic bootstrap
(the host launcher already omitted it); operational and physical checks remain.
Its worker report includes protocol and source
warnings. Recorded control decoding, `replay_control`, native ticks, applied
target comparison, clocks, joint/finger/link6 comparisons, round sequencing,
teardown and output isolation are unchanged. `ora motion replay` no longer
needs `--allow-kernel-drift`; the flag is retained for CLI compatibility.
Its historical optional private-object determinism metric remains a reported
metric, not a newly introduced gate. Existing live/native/align physical and
GT verification gates remain strict.

Provider tape candidates are **new formal runs**: their preflight and
bootstrap remain strict, including runtime pins. Comparison warnings do not
authorize cross-runtime production startup. Such a mode would need a separate
explicit diagnostic design. This distinction was confirmed for M7.6.

## Refusals that remain

- CAS/blob length/hash checks, frozen manifest hashes, step hash chains and
  request hashes check the integrity of stored evidence, not code versions.
- Parent/model/call/round/observation/controller identities, exact normalized
  action matching, cursor order, ambiguous responses and gaps remain strict.
- Unknown schemas/methods without a decoder are unsupported; missing evidence
  remains NotRecorded. No future state is imported to fill a hole.
- All existing physical restoration comparisons and required GT comparisons
  still gate capability authorization. Sensor differences remain statistics
  under M7.4b; they are not silently called byte-exact.
- Path allowlists, output separation, clean checkout, allocation/lease and
  process cleanup checks remain. Warning mode grants no extra robot capability.

## When to increment

Increment the number when changing public method semantics: signatures or
defaults, semantic envelopes or blob encoding, units/frames, source categories,
permission meaning, boundary/cursor semantics, or service comparison rules.
Record the migration and supported decoder versions with that change. Keeping
a decoder for an older version is an explicit implementation promise.

Do not increment for refactoring, a new commit, a tool/lib implementation hash,
logging of additional advisory evidence, or a renderer producing different
bytes. These identities and differences remain evidence independently of the
protocol number. Version 1 describes the existing nine-family data interface;
this change adds its declaration without changing robot-visible behavior.

## CPU validation / live follow-up

Tests cover matching, mismatching and absent versions; conflicting declarations;
old unversioned logs; differing method versions with valid original hashes;
corrupt hashes still refused; publication/replay results; production vs
diagnostic bootstrap; and changed kernels with exact vs divergent physical
replay. No simulator or GPU run is part of this implementation check. The
reviewer will validate a historical run on a router allocation.
