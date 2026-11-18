# State

Supervisor-owned state, below interface/tools and independent of the worker.
`keys.py` declares ownership and units; `store.py` gives copy-on-read accessors;
`snapshot.py` exports and loads complete publication boundaries. No simulator,
service, interface or orchestrator imports belong here.

Environment writers own `env.*` and `privileged.*`; execution writers own
`exec.*`; tools can write only finite JSON under `tool.*`. Robot reads reject
privileged keys. The temporary task-check grant requires both the key and the
check_task tool. RSI/supervisor readers can inspect privileged state. Accessors
are same-process capabilities, not a sandbox against malicious Python code.

Snapshots live at `state/steps/step-NNNN/snapshot.json`, align with the existing
policy-state hash chain and reference original policy-observations files. No
image copies or extra rendering. Each execute invalidates completeness until
an existing publication completes. Loading verifies identities, chains and
public file hashes without starting a simulator. Tool values that happen to
contain paths are never read as files. Loaded state is read-only.

Step 1 does not change tool APIs, budgets or the model-visible surface. Wiring
is owned by orchestrator; later data-layer accessors will read this same store.

`deltas.py` records `tool.*` changes between publications in
`state/tool-deltas.jsonl`. The owner binds one writer; trusted dispatch supplies
parent model/tool identities and the next I/O ordinal. Each snapshot includes a
hash-bound journal watermark. `before_call` loads only changes preceding the
requested parent into a private overlay. Original loaded state and source files
stay read-only. Failed recording freezes the writer and records a gap if
possible; it never changes live tool returns or state. Missing evidence is an
error for exact parent-boundary reconstruction, never an inferred state.


`before_interface` is the native-tick restorer's overlay: from the latest
complete snapshot, apply writes through the selected parent's interface ordinal
(before that invocation). The orchestrator must first verify the action-to-I/O
mapping. Writes in one gap retain state_write_ordinal order; later writes,
missing watermarks and affected gap intervals cannot supply an exact state.

`unsettled_publication` validates a publication watermark and identifies
in-flight parent/owner scopes. The --step API marks tool.* unavailable at these
boundaries instead of exposing a pending intent as settled policy state. The
raw snapshot and before-call/interface restoration remain unchanged.

`exec_ledger.py` records every change of `exec.ik_sampler_state` (the CuRobo
seed generator, after each worker RPC that moved it) in the hash-chained
`state/exec-ledger.jsonl`, bound with the tool journal and independent of
toolkit I/O. It is evidence (`read_rows` validates it); nothing restores from
it: a restore resumes the generators from the boundary snapshot's
`exec.ik_sampler_state`.
