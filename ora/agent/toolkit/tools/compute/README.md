# compute

Restricted numeric expressions over caller-supplied bindings.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool compute`.

The tool owns argument reading, escaped-newline normalization and provenance.
lib.numeric calls api.evaluate_numeric; trusted assembly binds the existing
restricted geometry kernel. The AST allowlist, child execution, CPU/memory/wall
time limits and output checks stay intact. No unrestricted eval API is added.

Replay recomputes from the selected parent arguments; no simulator or service
is needed. This local computation consumes no interface ordinal. A boundary
input index can identify this call even when calls.jsonl contains zero entries.
The tool and surrounding replay script remain in the coordinator process; the
numeric kernel's existing restricted child is not isolation of arbitrary tools.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.
