# privileged

Immutable data capability. Reads/writes are delegated to the owning layer on
every call. Assembly controls grants; no mutable state or filesystem access.

No live or final-state query supplies a missing value.

`live.py` retains the original StateError for missing values, including legacy
RSI diagnostics. `replay.py` is used only by `assemble_snapshot`: missing
exported keys become NotRecorded, while permission failures stay unchanged.
