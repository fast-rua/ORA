# Agent-core goldens

What the agent core sent and recorded on the scripted scenarios of
`tests/agent_core`, frozen. Each `<name>.json.gz` is written and read by
`tests/support/agent_core/golden.py`:

- `requests`: the sha256 and size of each request's exact bytes;
- `request_bodies`: the same requests as JSON, inline images replaced by their
  sha256 and length, so a mismatch is reported at its first JSON path;
- `event_types` and `events`: the agent trace in order and, per event, a
  sha256 of its fields without clocks and the hash chain over them;
- `extra`: what the scenario records besides (the call ids of an aggregated
  batch, the headers of a wire capture, an operator stop's error payload).

| Golden | Test | Scenario |
|---|---|---|
| `synthetic-<composition>-<retention>` | `test_synthetic_parity.py` | six actions of the default script, per context |
| `mark-batches-<composition>-<retention>` | `test_synthetic_parity.py` | two aggregated `mark_point` batches |
| `anthropic-wire-<recovery>` | `test_anthropic_parity.py` | the request bodies a loopback Anthropic Messages server received |
| `operator-stop` | `test_operator_stop.py` | `ora stop` during the fourth request |
| `compactor-bound` | `test_context_compactor_binding.py` | the policy and compactor bodies with a bound compactor, and the trace |

They were recorded from the Python Native core before ORA removed it (it
remains in the robot-harness repository), when the Pi core reproduced every
one of them byte for byte; `recorded_from` in each file says which core wrote
it. The Native core was the differential reference of the Pi port, so these
files are what is left of that reference.

## Re-recording

Only a deliberate, reviewed change of what the agent core sends or records
re-records a golden, from the Pi core:

```bash
python -m tests.support.agent_core.golden show NAME > before.json
ORA_AGENT_CORE_GOLDEN_BLESS=1 python -m pytest -q tests/agent_core/test_synthetic_parity.py
python -m tests.support.agent_core.golden show NAME > after.json
diff before.json after.json
```

Review the diff before committing: once a golden is blessed from the Pi core, a
change that breaks the model-visible bytes no longer shows up anywhere else.
