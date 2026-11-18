# Testing

[`tests/README.md`](../../tests/README.md) is the reference for the suite:
what each directory of the mirrored tree covers, the `tests/support/` and
conftest rules, the marker lanes, the commands, and what a run needs from its
environment. This page is about choosing and changing checks.

## Choose a check

Start with the changed contract, then run the modules that own it: every
test module states the contract it protects in its docstring, and
`tests/ora/<package path>/` covers `ora/<package path>/`.

| Change | Check |
|---|---|
| One contract | `uv run --locked pytest -q tests/ora/<package path>/test_<name>.py` |
| One package | `uv run --locked pytest -q tests/ora/<package path>/` |
| Anything in Python | `uv run --locked pytest -q -n auto -m "not agent_core"` (CI's tests job, `scripts/dev/check` lane A) |
| The agent core, its Python bridge (`ora/agent/pi/`) or anything that reaches the model-visible request | Also `uv run --locked pytest -q -m agent_core` and `uv run --locked node --test ora-pi/test/*.test.ts ora-pi/test/*.test.mjs` (CI's agent-core job, lanes B and C) |
| Context projection, request assembly or provider wire bytes | Also the opt-in replays of the sealed acceptance runs (`ORA_TEST_SEALED_*`); the Synthetic checks do not exercise those environments' tool surfaces |
| Before a pull request | `scripts/dev/check`: every lane in parallel, then the documentation build and the no-write check |

Inspect CPU, memory, existing jobs and simulator slots before scheduling checks,
and use the available CPU capacity for independent offline cases; the root
`conftest.py` runs BLAS and OpenMP with one thread per test process, so
pytest-xdist workers do not oversubscribe the host. Fixture-based robot checks
do not certify simulator physics or live service readiness.

## Simulator-only code

The suite reaches a simulator backend's host side without the simulator:
RunSpec builders compose and resolve against a scripted host profile,
preflight checks run against scripted probes, and the scripted qualification
tools (`ora qualify robocasa|robolab motion`) drive a scripted worker session
(`tests/support/orchestrator/sparse_motion.py`). Code that runs only inside a
simulator interpreter is not reached in process: the RoboCasa365 session and
scene construction (robosuite), the RoboLab bridge and session (Isaac Sim and
Kit), the RoboTwin play scenes and robot asset geometry (SAPIEN), the
RoboDojo robot asset geometry (Omniverse), and the LIBERO probe and preview
workers. Qualify a change there on the simulator host
([`docs/developer/authoring/qualification.md`](authoring/qualification.md));
a host-safe test of its pure parts belongs beside the backend's other tests.

## Families split by protected contract

| Family | Directory | Base contract | Children |
|---|---|---|---|
| `test_control_plane` | `tests/ora/orchestrator/` | Planning, leases and sealing | `preflight` in `tests/ora/orchestrator/preflight/`, `workspace` beside it, `viewer` in `tests/viewer/`; the former `cli` child is `tests/ora/app/test_main.py` |
| `test_runtime_v1` | `tests/ora/orchestrator/episode/` | Lifecycle and session/resource contracts | `state`, `tools`, `libero` beside it; `metrics` in `tests/ora/orchestrator/evaluation/` |
| `test_provider_adapters` | `tests/ora/agent/providers/` | Provider adapters, profiles and the shared Synthetic fixtures | `test_anthropic_provider`, `test_openai_provider`; `test_sft_export` in `tests/ora/agent/evidence/` |
| `test_lossless_context_runtime` | `tests/ora/agent/memory/` | Recoverable projection | `cost` in `tests/ora/agent/context/lossless/` |
| `test_robotwin_motion_v2` | `tests/ora/environment/backends/robotwin/` | Motion execution | `grasp`, `probe`, `profiles`, `sparse`, `surface` |

Each child states its own contract in its docstring. Shared helpers and
constants keep a single owner, a module in `tests/support/<area>/` that the
base module and its children import, and a fixture they share is registered in
the narrowest `conftest.py` above them. Do not duplicate constants into
children: an asset update must not leave independent copies of its expected
hash behind.

For another split or move, compare the complete test-function/decorator AST set
and the per-module pytest collection before and after, and the fixtures each
test sets up (`pytest --setup-plan`). Exercise cross-module helper consumers as
well as the moved tests. A collection count alone cannot detect replacement of
one test by another.

## Timing and parallel execution

Splitting improves navigation and gives file-based schedulers smaller work
units. It does not itself demonstrate faster wall time: uneven file costs can
still put most work in one shard.

When diagnosing a slow or failing check, inspect subprocess startup, installed
client versions, shard imbalance and the host's load before changing the test.
The tests that wait on real time (`tests/README.md` lists them) fail first on a
loaded host or a stalling file system, so reproduce such a failure on a quiet
host before treating it as a regression. Report the worker count, selected
cases, elapsed time and failures.

## Pinned assertions and readability

The module-size, parameter-delivery and ownership tests are ratchets. A
failure requires identifying which contract changed before editing an expected
value. Frozen profile hashes and tool inventories are deliberate
boundaries. Fix a stale probe when its implementation no longer checks the
contract; fix product code when it violates the existing contract.

`tests/repository/test_module_size_ratchet.py` governs product Python modules,
not test-file length. Names and docstrings describe the executable test; the
module docstring states the module contract. Prefer behavior checks over
source-layout assertions unless the source itself is the artifact under test,
such as delivered prompt text.
