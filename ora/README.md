# ora

## Responsibility

The ORA harness as one Python package: it runs a language model as the policy
of a simulated robot behind a trusted Supervisor, and plans, executes,
evaluates and seals every run with hash-bound evidence. The agent loop of the
default policy runs in the TypeScript core in [`ora-pi/`](../ora-pi/README.md);
everything else is here.

| Package | Holds | README |
|---|---|---|
| `contracts/` | The versioned action envelope and the pose, motion, refusal and receipt contracts that the agent and the environment share | [contracts](contracts/README.md) |
| `environment/` | Transport-neutral environment sessions, the isolated worker and its local RPC, Synthetic, host-safe benchmark and controller contracts, and the simulator backends (`backends/<sim>/`) | [environment](environment/README.md) |
| `agent/` | The policy side: the Python half of the Pi agent core, the tools, ToolSurface / prompt / embodiment composition, context rules, providers, resident model services and the agent trace's formats | [agent](agent/README.md) |
| `orchestrator/` | RunSpec to RunPlan, preflight, leases, launch, the Supervisor's episode loop, evaluation, sealing and reports; host-side backend code under `backends/<kind>/` | [orchestrator](orchestrator/README.md) |
| `app/` | The `ora` command: one parser, one command table, the terminal and web front ends | [app](app/README.md) |
| `policy/` | Data, not code: prompt layers, ToolSurfaces, embodiment and environment assets, the default baseline; every file pinned by `policy.lock` | [policy](policy/README.md) |
| `robot_client/` | The standard-library robot client (`ora-robot`) that a free-play agent's own scripts call | [robot_client](robot_client/README.md) |
| `version.py` | The version `ora --version` and the transcript headers print | |

## What belongs here, and what does not

Here: the product code and the data files it reads at run time. Code that
belongs to one simulator lives in exactly two places:
`environment/backends/<sim>/` (imported only inside that simulator's own
interpreter, by the isolated worker) and `orchestrator/backends/<kind>/` (the
host-side rules, preflight and qualification of that `runtime.kind`). Everything
else is shared by all environments.

Not here: the read-only run viewer (`viewer/`, its own package, which imports
`ora`), the TypeScript agent core and ORA's web page (`ora-pi/`), the tests
(`tests/`, a tree that mirrors this one), examples, documentation and generated
runs (the ignored `outputs/` workspace). `ora/__init__.py` re-exports nothing;
import the module that defines a name.

## Dependency direction

```text
app  >  orchestrator  >  agent | environment  >  contracts
policy, version: import no ora package; any package may import them
robot_client: standard library only; only orchestrator and app import it
```

`agent` and `environment` never import each other: they meet in `contracts`
and in the orchestrator, which binds a policy session to an environment
session. `tests/repository/test_structure.py` checks these rules (L1) and the
finer ones the sub-package READMEs state: `ora` never imports `viewer` (L2);
the orchestrator core reaches a backend only through
`orchestrator/backends/registry.py` (L3); the environment core imports its
backends only from the worker entry points `environment/worker.py` and
`environment/replay_worker.py` (L4); the agent imports no simulator backend,
orchestrator or app module (L5) and keeps its own layers (L6, L7); a package
with sub-packages imports nothing eagerly in its `__init__.py` (L9).

## Entry points

- `ora` (`ora.app.main:main`), `ora-remote`, `ora-robot` and `robotwin-qualify`:
  the console scripts `pyproject.toml` installs; `ora-viewer` comes from
  `viewer/`.
- `python -m ora.app`: the same command as `ora`, behind `scripts/ora`.
- Every module started with `python -m` or by path (the simulator worker, the
  supervisor, the probes) is listed in [ora/app/README.md](app/README.md#entry-points).

## Documentation

- [Architecture](../docs/architecture.md) and [code map](../docs/code-map.md)
- [How ORA works](../docs/how-ora-works.md)
- [Repository structure](../docs/developer/repository-structure.md): the
  layers, the import rules and the files that may not move
