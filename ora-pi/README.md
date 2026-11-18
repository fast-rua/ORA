# ora-pi

## Responsibility

ORA's default agent core, in TypeScript on the pinned
[Pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) packages,
and ORA's web page. For every `native_api` run the Supervisor starts
`node ora-pi/src/main.ts --rpc-fd N` under Node's permission model and drives
it over a JSON-lines RPC channel. The core owns the conversation: the provider
loop, request assembly, context management and the agent-side evidence. Tools,
the ToolSurface, robot semantics and all authority over the robot stay in
Python: the core calls back into `ora/agent/pi/` for every tool, observation
and static context slot, and may write only the run's agent evidence.

| Path | Holds |
|---|---|
| `src/main.ts`, `src/rpc.ts` | The entry point (headless under the Supervisor, or serving Pi's terminal UI and RPC to attached clients) and the RPC peer |
| `src/core/` | One policy session turn by turn (`session.ts`), the Pi integration (`pi_bridge.ts`), tool dispatch, retention hints, persistence for a restarted process, credentials for `/login` |
| `src/context/` | The Context pipeline, ported byte for byte from the retired Python core: `stages/` the stage driver and the stages lossless retention runs as (management v6 and v8; [README](src/context/stages/README.md)), `native.ts` the pipeline and the `rewrite_compact_v1` projection, the rollout compactor, request assembly and size, context packs and the compact view; `memory/` the Context artifacts |
| `src/provider/` | The OpenAI-compatible and Anthropic Messages wire projections, recovery, transport and the fake and replay providers |
| `src/evidence/` | `agent-events.jsonl`, content-addressed artifacts, the session manifest and write durability |
| `src/play/` | Free play (`ora play`): Pi's own agent as a coding agent, the image window, the robot socket for codemode scripts, a real cell's operator commands (`ora cell`, `cell.ts`) |
| `src/rpc/`, `src/tui/` | Pi's RPC protocol for clients attached to a run (`bin/ora-rpc-attach`) and the terminal UI behind `ora attach`, `ora interactive` and session export |
| `src/watch/` | The pi-tui front end of `ora watch` and `ora interactive --spec`, which starts `python -m ora.app.backend` |
| `src/jcs.ts`, `src/pyjson.ts` | JSON canonicalization and Python-compatible JSON for values that cross the language boundary |
| `test/` | The core's unit tests (`node --test`) |
| `web/` | ORA's web page for a live run (`ora web`) and replay (`ora replay`): a React client (`web/src/`) and its Node server (`web/server/`), with its own `package.json` and tests |

`context_token_estimator=image_aware_v1` estimates rewrite retention's requests
(the toolkit-only RoboDojo baseline `robodojo_toolkit_v1` selected it; that
baseline was retired on 2026-10-31 and the code stays for its sealed runs).
`context/native.ts` reuses the lossless image-pressure helper, records both raw
and image-aware estimates, and anchors usage calibration to the latter.

## What belongs here, and what does not

Here: the model loop and everything it needs that is robot-agnostic, and the
browser and terminal front ends of a run. Not here: tools, observation
projection rules, the ToolSurface and prompt composition, admission and
evaluation; they are Python (`ora/agent/`, `ora/orchestrator/`), and the core
receives their results over RPC. The robot-specific names the context port
needs are in one place, `src/context/domain_defaults.ts`.

## Dependency direction

The core depends on the Pi packages pinned in `package.json` and
`package-lock.json`, and on the Python side only through the RPC protocol
(`ora/agent/pi/rpc.py`, `core_process.py`, `tool_rpc.py`); it imports no
Python. Python reaches it by path (`ora/agent/pi/core_process.py` names
`ora-pi/src/main.ts`), as do `ora attach`, `ora web` and `ora watch --ui pi`.
`web/` talks to a run through the core's RPC socket and the run's evidence
directory.

## Entry points

- `node ora-pi/src/main.ts --rpc-fd N`: started by `ora/agent/pi/core_process.py`.
- `ora-pi/src/tui/attach.ts`, `view.ts`, `export.ts`, `pick.ts`: started by
  `ora attach`, `ora sessions export` and the launch pickers.
- `ora-pi/src/watch/index.mjs`: started by `ora watch --ui pi` and `ora
  interactive --spec`.
- `src/rpc/attach.ts` (`bin/ora-rpc-attach`): a stand-in for `pi --mode rpc`
  that attaches a Pi RPC client to a live run; `ora web` starts it for the
  page's server.
- `npm ci --prefix ora-pi` installs the core (Node.js >= 22.19);
  `node --test ora-pi/test/*.test.ts ora-pi/test/*.test.mjs` runs its tests.
  `ora web setup` installs and builds `web/`.

## Documentation

- [How ORA works: the agent core](../docs/how-ora-works.md#the-agent-core)
- [Agent core configuration](../docs/reference/configuration-and-evidence.md#agent-core)
- [Context assembly](../docs/context-assembly.md)
- [Interactive sessions](../docs/usage/interactive.md),
  [free play](../docs/usage/play.md) and the
  [web view](../docs/usage/remote.md#web-view)
- [ora/agent/README.md](../ora/agent/README.md), the Python half
