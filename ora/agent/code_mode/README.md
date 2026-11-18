# ora/agent/code_mode

## Responsibility

Code mode (plan `policy.code_mode`; RoboDojo's default since 2026-11-05, off by default elsewhere): the model acts by
writing Python that calls its tools. This package holds the parts that do not depend on
a policy session: the contract a RunSpec's `policy.code_mode` resolves to, the
tools the model is shown instead of the robot tools, the bubblewrap sandbox a
command runs in, the code that runs inside that sandbox, and the persistent
workspace.

| Unit | What it holds |
|---|---|
| `contract.py` | `resolve_contract` (`ora.code-mode-script.v1`: interface `coding`, `prompt` `v2` or `v3` (`v3` is written for, and needs, `library: extended`), `library` `base` or `extended`, per-script limits that may only tighten, and with `extended` the query allowance `QUERY_LIMITS`: `max_queries_per_script`, `query_time_s`; the retired interface `script` and prompt `v1` are refused), `with_reads` (`script_reads`, what the surface serves of `docs.READS`, recorded with `extended` only), which tools stay the model's own (`DIRECT_TOOLS`) and which become script functions (`script_functions`) |
| `coding.py` | The code tools: `read`, `write`, `edit` and `bash`, named and shaped as Pi's defaults and confined to `/work`; `bash`'s description lists the functions with their full descriptions and schemas (`base`) or as an index and a library card (`extended`) |
| `docs.py` | The script functions' documentation (`library: extended`), generated from their ToolSurface definitions (a toolkit tool's `description.json`) and result schemas: the index line a code tool's description lists per function (`index_line`: the call's nested shape, `call_shape`, its result's, `result_shape`, and its purpose, within `ARGUMENT_CHARS` and `RESULT_CHARS`), and the entry a script looks up by part (`robot.docs(name, part)`, `"all"` for the whole; with no part, and in `help(tools.<name>)`, `short`: at most `SHORT_CHARS`, then each other part's name and size), with the examples of `EXAMPLES` (what a tool's result is measured with), the `dry_run` part and argument of an arm-moving tool (`takes_dry_run`; its examples include one that uses `dry_run`) and the `explain` part (the stages of a tool that has a stage template); the index ends with one line per query and `query_entries` gives their entries (`query.<name>`, looked up as `query_<name>`). `move_pose`'s line is followed, where its schema takes `interaction` (`interaction_form`: in each arm's target, or flat on a single-arm surface), by `interaction_note`: label every move, when each value applies, and the call in that surface's spelling; a `move_base` with both forms by `BASE_NOTE`, the rule that joins them. An array of integers reads `[int; 2]`. The examples are written per surface (`examples(name, definition=...)`: the first camera its schema names, the first dry-run form it validates). Every generator takes `reads`, what the surface serves of `READS`, and leaves the rest out; `takes_interaction`, read off the definitions, does the same for `interaction` |
| `answers.py` | What a script reads of a call in its surface's terms (`library: extended`; the policy session, `agent/pi/code_mode.py`, asks): `refusal`, what a call is refused before its tool runs (a name of no tool, `unknown`: `allowed` the script functions without the model's own tools, and the import of an `ora_lib` or `robot` name; a `targets` wrapper around a tool whose schema takes one arm's fields directly, `unwrapped`, naming the form it takes); `item_errors`, the `error` of a failed answer whose items failed with codes of their own (`mark_point`'s marks); `predicted_axes`, a dry run's per-arm `predicted_approach_world` and `predicted_jaw_line_world` from the surface's controlled-frame convention (tool-call mode's dry-run projection, `session_core/domain.py`, applies it too); `preview_line`, a dry run's own line |
| `library_card.py` | After the functions in a code tool's description (`library: extended`), `card`: by introspection of `ora_lib` and `robot` (imported on the host for it, nothing executed), one line per public class (constructor, attributes, methods) and function (signature), under the import that reaches it; `fragment_parts`, each such name's entry, its signature and docstring, looked up as a function is; both leave `target`'s `interaction` out where the surface's `move_pose` does not take it |
| `runtime.py` | `CodeModeToolRuntime`: the episode's ToolRuntime with the code tools advertised after the base runtime's; it never runs a script itself; the code tools' result schemas (`extra_result_schemas`, with `allowance_left` and the `explain` of a failed call in `calls` for `library: extended`) |
| `sandbox.py` | `bwrap_command`: the fresh sandbox (no network, read-only interpreter and system, private `/tmp`, the workspace at `/work`); `sandbox_status` for preflight (`bwrap`, the user namespaces it needs, then every package of the interpreter's lock imported in the sandbox, by a program as a `bash` command runs one, at its locked version; the interpreter recorded; a refusal says what to install or set); `python_path` (`ORA_CODE_MODE_PYTHON`, else `default_python()`, where `ora setup --env robodojo` builds it); `prerequisite_checks` for `ora doctor` |
| `sandbox-requirements.in`, `sandbox-requirements.txt` | What the sandbox interpreter (`ORA_CODE_MODE_PYTHON`) holds: the direct packages (NumPy 1.26.4 and SciPy 1.13.1 exactly, Pillow, scikit-image, Shapely) and their lock, every package at an exact version with its hashes, for CPython 3.11 on manylinux x86_64; regenerated from this directory with the command in the lock's header (`uv pip compile sandbox-requirements.in ... --generate-hashes -o sandbox-requirements.txt`) |
| `shell.py` | `run_command`: one `bash` command in the sandbox, with the `robot` library on `PYTHONPATH` (with `library: extended`, `/ora/lib` alone: `robot`, `ora_lib` and `sitecustomize`, with `PYTHONSAFEPATH` set), its calls served in order over one socket; a program that ends while a call is answered ends the command `failed` (`ended_during_call`), not the run |
| `sitecustomize.py` | Inside the sandbox, standard library only, with `library: extended`: a `bash` program's own directories (the script's, or the working directory, and `/work`) on `sys.path` after the library, the standard library and site-packages, so a workspace file cannot stand for a module anything imports (`/work/inspect.py` broke `import numpy`), and is still found, and run, as itself. The price: `import <name>` of a workspace file named after a standard-library or installed module reaches that module |
| `robot.py` | Inside the sandbox (standard library): the `robot` library, speaking `ora.code-mode-rpc.v1` (tool calls, and the read-only `docs`, `explain` and `query` ops the policy session answers with `library: extended`); with `library: extended`, typed results (`Result`, with `pose_after` read from the mounted observation after an action, `reused`, the `call_id` and, for a failed call, the `explain` the policy session gives it) and the parameters it fills from typed values (`target`); `tools.<name>` also takes positional arguments, in the order of its signature (a `docs` lookup gives their names); `robot.explain(r)` asks what a call did; `robot.query(name, **arguments)` reads the robot model (typed poses written as lists); a dry run's answer is a `Plan` (a `Result` printed `Plan(...)`) that reads `dry_run`, `feasible`, `predicted_pose`, `end_joints_deg`, `route` and `screening` from the plan the policy session returned, and raises `PlanNotRun` on a run's field (`ok`, `status`, the receipt's) |
| `query_spec.py`, `query.py` | `robot.query` (`library: extended`): the seven queries (`ik`, `fk`, `screen_path`, `reach_check`, `probe_motion`, `align_plan`, `robot_state`) with their arguments, results and examples in one place (`query_spec.py`, nothing of ORA imported, which `docs.py` and the sandbox read), and the host side that checks a query's arguments and calls the toolkit lib over the episode's data interface, degrees in and out (`query.py`; `toolkit/host/query.py` binds the interface, `agent/pi/code_mode.py` serves the op and keeps the allowance) |
| `library.py` | The extended library's mounts (`library: extended` only): `ora_lib` bound read-only at `/ora/lib/ora_lib`, with `source: true` ORA's code read-only under `/ora/src` at its repository paths (`source_files`: the toolkit's lib and tools, the motion planner, the `ora` modules they import, each robot's public geometry), and `ObservationMount`, a hash-checked copy of the current published observation at `/ora/observation` (`ora.code-mode-observation.v1`), brought up to date after every call, with the masks of the segmented targets (`vision_ground`, `vision_segment`) recorded on it and, on a surface that delivers them, the public robot facts the policy's context carries for it |
| `record.py` | What an extended script asked of the library, on the record: the `explain_inputs` each call's `tool_result` records (`ora.code-mode-explain-inputs.v1`: the tool and its template's identity, the arguments, what the script read of the call, the stage events), `render` (a call's explanation from them alone), `recorded_inputs` (a call's inputs read back from the run's trace, so `explain` answers for an older call or after a restart) and `check_explanations` (every explanation a script read, rendered again from the record; the run's evaluation calls it); the `query` and `explain` requests themselves are recorded by the policy session (`agent/pi/code_mode.py`, `_code_mode_record_op`) as nested `robot.<op>` call pairs, which `evidence.agent_events.library_op` tells from a tool call |
| `reuse.py` | `KeptResults` (`library: extended`): the answers of a script's repeatable perception calls (`REUSABLE`: `vision_ground`, `vision_segment`, `geometry_query`) kept for the current observation, which the policy session returns marked `reused` to the same call |
| `ora_lib/` | Inside the sandbox (NumPy and SciPy): `frames` (`Point`, `Pixel`, `Pose`, `Direction`, `x.to(frame)` across `world`, `tcp:<arm>`, `camera:<name>`, `pixel:<name>`; `offset`, `direction`, `pixels_to_world`, `project`), `observation` (the mounted observation and its targets' masks), `card` (`frames_card()`, the text of how this scene's frames relate, written from the mounted observation), `geometry` (the toolkit lib's transform, projection, depth-window, `mask_to_cloud`, `depth_inliers` and `principal_geometry` arithmetic on plain arrays) and `vision` (`object_geometry(target)`: extent, principal axes, support height, height above support and mid height of a target; `localize_mask`, `mask_summary`) |
| `workspace.py` | The persistent workspace `<run>/native/code-workspace`: edits (all or none), reads, bounds and per-run snapshots under `<run>/native/code-workspace-objects` |

## What belongs here, and what does not

Here: the contract, the code tools' definitions and result schemas, the
sandbox and what runs inside it, and the workspace. Not here: running a script
inside an episode and recording its calls (the Pi policy session,
`agent/pi/code_mode.py`, through the robot socket's script path), admitting and
executing its actions (the Supervisor's `ToolActionExecutor`,
`orchestrator/episode/tool_execution.py` and `code_mode_execution.py`), the
plan's prompt layers and derived tool layer (`orchestrator/spec/composition.py`)
and the code planners (`ora/policy/planners/native/code-mode-*.md`, one pair per `prompt` generation, and from `v3` a
layer for each read only some surfaces serve).

## Dependency direction

`code_mode` sits at the rank of `memory` (rule L6): it may import `tools`,
`composition`, `toolkit`, `providers`, `model_services`, `evidence`,
`sessions`, `contracts` and `error_classes`, never `memory`, `context`,
`session_core`, `pi` or `tui`. `pi` and the orchestrator may import it. `robot.py`,
`sitecustomize.py` and `ora_lib/` run in the sandbox's own interpreter (`ORA_CODE_MODE_PYTHON`),
which has no ORA package: they import nothing of `ora` (`ora_lib`'s modules only
each other) and are bound by path. `ora_lib` cannot import
`ora/agent/toolkit/lib`, whose modules reach its data interface; it carries small
equivalents, held to the lib's functions by `tests/ora/agent/code_mode/ora_lib`. The
interpreter has no OpenCV either, so the lib's connected components and elliptical dilation are
written with SciPy (`geometry.depth_inliers`, `geometry.dilate_disc`) and held to OpenCV's.

## Entry points

- `contract.resolve_contract`: `policy.code_mode` as its contract, None when
  off (`orchestrator/spec/native_policy.py`). A contract records `library`
  only when it is `extended`; features of the library check
  `contract.get("library") == "extended"`.
- `docs.listing`, `docs.entry_parts`, `docs.short`, `library_card.fragment_parts`: the index and the card in `bash`'s
  description (`coding.py`) and what the policy session serves for `robot.docs` (`agent/pi/code_mode.py`),
  all from the result schemas the session and the plan read alike (`coding.advertised`'s `result_schemas`).
- `contract.script_reads`: what the surface serves a script beyond its tools (`dry_run`, `query`), decided once
  in resolution (`contract.with_reads`, `orchestrator/spec/native_policy.py`) and read by the prompt layers
  (`orchestrator/spec/composition.py`, `CODE_MODE_READ_LAYERS`), the descriptions and `robot.docs`. Nothing else may tell
  the model of a read; the runtime refusals stay, and go by the same judgement (`toolkit.host.motion.joint_trajectory_surface`,
  which names no backend).
- `query.prepare(name, arguments)`: a `robot.query` checked and put in the lib's terms; the call to make over a
  `DataInterface` (`toolkit/host/query.query_api` binds one that records nothing); the policy session
  charges it to the script's query allowance (`PiCodeMode._code_mode_charge_query`, which a dry run calls too).
- `runtime.build_code_mode_runtime`: wraps the episode's ToolRuntime when the
  plan carries a contract (`orchestrator/episode/coordinator.py`).
- `shell.run_command`: one command, served; `extended=True` (`library: extended`)
  binds `ora_lib` and `sitecustomize.py` and sets `ORA_CODE_MODE_LIBRARY=extended` and
  `PYTHONSAFEPATH` for its programs; `observation` binds the current observation.
- `ora_lib.frames_card`: the frames card a program prints (`from ora_lib import frames_card`): each camera's
  place and how the world's axes show in its image, which cameras move, each arm's place and local axes, all from the
  mounted observation. The calibration of a scene is per observation (`docs/robots/frames-and-units.md`), unknown when the
  prompt is composed, so the card is written where the observation is; the `prompt: v3` planner has the model print it first.
- `robot.explain(call)` (`robot.py`): the read-only `explain` op, answered by the policy session
  (`agent/pi/code_mode.py`) from `toolkit/explain.py`, the tool's `explain.json` and the stage events collected
  around the call (`toolkit/interface/stages.py`), kept in memory for the latest calls and recorded with each call's
  result for the others (`record.py`); a failed call carries its explanation by default.
- `record.check_explanations(run_root)`: every explanation a script read, rendered again from the record alone
  (`orchestrator/evaluation/checks.py` runs it for a plan with `library: extended`). Each `query` and `explain` request
  is recorded as a nested call of the script by `PiCodeMode._code_mode_record_op(op, name, arguments, answer, *,
  model_call, parent, state)`, which takes the next `state["seq"]`; an op handler finds the script call's ids in
  `state["model_call_id"]` and `state["tool_call_id"]`.
- `dry_run=True` on every tool that moves the robot (`robot.py` sends it as a plain argument; `agent/pi/code_mode.py`,
  `_code_mode_dry_run`, strips it, so it is never an argument of the tool): the plan the real call runs before it acts,
  returned instead of executed, with its preview (`toolkit/interface/dry_run.py`, `toolkit/tools/_dry_run.py`,
  `toolkit/lib/action_preview.py`). It is charged to the script's queries (`_code_mode_charge_query`), recorded as a
  read-only nested call (`origin.dry_run`), and never proposes an action; its summary and picture reach the model with
  the script's result. An action no plan can answer without moving is refused `dry_run_unavailable`.
- `library.observation_mount`: the observation's mount for one command,
  None unless `contract.get("library") == "extended"` (`agent/pi/code_mode.py`).
- `reuse.KeptResults`: the policy session's kept perception answers
  (`agent/pi/code_mode.py`, `_code_mode_call`), `library: extended` only.
- `sandbox.sandbox_status`: the `code_mode_sandbox` preflight check, a program run as `bash` runs one. The
  sandbox needs two host prerequisites, `bwrap` (on `PATH`, at `~/.local/bin/bwrap` or `ORA_CODE_MODE_BWRAP`) and
  unprivileged user namespaces (an empty sandbox with every namespace unshared must run `true`,
  `namespace_refusal`), and the interpreter, a Python 3.11 built from the lock
  `sandbox-requirements.txt` (`sandbox.LOCK_BUILD`: `uv venv --python 3.11 DIR && uv pip install --python
  DIR/bin/python --require-hashes --no-deps --only-binary :all: --compile-bytecode -r ora/agent/code_mode/sandbox-requirements.txt`;
  the sandbox cannot write bytecode, so it is compiled at the build). `ora setup --env robodojo` builds it at
  `default_python()`, `envs/code-mode/bin/python` under its prefix (`ORA_GPU_PREFIX`, else
  `~/.local/share/ora/gpu`), and writes `ORA_CODE_MODE_PYTHON` to `.env`; `python_path` takes
  `ORA_CODE_MODE_PYTHON` when it is set (an explicit value wins and must be executable), else that path. Each
  prerequisite missing refuses the plan with what to install or set (`BWRAP_REMEDY`, `USER_NAMESPACE_REMEDY`, the
  setup command or the build). The check then imports every locked package in the sandbox; one missing, at another
  version or failing to import refuses the plan with the lock, the package, both versions and that command. The
  check's detail records the interpreter (`interpreter`: executable, Python version, the locked packages' versions,
  the lock's sha256), kept in the run's `preflight.json`. None of this decides whether a plan runs code mode: the
  RunSpec and its baseline do, in resolution (`orchestrator/spec/defaults.py`).
- `sandbox.prerequisite_checks`: `ora doctor`'s `code_mode_bwrap`, `code_mode_user_namespaces` and
  `code_mode_interpreter` (the lock probe run by the interpreter itself, outside the sandbox), for an environment
  whose default runs code mode.

## Documentation

- [Public interfaces](../../../docs/reference/public-interfaces.md), code mode
- [Toolkit execution](../../../docs/reference/toolkit-execution.md): the executor a script's actions run through
- [Repository structure](../../../docs/developer/repository-structure.md): rule L6, where `code_mode` ranks
