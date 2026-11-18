"""Code mode in a benchmark episode (plan ``policy.code_mode``): the policy session's side.

The model's script call, ``bash``,
reaches ``tools.invoke`` like any tool call; ``tool_rpc._rpc_invoke`` hands it to
``_code_mode_execute`` inside the per-act cache (``_once``), so a restarted agent
core that re-sends the call gets the recorded result and no script runs twice.
``read``, ``write`` and ``edit`` work on the workspace only.

A command runs in a fresh sandbox (``agent/code_mode/sandbox.py``, ``shell.py``;
with ``library: extended`` also ``ora_lib`` and a copy of the current published
observation, brought up to date after each of its calls: ``library.py``)
as one parent invocation of the Supervisor's ``ToolActionExecutor``
(``orchestrator/episode/tool_execution.py``): every action it causes is admitted,
executed and published inside that invocation, and its ledger is the call's
``ora.tool-execution.v1`` evidence, which ends the turn when it spent anything
(``_rpc_invoke``, the agent core's execution check). Each of the script's
``tools.<name>(...)`` calls takes the robot socket's script path
(``robot_socket._script_call``): recorded under the reply's model call as
``<parent call id>#NNN`` with ``payload.origin`` ``{kind: code_mode,
parent_tool_call_id, seq}``. A legacy tool's submitted action is executed through
the parent (``_code_mode_action``); a synchronous toolkit tool runs its own
primitives inside the same parent. Per script at most ``max_actions_per_script``
actions (and no more than half of the robot decisions left, at least one) and
``max_calls_per_script`` other calls; the completion check and the script tools
themselves are refused (``script_tool_unavailable``), finishing as in free play
(``script_lifecycle_unavailable``), and where the model has the refused tool the
refusal says to call it directly. After the trusted terminal boundary the agent core
refuses the script tool itself (``episode_terminal``) in the words
``_code_mode_terminal_refusals`` gives it: no script runs any more, and the refusal
names the direct tools left.

The script reads each result as the model would (its causal projection), and
after an action its receipt. The model's result lists the script's value or
output, error and calls, with the images its calls returned as the tool images
(the last ``max_frames_per_script`` of them); an action's frame comes with the
fresh observation the next turn starts from. With ``library: extended`` the
script's calls return typed results (``agent/code_mode/robot.py``) and the result
also states what is left of the script's allowance (``allowance_left``: actions and
other calls). A committed skill sequence (RoboTwin's
closed-skill ``grasp_execute``) runs its stage loop as one action: its call's entry
carries the staged receipt, its post-close views are that call's images, and the
Supervisor's time on it does not count against the script's wall time.

With ``library: extended`` a script also looks its functions' documentation up
(``robot.docs(name, part)``, ``help(tools.<name>)``): the ``docs`` op is answered
here (``_code_mode_op``) from the entries ``code_mode/docs.py`` generates from the
same definitions the description's index lists. A lookup is no tool call: it is
not recorded and counts against no limit. Nor is a reused answer: the same
perception call with the same arguments on the same observation (``vision_ground``,
``vision_segment``, ``geometry_query``) returns the answer kept from the first, marked
``reused``, with no tool run and no images again; the call is listed in the result's
``calls`` with ``reused`` (``code_mode/reuse.py``, ``_code_mode_call``).

A script may also read the robot model before a move (``robot.query``, the ``query`` op, ``code_mode/query.py`` over the
tools' own data interface, ``toolkit/host/query.py``). A query is no tool call: no call limit counts it. It
counts against the script's query allowance (``state["queries"]`` and ``state["query_s"]``,
``_code_mode_charge_query``), which ``dry_run`` shares; ``allowance_left.queries`` says what is left.

It also asks what a call did (``robot.explain(r)``, the ``explain`` op): the session answers by call id
(``<parent call id>#NNN``, ``call_id`` in the script's result) with the rendering of the tool's stage template
(``toolkit/explain.py``) from the call's answer and the stage events its tool recorded (``toolkit/interface/stages.py``,
collected around the call). A failed call of a tool that has a template carries its explanation by default: as
``explain`` in the answer the script reads and in its entry of the result's ``calls`` (at most
``toolkit_explain.SCRIPT_CHARS`` characters of them per script). The stage events never enter a result.

What a script asks of the library is on the record (``code_mode/record.py``): each ``query`` and ``explain``
request, refused or answered, is a nested call of the script, as its tool calls are, written by ``_code_mode_record_op`` (named
``robot.<op>``, effect ``read_only``, ``origin.op``; it takes the next ``seq``, so the script's later calls number after
it) and counted against no call limit; a ``docs`` lookup is not recorded. Each extended call's result records the
inputs its explanation is rendered from (``payload.explain_inputs``): the session keeps the latest ``EXPLAIN_CALLS``
in memory and answers for any older call, or after a restart, from the record.
"""

from __future__ import annotations

import base64
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

from ora.agent.composition.surface.units import CAPABILITY as DEGREES, policy_degrees
from ora.agent.context.causal_value import causal_provider_value
from ora.agent.code_mode import answers as code_answers
from ora.agent.code_mode import coding
from ora.agent.code_mode import docs as code_docs
from ora.agent.code_mode import query as code_query
from ora.agent.code_mode import record as code_record
from ora.agent.code_mode import workspace as code_workspace
from ora.agent.code_mode.library import observation_mount
from ora.agent.code_mode.reuse import KeptResults
from ora.agent.code_mode.runtime import validate_command
from ora.agent.code_mode.shell import run_command
from ora.agent.code_mode.contract import (DIRECT_TOOLS, NOT_FROM_SCRIPTS, WITHHELD, query_allowance, script_functions,
                                          script_reads)
from ora.agent.code_mode.runtime import ALLOWANCE_FIELD
from ora.agent.code_mode.robot import Result as ScriptResult
from ora.agent.code_mode.sandbox import SandboxError
from ora.agent.context.observation.skill_views import materialize_skill_observations
from ora.agent.evidence.trace_verify import is_supervisor_refusal
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.toolkit import explain as toolkit_explain
from ora.agent.toolkit.interface import dry_run, stages
from ora.agent.toolkit.interface.types import ActionNotStarted
from ora.agent.error_classes import ToolError
from ora.contracts import SKILL_SEQUENCE_ACTION_SCHEMA

from .robot_socket import ScriptCall

ORIGIN_KIND = "code_mode"
#: How many of an episode's latest script calls ``explain`` answers for from memory (older ones from the record).
EXPLAIN_CALLS = 256


class PiCodeMode:
    """``PiPlanner``'s code tools (a mixin)."""

    code_mode: dict[str, Any] | None = None
    _code_mode_decisions: int | None = None
    _code_mode_documentation: tuple[str, dict[str, dict[str, str]]] | None = None
    _code_mode_kept: KeptResults | None = None
    _code_mode_calls: dict[str, dict[str, Any]] | None = None

    def _code_mode_new_act(self, remaining_budget: Mapping[str, Any]) -> None:
        decisions = remaining_budget.get("robot_decisions")
        self._code_mode_decisions = decisions if isinstance(decisions, int) and not isinstance(decisions, bool) else None

    def _code_mode_execute(self, name: str, arguments: Any, call_context: Mapping[str, Any]) -> dict[str, Any]:
        """A code-mode tool's raw result (``_rpc_invoke`` projects it and attaches the execution evidence)."""

        if name == coding.BASH_TOOL:
            return self._code_mode_bash(arguments, call_context)
        return self._code_mode_file(name, arguments)

    def _code_mode_tool(self, name: str) -> bool:
        return bool(self.code_mode) and name in coding.CODING_TOOLS

    @contextmanager
    def _code_mode_parent(self, name: str, call_context: Mapping[str, Any], cap: int) -> Iterator[Any]:
        """The script's parent invocation: every action it causes is executed and recorded inside it."""

        executor = getattr(self, "tool_action_executor", None)
        if executor is None:
            raise ToolError("code mode needs the Supervisor's action executor", code="code_mode_unbound",
                            retryable=False)
        with executor.invocation(name, dict(call_context), cap * executor.native_actions_per_action()) as parent:
            parent.reentrant = True  # a synchronous toolkit tool the script calls runs inside this parent
            yield parent

    def _code_mode_file(self, name: str, arguments: Any) -> dict[str, Any]:
        """``read``, ``write`` and ``edit``: the workspace, never the world."""

        root = code_workspace.workspace_root(self.run_root)
        try:
            if name == coding.READ_TOOL:
                return code_workspace.read_file(root, arguments)
            if name == coding.WRITE_TOOL:
                if not isinstance(arguments, Mapping) or set(arguments) != {"path", "content"}:
                    raise ToolError("write takes path and content", code="invalid_arguments", retryable=True)
                written = code_workspace.apply_edits(root, [dict(arguments)])
            else:
                if not isinstance(arguments, Mapping) or set(arguments) != {"path", "edits"} \
                        or not isinstance(arguments["edits"], list):
                    raise ToolError("edit takes path and edits", code="invalid_arguments", retryable=True)
                written = code_workspace.apply_edits(root, [
                    {"path": arguments["path"], "old_string": (edit or {}).get("oldText"),
                     "new_string": (edit or {}).get("newText", "")} for edit in arguments["edits"]])
            code_workspace.check_bounds(root)
            return {"ok": True, "path": written[0], "written": written}
        except ToolError as exc:
            return _tool_error(exc)

    def _code_mode_turn_instruction(self) -> str | None:
        """The turn rule when code is how the model acts (code mode); None otherwise."""

        if not self.code_mode:
            return None
        return ("bash must be the only tool call in its assistant response; a command whose programs executed an "
                "action ends the turn with its output and the next actual observation, otherwise the turn "
                "continues. read, write and edit may be called alone or together.")

    def _code_mode_model_tools(self) -> list[str]:
        """The names of the model's own tools, in the order it is given them."""

        return [definition["function"]["name"] for definition in self.gateway.definitions()]

    def _code_mode_call_directly(self, name: str) -> str:
        """The end of a script's refusal of ``name``: where the model has the tool, to call it directly."""

        return (f"; call {name} directly, as its own tool call outside the script"
                if name in self._code_mode_model_tools() else "")

    def _code_mode_refusal(self, canonical: str, name: str, arguments: Any) -> ToolError | None:
        """``answers.refusal`` (extended): a name of no tool, or a ``targets`` wrapper the tool's schema has no room for."""
        tools = getattr(self, "tools", None)
        return code_answers.refusal(canonical, name, arguments, self._script_names.values(),
                                    reads=script_reads(self.code_mode), definitions=getattr(tools, "script_definitions", tuple),
                                    lifecycle=getattr(getattr(tools, "surface", None), "lifecycle_tools", ()))

    def _code_mode_terminal_refusals(self) -> dict[str, dict[str, Any]]:
        """The agent core's ``episode_terminal`` refusal of ``bash``, by name (code mode; empty otherwise).

        At the trusted terminal boundary the core refuses every world mutation, and ``bash`` is one whatever
        its command does, so no command runs any more; the refusal names what is left, the model's direct
        tools (those of ``DIRECT_TOOLS`` and the lifecycle tools it has), in ``constraints.allowed`` too. The list
        holds the direct tools only: the workspace tools, which do not change the world, still answer as well.
        """

        if not self.code_mode:
            return {}
        model_tools = self._code_mode_model_tools()
        lifecycle = set(getattr(getattr(self.tools, "surface", None), "lifecycle_tools", ()) or ())
        direct = [name for name in model_tools if name in DIRECT_TOOLS | lifecycle]
        message = ("world mutation is unavailable after the trusted terminal boundary, so no script or command runs "
                   "any more (a read-only one included)"
                   + (f"; call the tools left directly: {', '.join(direct)}" if direct else ""))
        return {name: {"message": message, "constraints": {"allowed": direct}}
                for name in model_tools if name == coding.BASH_TOOL}

    def _code_mode_allowance(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """What is left of the script's allowance (``library: extended``; nothing otherwise): world-changing calls
        against its action limit, other calls against ``max_calls_per_script``."""

        if self.code_mode.get("library") != "extended":
            return {}
        queries, seconds = query_allowance(self.code_mode)
        return {ALLOWANCE_FIELD: {
            "actions": max(0, int(state["cap"]) - int(state["actions"])),
            "calls": max(0, int(self.code_mode["max_calls_per_script"]) - int(state["calls"])),
            "queries": {"count": max(0, queries - int(state.get("queries", 0))),
                        "seconds": round(max(0.0, seconds - float(state.get("query_s", 0.0))), 3)},
        }}

    def _code_mode_charge_query(self, state: dict[str, Any], seconds: float = 0.0, *, count: int = 1) -> None:
        """Charge queries against the script's allowance (``library: extended``; ``state["queries"]`` and ``state["query_s"]``,
        limits ``query_allowance(self.code_mode)``). Called before the work as ``(state, 0.0)`` it admits one query: it raises
        ``script_query_limit`` (with ``constraints``) when the queries or the seconds are already spent, else counts it. Called
        after the work as ``(state, elapsed, count=0)`` it adds the seconds and never raises; a caller that charges once,
        after the work, passes the seconds with the default ``count=1``. The last query may overrun the seconds."""

        made, spent = int(state.get("queries", 0)), float(state.get("query_s", 0.0))
        if count:
            limit, budget = query_allowance(self.code_mode or {})
            if made >= limit or spent >= budget:
                raise ToolError(
                    f"this script already made {made} queries, its limit" if made >= limit else
                    f"this script already spent {spent:.1f} s on queries, its limit of {budget} s",
                    code="script_query_limit", retryable=False,
                    constraints={"max_queries": limit, "query_time_s": budget, "queries": made,
                                 "query_s": round(spent, 3)})
            state["queries"] = made + count
        state["query_s"] = spent + max(0.0, float(seconds))

    def _code_mode_ops(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """The read-only ops a script's sandbox may send besides its calls: ``docs``, ``explain`` and ``query`` with
        ``library: extended``; a query is charged to the script's ``state``, and a request is recorded under the script
        call ``state`` names (``model_call_id``, ``tool_call_id``)."""

        if (self.code_mode or {}).get("library") != "extended":
            return {}
        return {"handle_op": lambda message: self._code_mode_op(message, state)}

    def _code_mode_op(self, message: Mapping[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
        """A script's ``docs`` op: a function's documentation entry, or one part of it (``code_mode/docs.py``); with no
        name the index, as the script tool's description lists it. ``explain`` and ``query`` are answered
        elsewhere and, inside a script (``state`` names its call), recorded with what was asked
        (``_code_mode_record_op``)."""

        op = message.get("op")
        if op in ("explain", "query"):
            state = {} if state is None else state
            if op == "explain":
                name, arguments, answer = None, {"call_id": message.get("call_id")}, self._code_mode_explain(message)
            else:
                name, arguments = message.get("name"), message.get("arguments")
                answer = self._code_mode_query(message, state)
            if "tool_call_id" in state:
                self._code_mode_record_op(op, name, arguments, answer, model_call=state["model_call_id"],
                                          parent=state["tool_call_id"], state=state)
            return answer
        if op != "docs":
            return {"ok": False, "error": {"code": "op_unavailable", "retryable": False,
                                           "message": f"{message.get('op')} is not served in this code mode"}}
        index, entries = self._code_mode_docs()
        name, part = message.get("name"), message.get("part")
        if name is None:
            return {"ok": True, "text": index}
        spelled = {**{code_docs.wire_name(key): key for key in entries}, **{key: key for key in entries}}
        canonical = spelled.get(str(name))
        reads = script_reads(self.code_mode)
        parts = (entries[canonical] if canonical is not None  # or a public name of the library
                 else code_docs.fragment_parts(str(name), reads=reads, interaction=code_docs.labelled(entries)))
        if parts is None:
            own = str(name) in self._code_mode_model_tools()
            allowed = [*(code_docs.wire_name(key) for key in entries), *code_docs.library_names(reads)]
            near = code_docs.nearest(str(name), allowed)
            return {"ok": False, "error": {
                "code": "unknown_function", "retryable": False,
                "message": (f"{name} is one of your own tools, not a function; its description is in your tool list"
                            if own else f"no function {name!r}"
                            + (f"; the nearest: {', '.join(near)}" if near else "; robot.docs() lists them")),
                "constraints": {"allowed": allowed, "nearest": near}}}
        canonical = canonical or str(name)
        if part is None:
            return {"ok": True, "name": canonical, "text": code_docs.short(code_docs.wire_name(canonical), parts)}
        if part == code_docs.ALL:
            return {"ok": True, "name": canonical, "part": part, "text": code_docs.whole(parts)}
        if part not in parts:
            return {"ok": False, "error": {"code": "unknown_part", "retryable": False,
                                           "message": f"{canonical} has no part {part!r}",
                                           "constraints": {"allowed": [*parts, code_docs.ALL]}}}
        return {"ok": True, "name": canonical, "part": part, "text": parts[part] + "\n"}

    def _code_mode_query(self, message: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """A script's ``query`` op: arguments checked, the data interface bound to the current observation, then the query
        admitted against the allowance and made. Nothing before the admission costs a query."""

        from ora.agent.toolkit.host.query import query_api

        try:
            arguments = message.get("arguments")
            call = code_query.prepare(message.get("name"), {} if arguments is None else arguments)
            api = query_api(self.tools)
            self._code_mode_charge_query(state, 0.0)
            started = time.monotonic()
            try:
                result = call(api)
            finally:
                self._code_mode_charge_query(state, time.monotonic() - started, count=0)
        except ToolError as exc:
            return _tool_error(exc)
        except Exception as exc:  # noqa: BLE001 - the script waits for an answer
            return _tool_error(ToolError(f"the query failed in the policy session ({type(exc).__name__})",
                                         code="internal_tool_error", retryable=False))
        return {"ok": True, "name": message["name"], "result": result}

    def _code_mode_explain(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """A script's ``explain`` op: what the call ``call_id`` did, in causal order (``toolkit/explain.py``), from the
        inputs kept in memory or, for an older call or after a restart, recorded with its result."""

        call_id = message.get("call_id")
        inputs = (self._code_mode_calls or {}).get(call_id) if isinstance(call_id, str) else None
        if inputs is None and isinstance(call_id, str):
            inputs = code_record.recorded_inputs(self.run_root, call_id)
        if inputs is None:
            return {"ok": False, "error": {
                "code": "unknown_call", "retryable": False,
                "message": f"no script call {call_id!r} in this episode's record; explain takes the result of a call "
                           "or its call_id"}}
        explained = code_record.render(inputs)
        if explained is None:
            return {"ok": False, "error": {
                "code": "no_explanation", "retryable": False,
                "message": f"{inputs['tool']} has no explanation yet; explain covers {', '.join(toolkit_explain.tools())}",
                "constraints": {"allowed": list(toolkit_explain.tools())}}}
        return {"ok": True, "call_id": call_id, "text": explained.text}

    def _code_mode_remember(self, call_id: str, inputs: Mapping[str, Any]) -> toolkit_explain.Explanation | None:
        """Keep a call's explain inputs for ``explain`` (the latest ``EXPLAIN_CALLS``); its explanation if its tool
        has a template."""

        if self._code_mode_calls is None:
            self._code_mode_calls = {}
        self._code_mode_calls[call_id] = dict(inputs)
        while len(self._code_mode_calls) > EXPLAIN_CALLS:
            del self._code_mode_calls[next(iter(self._code_mode_calls))]
        return code_record.render(inputs)

    def _code_mode_explain_record(self, tool: str, arguments: Mapping[str, Any], answer: Mapping[str, Any],
                                  events: list[dict[str, Any]], held: dict[str, Any]) -> dict[str, Any]:
        """A call's explain inputs (``code_mode/record.py``), held for ``_code_mode_remember`` and returned as the field
        its ``tool_result`` records them in."""

        held["inputs"] = code_record.explain_inputs(tool, arguments, answer, events)
        return {code_record.EXPLAIN_INPUTS_FIELD: self.artifacts.put_json(held["inputs"],
                                                                          kind=code_record.EXPLAIN_INPUTS_KIND)}

    def _code_mode_record_op(self, op: str, name: Any, arguments: Any, answer: Mapping[str, Any], *, model_call: str,
                             parent: str, state: dict[str, Any]) -> str:
        """Record a script's library request as a nested call of the script, as its tool calls are; the record's id.

        ``op`` is ``query`` or ``explain``. The ``tool_call_proposed``/``tool_result`` pair is named
        ``robot.<op>`` (on no tool surface: ``evidence.agent_events.library_op`` tells it from a tool call), has the
        effect ``read_only`` and ``origin`` ``{kind, parent_tool_call_id, seq, op}``; its arguments are what was asked,
        ``{"name": name, "arguments": arguments}`` (``name`` is the query, None for ``explain``), and its
        result is ``answer``, a refusal too. It takes the next ``seq`` of the script's calls (``state["seq"]``), so the
        script's later calls number after it; no call limit counts it."""

        state["seq"] += 1
        seq = state["seq"]
        call_id, recorded = f"{parent}#{seq:03d}", f"robot.{op}"
        origin = {"kind": ORIGIN_KIND, "parent_tool_call_id": parent, "seq": seq, "op": op}
        step = self._robot_step()
        proposal = self._script_record("tool_call_proposed", {
            "causal_parent_id": self.last_fresh_observation_event_id, "observation_step": step,
            "model_call_id": model_call, "tool_call_id": call_id,
            "payload": {"name": recorded, "arguments": {"name": name, "arguments": arguments}, "effect": "read_only",
                        "batch_valid": True, "origin": origin},
        })
        self._script_record("tool_result", {
            "observation_step": step, "model_call_id": model_call, "tool_call_id": call_id,
            "causal_parent_id": proposal["event_id"],
            "payload": {"name": recorded, "ok": answer.get("ok") is True, "origin": origin},
            "artifact_ref": self.artifacts.put_json(dict(answer), kind="native_tool_result"),
        })
        return call_id

    def _code_mode_docs(self) -> tuple[str, dict[str, dict[str, str]]]:
        """The index and every function's entry by part, from the definitions scripts call (built once)."""

        if self._code_mode_documentation is None:
            surface = getattr(self.tools, "surface", None)
            lifecycle = tuple(getattr(surface, "lifecycle_tools", ()) or ())
            results = getattr(surface, "result_schemas", None) or {}
            functions = script_functions(list(self.tools.script_definitions()), lifecycle)
            reads = script_reads(self.code_mode)
            entries = {
                definition["name"]: code_docs.entry_parts(
                    definition, results.get(definition["name"]),
                    examples=code_docs.examples(definition["name"], reads=reads, definition=definition),
                    explain=(toolkit_explain.documentation(spec) if (spec := toolkit_explain.template(
                        definition["name"])) else None),
                    reads=reads)
                for definition in functions}
            self._code_mode_documentation = (
                code_docs.listing(functions, results, reads=reads),
                {**entries, **(code_docs.query_entries() if "query" in reads else {})})
        return self._code_mode_documentation

    def _code_mode_cap(self) -> int:
        cap = int(self.code_mode["max_actions_per_script"])
        if self._code_mode_decisions is not None:
            left = self._code_mode_decisions - (self.robot.actions_this_turn if self.robot is not None else 0)
            cap = min(cap, max(1, left // 2))
        return cap

    def _code_mode_bash(self, arguments: Any, call_context: Mapping[str, Any]) -> dict[str, Any]:
        """``bash`` (plan ``policy.code_mode``): one command in the workspace, its robot calls served."""

        contract = self.code_mode
        try:
            command = validate_command(arguments, contract)
            root = code_workspace.workspace_root(self.run_root)
            code_workspace.check_bounds(root)
        except ToolError as exc:
            return _tool_error(exc)
        model_call, parent = str(call_context["model_call_id"]), str(call_context["tool_call_id"])
        cap, play = self._code_mode_cap(), getattr(self, "play", None) is not None  # free play: Esc stops a script
        state: dict[str, Any] = {"seq": 0, "actions": 0, "calls": 0, "cap": cap, "log": [], "frames": [],
                                 "uncounted_s": 0.0, "queries": 0, "query_s": 0.0,
                                 "model_call_id": model_call, "tool_call_id": parent}
        wall = min(float(contract["script_wall_time_s"]), float(arguments.get("timeout") or contract["script_wall_time_s"]))
        self._serving += 1  # a core lost while a script call records is never resumed (core_process._call)
        try:
            with self._code_mode_parent(coding.BASH_TOOL, call_context, cap) as invocation, \
                    observation_mount(contract, self.tools.control_root) as observation:
                state["parent"] = invocation
                outcome = run_command(command, handle_call=lambda name, call_arguments: self._code_mode_call(
                    name, call_arguments, model_call=model_call, parent=parent, state=state),
                    wall_s=wall, cpu_s=int(contract["script_cpu_time_s"]),
                    memory_mib=int(contract["script_memory_mib"]), workspace=root,
                    uncounted_s=lambda: state["uncounted_s"],
                    extended=contract.get("library") == "extended", observation=observation,
                    source="source" in script_reads(contract), **self._code_mode_ops(state),
                    **({"should_stop": self.tool_action_executor.stop_requested} if play else {}))
        except SandboxError as exc:
            return _tool_error(ToolError(f"the script sandbox is unavailable on this host: {exc}",
                                         code="code_mode_sandbox_unavailable", retryable=False))
        finally:
            self._serving -= 1
        limit = int(contract["max_output_chars"])
        output = outcome.output if len(outcome.output) <= limit else \
            f"[... {len(outcome.output) - limit} characters cut ...]\n" + outcome.output[-limit:]
        result: dict[str, Any] = {
            "ok": outcome.status in {"completed", "handed_back"},
            "status": outcome.status,
            "exit_code": outcome.exit_code,
            **({"handed_back": outcome.handed_back} if outcome.handed_back else {}),
            **({"error_detail": outcome.error} if outcome.error else {}),
            "output": output,
            "calls": state["log"],
            "actions_executed": state["actions"],
            "action_limit": cap,
            **self._code_mode_allowance(state),
            "wall_s": outcome.wall_s,
            "workspace": code_workspace.snapshot(root, self.run_root / "native" / "code-workspace-objects")[:64],
        }
        frames = state["frames"][-int(contract["max_frames_per_script"]):] if contract["max_frames_per_script"] else []
        if frames:
            result["images_base64"] = [base64.b64encode(Path(path).read_bytes()).decode("ascii") for path in frames]
        return result

    def _code_mode_call(self, name: str, arguments: Mapping[str, Any], *, model_call: str, parent: str,
                        state: dict[str, Any]) -> dict[str, Any]:
        """One tool call of a script: limits first, then the free-play script path."""

        state["seq"] += 1
        seq = state["seq"]
        canonical = self._script_names.get(name, name)
        extended = self.code_mode.get("library") == "extended"
        refusal: Exception | None = self._code_mode_refusal(canonical, name, arguments) if extended else None
        effect: str | None = None
        if self._code_mode_dry_asked(canonical, arguments):  # dry_run=...: never the tool's own argument
            arguments = dict(arguments)
            asked = arguments.pop(dry_run.ARGUMENT)
            if asked is True:
                return self._code_mode_dry_run(canonical, name, arguments, seq=seq, model_call=model_call,
                                               parent=parent, state=state, refusal=refusal)
            if asked is not None and asked is not False:
                refusal = ToolError("dry_run must be true or false", code="invalid_arguments", retryable=True,
                                    field=dry_run.ARGUMENT, received=asked)
        if canonical in NOT_FROM_SCRIPTS | WITHHELD:
            way = ("; compute in the script itself (np, Rotation, math)" if canonical in WITHHELD
                   else self._code_mode_call_directly(canonical))
            refusal = ToolError(f"{canonical} cannot be called from a script{way}",
                                code="script_tool_unavailable", retryable=False)
        else:
            try:
                effect = self.gateway.classify_effect(canonical, self.gateway.arguments(canonical, dict(arguments)))
            except Exception:  # unknown tool or bad arguments: _script_call refuses and records it
                effect = None
            kept = self._code_mode_kept_answer(canonical, arguments, effect)
            if kept is not None:  # the same look at this observation again: no tool runs, no limit counts
                state["log"].append({"call": seq, "tool": canonical, "ok": True, "reused": True})
                return kept
            if effect == "lifecycle":
                # Free play's refusal (robot_socket._script_call), saying where the model finishes instead.
                refusal = ToolError("a lifecycle tool is the agent's own decision; a script cannot call it"
                                    + self._code_mode_call_directly(canonical),
                                    code="script_lifecycle_unavailable", retryable=False)
            elif effect == "mutation" and state["actions"] >= state["cap"]:
                refusal = ToolError(
                    f"this script already ran {state['actions']} world-changing call(s), its limit; return, and "
                    "act again on the next observation", code="script_action_limit", retryable=False,
                    constraints={"max_actions": state["cap"]})
            elif effect != "mutation" and state["calls"] >= int(self.code_mode["max_calls_per_script"]):
                refusal = ToolError(f"this script already made {state['calls']} other call(s), its limit",
                                    code="script_call_limit", retryable=False,
                                    constraints={"max_calls": self.code_mode["max_calls_per_script"]})
        origin = {"kind": ORIGIN_KIND, "parent_tool_call_id": parent, "seq": seq}
        before = len(_executed(state.get("parent")))
        held: dict[str, Any] = {}
        started = time.monotonic()

        def answered(outcome: dict[str, Any]) -> dict[str, Any]:
            """What the script reads of the call, held for below, and its explain inputs, recorded with its result."""

            executed = len(_executed(state.get("parent"))) - before
            held["answer"] = self._code_mode_answer(canonical, outcome,
                                                    index=state["actions"] + executed if executed > 0 else None)
            return self._code_mode_explain_record(canonical, arguments, held["answer"], events, held)

        with stages.collecting() if extended else nullcontext([]) as events:  # the tool's stage events, for explain
            outcome = self._script_call(
                ScriptCall(connection=0, request_id=seq, name=name, arguments=dict(arguments), origin={}),
                ids=(model_call, f"{parent}#{seq:03d}"), origin=origin,
                preset=None if refusal is None else (effect, refusal),
                **({"result_fields": answered} if extended else {}),
            )
        # The parent's ledger counts what ran: a legacy submission executed through it, or a synchronous
        # toolkit tool's own primitives.
        executed = len(_executed(state.get("parent"))) - before
        acted = executed > 0
        if acted:
            state["actions"] += executed
        else:
            state["calls"] += 1
        # An action's time is the Supervisor's, not the script's: a long move does not spend the script's wall time.
        state["uncounted_s"] += float(outcome.get("uncounted_s") or (time.monotonic() - started if acted else 0.0))
        answer = held["answer"] if extended else self._code_mode_answer(canonical, outcome,
                                                                        index=state["actions"] if acted else None)
        failed = code_answers.item_errors(answer) if extended else None  # after the explain inputs: the tool's own
        answer.update({"error": failed} if failed else {})
        entry: dict[str, Any] = {"call": seq, "tool": canonical, "ok": answer["ok"], **({"summary": failed["message"]}
                                                                                          if failed else {})}
        returned = [item["path"] for item in outcome.get("images") or [] if isinstance(item, Mapping) and item.get("path")]
        if returned:
            entry["images"] = len(returned)
            state["frames"].extend(returned)
        if acted:
            entry["action"] = state["actions"]
            if outcome.get("sequence") and answer.get("receipt") is not None:
                # The staged receipt (stages, stop_reason, verification) tool-call mode shows on the next turn:
                # the model reads it here even when the script does not print it or ends before it could.
                entry["receipt"] = answer["receipt"]
            seen = ScriptResult(canonical, answer)
            if not seen.ok and seen.summary:
                # A failed action's line, a planner refusal's own words included: the model reads it here even
                # when the script prints only a code, as it reads the staged receipt.
                entry["summary"] = seen.summary
        if isinstance(answer.get("error"), Mapping):
            entry["error"] = answer["error"].get("code")
        if extended:
            answer["call_id"] = f"{parent}#{seq:03d}"
            explained = self._code_mode_remember(answer["call_id"], held["inputs"])
            if explained is not None and (explained.failed or not answer["ok"]):  # a failed call explains itself
                answer["explain"] = code_record.attached(explained.text)
                if state.get("explained", 0) + len(answer["explain"]) <= toolkit_explain.SCRIPT_CHARS:
                    entry["explain"] = answer["explain"]  # the result's calls carry a bounded share
                    state["explained"] = state.get("explained", 0) + len(answer["explain"])
        state["log"].append(entry)
        if not acted:
            self._code_mode_keep(canonical, arguments, effect, answer)
        return answer

    def _code_mode_dry_asked(self, name: str, arguments: Mapping[str, Any]) -> bool:
        """Whether a call names ``dry_run`` for a tool that has a dry run."""

        return (bool(self.code_mode) and name.rsplit(".", 1)[-1] in dry_run.TOOLS
                and isinstance(arguments, Mapping) and dry_run.ARGUMENT in arguments)

    def _code_mode_dry_run(self, canonical: str, name: str, arguments: Mapping[str, Any], *, seq: int,
                           model_call: str, parent: str, state: dict[str, Any],
                           refusal: Exception | None = None) -> dict[str, Any]:
        """A script's dry run of a motion tool: the plan the real call would run, not its effect.

        The flag reaches the tool out of band (``dry_run.dry``), never as an argument, and only where the surface
        plans on the policy side (a toolkit tool; the environment plans a legacy tool's move when it executes, so a
        dry run is refused ``dry_run_unavailable`` naming why, and nothing runs). The call is recorded as a nested
        call of the script with ``origin.dry_run`` and the effect ``read_only``: it never proposes an action, spends
        no native action and counts against the script's queries (``_code_mode_charge_query``), not its calls or
        actions. A plan the real call would refuse is the answer (``feasible: false``), not an error."""

        call_id = f"{parent}#{seq:03d}"
        if refusal is None:
            try:
                arguments = dict(self.gateway.arguments(canonical, dict(arguments)))
            except Exception as exc:  # the call the tool would refuse
                refusal = exc
        charged = False
        if refusal is None:
            try:
                self._code_mode_charge_query(state, 0.0)
                charged = True
            except ToolError as exc:
                refusal = exc
        started = time.monotonic()
        held: dict[str, Any] = {}
        started = time.monotonic()

        def answered(outcome: dict[str, Any]) -> dict[str, Any]:
            held["answer"] = self._code_mode_dry_answer(canonical, outcome)
            return self._code_mode_explain_record(canonical, arguments, held["answer"], events, held)

        with stages.collecting() as events, dry_run.dry():
            outcome = self._script_call(
                ScriptCall(connection=0, request_id=seq, name=name, arguments=dict(arguments), origin={}),
                ids=(model_call, call_id), origin={"kind": ORIGIN_KIND, "parent_tool_call_id": parent, "seq": seq,
                                                   "dry_run": True},
                preset=("read_only", refusal), result_fields=answered)
        if charged:
            self._code_mode_charge_query(state, time.monotonic() - started, count=0)
        state["uncounted_s"] += float(outcome.get("uncounted_s") or 0.0)
        answer = held["answer"]
        answer["call_id"] = call_id
        entry: dict[str, Any] = {"call": seq, "tool": canonical, "ok": answer["ok"], "dry_run": True}
        if isinstance(answer.get("error"), Mapping):
            entry["error"] = answer["error"].get("code")
        # What the model reads whether the script prints it or not: the numbers here, the picture with the script's frames.
        if isinstance(code_answers.preview_line(answer), str):
            entry["summary"] = code_answers.preview_line(answer)
        returned = [item["path"] for item in outcome.get("images") or [] if isinstance(item, Mapping) and item.get("path")]
        if returned:
            entry["images"] = len(returned)
            state["frames"].extend(returned)
        explained = self._code_mode_remember(call_id, held["inputs"])
        if explained is not None:  # a dry run explains itself: its plan is the answer
            answer["explain"] = code_record.attached(explained.text)
            if explained.failed and state.get("explained", 0) + len(answer["explain"]) <= toolkit_explain.SCRIPT_CHARS:
                entry["explain"] = answer["explain"]
                state["explained"] = state.get("explained", 0) + len(answer["explain"])
        state["log"].append(entry)
        return answer

    def _code_mode_dry_answer(self, name: str, outcome: Mapping[str, Any]) -> dict[str, Any]:
        """What the script reads of a dry run: the plan's record in degrees as the surface delivers angles, through the
        causal filter; a refused call reads as any refused call does."""

        result = outcome.get("result") or {}
        if isinstance(result.get("error"), Mapping) or result.get("dry_run") is not True:
            return {**self._code_mode_answer(name, outcome, index=None), "dry_run": True}
        capabilities = getattr(getattr(self.tools, "surface", None), "capabilities", ()) or ()
        record = causal_provider_value(policy_degrees(result) if DEGREES in capabilities else dict(result))
        # Each arm's predicted axes in world, as tool-call mode gives them (``session_core/domain.py``).
        record = code_answers.predicted_axes(record, getattr(getattr(self.tools, "surface", None), "view_config", None))
        return {"ok": bool(outcome.get("ok")), "tool": name, "dry_run": True, "result": record}

    def _code_mode_kept_call(self, name: str, arguments: Mapping[str, Any],
                             effect: str | None) -> tuple[KeptResults, str | None, dict[str, Any]] | None:
        """What keeping or reusing a call needs (``library: extended``, a call the surface classes read-only): the
        kept results, the current observation's id and the call's arguments as the tool receives them."""

        if effect != "read_only" or (self.code_mode or {}).get("library") != "extended":
            return None
        try:
            normalized = dict(self.gateway.arguments(name, dict(arguments)))
        except Exception:  # the call is refused where it runs
            return None
        if self._code_mode_kept is None:
            self._code_mode_kept = KeptResults()
        return self._code_mode_kept, self._current_observation_id(), normalized

    def _code_mode_kept_answer(self, name: str, arguments: Mapping[str, Any],
                               effect: str | None) -> dict[str, Any] | None:
        """The kept answer to a repeat of a perception call on the current observation, marked reused
        (``code_mode/reuse.py``); None for any other call."""

        call = self._code_mode_kept_call(name, arguments, effect)
        return None if call is None else call[0].get(call[1], name, call[2])

    def _code_mode_keep(self, name: str, arguments: Mapping[str, Any], effect: str | None,
                        answer: Mapping[str, Any]) -> None:
        """Keep a perception call's answer for the observation it was made on."""

        call = self._code_mode_kept_call(name, arguments, effect)
        if call is not None:
            call[0].keep(call[1], name, call[2], answer)

    def _code_mode_answer(self, name: str, outcome: Mapping[str, Any], *, index: int | None) -> dict[str, Any]:
        """What the script reads: the projected result; after an action its number and its receipt.

        The result is projected as the model would read it and the receipt passes the causal filter of
        the model's context, so the script holds nothing the model may not read (no event ids, paths or
        clocks); an action is named by its number in the script.
        """

        step = outcome.get("observation_step")
        projected = self._project(name, outcome.get("result") or {}, call_id=str(outcome.get("call_id")),
                                  step=step if isinstance(step, int) else -1)
        visible = projected["visible"]
        answer: dict[str, Any] = {"ok": bool(outcome.get("ok")), "tool": name, "result": visible}
        if isinstance(visible.get("error"), Mapping):
            answer["error"] = visible["error"]
        if isinstance(outcome.get("action_error"), Mapping):
            answer.update({"ok": False, "error": dict(outcome["action_error"])})
        if index is not None:
            answer["action"] = index
            if outcome.get("receipt") is not None:
                answer["receipt"] = causal_provider_value(outcome.get("receipt"))
        return answer

    def _code_mode_action(self) -> dict[str, Any]:
        """Execute a legacy tool's submitted action inside the script's parent invocation.

        A LIBERO or RoboTwin tool persists its proposal and returns ``action_submitted``; here the
        Supervisor's executor admits, executes and publishes it at once, as a synchronous toolkit tool's
        primitive, so the parent's ledger holds it. RoboTwin's closed-skill ``grasp_execute`` persists a
        committed skill sequence, whose stage loop runs here as that one action; the stage loop's time
        (``uncounted_s``) is the Supervisor's, not the script's, and the result adds its post-close views.

        An action that did not start (``ActionNotStarted``), or that the executor refused (its ledger keeps
        the refusal), leaves no proposal behind, nor a committed skill's ledger, and reaches the script as a
        refusal: the observation stays open for the script's next action, which carries the same identity.
        Any other failure may follow the dispatch, so it interrupts the parent invocation.
        """

        executor = self.tool_action_executor
        path = self.tools.control_root / "submitted-action.json"
        action = json.loads(path.read_text(encoding="utf-8"))
        sequence = action.get("schema") == SKILL_SEQUENCE_ACTION_SCHEMA
        started = time.monotonic()
        try:
            executed = executor.execute_action(action, executor.native_actions_per_action())
        except ActionNotStarted as exc:  # nothing reached the world
            self._code_mode_withdraw(path, action, sequence=sequence)
            return {"action_error": {"code": "action_not_started", "message": str(exc)}}
        receipt = dict(executed.get("receipt") or {})
        if is_supervisor_refusal(receipt):  # nothing ran and nothing was published
            self._code_mode_withdraw(path, action, sequence=sequence)
            return {"action_error": {"code": str(receipt.get("reason") or "action_refused"),
                                     "message": "the Supervisor refused the action"}, "receipt": receipt}
        result = {"receipt": receipt, "remaining_budget": executed.get("remaining_budget")}
        if sequence:
            result.update(sequence=True, uncounted_s=time.monotonic() - started, images=self._code_mode_skill_views())
        return result

    def _code_mode_withdraw(self, path: Path, action: Mapping[str, Any], *, sequence: bool) -> None:
        """Remove what a proposal that never ran left behind: its file, its closed observation and, for a
        committed skill, the ledger holding it as running, which would refuse every later commit
        (``skill_sequence_active``). The ledger reports the way the stage loop's every exit does."""

        path.unlink(missing_ok=True)
        self._code_mode_reopen(action)
        if sequence:
            self.tools.build_skill_receipt(action=action, status="aborted", stages=[],
                                           observation={"observation_id": action["observation_id"]},
                                           stop_reason="sequence_not_started")

    def _code_mode_skill_views(self) -> list[dict[str, Any]]:
        """A committed grasp's post-close views (``skill_views.py``) as images of its call, where the next
        observation does not show them (``toolkit_observation_receipts`` false, RoboTwin's code mode)."""

        surface = getattr(self.tools, "surface", None)
        if getattr(self, "toolkit_observation_receipts", True) or surface is None \
                or not surface.supports("prepared_grasp_execution_v2"):
            return []
        views = materialize_skill_observations(receipt=self.pending_receipt_body, run_root=self.run_root,
                                               workspace=self.run_root / "native",
                                               camera_names=surface.view_config["automatic_actual_views"])
        return [{"path": str(self.run_root / item["run_path"]), "sha256": item["sha256"], "media_type": "image/png"}
                for item in views]

    def _code_mode_reopen(self, action: Mapping[str, Any]) -> None:
        """Reopen the observation a not-started or refused proposal closed (``_persist_action``), so the script
        and the model may act on it again; an observation published since is left alone."""

        pointer = self.tools.control_root / "current-observation.json"
        control = json.loads(pointer.read_text(encoding="utf-8"))
        if control.get("open") is True or control.get("observation_id") != action.get("observation_id"):
            return
        control["open"] = True
        temporary = pointer.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(pointer)


def _executed(parent: Any) -> list[Mapping[str, Any]]:
    """The actions a parent invocation's ledger records as run (refusals excluded)."""

    value = getattr(getattr(parent, "log", None), "value", None)
    actions = value.get("actions") if isinstance(value, Mapping) else None
    return [item for item in actions or [] if isinstance(item, Mapping) and not item.get("refusal")]


__all__ = ["ORIGIN_KIND", "PiCodeMode"]
