"""Coding tools (plan ``policy.code_mode``): the model works like a coding agent.

The model's tools are those of a coding agent, named and
shaped as Pi's defaults -- ``read``, ``write``, ``edit``, ``bash`` -- confined to
the persistent workspace ``/work``, plus the steps that stay the model's own
(finishing, the completion check, the issue report, the artifact reader). Every
robot tool is a function of the ``robot`` library: any Python process a ``bash``
command starts can ``from robot import tools, handoff`` and call
``tools.<name>(**arguments)``. ``bash`` runs in the sandbox (``sandbox.py``,
``shell.py``) and the Supervisor admits every action one at a time; a command
whose programs acted ends the turn.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .contract import (DIRECT_TOOLS, NOT_FROM_SCRIPTS, WITHHELD, current_generation, query_allowance, script_functions,
                       script_library, script_reads, wire_name)
from .docs import interaction_form, listing as docs_listing

READ_TOOL, WRITE_TOOL, EDIT_TOOL, BASH_TOOL = "read", "write", "edit", "bash"
FILE_TOOLS = frozenset({READ_TOOL, WRITE_TOOL, EDIT_TOOL})
CODING_TOOLS = FILE_TOOLS | {BASH_TOOL}

BASH_DESCRIPTION = """\
Execute a bash command in your persistent workspace /work (the current directory, kept for the whole episode). \
Returns the combined stdout and stderr, truncated to its last {chars} characters. Commands run in a sandbox with no \
network and nothing outside /work and /tmp; `python` is Python 3.11 with NumPy and SciPy. This is how you act: \
every robot capability -- looking, perception, measurement, every motion -- is a function of the `robot` library, \
which any Python program you run can import:

    {example}   # a dict: result, and for an action its receipt{receipt}

A call the Supervisor refuses, or that fails, raises `robot.ToolRefused` (`.code`, `.answer`); actions already \
executed are never undone, and each one is admitted and executed one at a time, exactly as if you had called it \
yourself. After an action the world has moved on: the perception and measurement functions called next see the new \
state. `handoff(reason, **data)` stops the command there and gives you the reason and data first. Keep your policy \
as modules in /work and run them (`python -m policy`, `python step.py`); there is no separate calculation function: \
compute in your code. Per command: at most {actions} actions (and at most half of the robot decisions left), \
{calls} other robot calls{queries}, {seconds} s{resources}.{rules} If a command's programs executed any action, your \
turn ends with its result and the next actual observation; otherwise this turn continues. The `robot` library's \
functions:"""
#: BASH_DESCRIPTION's limits that only ``library: extended`` states (``_limits``), so a ``base`` description keeps its
#: text: the query allowance (``contract.QUERY_LIMITS``) after the calls, the programs' CPU time and memory
#: (``script_cpu_time_s``, ``script_memory_mib``) after the wall time, then that ``/tmp`` is the command's own
#: (``sandbox.py`` mounts a fresh tmpfs each time) and which of the model's own tools a script cannot call
#: (``contract.NOT_FROM_SCRIPTS`` and the lifecycle tools).
LIMITS_WORDING = {
    "queries": ", {count} {spent} ({seconds} s of them)",
    "resources": " of wall time, {cpu} s of CPU time and {memory} MiB of memory for its programs",
    "rules": " `/tmp` is a fresh, empty directory for every command: keep what a later command reads in /work.",
    "direct": " A script cannot call {names}: call {them} directly, outside the script.",
}
#: What BASH_DESCRIPTION's example says a call returns after its receipt, by ``prompt``: v2 names where a tool that
#: executes inside its own call (RoboDojo's synchronous toolkit) carries the receipt (the retired v1 promised a robot
#: state the session never returns).
BASH_RECEIPT_WORDING = {"v2": " (a tool that executes inside its own call carries it as result.receipt)"}
BASH_RECEIPT_WORDING["v3"] = BASH_RECEIPT_WORDING["v2"]
#: BASH_DESCRIPTION's example. Where ``move_pose`` takes ``targets``, the index beside it (``library: extended``) shows
#: that call; ``base`` keeps the sketch it has always had, its schemas follow it.
BASH_EXAMPLE = "from robot import tools, handoff\n    r = tools.move_pose(position_m=[...], ...)"
BASH_TARGETS_EXAMPLE = 'from robot import tools, handoff, target\n    r = tools.move_pose(targets={"left": target(p_world)})'
#: After BASH_DESCRIPTION in place of every function's description and schema, with ``library: extended``: how to read
#: the lines that follow, what ``robot.docs`` is for, then the index and the card (``docs.py``).
BASH_INDEX_HEADER = """ one line each: `tools.<name>(arguments) -> result: purpose`. Each line gives the call's \
arguments (by keyword; `?` marks an optional one, `{...}` a dict, `a|b` the values allowed, `[3]` three numbers, \
`[int; 2]` two integers, `[x; 1-8]` one to eight of x, `...` what the line leaves out) and its result. A call \
returns a `Result`, the dict described above with its outcome as fields (the library's lines below list them); \
`r["result"]` holds what `->` shows. `robot.docs("<name>")` (`import robot`), a string to print, also shown by \
`help(tools.<name>)`, is for what a line does not settle: a function's description, units and frames, and the sizes \
of its other parts, each its own lookup (`robot.docs("<name>", "result")`, `robot.docs("<name>", "all")` for all of \
it). It knows the library's names too (`robot.docs("Point")`). A lookup is no robot call and counts against no limit, \
but a reply spent on it is a model call: put it in the same command as the work that needs it, several in one \
command. `target` comes from `robot`, not `ora_lib`:
"""

READ_DESCRIPTION = """\
Read a file of your workspace /work with line numbers; `offset` is the first line (1-indexed), `limit` the number \
of lines (default 400). An empty `path` lists the workspace's files."""
WRITE_DESCRIPTION = """\
Write a file of your workspace /work. Creates it if it does not exist, overwrites it if it does; creates parent \
directories."""
EDIT_DESCRIPTION = """\
Edit a file of your workspace /work by exact text replacement. Each `edits[].oldText` must match a unique region of \
the file; the edits apply in order, all or none."""


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


def definitions(contract: Mapping[str, Any], functions: list[Mapping[str, Any]],
                result_schemas: Mapping[str, Any] | None = None,
                direct: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The code tools' definitions; ``direct`` names the model's own tools a script cannot call (``advertised``)."""

    path = {"type": "string", "description": "A path relative to /work."}
    return [
        {"name": READ_TOOL, "description": READ_DESCRIPTION, "inputSchema": _object({
            "path": path, "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000}}, ["path"])},
        {"name": WRITE_TOOL, "description": WRITE_DESCRIPTION, "inputSchema": _object({
            "path": path, "content": {"type": "string"}}, ["path", "content"])},
        {"name": EDIT_TOOL, "description": EDIT_DESCRIPTION, "inputSchema": _object({
            "path": path, "edits": {"type": "array", "minItems": 1, "maxItems": 32, "items": _object({
                "oldText": {"type": "string", "minLength": 1}, "newText": {"type": "string"}},
                ["oldText", "newText"])}}, ["path", "edits"])},
        {"name": BASH_TOOL, "description": BASH_DESCRIPTION.format(
            chars=contract["max_output_chars"], actions=contract["max_actions_per_script"],
            calls=contract["max_calls_per_script"], seconds=contract["script_wall_time_s"],
            receipt=BASH_RECEIPT_WORDING[current_generation(contract)], example=_example(functions, contract),
            **_limits(contract, direct),
        ) + _functions(functions, contract, result_schemas), "inputSchema": _object({
            "command": {"type": "string", "minLength": 1, "maxLength": int(contract["max_script_chars"])},
            "timeout": {"type": "integer", "minimum": 1, "maximum": int(contract["script_wall_time_s"])}},
            ["command"])},
    ]


def _limits(contract: Mapping[str, Any], direct: Sequence[str]) -> dict[str, str]:
    """BASH_DESCRIPTION's per-command limits beyond its calls (``library: extended``; nothing otherwise): the query
    allowance where queries or dry runs are served, the programs' CPU time and memory, ``/tmp`` and the model's own
    tools a script cannot call (``LIMITS_WORDING``), each from the contract."""

    if script_library(contract) != "extended":
        return {"queries": "", "resources": "", "rules": ""}
    reads = script_reads(contract)
    spent = " and ".join(word for read, word in (("query", "queries"), ("dry_run", "dry runs")) if read in reads)
    count, seconds = query_allowance(contract)
    return {"queries": LIMITS_WORDING["queries"].format(count=count, seconds=seconds, spent=spent) if spent else "",
            "resources": LIMITS_WORDING["resources"].format(cpu=contract["script_cpu_time_s"],
                                                            memory=contract["script_memory_mib"]),
            "rules": LIMITS_WORDING["rules"] + (LIMITS_WORDING["direct"].format(
                names=" or ".join(f"`{name}`" for name in direct), them="it" if len(direct) == 1 else "them")
                if direct else "")}


def _example(functions: list[Mapping[str, Any]], contract: Mapping[str, Any]) -> str:
    """BASH_DESCRIPTION's example: a call that runs on this surface, with ``library: extended``."""

    arms = script_library(contract) == "extended" and any(interaction_form(item) == "arm" for item in functions)
    return BASH_TARGETS_EXAMPLE if arms else BASH_EXAMPLE


def _api(functions: list[Mapping[str, Any]]) -> str:
    return "".join(f"\n\n### tools.{wire_name(definition['name'])}\n{definition.get('description', '').strip()}\n"
                   "Arguments: " + json.dumps(definition.get("inputSchema") or {}, sort_keys=True, separators=(",", ":"))
                   for definition in functions)


def _functions(functions: list[Mapping[str, Any]], contract: Mapping[str, Any],
               result_schemas: Mapping[str, Any] | None = None) -> str:
    """The ``robot`` library's functions as ``bash``'s description lists them: each one's description and schema
    (``base``), or the index and the library card (``extended``)."""

    if script_library(contract) == "extended" and functions:
        return BASH_INDEX_HEADER + docs_listing(functions, result_schemas, reads=script_reads(contract))
    return _api(functions)


def advertised(base: list[Mapping[str, Any]], contract: Mapping[str, Any], lifecycle_tools: tuple[str, ...] = (),
               result_schemas: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """What the model sees: the direct tools it keeps, then ``read``, ``write``, ``edit`` and ``bash``, whose description
    lists the others as functions (``result_schemas`` give the index their results)."""

    kept = [dict(definition) for definition in base
            if definition["name"] in (DIRECT_TOOLS | set(lifecycle_tools)) - WITHHELD]
    direct = [definition["name"] for definition in kept if definition["name"] in NOT_FROM_SCRIPTS | set(lifecycle_tools)]
    return [*kept, *definitions(contract, script_functions(base, lifecycle_tools), result_schemas, direct)]


BASH_RESULT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {
    "ok": {"type": "boolean"},
    "status": {"enum": ["completed", "failed", "handed_back", "timeout", "crashed"]},
    "exit_code": {"type": ["integer", "null"]},
    "output": {"type": "string"},
    "handed_back": {"type": "object"},
    "error_detail": {"type": "object"},
    "calls": {"type": "array"},
    "actions_executed": {"type": "integer"},
    "action_limit": {"type": "integer"},
    "wall_s": {"type": "number"},
    "workspace": {"type": "array"},
}}
FILE_RESULT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {
    "ok": {"type": "boolean"}, "path": {"type": "string"}, "content": {"type": "string"},
    "lines": {"type": "integer"}, "files": {"type": "array"}, "written": {"type": "array"}}}


def result_schemas() -> dict[str, Any]:
    return {BASH_TOOL: dict(BASH_RESULT_SCHEMA), **{name: dict(FILE_RESULT_SCHEMA) for name in FILE_TOOLS}}


__all__ = ["BASH_TOOL", "CODING_TOOLS", "EDIT_TOOL", "FILE_TOOLS", "READ_TOOL", "WRITE_TOOL", "advertised",
           "result_schemas"]
