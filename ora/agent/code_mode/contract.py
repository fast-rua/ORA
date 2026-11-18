"""Code mode in a benchmark episode (plan ``policy.code_mode``): its contract.

``policy.code_mode`` resolves to the script contract (``ora.code-mode-script.v1``)
the Supervisor's side enforces per command; the code tools (``coding.py``) are added to
the advertised ToolSurface as a derived layer (``orchestrator/spec/composition.py``), so
the plan, its context manifest and the model see the same tool list.

Code is the model's way of acting: the model keeps as tools only what a program cannot or
should not do for it (``DIRECT_TOOLS`` and the surface's lifecycle tools) and a coding
agent's tools on a persistent workspace, ``/work``, whose ``bash`` lists every other tool
as a function of the ``robot`` library with its description and arguments; the runtime still
runs them all, for programs. A program may stop and hand control back to the model
(``handoff(reason, **data)``), and it is the calculation capability: the ``compute`` tool is
withheld from the model and from programs (``WITHHELD``).

A resolved contract records this as ``FLAGS``, each one true; they stay because a plan's identity hashes the contract.
The interface ``script`` (``run_python`` and ``read_file``, a contract without ``coding_tools``) ran in none of 119
cohorts and was retired on 2026-11-01, so such a contract resolves and composes only at its launch commit, and its
sealed runs keep their recorded definitions.

``prompt`` names the generation of code mode's model-visible text: its prompt layers
(``orchestrator/spec/composition.py``) and the wording of ``bash``'s description.
``v2`` states the closed loop (perceive, compute, act and check the receipt in one script),
does not promise a robot state the session never returns, and describes the
``round_static_facts`` message only where the context composition sends it; ``v3`` rewrites the planner as
a robot-programming environment for ``library: extended`` (a procedure with trigger lines for the library's
functions, typed results and a worked script) and keeps ``v2``'s core and the layers after it. A resolved contract
records ``prompt``. One without it is ``v1``, what every code-mode run sealed before ``v2``
existed saw; ``v1`` was retired on 2026-10-31 (the v1/v2 comparison is over), so such a
contract resolves and composes only at its launch commit, and its sealed runs keep their
recorded definitions.

``library`` names what a script is given besides the tools. ``base`` is
code mode as it was before the switch, kept as the control: ``bash``'s description lists every function
with its full description and input schema (``coding._api``). ``extended`` gives the code-mode library: the description
carries an index, one line per function with its arguments' and its result's shape, and a card of the library, and a
script looks up what those do not settle (``robot.docs(name)``, ``help(tools.<name>)``), all generated (``docs.py``). A resolved
contract records ``library`` only when it is ``extended``: one without it is ``base``, so every contract resolved
before the switch, and a new ``base`` one, keeps its model-visible identity; features of the library check
``contract.get("library") == "extended"``. The query allowance (``QUERY_LIMITS``: ``max_queries_per_script``,
``query_time_s``) is a limit of that library too, recorded with it and refused with ``base``.

``source: true`` (``library: extended``) also mounts ORA's code for a script to read (``library.source_files``).
A resolved contract records it only when true, so no earlier contract changes; which code it was is the plan's commit.

``preview: true`` made a moving tool's dry run how a move happened (the script planned, then ran the plan with
``.go()``); it was retired on 2026-11-05, when the code-mode comparison kept the dry run optional (``dry_run=True`` on
any moving tool), so such a contract resolves and composes only at its launch commit, and its sealed runs keep their
recorded definitions.

``script_reads`` (``library: extended``) lists what the surface serves of ``docs.READS``: ``query``
where its tools run synchronously on the toolkit's data interface, ``dry_run`` where its moves are also planned on the
policy side, ``source`` where the contract asks for it and its tools run that way. A surface whose environment plans its
moves runs on the toolkit too and serves what it served on the handlers: not every such backend answers a ``reach_check``.
It is decided once, in resolution (``with_reads``, ``orchestrator/spec/native_policy.py``), and recorded in the contract
because three readers need it and only the contract reaches all three: the prompt layers, ``bash``'s description
(composed for the plan and again by the runtime, which must agree) and the session's ``robot.docs``. A contract records
it only where it is not empty, so a ``base`` contract never has it; what is not listed is not told to the model (the
runtime refusals stay).
"""

from __future__ import annotations

from typing import Any, Mapping

from .docs import READS

CONTRACT = "ora.code-mode-script.v1"
#: Generations of code mode's model-visible text (``prompt``); new RunSpecs get ``DEFAULT_PROMPT``.
PROMPTS = ("v2", "v3")
DEFAULT_PROMPT = "v2"
#: The library a generation's text states and needs: ``v3`` names ``robot.docs``, typed results and ``ora_lib``,
#: which a ``base`` script does not have, so it is refused there. A ``base`` contract that names no ``prompt`` gets
#: ``BASE_PROMPT``, the newest generation that needs no library, whatever ``DEFAULT_PROMPT`` is.
PROMPT_LIBRARY = {"v3": "extended"}
BASE_PROMPT = "v2"
#: What a script is given besides the tools (``library``); new RunSpecs get ``DEFAULT_LIBRARY``.
LIBRARIES = ("base", "extended")
DEFAULT_LIBRARY = "base"
#: Per-script limits. 12 world-changing calls are one pick-and-place plus one retry; the
#: wall time is the script's own: the time its actions take does not count (``uncounted_s``).
LIMITS: dict[str, int] = {
    "max_actions_per_script": 12,
    "max_calls_per_script": 24,
    "script_wall_time_s": 300,
    "script_cpu_time_s": 120,
    "script_memory_mib": 2048,
    "max_script_chars": 20000,
    "max_output_chars": 20000,
    "max_frames_per_script": 4,
}
#: The per-script query allowance of ``library: extended`` (``robot.query``, and the dry runs that share it): a count and seconds, tightened like the limits above and recorded only with that library, so a ``base`` contract
#: keeps the limits it always had. A query is a read of the environment's robot model: on RoboDojo's worker, in recent
#: runs, an ``fk`` took 0.03 s, a ``screen_path`` 0.04 s and an ``ik_chain`` (20 poses at the median, 90 at most) 0.2 s,
#: 0.8 s at the 90th percentile and 3.6 s at worst, and a ``reach_check`` about 0.45 s a pose in the planner's world, so
#: 16 poses take about 7.7 s. 180 s lets a script spend its 32 queries on about 12 poses each; a planner call may take
#: seconds. 32 is a little more than the 24 calls a script makes, so a script can check each step before it takes it
#: and still look around.
QUERY_LIMITS: dict[str, int] = {
    "max_queries_per_script": 32,
    "query_time_s": 180,
}
#: Limits that may be tightened to nothing (no frames, no queries).
ZERO_LIMITS = frozenset({"max_frames_per_script", "max_queries_per_script"})
#: Tools a script never calls: finishing is the model's own decision, the completion
#: check would turn the public contract into a search signal, and scripts do not nest.
NOT_FROM_SCRIPTS = frozenset({"check_task", "bash", "read", "write", "edit"})
#: Tools the model keeps besides the code tools and the lifecycle tools: the completion check,
#: the issue report and the context artifact reader are the model's own steps.
DIRECT_TOOLS = frozenset({"check_task", "report_issue", "artifact_read", "robot.read_artifact"})
#: Tools that, on a surface without synchronous toolkit execution, commit a skill sequence for the stage loop: a
#: script's parent runs it as one action where the worker does not cap one action's native budget (RoboTwin), and
#: resolution refuses the surface where it does (RoboDojo's closed skills, ``spec/native_policy.py``).
SEQUENCE_TOOLS = frozenset({"grasp_execute"})
SYNCHRONOUS_EXECUTION = "toolkit_synchronous_execution_v1"
#: The workspace and its bounds.
WORKSPACE_MOUNT = "/work"
WORKSPACE_MAX_BYTES = 16 * 1024 * 1024
WORKSPACE_MAX_FILES = 256
MAX_EDITS = 32
#: Tools neither the model nor a program may call (besides ``NOT_FROM_SCRIPTS``): the program itself computes.
WITHHELD = frozenset({"compute"})


def wire_name(name: str) -> str:
    return name.replace(".", "_")


def script_functions(definitions: list[Mapping[str, Any]],
                     lifecycle_tools: tuple[str, ...] = ()) -> list[Mapping[str, Any]]:
    """The definitions that become functions of the ``robot`` library: all but the model's own and ``WITHHELD``."""

    return [definition for definition in definitions
            if definition["name"] not in DIRECT_TOOLS | set(lifecycle_tools) | WITHHELD]


def prompt_generation(contract: Mapping[str, Any]) -> str:
    """The contract's ``prompt``; a resolved contract that names none is ``v1``."""

    return str(contract.get("prompt", "v1"))


def script_library(contract: Mapping[str, Any]) -> str:
    """The contract's ``library``; a resolved contract that names none is ``base``."""

    return str(contract.get("library", "base"))


def script_reads(contract: Mapping[str, Any]) -> frozenset[str]:
    """What this contract's surface serves of ``READS``."""

    return frozenset(contract.get(READS_FIELD) or ())


def with_reads(contract: dict[str, Any], *, toolkit: bool, moves: bool) -> dict[str, Any]:
    """The resolved contract with ``script_reads`` for its surface: ``toolkit``, its tools run synchronously on the
    toolkit's data interface (``query``); ``moves``, it has an action that moves the robot (``dry_run``). Recorded with
    ``library: extended`` only."""

    contract = {key: item for key, item in contract.items() if key != READS_FIELD}
    reads = [*(["dry_run"] if moves else []), *(["query"] if toolkit else []),
             *(["source"] if toolkit and contract.get("source") else [])]
    return {**contract, READS_FIELD: reads} if reads and script_library(contract) == "extended" else contract


def current_generation(contract: Mapping[str, Any]) -> str:
    """The contract's ``prompt``, refused unless this tree composes it (``PROMPTS``, the ``coding`` interface)."""

    if not contract.get("coding_tools"):
        raise ValueError(RETIRED_INTERFACE)
    prompt = prompt_generation(contract)
    if prompt not in PROMPTS:
        raise ValueError(f"policy.code_mode.prompt {prompt} was retired on 2026-10-31; a plan that names it "
                         f"composes only at its launch commit (this tree composes {', '.join(PROMPTS)})")
    return prompt


def query_allowance(contract: Mapping[str, Any]) -> tuple[int, int]:
    """The script's query allowance, ``(queries, seconds)``: the contract's, or ``QUERY_LIMITS`` where it names none."""

    return (int(contract.get("max_queries_per_script", QUERY_LIMITS["max_queries_per_script"])),
            int(contract.get("query_time_s", QUERY_LIMITS["query_time_s"])))


INTERFACES = ("coding",)
RETIRED_INTERFACE = ("policy.code_mode.interface script (run_python and read_file) was retired on 2026-11-01; a plan "
                     "that names it composes only at its launch commit and its sealed runs keep their recorded "
                     "definitions. Use interface coding, the default")
#: The retired forced preview: a contract's ``preview`` flag and the ``preview`` read it put in ``script_reads``.
RETIRED_PREVIEW = "preview"
RETIRED_PREVIEW_MESSAGE = ("policy.code_mode.preview (plans first, run with .go()) was retired on 2026-11-05; a plan "
                           "that names it composes only at its launch commit and its sealed runs keep their recorded "
                           "definitions. A dry run stays optional on every moving tool (dry_run=True)")
#: What a resolved contract records of the one interface, each one true (the module's docstring says why).
FLAGS = ("code_only", "handoff", "script_math", "workspace", "coding_tools")
#: The contract's field that lists what the surface serves of ``READS`` (``with_reads``, never a plan's to set).
READS_FIELD = "script_reads"


def resolve_contract(value: Any) -> dict[str, Any] | None:
    """``policy.code_mode`` as its contract: None when off.

    ``true`` or ``{"interface": "coding", "prompt": "v2" | "v3", "library": "base" | "extended", "source": true,
    <limits>}``:
    ``coding``, the one interface, gives the model a coding agent's tools (``coding.py``), a persistent workspace,
    ``handoff`` and code to compute in (``FLAGS``); the retired ``script`` is refused. ``prompt`` defaults to
    ``DEFAULT_PROMPT`` (``BASE_PROMPT`` for a ``base`` library that ``DEFAULT_PROMPT`` does not fit) and is recorded;
    ``v3`` needs ``library: extended``; ``library`` defaults to ``DEFAULT_LIBRARY`` and is recorded unless it is
    ``base``; ``source`` needs ``extended`` and is recorded when true. Limits may only tighten. A resolved contract (it
    names ``contract``) resolves to itself, its ``script_reads`` kept; one without ``coding_tools`` is the retired
    ``script`` and is refused, one without ``prompt`` is the retired ``v1`` and is refused, one without ``library`` is
    ``base``; ``preview``, set or read, is retired and refused.
    """

    if value is False or value is None:
        return None
    if value is True:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("policy.code_mode must be true, false or a mapping")
    reads: list[str] = []
    if "contract" in value:  # already resolved
        if value["contract"] != CONTRACT:
            raise ValueError(f"policy.code_mode.contract must be {CONTRACT}")
        requested = {key: item for key, item in value.items() if key not in (*FLAGS, "contract", READS_FIELD)}
        reads = list(value.get(READS_FIELD) or ())
        if RETIRED_PREVIEW in reads:
            raise ValueError(RETIRED_PREVIEW_MESSAGE)
        if set(reads) - set(READS):
            raise ValueError(f"policy.code_mode.{READS_FIELD} must name only {', '.join(READS)} "
                             "(inspect was retired on 2026-11-01)")
        interface = "coding" if value.get("coding_tools") else "script"
        requested["interface"] = interface
        requested["prompt"] = prompt_generation(value)
        requested["library"] = script_library(value)
        value = requested
    if RETIRED_PREVIEW in value:
        raise ValueError(RETIRED_PREVIEW_MESSAGE)
    unknown = sorted(set(value) - set(LIMITS) - set(QUERY_LIMITS) - {"interface", "prompt", "library", "source"})
    if unknown:
        raise ValueError(f"unsupported policy.code_mode keys: {', '.join(unknown)}")
    interface = value.get("interface", "coding")
    if interface == "script":
        raise ValueError(RETIRED_INTERFACE)
    if interface not in INTERFACES:
        raise ValueError(f"policy.code_mode.interface must be one of {', '.join(INTERFACES)}")
    library = value.get("library", DEFAULT_LIBRARY)
    if library not in LIBRARIES:
        raise ValueError(f"policy.code_mode.library must be one of {', '.join(LIBRARIES)}")
    prompt = value.get("prompt", DEFAULT_PROMPT if PROMPT_LIBRARY.get(DEFAULT_PROMPT, library) == library
                       else BASE_PROMPT)
    if prompt not in PROMPTS:
        raise ValueError(f"policy.code_mode.prompt must be one of {', '.join(PROMPTS)} (v1 was retired on 2026-10-31)")
    if PROMPT_LIBRARY.get(prompt, library) != library:
        raise ValueError(f"policy.code_mode.prompt {prompt} is written for library {PROMPT_LIBRARY[prompt]} (robot.docs, "
                         f"typed results, ora_lib), which library {library} does not have: set "
                         f"policy.code_mode.library to {PROMPT_LIBRARY[prompt]}, or use prompt {BASE_PROMPT} with base")
    flags = {name: value.get(name, False) for name in ("source",)}
    for name, flag in flags.items():
        if not isinstance(flag, bool):
            raise ValueError(f"policy.code_mode.{name} must be true or false")
        if flag and library != "extended":
            raise ValueError(f"policy.code_mode.{name} needs library extended, which library {library} is not")
    stray = sorted(set(value) & set(QUERY_LIMITS)) if library != "extended" else []
    if stray:
        raise ValueError(f"policy.code_mode.{stray[0]} is the query allowance of library extended (robot.query), "
                         f"which library {library} does not have")
    contract: dict[str, Any] = {"contract": CONTRACT, "code_only": True, "handoff": True, "script_math": True,
                                "workspace": True, "coding_tools": True, "prompt": prompt,
                                # base is recorded as absent, as every contract resolved before the library switch is
                                **({"library": library} if library != "base" else {}),
                                **{name: True for name, flag in flags.items() if flag}}
    for key, default in {**LIMITS, **(QUERY_LIMITS if library == "extended" else {})}.items():
        limit = value.get(key, default)
        floor = 0 if key in ZERO_LIMITS else 1
        if isinstance(limit, bool) or not isinstance(limit, int) or not floor <= limit <= default:
            raise ValueError(f"policy.code_mode.{key} must be an integer from {floor} to {default}")
        contract[key] = limit
    return {**contract, READS_FIELD: reads} if reads and library == "extended" else contract


__all__ = ["BASE_PROMPT", "CONTRACT", "DEFAULT_LIBRARY", "DEFAULT_PROMPT", "DIRECT_TOOLS", "FLAGS", "INTERFACES",
           "LIBRARIES", "LIMITS", "NOT_FROM_SCRIPTS", "PROMPTS", "PROMPT_LIBRARY", "QUERY_LIMITS", "READS_FIELD",
           "RETIRED_INTERFACE", "RETIRED_PREVIEW_MESSAGE", "WITHHELD", "WORKSPACE_MOUNT", "current_generation",
           "prompt_generation", "query_allowance", "resolve_contract", "script_functions", "script_library",
           "script_reads", "wire_name", "with_reads"]
